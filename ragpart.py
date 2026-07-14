import fitz  # PyMuPDF
import os
import re
import uuid
import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
import sys
import json

pinecone_environment = "us-east-1"

# Pinecone index name
index_name = "llama3"

# LLM used for answering queries (served via Hugging Face Inference Providers)
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Reranker applied to the fused dense+sparse candidates
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Chunking. all-MiniLM-L6-v2 truncates at 256 tokens and arXiv text runs about
# 1.5 tokens/word, so a chunk must stay under ~170 words or the tail of it is
# never embedded and becomes invisible to dense search. Do not raise these
# without also switching to an embedding model with a longer context window.
CHUNK_WORDS = 150
CHUNK_OVERLAP_WORDS = 30

# Hybrid retrieval settings
DENSE_TOP_K = 20    # candidates pulled from Pinecone
SPARSE_TOP_K = 20   # candidates pulled from BM25
RRF_K = 60          # Reciprocal Rank Fusion damping constant
FINAL_TOP_N = 8     # chunks handed to the LLM after reranking

EMBED_BATCH_SIZE = 64


def get_secret(key):
    try:
        value = st.secrets["general"][key]
    except (KeyError, FileNotFoundError):
        value = None
    if not value:
        st.error(
            f"Missing `{key}`. Add it to `.streamlit/secrets.toml` under the "
            "`[general]` section."
        )
        st.stop()
    return value


@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=get_secret("PINECONE_API_KEY"))


@st.cache_resource
def get_embedding_model():
    # Torch defaults to half the cores; using all of them is ~1.2x faster and
    # embedding is the slowest part of indexing.
    torch.set_num_threads(os.cpu_count() or 1)
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)


def create_index():
    """Get (or create) the shared index and return it with a fresh namespace.

    Creating a serverless index costs ~10s, and the old code paid that on every
    single indexing run by deleting and recreating it. The index is now created
    once and reused; each session writes into its own namespace instead, which
    clears in ~0.3s. Namespaces also isolate concurrent users -- previously two
    people indexing at the same time would wipe each other's vectors.
    """
    pc = get_pinecone_client()
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_environment
            )
        )
    return pc.Index(index_name)


def new_namespace():
    return f"session-{uuid.uuid4().hex[:12]}"


def clear_namespace(index, namespace):
    """Drop a session's vectors. Safe to call on a namespace that never existed."""
    if not namespace:
        return
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception:
        # Pinecone 404s when the namespace has no vectors; nothing to clean up.
        pass

def extract_text_from_pdf(pdf_file):
    if isinstance(pdf_file, str):
        doc = fitz.open(pdf_file)  # Open the PDF file using the file path
    else:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def title_based_chunking(text):
    chunks = re.split(r'(?<=\n)\s*(?=\w)', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def section_based_chunking(text):
    sections = re.split(r'\n\s*\n', text)  # Split by blank lines
    return [section.strip() for section in sections if section.strip()]

def semantic_chunking(text, max_chunk_size=CHUNK_WORDS, overlap=CHUNK_OVERLAP_WORDS):
    """Split text into overlapping word windows sized to fit the embedding
    model's 256-token window. arXiv prose runs ~1.5 tokens/word, so 150 words
    is ~220 tokens. Oversized chunks are silently truncated by the model, which
    would leave most of a chunk unrepresented in its own embedding."""
    words = text.split()
    if not words:
        return []

    step = max_chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        window = words[i:i + max_chunk_size]
        if not window:
            break
        chunks.append(' '.join(window))
        if i + max_chunk_size >= len(words):
            break  # this window reached the end; a further step would repeat text
    return chunks

def enforce_token_limit(chunks):
    """Halve any chunk that still exceeds the embedding model's token window.

    A word budget alone is not enough: equations, citations and long identifiers
    tokenize far denser than prose (up to ~2.5 tokens/word), so some 150-word
    chunks still overflow. Anything past the window is dropped by the model and
    would be invisible to dense search, so split those chunks instead.
    """
    model = get_embedding_model()
    tokenizer, limit = model.tokenizer, model.max_seq_length

    result = []
    queue = list(chunks)
    while queue:
        chunk = queue.pop(0)
        words = chunk.split()
        # Stop splitting tiny chunks even if they somehow still tokenize long,
        # otherwise a pathological chunk could recurse forever.
        if len(words) <= 20 or len(tokenizer.encode(chunk)) <= limit:
            result.append(chunk)
        else:
            mid = len(words) // 2
            queue.insert(0, ' '.join(words[mid:]))
            queue.insert(0, ' '.join(words[:mid]))
    return result


def combined_chunking(text):
    title_chunks = title_based_chunking(text)
    final_chunks = []
    for chunk in title_chunks:
        section_chunks = section_based_chunking(chunk)
        for section_chunk in section_chunks:
            semantic_chunks = semantic_chunking(section_chunk)
            final_chunks.extend(semantic_chunks)
    # Must run before the chunk list is used for BM25 and Pinecone, since both
    # index by position in this list.
    return enforce_token_limit(final_chunks)

def bm25_tokenize(text):
    # Shared by corpus and query so BM25 scores both sides the same way.
    return re.findall(r"\w+", text.lower())


def build_bm25(chunks):
    return BM25Okapi([bm25_tokenize(chunk) for chunk in chunks])


def embed_chunks(chunks, progress=None):
    """Embed in batches so the caller can report progress. Encoding is the slow
    part of indexing (roughly 30-40 ms per chunk on CPU)."""
    model = get_embedding_model()
    embeddings = []
    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start:start + EMBED_BATCH_SIZE]
        embeddings.extend(model.encode(batch, batch_size=EMBED_BATCH_SIZE))
        if progress:
            progress(min(1.0, (start + len(batch)) / len(chunks)))
    return embeddings


def store_chunks_in_pinecone(chunks, index, max_batch_size_mb=2, progress=None, namespace=None):
    chunk_embeddings = embed_chunks(chunks, progress=progress)
    # "idx" is the chunk's position in the corpus list, which is what lets dense
    # hits be fused with BM25 hits by position.
    vectors = [{"id": f"chunk-{i}", "values": embedding.tolist(),
                "metadata": {"content": chunk, "type": "chunk", "idx": i}}
               for i, (embedding, chunk) in enumerate(zip(chunk_embeddings, chunks))]

    # Split vectors into batches that are under the maximum batch size
    max_batch_size_bytes = max_batch_size_mb * 1024 * 1024
    current_batch = []
    current_batch_size = 0

    for vector in vectors:
        vector_size = sys.getsizeof(json.dumps(vector))
        if current_batch_size + vector_size > max_batch_size_bytes:
            index.upsert(vectors=current_batch, namespace=namespace)
            current_batch = [vector]
            current_batch_size = vector_size
        else:
            current_batch.append(vector)
            current_batch_size += vector_size

    if current_batch:
        index.upsert(vectors=current_batch, namespace=namespace)

def dense_search(query, index, top_k=DENSE_TOP_K, namespace=None):
    """Semantic search via Pinecone. Returns corpus indices, best first."""
    query_embedding = get_embedding_model().encode([query])[0].tolist()
    search_results = index.query(vector=query_embedding, top_k=top_k,
                                 include_metadata=True, namespace=namespace)
    return [int(match['metadata']['idx']) for match in search_results['matches']]


def sparse_search(query, bm25, top_k=SPARSE_TOP_K):
    """Keyword search via BM25. Returns corpus indices, best first."""
    scores = bm25.get_scores(bm25_tokenize(query))
    ranked = np.argsort(scores)[::-1][:top_k]
    return [int(i) for i in ranked if scores[i] > 0]


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """Merge ranked lists of corpus indices into one, scoring each entry by
    1/(k + rank). Rank position is used rather than the underlying scores,
    since cosine similarity and BM25 scores are not on a comparable scale."""
    fused_scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores, key=fused_scores.get, reverse=True)


def get_relevant_chunks(query, index, chunks, bm25, top_n=FINAL_TOP_N, namespace=None):
    """Hybrid retrieval: dense + BM25 candidates, fused with RRF, then reranked
    by a cross-encoder down to the chunks worth sending to the LLM."""
    dense_hits = dense_search(query, index, namespace=namespace)
    sparse_hits = sparse_search(query, bm25) if bm25 is not None else []

    candidate_indices = reciprocal_rank_fusion([dense_hits, sparse_hits])
    candidates = [chunks[i] for i in candidate_indices if 0 <= i < len(chunks)]
    if not candidates:
        return []

    ranked = get_reranker().rank(query, candidates, top_k=top_n, return_documents=True)
    return [result['text'] for result in ranked]

def generate_response_from_chunks(chunks, query):
    combined_content = "\n".join([f"Chunk:\n{chunk}" for chunk in chunks])
    prompt_template = (
        "You are an AI research assistant. Your job is to help users understand and extract key insights from research papers. "
        "You will be given a query and context from multiple research papers. Based on this information, provide accurate, concise, and helpful responses. "
        "Here is the context from the research papers and the user's query:\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Please provide a detailed and informative response based on the given context. Make sure your response is complete and ends with 'End of response.'."
    )
    user_query = prompt_template.format(context=combined_content, query=query)
    
    huggingface_token = get_secret("HUGGINGFACE_TOKEN")
    client = InferenceClient(model=LLM_MODEL, token=huggingface_token)

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": user_query}],
            max_tokens=500,
            stream=False
        )
    except HfHubHTTPError as e:
        st.error(f"Hugging Face inference request failed: {e}")
        return "No response received."

    if response.choices:
        content = response.choices[0].message.content
        if 'End of response.' in content:
            content = content.split('End of response.')[0].strip()
        return content
    else:
        return "No response received."
