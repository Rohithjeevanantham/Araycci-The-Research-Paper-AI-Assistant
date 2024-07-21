import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient

# Initialize Pinecone
pinecone_api_key = "b887f4da-c8c8-4e25-954b-1c0c15df7312"
pinecone_environment = "us-east-1"
pc = Pinecone(api_key=pinecone_api_key)

# Model initialization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Pinecone index name
index_name = "llama3"

def create_index():
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
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

def semantic_chunking(text, max_chunk_size=512, overlap=128):
    words = text.split(' ')
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + max_chunk_size])
        if i > 0:
            previous_chunk = ' '.join(words[max(0, i - overlap):i])
            chunk = previous_chunk + ' ' + chunk
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks

def combined_chunking(text):
    title_chunks = title_based_chunking(text)
    final_chunks = []
    for chunk in title_chunks:
        section_chunks = section_based_chunking(chunk)
        for section_chunk in section_chunks:
            semantic_chunks = semantic_chunking(section_chunk)
            final_chunks.extend(semantic_chunks)
    return final_chunks

def store_chunks_in_pinecone(chunks, index):
    chunk_embeddings = model.encode(chunks)
    vectors = [{"id": f"chunk-{i}", "values": embedding.tolist(), "metadata": {"content": chunk, "type": "chunk"}}
               for i, (embedding, chunk) in enumerate(zip(chunk_embeddings, chunks))]
    index.upsert(vectors)

def get_relevant_chunks(query, index, top_k=5):
    query_embedding = model.encode([query])[0].tolist()
    search_results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    chunks = [result['metadata']['content'] for result in search_results['matches']]
    return chunks

def generate_response_from_chunks(chunks, query):
    combined_content = "\n".join([f"Chunk:\n{chunk}" for chunk in chunks])
    prompt_template = (
        "You are an AI research assistant. Your job is to help users understand and extract key insights from research papers. "
        "You will be given a query and context from multiple research papers. Based on this information, provide accurate, concise, and helpful responses. "
        "Here is the context from the research papers and the user's query:\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}\n\n"
        "Please provide a detailed and informative response based on the given context."
    )
    user_query = prompt_template.format(context=combined_content, query=query)
    client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token="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")
    
    # Token limit constants
    max_total_tokens = 8192
    
    # Calculate the number of input tokens
    input_tokens = len(user_query.split())
    
    # Calculate the remaining tokens for output generation
    max_new_tokens = max_total_tokens - input_tokens
    
    if max_new_tokens <= 0:
        return "Input text is too long. Please reduce the size of the input."
    
    response = client.chat_completion(
        messages=[{"role": "user", "content": user_query}], 
        max_tokens=max_new_tokens, 
        stream=False
    )
    
    return response['choices'][0]['message']['content'] if response['choices'] else "No response received."



def process_pdfs(pdf_files, query, index):
    nested_texts = []

    # Extract and clean text from each PDF, store in a nested list
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        cleaned_text = clean_text(text)
        nested_texts.append(cleaned_text)

    # Chunk each list of texts and store in Pinecone
    for text in nested_texts:
        chunks = combined_chunking(text)
        store_chunks_in_pinecone(chunks, index)

    relevant_chunks = get_relevant_chunks(query, index)
    response = generate_response_from_chunks(relevant_chunks, query)
    return response