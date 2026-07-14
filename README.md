# Araycci - The Research Paper AI Assistant
## Overview

Araycci Research Paper AI Assistant is an interactive research assistant built using Streamlit. This application allows users to process and cluster local PDF files, search and download research papers from ArXiv, and ask questions about the content. The bot provides answers based on the processed text and supports translation and audio generation for responses.

## Features

- **Local PDF Processing**: Upload and process local PDF files.
- **Web Search**: Search for research papers on ArXiv and download them.
- **Clustering**: Cluster papers by similarity (TF-IDF + KMeans) so you can index one topic at a time.
- **Hybrid Retrieval-Augmented Generation (RAG)**: Dense vector search *and* BM25 keyword search, fused with Reciprocal Rank Fusion and reranked by a cross-encoder — see [How retrieval works](#how-retrieval-works).
- **Translation**: Translate responses into English, French, or Spanish.
- **Audio Generation**: Generate audio responses for the translated text.

## How retrieval works

A query is answered by four stages rather than a single vector lookup:

```
query ──┬─ dense  : MiniLM embedding → Pinecone            → top 20 chunks
        └─ sparse : BM25 over the session's chunk corpus   → top 20 chunks
                              │
                    Reciprocal Rank Fusion (k=60)
                              │
                  cross-encoder rerank → top 5 chunks
                              │
                        Llama 3.1 8B Instruct
```

**Why hybrid?** Dense (semantic) search alone misses exact tokens — author names, method names, acronyms, numeric results — which are exactly what research-paper questions hinge on. BM25 catches those literal matches, while dense search catches paraphrases. Fusion is done with Reciprocal Rank Fusion because cosine similarity and BM25 scores live on different scales and cannot be blended directly; RRF combines them by *rank position* instead, so no tuning constant is needed.

**Why rerank?** BM25 and vector search are recall-oriented — they always return *something*. The cross-encoder reads each candidate chunk together with the query (rather than comparing precomputed embeddings) and is far more accurate at judging relevance, so it filters the ~40 fused candidates down to the 5 worth spending LLM context on.

Models used (all defined at the top of `ragpart.py`):

| Role | Model |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (384-dim) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Answer generation | `meta-llama/Llama-3.1-8B-Instruct` |

Retrieval depth (`DENSE_TOP_K`, `SPARSE_TOP_K`, `FINAL_TOP_N`) is tunable via constants in `ragpart.py`.

## Installation

### Prerequisites

- Python 3.9 or higher (3.10+ recommended; `huggingface_hub` v1 dropped 3.8)
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository**:
    ```sh
    git clone https://github.com/rohithjeevanantham/araycci-the-research-paper-ai-assistant.git
    cd araycci-the-research-paper-ai-assistant
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up your secrets**:
    Create `.streamlit/secrets.toml` with a `[general]` section (the section header is required — the app reads `st.secrets["general"][...]`):
    ```toml
    [general]
    PINECONE_API_KEY = "your-pinecone-api-key"
    HUGGINGFACE_TOKEN = "hf_your-huggingface-token"
    ```
    - **Pinecone API Key**: used for the vector index storing document chunks.
    - **Hugging Face Token**: a **read** token is sufficient — the app only runs inference and downloads public models, it never writes to the Hub.

    > **Llama 3.1 is a gated model.** A valid token is not enough on its own: log in to Hugging Face, open the [model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and accept Meta's license via the banner at the top of the page ("You need to agree to share your contact information to access this model") using the same account that owns your token. Until access is granted you will get a `403` that looks like an auth error but is really a gating error. If you use a *fine-grained* token instead of a classic read token, it must also have the "Make calls to Inference Providers" permission enabled.
    >
    > To use an ungated model instead, change `LLM_MODEL` in `ragpart.py` (e.g. to `Qwen/Qwen2.5-7B-Instruct`) — the rest of the pipeline is model-agnostic.

    > **⚠️ Never commit `secrets.toml`.** It is listed in `.gitignore`, but `.gitignore` has no effect on files git already tracks. Verify with `git ls-files .streamlit/` — if the file is listed, untrack it with `git rm --cached .streamlit/secrets.toml` (this keeps your local copy) and rotate any keys that were previously pushed.

## Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Open the provided URL** in your browser to access the app.

## App Functionality

### Sidebar

- **Choose Language**: Select the language for responses (English, French, Spanish).
- **Pick Source of Papers**: Choose between processing local PDFs or searching the web (ArXiv).

### Local PDF Processing

1. **Upload PDF Files**: Use the file uploader to select one or more PDF files from your local system.
2. **Cluster By Similarity (optional)**: Toggle this option to cluster the text content by similarity for better organization.
3. **View Clusters and Select**: If clustering is enabled, view the clusters and select one for processing.
4. **Process and Index**: The selected cluster or entire PDFs will be processed. The text will be chunked and stored in Pinecone for query-based retrieval.

### Web Search

1. **Enter Search Query**: Input a search query for ArXiv papers.
2. **Set Maximum Results**: Adjust the slider to set the maximum number of search results.
3. **View and Select Papers**: View the search results and tick the papers you want.
4. **Download**: The selected papers are downloaded and bundled into a ZIP.

> **Note:** The Web tab *fetches* papers, it does not index them. Unzip the downloaded PDFs, then switch to **Local** and upload them to make them queryable. The app prompts you with "You May Now Switch To Local To Proceed".

### Query Handling

Available once papers have been indexed via the **Local** tab.

1. **Ask a Question**: Enter your question in the input box and press **Ask**.
2. **Generate Response**: Hybrid retrieval fetches the most relevant chunks and Llama 3.1 answers from them.
3. **Translation and Audio Generation**: If a language other than English is selected, your question is translated to English for retrieval and the answer is translated back, then read aloud.
4. **Download Audio**: Download the spoken answer as an MP3.
5. **End conversation**: Clears the index, the chunk corpus, and the BM25 index, resetting the session.

## Modules

- **ragpart.py**: Handles hybrid retrieval-augmented generation, chunking, and text processing. Answers are generated with `meta-llama/Llama-3.1-8B-Instruct` via Hugging Face Inference Providers.
  - `generate_response_from_chunks`: Generates responses based on relevant text chunks.
  - `get_relevant_chunks`: Hybrid retrieval — runs `dense_search` and `sparse_search`, fuses them with `reciprocal_rank_fusion`, then reranks with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`).
  - `dense_search`: Semantic search over the Pinecone vector index.
  - `sparse_search`: BM25 keyword search over the session's chunk corpus.
  - `reciprocal_rank_fusion`: Merges the dense and sparse rankings without needing comparable scores.
  - `build_bm25`: Builds the BM25 index from the chunk corpus.
  - `create_index`: Creates a Pinecone index.
  - `extract_text_from_pdf`: Extracts text from PDF files.
  - `clean_text`: Cleans the extracted text.
  - `store_chunks_in_pinecone`: Stores chunks in Pinecone.
  - `combined_chunking`: Performs advanced chunking to prevent context loss.

- **app.py**: The Streamlit UI and session flow.
  - `index_chunks`: Indexes a chunk corpus for *both* halves of hybrid retrieval — dense vectors into Pinecone, and the chunk list plus its BM25 index into `st.session_state`.
  - `process_local_pdfs`: Extracts and chunks uploaded PDFs (accepts either raw uploads or a clustered DataFrame).
  - `handle_query_response`: Runs retrieval, generation, translation, and audio.
  - `reset_page`: Clears all session state, including the chunk corpus and BM25 index.

- **translate.py**: Provides translation and text-to-speech functionalities.
  - `translate`: Translates text between English and the selected language.
  - `generate_audio`: Generates MP3 audio from text via gTTS.

- **arxiv.py**: Handles searching, downloading, and clustering ArXiv papers.
  - `search_arxiv`: Searches ArXiv for papers based on a query.
  - `process_docs2`: Downloads the selected papers and bundles them into a ZIP.
  - `clustering`: TF-IDF + KMeans clustering, with `k` chosen by silhouette score and a PCA scatter plot of the result.
  - `text_from_file_uploader`: Extracts text from uploaded files.
  - `tokenize_text`: Tokenizes and removes stopwords, for clustering.

> **Note:** the chunk corpus lives in `st.session_state` because BM25 scores against local text — unlike dense search, it cannot query Pinecone. `index_chunks` and `reset_page` keep the Pinecone index, the corpus, and the BM25 index in lockstep.

## Contributing

We welcome contributions to improve Arayacci Research Paper Bot. To contribute:

1. **Fork the repository**.
2. **Create a new branch**:
    ```sh
    git checkout -b feature-branch
    ```
3. **Make your changes** and commit them:
    ```sh
    git commit -am 'Add new feature'
    ```
4. **Push to the branch**:
    ```sh
    git push origin feature-branch
    ```
5. **Create a new Pull Request**.

## Team

The Araycci Research Paper AI Assistant was developed by:

- **Ananth Shyam**
- **Rohith Jeevanantham**
- **Samyuktha**
- **Arush Ajith**
- **Aditya V**
- **Avinash M**


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/): The web framework used for building the app.
- [Hugging Face](https://huggingface.co/): Inference Providers for Llama 3.1, plus the embedding and reranker models.
- [Sentence Transformers](https://sbert.net/): Bi-encoder embeddings and the cross-encoder reranker.
- [Pinecone](https://www.pinecone.io/): Vector database used for dense retrieval.
- [rank_bm25](https://github.com/dorianbrown/rank_bm25): BM25 implementation powering sparse retrieval.
- [ArXiv](https://arxiv.org/): Source of research papers.

---

*Happy researching!*
