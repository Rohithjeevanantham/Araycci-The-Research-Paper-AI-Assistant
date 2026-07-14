# Araycci - The Research Paper AI Assistant
## Overview

Araycci Research Paper AI Assistant is an interactive research assistant built using Streamlit. This application allows users to process and cluster local PDF files, search and download research papers from ArXiv, and ask questions about the content. The bot provides answers based on the processed text and supports translation and audio generation for responses.

## Features

- **Local PDF Processing**: Upload and process local PDF files.
- **Web Search**: Search for research papers on ArXiv and download them.
- **Clustering**: Cluster text content for better organization and analysis.
- **Retrieval-Augmented Generation (RAG)**: Answer queries based on the processed text.
- **Translation**: Translate responses into English, French, or Spanish.
- **Audio Generation**: Generate audio responses for the translated text.

## Installation

### Prerequisites

- Python 3.8 or higher
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
    Create `.streamlit/secrets.toml` (gitignored) with:
    ```toml
    [general]
    PINECONE_API_KEY = "your-pinecone-api-key"
    HUGGINGFACE_TOKEN = "your-huggingface-token"
    ```
    - **Pinecone API Key**: used for the vector index storing document chunks.
    - **Hugging Face Token**: used to query `meta-llama/Llama-3.1-8B-Instruct` via Hugging Face Inference Providers (the token needs access to the Llama model, which requires accepting Meta's license on the model page).

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
3. **View and Select Papers**: View the search results and select papers to download.
4. **Download and Process**: Download and process the selected papers. Processed text will be stored for query-based retrieval.

### Query Handling

1. **Ask a Question**: Enter your question in the input box.
2. **Generate Response**: The bot will retrieve relevant text chunks and generate a response.
3. **Translation and Audio Generation**: If a language other than English is selected, the response will be translated and an audio file will be generated.
4. **Download Audio**: Download the audio response if desired.

## Modules

- **ragpart.py**: Handles retrieval-augmented generation, chunking, and text processing. Answers are generated with `meta-llama/Llama-3.1-8B-Instruct` via Hugging Face Inference Providers.
  - `generate_response_from_chunks`: Generates responses based on relevant text chunks.
  - `get_relevant_chunks`: Retrieves relevant chunks for a given query.
  - `create_index`: Creates a Pinecone index.
  - `extract_text_from_pdf`: Extracts text from PDF files.
  - `clean_text`: Cleans the extracted text.
  - `store_chunks_in_pinecone`: Stores chunks in Pinecone.
  - `combined_chunking`: Performs advanced chunking to prevent context loss.

- **translate.py**: Provides translation and text-to-speech functionalities.
  - `translate`: Translates text into the specified language.
  - `generate_audio`: Generates audio from text in the specified language.

- **arxiv.py**: Handles searching, downloading, and processing ArXiv papers.
  - `search_arxiv`: Searches ArXiv for papers based on a query.
  - `process_docs2`: Processes and downloads selected ArXiv papers.
  - `clustering`: Clusters the text content for better organization.
  - `text_from_file_uploader`: Extracts text from uploaded files.
  - `tokenize_text`: Tokenizes the text for clustering.

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
- [Hugging Face](https://huggingface.co/): Provides models and tools for NLP.
- [Pinecone](https://www.pinecone.io/): Vector database service used for storing and retrieving text chunks.
- [ArXiv](https://arxiv.org/): Source of research papers.

---

*Happy researching!*
