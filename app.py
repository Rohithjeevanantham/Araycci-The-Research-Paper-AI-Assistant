import streamlit as st
import pandas as pd
from ragpart import (generate_response_from_chunks, get_relevant_chunks, create_index,
                     extract_text_from_pdf, clean_text, store_chunks_in_pinecone,
                     combined_chunking, build_bm25, get_embedding_model, get_reranker,
                     new_namespace, clear_namespace)
from translate import translate, generate_audio
from arxiv import search_arxiv, process_docs2, clustering, text_from_file_uploader, tokenize_text

# Initialize session state
if 'index' not in st.session_state:
    st.session_state.index = None
if 'search' not in st.session_state:
    st.session_state.search = []
if 'query' not in st.session_state:
    st.session_state.query = None
if 'download' not in st.session_state:
    st.session_state.download = False
if 'papers_downloaded' not in st.session_state:
    st.session_state.papers_downloaded = False
if 'result_df' not in st.session_state:
    st.session_state.result_df = None
if 'fig' not in st.session_state:
    st.session_state.fig = None
if 'selected_cluster' not in st.session_state:
    st.session_state.selected_cluster = None
if 'selected_indices' not in st.session_state:
    st.session_state.selected_indices = []
if 'cluster' not in st.session_state:
    st.session_state.cluster = None
# Chunk corpus and its BM25 index: the sparse half of hybrid retrieval scores
# against these, so they have to outlive the indexing step.
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'bm25' not in st.session_state:
    st.session_state.bm25 = None
if 'select_all' not in st.session_state:
    st.session_state.select_all = False
# Each session writes its vectors into its own Pinecone namespace, so users
# don't overwrite each other and cleanup is a fast namespace delete.
if 'namespace' not in st.session_state:
    st.session_state.namespace = None

def reset_page():
    # Drop this session's vectors first -- this needs the index and namespace,
    # which the lines below are about to clear.
    if st.session_state.index is not None and st.session_state.namespace:
        clear_namespace(st.session_state.index, st.session_state.namespace)

    st.session_state.index = None
    st.session_state.namespace = None
    st.session_state.search = []
    st.session_state.query = None
    st.session_state.papers_downloaded = False
    st.session_state.result_df = None
    st.session_state.fig = None
    st.session_state.selected_cluster = None
    st.session_state.selected_indices = []
    st.session_state.cluster = None
    st.session_state.chunks = None
    st.session_state.bm25 = None

    # Widget keys must be deleted, not assigned: reset_page() can run from the
    # "End conversation" button, which renders *after* these checkboxes, and
    # Streamlit forbids writing to a widget's state once it is instantiated.
    # Deleting the key resets the widget to its default on the next run.
    st.session_state.pop("select_all", None)
    # Only the numbered checkbox keys (selected_0, selected_1, ...) -- not
    # "selected_indices", which is ordinary state the app still reads.
    stale = [k for k in st.session_state
             if k.startswith("selected_") and k.split("_", 1)[1].isdigit()]
    for key in stale:
        st.session_state.pop(key, None)

# Streamlit app
st.sidebar.image("logo.jpg")
st.title("Araycci Research Paper Bot")
st.sidebar.title("PDF Research Assistant")

lang = st.sidebar.radio("Choose", ["English", "French", "Spanish"])

Source = st.radio(
    "Pick Source of Papers",
    ["Local", "Web"],
    index=0,
    on_change=reset_page
)


# Load the models now, while the user is still choosing papers, rather than
# making them wait for it once indexing starts. Both are cached, so this is a
# no-op on every rerun after the first.
with st.spinner("Warming up models (first run downloads ~150 MB)..."):
    get_embedding_model()
    get_reranker()



@st.cache_data(show_spinner=False)
def chunks_for_text(text):
    """Chunking keyed on the text itself, so re-indexing the same papers (e.g.
    after picking a different cluster) skips the work entirely."""
    return combined_chunking(clean_text(text))


def process_local_pdfs(data):
    combined_chunks = []

    # Check if data is a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_dict()
        data = data['text']

    # If data is a list of uploaded files
    for pdf_file in data:
        if isinstance(data, dict) and isinstance(data[pdf_file], str):
            text = data[pdf_file]
        else:
            text = extract_text_from_pdf(pdf_file)

        combined_chunks.extend(chunks_for_text(text))

    return combined_chunks


def index_chunks(combined_chunks):
    """Index chunks for both halves of hybrid retrieval: dense vectors into
    Pinecone, and the chunk corpus + BM25 index into the session.

    Indexing takes tens of seconds (model load, then ~30-40 ms of embedding per
    chunk), so each stage reports progress rather than hiding behind one spinner.
    """
    n = len(combined_chunks)
    status = st.status(f"Indexing {n} chunks...", expanded=True)

    with status:
        st.write("Connecting to Pinecone...")
        st.session_state.index = create_index()
        if not st.session_state.index:
            status.update(label="Indexing failed", state="error")
            st.error("Failed to create Pinecone index.")
            return

        # Reuse this session's namespace, clearing any vectors from a previous
        # indexing run so stale chunks can't be retrieved.
        if st.session_state.namespace:
            clear_namespace(st.session_state.index, st.session_state.namespace)
        else:
            st.session_state.namespace = new_namespace()

        st.write(f"Embedding and uploading {n} chunks...")
        bar = st.progress(0.0)
        store_chunks_in_pinecone(
            combined_chunks,
            st.session_state.index,
            progress=lambda frac: bar.progress(frac, text=f"{int(frac * n)} / {n} chunks"),
            namespace=st.session_state.namespace,
        )

        st.write("Building BM25 keyword index...")
        st.session_state.chunks = combined_chunks
        st.session_state.bm25 = build_bm25(combined_chunks)

    st.session_state.papers_downloaded = True
    status.update(label=f"Indexed {n} chunks. Ready for questions.", state="complete")
    st.success("PDF processed and indexed successfully!")


def download_and_process_arxiv(selection, arxiv_results):
    zip_file = process_docs2(selection, arxiv_results)
    st.download_button(
        label="Download ZIP",
        data=zip_file,
        file_name="pdfs.zip",
        mime="application/zip"
    )

def handle_query_response(query, lang):
    relevant_chunks = get_relevant_chunks(
        query,
        st.session_state.index,
        st.session_state.chunks,
        st.session_state.bm25,
        namespace=st.session_state.namespace,
    )
    response = generate_response_from_chunks(relevant_chunks, query)
    if lang != "English":
        translated_response = translate(response, lang, True)
        st.write(translated_response)
        audio_io = generate_audio(translated_response, lang)
    else:
        st.write(response)
        audio_io = generate_audio(response, lang)
    st.audio(audio_io, format='audio/mp3')
    st.download_button(label="Download Audio Response", data=audio_io, file_name="response.mp3", mime="audio/mp3")

# Handle Local PDF Processing
if Source == "Local":
    data = st.sidebar.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=True)
    if data and not st.session_state.papers_downloaded:
        
        if st.toggle("Cluster By Similarity", value=True):
            pdf_texts = text_from_file_uploader(data)
            processed_documents = tokenize_text(pdf_texts)
            result_df, fig = clustering(pdf_texts, processed_documents)
            if fig != "Error":
                st.pyplot(fig)
                st.write(result_df)
                selected_cluster = st.text_input("Enter Cluster number")
                if st.button("Process Cluster") and selected_cluster:
                    st.write(f"Processing cluster: {selected_cluster}")
                    selected_cluster = int(selected_cluster)
                    result_df = result_df[result_df['Cluster'] == selected_cluster]

                    with st.spinner("Processing PDFs..."):
                        index_chunks(process_local_pdfs(result_df))

            else:
                st.write("Too Few Papers for Clustering")

        else:
            with st.spinner("Processing PDFs..."):
                index_chunks(process_local_pdfs(data))

def toggle_select_all():
    """Tick/untick every paper to match the 'Select all' box. Runs as an
    on_change callback, i.e. before the rerun that rebuilds the checkboxes --
    Streamlit forbids writing to a widget's state after it is instantiated."""
    for i in range(len(st.session_state.search)):
        st.session_state[f"selected_{i}"] = st.session_state.select_all


# Handle Web Search and Download
if Source == "Web":
    search = st.text_input("Enter the search query: ")
    max_results = st.slider("Maximum results:", 10, 100)
    if st.button("Search"):
        # Drop the previous search's checkbox state, or stale ticks carry over
        # onto the new results.
        for i in range(len(st.session_state.search)):
            st.session_state.pop(f"selected_{i}", None)
        st.session_state.select_all = False
        st.session_state.search = search_arxiv(search, max_results)
        st.session_state.selected_indices = []  # Reset selection on new search
        st.session_state.download = False

    if st.session_state.search:
        arxiv_results = st.session_state.search
        selection = {}

        st.checkbox(
            f"Select all {len(arxiv_results)} results",
            key="select_all",
            on_change=toggle_select_all,
        )

        for i, result in enumerate(arxiv_results):
            st.subheader(f"{i+1}. {result['title']} ({result['published']})")
            st.write(f"**Authors:** {', '.join(result['authors'])}")
            st.write(f"**Summary:** {result['summary']}")
            st.write(f"**Link:** [arXiv Paper]({result['link']})")

            selection[f"selected_{i}"] = st.checkbox("Download Paper", key=f"selected_{i}")

        selected_indices = [i for i in range(len(arxiv_results)) if selection[f"selected_{i}"]]

        if st.button("Download Selection"):
            st.session_state.download = True
            st.session_state.selected_indices = selected_indices
            st.write(f"Selected indices: {st.session_state.selected_indices}")

        if st.session_state.download and st.session_state.selected_indices:
            if not st.session_state.papers_downloaded:
                with st.spinner("Downloading and processing papers..."):
                    download_and_process_arxiv(st.session_state.selected_indices, arxiv_results)
                st.success('Files Zipped And Ready To Download')
                st.session_state.papers_downloaded = True
            else:
                st.write("You May Now Switch To Local To Proceed")
              
# Query handling
if st.session_state.index:
    query = st.text_input("Enter your question:")
    if query:
        if lang!="English":
            translated_query = translate(query, lang, False)
            st.session_state.query = translated_query
        else:
            st.session_state.query = query
    if st.button("Ask") and st.session_state.query:
        with st.spinner("Searching for answers..."):
            handle_query_response(st.session_state.query, lang)
        
    if st.button("End conversation"):
        reset_page()
        st.rerun()
