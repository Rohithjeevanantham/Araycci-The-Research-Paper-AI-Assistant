import io
import zipfile

import streamlit as st
import pandas as pd
from ragpart import (generate_response_from_chunks, get_relevant_chunks, create_index,
                     extract_text_from_pdf, clean_text, store_chunks_in_pinecone,
                     combined_chunking, build_bm25, get_embedding_model, get_reranker,
                     new_namespace, clear_namespace, condense_query)
from storage import save_corpus, load_corpus, delete_corpus
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
# Each session writes its vectors into its own Pinecone namespace, so users
# don't overwrite each other and cleanup is a fast namespace delete.
if 'namespace' not in st.session_state:
    st.session_state.namespace = None
# Conversation so far. Each turn holds the English question/answer (what the LLM
# is given) plus what was shown to the user, which differ when a language other
# than English is selected.
if 'history' not in st.session_state:
    st.session_state.history = []

def reset_page():
    # Drop this session's vectors and its persisted corpus first -- both need
    # the namespace, which the lines below are about to clear.
    if st.session_state.index is not None and st.session_state.namespace:
        clear_namespace(st.session_state.index, st.session_state.namespace)
    if st.session_state.namespace:
        try:
            delete_corpus(st.session_state.namespace)
        except Exception as e:
            # st.toast, not st.warning: this runs as a widget callback and just
            # before a forced rerun, and ordinary output written there has no
            # stable place to land. Toasts are queued across the rerun.
            st.toast(f"Could not remove the saved corpus for this session ({e}).", icon="⚠️")

    # Drop the resume token too, or reopening the old URL would point at a
    # namespace whose corpus and vectors were just deleted.
    st.query_params.pop("ns", None)

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
    st.session_state.history = []

    # Widget keys must be deleted, not assigned: reset_page() can run from the
    # "End conversation" button, which renders *after* these checkboxes, and
    # Streamlit forbids writing to a widget's state once it is instantiated.
    # Deleting the key resets the widget to its default on the next run.
    # Only the numbered checkbox keys (selected_0, selected_1, ...) -- not
    # "selected_indices", which is ordinary state the app still reads.
    stale = [k for k in st.session_state
             if k.startswith("selected_") and k.split("_", 1)[1].isdigit()]
    for key in stale:
        st.session_state.pop(key, None)


def resume_session_from_url():
    """Rebuild a previous session's retrieval state from the `?ns=` URL token.

    Session state dies with the process, so a restart would otherwise strand
    already-indexed papers: the corpus and BM25 index are gone, and with the
    namespace gone too the app cannot even tell which Pinecone vectors were
    its own. The token in the URL is what survives, and it is enough to reload
    the corpus, rebuild BM25, and point back at the right namespace.

    Only ever resumes the namespace the current tab's URL already names, so
    someone opening the bare app URL never inherits another user's papers.
    """
    if st.session_state.namespace is not None or "ns" not in st.query_params:
        return

    token = st.query_params["ns"]
    chunks = load_corpus(token)
    if not chunks:
        st.query_params.pop("ns", None)  # nothing stored: stale or bogus token
        return

    index = create_index()
    if not index:
        st.warning("Could not reconnect to Pinecone to resume your previous session — starting fresh.")
        st.query_params.pop("ns", None)
        return

    st.session_state.index = index
    st.session_state.namespace = token
    st.session_state.chunks = chunks
    st.session_state.bm25 = build_bm25(chunks)
    st.session_state.papers_downloaded = True


resume_session_from_url()

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


def pdf_buffer(data, name):
    """A PDF as the rest of the app expects it: something with .read() and .name."""
    buf = io.BytesIO(data)
    buf.name = name  # clustering labels each paper by its filename
    return buf


def expand_uploads(uploaded_files):
    """Flatten the upload into individual PDFs, unpacking any ZIP archives.

    The Web tab hands papers back as a ZIP, so Local accepts one directly rather
    than sending the user through their file manager to unpack it first.

    Returns buffers rather than the raw UploadedFile because clustering and
    chunking both read the same paper, and a read pointer left at EOF by the
    first of them yields an empty PDF for the second.
    """
    pdfs, bad_zips = [], []

    for upload in uploaded_files:
        if not upload.name.lower().endswith(".zip"):
            pdfs.append(pdf_buffer(upload.getvalue(), upload.name))
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as archive:
                for entry in archive.infolist():
                    # __MACOSX carries resource forks that masquerade as PDFs.
                    if entry.is_dir() or entry.filename.startswith("__MACOSX/"):
                        continue
                    if not entry.filename.lower().endswith(".pdf"):
                        continue
                    # Zip entries are paths; papers are named by their basename.
                    name = entry.filename.rsplit("/", 1)[-1]
                    pdfs.append(pdf_buffer(archive.read(entry), name))
        except zipfile.BadZipFile:
            bad_zips.append(upload.name)

    return pdfs, bad_zips


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

        # Persisted from the same list that was just upserted, so the stored
        # order matches the positional ids Pinecone holds. Best-effort: the
        # expensive half is already done and this session works regardless --
        # only the ability to resume after a restart is at stake.
        try:
            save_corpus(st.session_state.namespace, combined_chunks)
        except Exception as e:
            st.warning(f"Could not save the corpus for later ({e}). This session "
                       "still works, but restarting the app will lose it.")

    st.session_state.papers_downloaded = True
    # The token that lets this tab pick up where it left off after a restart.
    st.query_params["ns"] = st.session_state.namespace
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

def handle_query_response(query, lang, display_query):
    """Answer one turn and append it to the conversation history.

    `query` is always English (the app translates before it gets here), which is
    what retrieval and the LLM see. `display_query` is what the user actually
    typed, which is what gets shown back to them.
    """
    history = st.session_state.history

    # Follow-ups like "what about its limitations?" carry their subject in an
    # earlier turn, and retrieval has no memory -- so resolve the question
    # against the conversation before searching with it.
    search_query = condense_query(query, history)

    relevant_chunks = get_relevant_chunks(
        search_query,
        st.session_state.index,
        st.session_state.chunks,
        st.session_state.bm25,
        namespace=st.session_state.namespace,
    )

    # Generate from the question the user actually asked, not the rewritten one:
    # the rewrite is tuned for search and drops instructions like "summarise that
    # in one sentence". The model has the history, so it can resolve the reference
    # itself.
    response = generate_response_from_chunks(relevant_chunks, query, history=history)

    if lang != "English":
        display_response = translate(response, lang, True)
    else:
        display_response = response

    audio_bytes = generate_audio(display_response, lang).getvalue()

    history.append({
        "question": query,                  # English: fed back to the LLM
        "answer": response,                 # English: fed back to the LLM
        "display_question": display_query,  # what the user typed
        "display_answer": display_response,
        "audio": audio_bytes,
        "rewritten": search_query if search_query != query else None,
    })


def render_history():
    """Show every turn of the current session, oldest first."""
    for i, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["display_question"])
        with st.chat_message("assistant"):
            st.write(turn["display_answer"])
            if turn["rewritten"]:
                st.caption(f"Searched for: {turn['rewritten']}")
            st.audio(turn["audio"], format="audio/mp3")
            st.download_button(
                label="Download Audio Response",
                data=turn["audio"],
                file_name=f"response_{i + 1}.mp3",
                mime="audio/mp3",
                # Every widget needs a unique key, or the second turn's button
                # collides with the first's and Streamlit raises.
                key=f"audio_dl_{i}",
            )

# Handle Local PDF Processing
if Source == "Local":
    data = st.sidebar.file_uploader(
        "Upload PDFs or a ZIP of PDFs",
        type=["pdf", "zip"],
        accept_multiple_files=True,
    )
    if data and not st.session_state.papers_downloaded:
        pdfs, bad_zips = expand_uploads(data)

        for name in bad_zips:
            st.error(f"Could not open `{name}` — it is not a valid ZIP file.")

        if any(upload.name.lower().endswith(".zip") for upload in data):
            st.caption(f"{len(pdfs)} PDFs found in the upload.")

        if not pdfs:
            st.warning("No PDFs found in the upload.")

        elif st.toggle("Cluster By Similarity", value=True):
            pdf_texts = text_from_file_uploader(pdfs)
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
                index_chunks(process_local_pdfs(pdfs))

def set_all_selected(selected):
    """Tick or untick every paper at once. Runs as a button callback, i.e. before
    the rerun that rebuilds the checkboxes -- Streamlit forbids writing to a
    widget's state after it is instantiated."""
    for i in range(len(st.session_state.search)):
        st.session_state[f"selected_{i}"] = selected


# Handle Web Search and Download
if Source == "Web":
    search = st.text_input("Enter the search query: ")
    max_results = st.slider("Maximum results:", 10, 100)
    if st.button("Search"):
        # Drop the previous search's checkbox state, or stale ticks carry over
        # onto the new results.
        for i in range(len(st.session_state.search)):
            st.session_state.pop(f"selected_{i}", None)
        st.session_state.search = search_arxiv(search, max_results)
        st.session_state.selected_indices = []  # Reset selection on new search
        st.session_state.download = False

    if st.session_state.search:
        arxiv_results = st.session_state.search
        selection = {}
        n_selected = sum(
            bool(st.session_state.get(f"selected_{i}")) for i in range(len(arxiv_results))
        )

        # Bulk actions live in a bordered bar directly under Search, where the eye
        # lands once results appear. This used to be a bare checkbox, which read as
        # just another paper row and was missed entirely.
        with st.container(border=True):
            pick, clear, count = st.columns([1, 1, 2], vertical_alignment="center")
            pick.button("Select all", on_click=set_all_selected, args=(True,))
            clear.button("Clear all", on_click=set_all_selected, args=(False,))
            count.markdown(f"**{n_selected} of {len(arxiv_results)}** papers selected")

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
    render_history()

    query = st.text_input("Enter your question:")

    if st.button("Ask") and query:
        # Retrieval and the LLM work in English; translate the question on the
        # way in, and the answer back on the way out.
        english_query = query if lang == "English" else translate(query, lang, False)
        st.session_state.query = english_query

        with st.spinner("Searching for answers..."):
            handle_query_response(english_query, lang, display_query=query)
        # Re-run so the new turn renders through render_history() with the rest
        # of the conversation, instead of being appended below the input box.
        st.rerun()

    if st.button("End conversation"):
        reset_page()
        st.rerun()
