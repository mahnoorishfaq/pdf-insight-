import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer

from src.pdf_processor import process_pdf
from src.embeddings import VectorStore, DEFAULT_MODEL_NAME
from src.llm_engine import answer_question, summarize_chunks, GeminiAPIError

load_dotenv()
ENV_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(
    page_title="PDF Insight",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Cached globally: the model is stateless and expensive to load, so
    every session can safely share one instance. The per-document FAISS
    index itself is NOT cached here — it lives in st.session_state so
    concurrent users never share or clobber each other's document."""
    return SentenceTransformer(DEFAULT_MODEL_NAME)


def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        chunks = process_pdf(tmp_path)
    finally:
        os.remove(tmp_path)

    store = VectorStore(model=get_embedding_model())
    store.build(chunks)
    return chunks, store


def main():
    st.title("📄 PDF Insight")
    st.caption(
        "Upload a PDF, then ask it questions or generate a summary — "
        "local semantic search paired with Gemini for generation."
    )

    if "chunks" not in st.session_state:
        st.session_state.chunks = None
        st.session_state.store = None
        st.session_state.file_name = None
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("1. Gemini API key")
        if ENV_API_KEY:
            st.success("Using API key from your local .env file.")
            api_key = ENV_API_KEY
            with st.expander("Use a different key instead"):
                override_key = st.text_input(
                    "Gemini API key", type="password",
                    help="Overrides the .env key for this session only.",
                )
                if override_key:
                    api_key = override_key
        else:
            api_key = st.text_input(
                "Enter your Gemini API key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey. "
                     "It's only used for this session and never saved to disk.",
            )
            if not api_key:
                st.warning("Enter an API key to enable Q&A and summarization.")

        st.header("2. Upload a document")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

        if uploaded_file is not None and uploaded_file.name != st.session_state.file_name:
            with st.spinner("Reading and indexing your document..."):
                chunks, store = process_uploaded_file(uploaded_file)
            st.session_state.chunks = chunks
            st.session_state.store = store
            st.session_state.file_name = uploaded_file.name
            st.session_state.chat_history = []
            if chunks:
                st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'.")
            else:
                st.warning(
                    f"No extractable text found in '{uploaded_file.name}'. "
                    "It may be a scanned or image-only PDF without a text layer."
                )

        if st.session_state.chunks:
            st.metric("Chunks indexed", len(st.session_state.chunks))
            if st.button("Clear document"):
                st.session_state.chunks = None
                st.session_state.store = None
                st.session_state.file_name = None
                st.session_state.chat_history = []
                st.rerun()

    if not st.session_state.chunks:
        st.info("👈 Upload a PDF from the sidebar to get started.")
        return

    tab_qa, tab_summary = st.tabs(["💬 Ask Questions", "📝 Summarize"])

    with tab_qa:
        st.subheader("Ask a question about the document")
        question = st.text_input("Your question", placeholder="e.g. What is the main conclusion of this report?")

        if st.button("Ask", type="primary", disabled=not api_key) and question.strip():
            with st.spinner("Searching document and generating an answer..."):
                retrieved = st.session_state.store.search(question, top_k=4)
                try:
                    result = answer_question(question, retrieved, api_key)
                except GeminiAPIError as exc:
                    result = None
                    st.error(str(exc))
            if result is not None:
                st.session_state.chat_history.append((question, result))

        for q, result in reversed(st.session_state.chat_history):
            with st.container(border=True):
                st.markdown(f"**Q: {q}**")
                st.markdown(f"**A:** {result['answer']}")
                if result["pages"]:
                    pages_str = ", ".join(str(p) for p in result["pages"])
                    st.caption(f"Source pages: {pages_str}")

    with tab_summary:
        st.subheader("Document summary")
        if st.button("Generate summary", disabled=not api_key):
            with st.spinner("Summarizing document..."):
                try:
                    summary = summarize_chunks(st.session_state.chunks, api_key)
                except GeminiAPIError as exc:
                    summary = None
                    st.error(str(exc))
            if summary is not None:
                st.write(summary)


if __name__ == "__main__":
    main()
