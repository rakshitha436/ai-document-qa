import streamlit as st
import os
from pdf_loader import load_pdf
from text_chunker import split_into_chunks
from embeddings import create_vector_store, save_vector_store
from qa_pipeline import build_qa_chain, ask_question

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📘",
    layout="centered"
)

# ── App title ─────────────────────────────────────────────────────────────────
st.title("📘 AI Document Question Answering")
st.markdown("Upload a PDF and ask questions about it using AI!")
st.divider()

# ── Check for Google API key ──────────────────────────────────────────────────
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY is not set.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ── PDF Upload ────────────────────────────────────────────────────────────────
st.subheader("📄 Step 1: Upload your PDF")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

if uploaded_file is not None:
    if st.session_state.pdf_name != uploaded_file.name:
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("📖 Reading and processing your PDF..."):
            try:
                pages = load_pdf(temp_pdf_path)
                chunks = split_into_chunks(pages)
                vector_store = create_vector_store(chunks)
                save_vector_store(vector_store)
                st.session_state.qa_chain = build_qa_chain(vector_store)
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                os.remove(temp_pdf_path)
                st.success(f"✅ '{uploaded_file.name}' processed! ({len(pages)} page(s), {len(chunks)} chunks)")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)

# ── Q&A Section ───────────────────────────────────────────────────────────────
if st.session_state.qa_chain is not None:
    st.divider()
    st.subheader("💬 Step 2: Ask questions about your document")

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])

    user_question = st.chat_input("Ask a question about your document...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating answer..."):
                try:
                    answer = ask_question(st.session_state.qa_chain, user_question)
                    st.write(answer)
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": answer
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
else:
    st.info("👆 Please upload a PDF file above to get started!")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About this app")
    st.markdown("""
    This app uses **RAG** to answer questions about your documents.
    
    **How it works:**
    1. 📄 Upload a PDF
    2. 🔢 Text is split into chunks
    3. 🧠 Chunks are converted to embeddings
    4. 💾 Stored in FAISS vector database
    5. ❓ Ask a question
    6. 🔍 Similar chunks are retrieved
    7. 🤖 Gemini AI generates the answer
    """)

    st.divider()

    if st.session_state.pdf_name:
        st.success(f"📄 Loaded: {st.session_state.pdf_name}")
        if st.button("🗑️ Clear & Upload New PDF"):
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.pdf_name = None
            st.rerun()

    st.divider()
    st.markdown("Built with LangChain + FAISS + Google Gemini")
