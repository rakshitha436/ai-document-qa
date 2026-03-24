import streamlit as st
import os
from pdf_loader import load_pdf
from text_chunker import split_into_chunks
from embeddings import create_vector_store, save_vector_store
from qa_pipeline import build_qa_chain, ask_question

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyDoc",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root variables ── */
    :root {
        --navy:       #0D1B2A;
        --navy-mid:   #1B2E42;
        --navy-light: #243447;
        --gold:       #C9A84C;
        --gold-light: #E2C97E;
        --cream:      #F5F0E8;
        --cream-dark: #EDE6D6;
        --text-dark:  #0D1B2A;
        --text-mid:   #3D5166;
        --text-light: #8A9BB0;
        --white:      #FFFFFF;
        --border:     #D8CEBC;
    }

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text-dark);
    }

    .stApp {
        background-color: var(--cream);
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: var(--navy) !important;
        border-right: 1px solid var(--navy-light);
    }

    section[data-testid="stSidebar"] * {
        color: var(--cream) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #A8B8C8 !important;
        font-size: 0.85rem;
        line-height: 1.7;
    }

    /* ── Sidebar logo area ── */
    .sidebar-logo {
        padding: 2rem 1rem 1.5rem;
        border-bottom: 1px solid var(--navy-light);
        margin-bottom: 1.5rem;
    }

    .sidebar-logo h1 {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--white) !important;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .sidebar-logo span {
        color: var(--gold) !important;
    }

    .sidebar-tagline {
        font-size: 0.75rem;
        color: var(--text-light) !important;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }

    /* ── Sidebar steps ── */
    .sidebar-step {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        padding: 0.6rem 0;
    }

    .step-number {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: var(--navy-light);
        border: 1px solid var(--gold);
        color: var(--gold) !important;
        font-size: 0.7rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 1px;
    }

    .step-text {
        font-size: 0.83rem;
        color: #A8B8C8 !important;
        line-height: 1.5;
    }

    /* ── Main content area ── */
    .main-header {
        padding: 3rem 0 2rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2.5rem;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--navy);
        letter-spacing: -1px;
        line-height: 1.1;
        margin: 0;
    }

    .main-title span {
        color: var(--gold);
    }

    .main-subtitle {
        font-size: 1rem;
        color: var(--text-mid);
        margin-top: 0.6rem;
        font-weight: 300;
        letter-spacing: 0.2px;
    }

    /* ── Section headings ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: var(--gold);
        margin-bottom: 0.8rem;
    }

    /* ── Upload area ── */
    .upload-container {
        background: var(--white);
        border: 1.5px solid var(--border);
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: border-color 0.2s ease;
    }

    .upload-container:hover {
        border-color: var(--gold);
    }

    /* ── File uploader styling ── */
    [data-testid="stFileUploader"] {
        background: transparent;
    }

    [data-testid="stFileUploader"] > div {
        border: 1.5px dashed var(--border) !important;
        background: var(--cream) !important;
        border-radius: 6px !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease;
    }

    [data-testid="stFileUploader"] > div:hover {
        border-color: var(--gold) !important;
        background: var(--cream-dark) !important;
    }

    [data-testid="stFileUploader"] label {
        color: var(--text-mid) !important;
        font-size: 0.9rem;
    }

    /* ── Success / info / error boxes ── */
    .stAlert {
        border-radius: 6px !important;
        font-size: 0.88rem;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border);
        padding: 1.2rem 0 !important;
    }

    [data-testid="stChatMessage"]:last-child {
        border-bottom: none;
    }

    /* User message bubble */
    [data-testid="stChatMessage"][data-testid*="user"] {
        background: var(--white) !important;
        border-radius: 6px;
        padding: 1rem 1.2rem !important;
        border: 1px solid var(--border) !important;
        margin-bottom: 0.5rem;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: var(--white) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 6px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--navy) !important;
        box-shadow: 0 0 0 3px rgba(13, 27, 42, 0.08) !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: var(--gold) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--navy) !important;
        color: var(--cream) !important;
        border: none !important;
        border-radius: 5px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        padding: 0.5rem 1.2rem !important;
        transition: background 0.2s ease !important;
    }

    .stButton > button:hover {
        background: var(--navy-mid) !important;
    }

    /* ── Divider ── */
    hr {
        border-color: var(--border) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(201, 168, 76, 0.12);
        border: 1px solid rgba(201, 168, 76, 0.3);
        color: #8B6914;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        letter-spacing: 0.3px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--gold);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-size: 0.82rem !important;
        color: var(--text-mid) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: var(--cream); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-light); }

    /* ── Q&A container ── */
    .qa-container {
        background: var(--white);
        border: 1.5px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem 2rem;
        min-height: 300px;
    }

    /* ── Empty state ── */
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: var(--text-light);
    }

    .empty-state-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.1rem;
        color: var(--text-mid);
        margin-bottom: 0.5rem;
    }

    .empty-state-sub {
        font-size: 0.85rem;
        color: var(--text-light);
    }

    /* ── Column layout ── */
    .block-container {
        padding: 0 2rem !important;
        max-width: 1200px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Check for Google API key ──────────────────────────────────────────────────
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not configured. Please set it in Streamlit secrets.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1>Ask<span>My</span>Doc</h1>
        <div class="sidebar-tagline">Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**How it works**")
    steps = [
        ("1", "Upload a PDF document"),
        ("2", "Text is extracted and split into chunks"),
        ("3", "Each chunk is converted to an embedding vector"),
        ("4", "Vectors are stored in a FAISS database"),
        ("5", "Your question is matched to relevant chunks"),
        ("6", "Gemini AI generates a contextual answer"),
    ]

    for num, text in steps:
        st.markdown(f"""
        <div class="sidebar-step">
            <div class="step-number">{num}</div>
            <div class="step-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div class="status-badge">
            <div class="status-dot"></div>
            {st.session_state.pdf_name}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Clear Document"):
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.pdf_name = None
            st.session_state.doc_stats = None
            st.rerun()
    else:
        st.markdown('<p style="font-size:0.82rem; color:#6B8099;">No document loaded</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.75rem; color:#4A6278;">Built with LangChain, FAISS<br>and Google Gemini</p>', unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Ask<span>My</span>Doc</h1>
    <p class="main-subtitle">Upload a document. Ask anything. Get precise, source-backed answers.</p>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.6], gap="large")

# ── LEFT COLUMN: Upload ───────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="section-label">Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select a PDF file to analyse",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            temp_pdf_path = f"temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing document..."):
                try:
                    pages = load_pdf(temp_pdf_path)
                    chunks = split_into_chunks(pages)
                    vector_store = create_vector_store(chunks)
                    save_vector_store(vector_store)
                    st.session_state.qa_chain = build_qa_chain(vector_store)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.session_state.doc_stats = {
                        "pages": len(pages),
                        "chunks": len(chunks)
                    }
                    os.remove(temp_pdf_path)
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    if os.path.exists(temp_pdf_path):
                        os.remove(temp_pdf_path)

    # Show document stats if loaded
    if st.session_state.doc_stats:
        st.markdown("---")
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.metric("Pages", st.session_state.doc_stats["pages"])
        with stat_col2:
            st.metric("Chunks", st.session_state.doc_stats["chunks"])

        st.markdown("---")
        st.markdown('<div class="section-label">Suggested Questions</div>', unsafe_allow_html=True)
        suggestions = [
            "What is the main topic of this document?",
            "Summarise the key points.",
            "What are the most important details?",
        ]
        for s in suggestions:
            st.markdown(f'<p style="font-size:0.82rem; color:#3D5166; padding: 0.3rem 0; border-bottom: 1px solid #EDE6D6;">— {s}</p>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="margin-top:1.5rem; padding: 1.2rem; background: #EDE6D6; border-radius: 6px;">
            <p style="font-size:0.83rem; color:#3D5166; margin:0; line-height:1.6;">
                Upload a PDF to begin. Supported documents include reports, research papers, contracts, resumes, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT COLUMN: Q&A ─────────────────────────────────────────────────────────
with col2:
    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)

    if st.session_state.qa_chain is not None:
        # Chat history display
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-state-title">Document ready</div>
                    <div class="empty-state-sub">Type your question below to begin</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for chat in st.session_state.chat_history:
                    with st.chat_message("user"):
                        st.markdown(f'<span style="font-size:0.92rem;">{chat["question"]}</span>', unsafe_allow_html=True)
                    with st.chat_message("assistant"):
                        st.markdown(f'<span style="font-size:0.92rem;">{chat["answer"]}</span>', unsafe_allow_html=True)

        # Chat input
        user_question = st.chat_input("Ask a question about your document...")

        if user_question:
            with st.chat_message("user"):
                st.markdown(f'<span style="font-size:0.92rem;">{user_question}</span>', unsafe_allow_html=True)

            with st.chat_message("assistant"):
                with st.spinner("Searching document..."):
                    try:
                        answer = ask_question(st.session_state.qa_chain, user_question)
                        st.markdown(f'<span style="font-size:0.92rem;">{answer}</span>', unsafe_allow_html=True)
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": answer
                        })
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
    else:
        st.markdown("""
        <div class="empty-state" style="padding: 5rem 1rem;">
            <div class="empty-state-title">No document loaded</div>
            <div class="empty-state-sub">Upload a PDF on the left to start asking questions</div>
        </div>
        """, unsafe_allow_html=True)
