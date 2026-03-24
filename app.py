import streamlit as st
import os
from pdf_loader import load_pdf
from text_chunker import split_into_chunks
from embeddings import create_vector_store, save_vector_store
from qa_pipeline import build_qa_chain, ask_question

st.set_page_config(
    page_title="AskMyDoc",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;700&family=Google+Sans+Display:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=Syne:wght@400;500;700&display=swap');

:root {
    --bg-dark:    #0F0F10;
    --bg-card:    #1A1A1F;
    --bg-hover:   #22222A;
    --border:     rgba(255,255,255,0.08);
    --border-glow:rgba(138,180,248,0.3);
    --text-white: #E8EAED;
    --text-muted: #9AA0A6;
    --text-dim:   #5F6368;
    --blue:       #8AB4F8;
    --purple:     #C58AF9;
    --teal:       #78D9D1;
    --pink:       #F28B82;
    --gold:       #FDD663;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-white) !important;
}

.stApp { background: var(--bg-dark) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Animated gradient background */
.stApp::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(138,180,248,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(197,138,249,0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(120,217,209,0.02) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D0D0F !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-white) !important; }
section[data-testid="stSidebar"] p { color: var(--text-muted) !important; font-size: 0.85rem !important; }

.logo-wrap {
    padding: 2rem 1.2rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}

.logo-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--blue) 0%, var(--purple) 50%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.5px;
    margin: 0;
    animation: shimmer 4s ease-in-out infinite;
    background-size: 200% 200%;
}

@keyframes shimmer {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.logo-sub {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-dim) !important;
    margin-top: 0.2rem;
}

.sidebar-step {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.55rem 0;
}

.step-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    flex-shrink: 0;
    margin-top: 6px;
}

.step-txt {
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    line-height: 1.5;
}

/* Main header */
.hero {
    padding: 3.5rem 0 2.5rem;
    text-align: center;
    position: relative;
}

.hero-sparkle {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(90deg, var(--blue), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    display: block;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1.1;
    margin: 0 0 0.8rem;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--blue) 40%, var(--purple) 70%, var(--teal) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 300% 300%;
    animation: titleFlow 6s ease infinite;
}

@keyframes titleFlow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.3px;
    margin: 0;
}

/* Glowing divider */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--blue), var(--purple), var(--teal), transparent);
    margin: 0.5rem 0 2.5rem;
    opacity: 0.4;
}

/* Cards */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s ease;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(138,180,248,0.4), rgba(197,138,249,0.4), transparent);
}

.glass-card:hover {
    border-color: rgba(138,180,248,0.2);
}

/* Section label */
.sec-label {
    font-size: 0.68rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    background: linear-gradient(90deg, var(--blue), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    display: block;
    font-weight: 500;
}

/* File uploader */
[data-testid="stFileUploader"] > div {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(138,180,248,0.25) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(138,180,248,0.5) !important;
    background: var(--bg-hover) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(138,180,248,0.1), rgba(197,138,249,0.1)) !important;
    color: var(--blue) !important;
    border: 1px solid rgba(138,180,248,0.3) !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(138,180,248,0.2), rgba(197,138,249,0.2)) !important;
    border-color: rgba(138,180,248,0.5) !important;
    box-shadow: 0 0 20px rgba(138,180,248,0.15) !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 1.2rem 0 !important;
}
[data-testid="stChatMessage"]:last-child {
    border-bottom: none !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-white) !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(138,180,248,0.4) !important;
    box-shadow: 0 0 0 3px rgba(138,180,248,0.08) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-white) !important;
    background: transparent !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    background: linear-gradient(135deg, var(--blue), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* Alerts */
.stAlert { border-radius: 10px !important; }
.stSuccess { background: rgba(120,217,209,0.08) !important; border: 1px solid rgba(120,217,209,0.2) !important; }
.stError { background: rgba(242,139,130,0.08) !important; border: 1px solid rgba(242,139,130,0.2) !important; }
.stInfo { background: rgba(138,180,248,0.08) !important; border: 1px solid rgba(138,180,248,0.2) !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: rgba(138,180,248,0.2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(138,180,248,0.4); }

/* Status badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(120,217,209,0.08);
    border: 1px solid rgba(120,217,209,0.2);
    color: var(--teal);
    font-size: 0.78rem;
    font-weight: 500;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
}

.pulse-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--teal);
    animation: pulse 2s ease infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
}

.empty-icon {
    font-size: 2rem;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    display: block;
}

.empty-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
    font-weight: 500;
}

.empty-sub {
    font-size: 0.83rem;
    color: var(--text-dim);
}

/* Suggestion chips */
.chip {
    display: inline-block;
    background: var(--bg-hover);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.35rem 0.8rem;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0.2rem;
    cursor: default;
    transition: border-color 0.2s ease;
}
.chip:hover {
    border-color: rgba(138,180,248,0.3);
    color: var(--blue);
}

/* HR */
hr { border-color: var(--border) !important; }

/* Block container */
.block-container {
    padding: 0 2rem !important;
    max-width: 1200px !important;
}
</style>
""", unsafe_allow_html=True)

# ── API Key check ─────────────────────────────────────────────────────────────
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not configured. Please set it in Streamlit secrets.")
    st.stop()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("qa_chain", None), ("chat_history", []), ("pdf_name", None), ("doc_stats", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
        <div class="logo-text">AskMyDoc</div>
        <div class="logo-sub">Document Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sec-label">How it works</span>', unsafe_allow_html=True)

    steps = [
        "Upload a PDF document",
        "Text is extracted and chunked",
        "Chunks are embedded into vectors",
        "Stored in a FAISS vector database",
        "Your question matches relevant chunks",
        "Gemini AI generates the answer",
    ]
    for text in steps:
        st.markdown(f"""
        <div class="sidebar-step">
            <div class="step-dot"></div>
            <div class="step-txt">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.pdf_name:
        st.markdown(f"""
        <div class="status-pill">
            <div class="pulse-dot"></div>
            {st.session_state.pdf_name}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Clear document"):
            for key in ["qa_chain", "chat_history", "pdf_name", "doc_stats"]:
                st.session_state[key] = None if key != "chat_history" else []
            st.rerun()
    else:
        st.markdown('<p style="font-size:0.8rem;color:#5F6368;">No document loaded</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#5F6368;line-height:1.8;">Built with LangChain · FAISS<br>Google Gemini · Streamlit</p>', unsafe_allow_html=True)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-sparkle">✦ &nbsp; Powered by Gemini AI &nbsp; ✦</span>
    <h1 class="hero-title">AskMyDoc</h1>
    <p class="hero-sub">Upload any document. Ask anything. Get intelligent, precise answers.</p>
</div>
<div class="glow-divider"></div>
""", unsafe_allow_html=True)

# ── Two column layout ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.7], gap="large")

# ── LEFT: Upload ──────────────────────────────────────────────────────────────
with col1:
    st.markdown('<span class="sec-label">Document</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select a PDF",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Processing document..."):
                try:
                    pages = load_pdf(temp_path)
                    chunks = split_into_chunks(pages)
                    vector_store = create_vector_store(chunks)
                    save_vector_store(vector_store)
                    st.session_state.qa_chain = build_qa_chain(vector_store)
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.session_state.doc_stats = {"pages": len(pages), "chunks": len(chunks)}
                    os.remove(temp_path)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    if st.session_state.doc_stats:
        st.success(f"Document ready — {st.session_state.doc_stats['pages']} page(s) processed")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pages", st.session_state.doc_stats["pages"])
        with c2:
            st.metric("Chunks", st.session_state.doc_stats["chunks"])

        st.markdown("---")
        st.markdown('<span class="sec-label">Try asking</span>', unsafe_allow_html=True)
        suggestions = [
            "Summarise this document",
            "What are the key points?",
            "What skills are mentioned?",
            "What is the main conclusion?",
        ]
        for s in suggestions:
            st.markdown(f'<span class="chip">{s}</span>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:1rem; padding:1.2rem; background:rgba(138,180,248,0.04); border:1px solid rgba(138,180,248,0.1); border-radius:12px;">
            <p style="font-size:0.83rem; color:#9AA0A6; margin:0; line-height:1.7;">
                Upload a PDF to begin. Supports research papers, resumes, contracts, reports, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Q&A ────────────────────────────────────────────────────────────────
with col2:
    st.markdown('<span class="sec-label">Conversation</span>', unsafe_allow_html=True)

    if st.session_state.qa_chain is not None:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-icon">✦</span>
                <div class="empty-title">Document loaded and ready</div>
                <div class="empty-sub">Type your first question below</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(f'<span style="font-size:0.92rem;color:#E8EAED;">{chat["question"]}</span>', unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    st.markdown(f'<span style="font-size:0.92rem;color:#E8EAED;">{chat["answer"]}</span>', unsafe_allow_html=True)

        user_question = st.chat_input("Ask anything about your document...")
        if user_question:
            with st.chat_message("user"):
                st.markdown(f'<span style="font-size:0.92rem;color:#E8EAED;">{user_question}</span>', unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating answer..."):
                    try:
                        answer = ask_question(st.session_state.qa_chain, user_question)
                        st.markdown(f'<span style="font-size:0.92rem;color:#E8EAED;">{answer}</span>', unsafe_allow_html=True)
                        st.session_state.chat_history.append({"question": user_question, "answer": answer})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    else:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">✦</span>
            <div class="empty-title">No document loaded</div>
            <div class="empty-sub">Upload a PDF on the left to start asking questions</div>
        </div>
        """, unsafe_allow_html=True)
