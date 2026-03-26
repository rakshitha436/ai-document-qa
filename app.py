import streamlit as st
import os
from pdf_loader import load_pdf
from text_chunker import split_into_chunks
from embeddings import create_vector_store, save_vector_store
from qa_pipeline import build_qa_chain, ask_question

st.set_page_config(
    page_title="AskMyDoc",
    page_icon="⚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600&family=Share+Tech+Mono&display=swap');

:root {
    --bg:         #050508;
    --bg-card:    #0A0A10;
    --bg-panel:   #0D0D16;
    --neon-cyan:  #00F5FF;
    --neon-purple:#BF5FFF;
    --neon-pink:  #FF2D78;
    --neon-gold:  #FFD700;
    --neon-green: #00FF88;
    --text-main:  #E0E8FF;
    --text-muted: #6A7A9A;
    --text-dim:   #3A4A6A;
    --border:     rgba(0,245,255,0.08);
    --border-hot: rgba(0,245,255,0.3);
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text-main) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0) 0px,
        rgba(0,0,0,0) 2px,
        rgba(0,245,255,0.01) 2px,
        rgba(0,245,255,0.01) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse at 10% 10%, rgba(0,245,255,0.06) 0%, transparent 40%),
        radial-gradient(ellipse at 90% 90%, rgba(191,95,255,0.06) 0%, transparent 40%),
        radial-gradient(ellipse at 50% 0%, rgba(255,45,120,0.04) 0%, transparent 30%),
        radial-gradient(ellipse at 0% 100%, rgba(0,255,136,0.03) 0%, transparent 30%);
    pointer-events: none;
    z-index: 0;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border-hot) !important;
    box-shadow: 4px 0 30px rgba(0,245,255,0.05) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-main) !important; }
section[data-testid="stSidebar"] p { color: var(--text-muted) !important; font-size: 0.84rem !important; }

.logo-wrap {
    padding: 2rem 1.2rem 1.5rem;
    border-bottom: 1px solid rgba(0,245,255,0.15);
    margin-bottom: 1rem;
    position: relative;
}

.logo-wrap::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-purple), transparent);
    animation: borderScan 3s ease-in-out infinite;
}

@keyframes borderScan {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 1; }
}

.logo-text {
    font-family: 'Orbitron', monospace;
    font-size: 1.4rem;
    font-weight: 900;
    letter-spacing: 3px;
    text-transform: uppercase;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple), var(--neon-pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 200% auto;
    animation: textScan 4s linear infinite;
    margin: 0;
}

@keyframes textScan {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.logo-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 3px;
    color: var(--neon-cyan) !important;
    opacity: 0.6;
    margin-top: 0.3rem;
    text-transform: uppercase;
}

.sidebar-step {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.5rem 0;
}

.step-num {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: var(--neon-cyan);
    opacity: 0.7;
    width: 16px;
    flex-shrink: 0;
    margin-top: 2px;
}

.step-txt {
    font-size: 0.82rem;
    color: var(--text-muted) !important;
    line-height: 1.5;
    font-weight: 400;
}

/* ── API KEY SCREEN ── */
.key-screen {
    max-width: 480px;
    margin: 6rem auto;
    text-align: center;
}

.key-title {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 4px;
    text-transform: uppercase;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--neon-cyan) 40%, var(--neon-purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}

.key-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin-bottom: 2rem;
    text-transform: uppercase;
}

.key-card {
    background: var(--bg-card);
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 4px;
    padding: 2rem;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 20px 100%, 0 calc(100% - 20px));
}

.key-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, var(--neon-cyan), var(--neon-purple));
}

.key-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 1px;
    height: 100%;
    background: linear-gradient(180deg, var(--neon-cyan), transparent);
}

.key-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 2px;
    color: var(--neon-cyan);
    text-transform: uppercase;
    text-align: left;
    margin-bottom: 0.5rem;
    display: block;
}

.key-hint {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 1px;
    color: var(--text-dim);
    margin-top: 0.8rem;
    text-align: left;
}

.key-hint a {
    color: var(--neon-cyan) !important;
    text-decoration: none;
}

/* ── HERO SECTION ── */
.hero {
    padding: 3rem 0 2rem;
    text-align: center;
    position: relative;
}

.hero-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    color: var(--neon-cyan);
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    display: block;
    opacity: 0.8;
    animation: flicker 5s ease-in-out infinite;
}

@keyframes flicker {
    0%, 95%, 100% { opacity: 0.8; }
    96% { opacity: 0.3; }
    97% { opacity: 0.8; }
    98% { opacity: 0.4; }
    99% { opacity: 0.8; }
}

.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: 6px;
    text-transform: uppercase;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--neon-cyan) 30%, var(--neon-purple) 60%, var(--neon-pink) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 300% 300%;
    animation: epicFlow 5s ease infinite;
    margin: 0 0 0.5rem;
    line-height: 1.1;
    filter: drop-shadow(0 0 30px rgba(0,245,255,0.3));
}

@keyframes epicFlow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 1px;
    margin: 0.5rem 0 0;
}

.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-purple), var(--neon-pink), transparent);
    margin: 0.5rem 0 2.5rem;
    position: relative;
    animation: dividerPulse 3s ease-in-out infinite;
}

@keyframes dividerPulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

.neon-divider::after {
    content: '';
    position: absolute;
    top: -2px; left: 50%;
    transform: translateX(-50%);
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--neon-cyan);
    box-shadow: 0 0 10px var(--neon-cyan), 0 0 20px var(--neon-cyan);
    animation: dotPulse 3s ease-in-out infinite;
}

@keyframes dotPulse {
    0%, 100% { box-shadow: 0 0 10px var(--neon-cyan), 0 0 20px var(--neon-cyan); }
    50% { box-shadow: 0 0 20px var(--neon-cyan), 0 0 40px var(--neon-cyan), 0 0 60px var(--neon-purple); }
}

.sec-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--neon-cyan);
    margin-bottom: 1rem;
    display: block;
    opacity: 0.8;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] > div {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 4px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.1), inset 0 0 20px rgba(0,245,255,0.03) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
}

/* ── TEXT INPUT ── */
[data-testid="stTextInput"] input {
    background: rgba(0,245,255,0.03) !important;
    border: 1px solid rgba(0,245,255,0.25) !important;
    border-radius: 2px !important;
    color: var(--text-main) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: var(--text-dim) !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    color: var(--neon-cyan) !important;
    border: 1px solid var(--neon-cyan) !important;
    border-radius: 2px !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 8px 100%, 0 calc(100% - 8px)) !important;
    width: 100% !important;
    padding: 0.6rem !important;
}
.stButton > button:hover {
    background: rgba(0,245,255,0.08) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.3), inset 0 0 20px rgba(0,245,255,0.05) !important;
    color: #FFFFFF !important;
}

/* ── CHAT MESSAGES ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(0,245,255,0.06) !important;
    padding: 1rem 0 !important;
}
[data-testid="stChatMessage"]:last-child { border-bottom: none !important; }

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 2px !important;
    color: var(--text-main) !important;
    font-family: 'Rajdhani', sans-serif !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 30px rgba(0,245,255,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text-main) !important;
    background: transparent !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 0.95rem !important;
}

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,245,255,0.15) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 0 100%) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: var(--neon-cyan) !important;
    font-size: 1.5rem !important;
    text-shadow: 0 0 20px rgba(0,245,255,0.5) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* ── ALERTS ── */
div[data-testid="stAlert"] {
    background: rgba(0,245,255,0.04) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-left: 3px solid var(--neon-cyan) !important;
    border-radius: 0 !important;
}

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--neon-cyan) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.3); }
::-webkit-scrollbar-thumb:hover { background: var(--neon-cyan); }

/* ── STATUS PILL ── */
.status-hud {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,255,136,0.06);
    border: 1px solid rgba(0,255,136,0.3);
    color: var(--neon-green);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    padding: 0.35rem 1rem;
    clip-path: polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px);
}

.hud-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--neon-green);
    box-shadow: 0 0 8px var(--neon-green);
    animation: hudPulse 1.5s ease infinite;
}

@keyframes hudPulse {
    0%, 100% { box-shadow: 0 0 4px var(--neon-green); opacity: 1; }
    50% { box-shadow: 0 0 12px var(--neon-green), 0 0 24px var(--neon-green); opacity: 0.7; }
}

/* ── SUGGESTION CHIPS ── */
.chip {
    display: inline-block;
    background: rgba(0,245,255,0.04);
    border: 1px solid rgba(0,245,255,0.15);
    color: var(--text-muted);
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    padding: 0.3rem 0.8rem;
    margin: 0.2rem;
    clip-path: polygon(6px 0, 100% 0, 100% calc(100% - 6px), calc(100% - 6px) 100%, 0 100%, 0 6px);
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 4rem 1rem;
}
.empty-icon {
    font-family: 'Orbitron', monospace;
    font-size: 2.5rem;
    color: var(--neon-cyan);
    margin-bottom: 1rem;
    display: block;
    opacity: 0.3;
    animation: iconPulse 3s ease infinite;
}
@keyframes iconPulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.05); }
}
.empty-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    letter-spacing: 3px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.empty-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 1px;
}

hr { border-color: rgba(0,245,255,0.1) !important; }

.block-container {
    padding: 0 2rem !important;
    max-width: 1200px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("qa_chain", None),
    ("chat_history", []),
    ("pdf_name", None),
    ("doc_stats", None),
    ("api_key", None),
    ("api_key_verified", False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Check for API key (environment or session) ────────────────────────────────
env_key = os.getenv("GOOGLE_API_KEY")
if env_key:
    st.session_state.api_key = env_key
    st.session_state.api_key_verified = True

# ── API KEY INPUT SCREEN ──────────────────────────────────────────────────────
if not st.session_state.api_key_verified:

    # Hide sidebar on key screen
    st.markdown("""
    <style>
        section[data-testid="stSidebar"] { display: none !important; }
        .block-container { max-width: 600px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="key-screen">
        <div class="key-title">AskMyDoc</div>
        <div class="key-sub">// Intelligence System — Enter Access Key</div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div class="key-card">
            <span class="key-label">// Gemini API Key</span>
        </div>
        """, unsafe_allow_html=True)

        api_input = st.text_input(
            "API Key",
            placeholder="AIza...",
            type="password",
            label_visibility="collapsed"
        )

        st.markdown("""
        <div class="key-hint">
            Get your free key at
            <a href="https://aistudio.google.com" target="_blank">aistudio.google.com</a>
            <br>Your key is never stored or shared.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Initialise System"):
            if api_input and api_input.startswith("AIza"):
                st.session_state.api_key = api_input
                st.session_state.api_key_verified = True
                os.environ["GOOGLE_API_KEY"] = api_input
                st.rerun()
            else:
                st.error("[ ERROR ] Invalid API key. Must start with 'AIza...'")

    st.stop()

# ── Set API key in environment ────────────────────────────────────────────────
os.environ["GOOGLE_API_KEY"] = st.session_state.api_key

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-wrap">
        <div class="logo-text">AskMyDoc</div>
        <div class="logo-sub">// Intelligence System v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sec-label">// System Protocol</span>', unsafe_allow_html=True)

    steps = [
        "Upload a PDF document",
        "Text extracted and chunked",
        "Chunks embedded into vectors",
        "Stored in FAISS database",
        "Question matched to relevant chunks",
        "Gemini AI generates the answer",
    ]
    for i, text in enumerate(steps, 1):
        st.markdown(f"""
        <div class="sidebar-step">
            <div class="step-num">0{i}</div>
            <div class="step-txt">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.pdf_name:
        name = st.session_state.pdf_name[:20] + "..." if len(st.session_state.pdf_name) > 20 else st.session_state.pdf_name
        st.markdown(f"""
        <div class="status-hud">
            <div class="hud-dot"></div>
            {name}
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button("Eject Document"):
            for key in ["qa_chain", "chat_history", "pdf_name", "doc_stats"]:
                st.session_state[key] = None if key != "chat_history" else []
            st.rerun()
    else:
        st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.72rem;color:#3A4A6A;letter-spacing:1px;">[ NO DOCUMENT LOADED ]</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Option to change API key
    if st.button("Change API Key"):
        st.session_state.api_key = None
        st.session_state.api_key_verified = False
        st.session_state.qa_chain = None
        st.session_state.chat_history = []
        st.session_state.pdf_name = None
        st.session_state.doc_stats = None
        st.rerun()

    st.markdown('<p style="font-family:Share Tech Mono,monospace;font-size:0.65rem;color:#3A4A6A;line-height:2;letter-spacing:1px;">LANGCHAIN · FAISS<br>GOOGLE GEMINI · STREAMLIT</p>', unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-badge">[ SYSTEM ONLINE ] · POWERED BY GEMINI AI · [ RAG ENGINE ACTIVE ]</span>
    <h1 class="hero-title">AskMyDoc</h1>
    <p class="hero-sub">Load your document. Query the intelligence. Get precise answers.</p>
</div>
<div class="neon-divider"></div>
""", unsafe_allow_html=True)

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.7], gap="large")

# ── LEFT: UPLOAD ──────────────────────────────────────────────────────────────
with col1:
    st.markdown('<span class="sec-label">// Document Upload</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select PDF",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Initialising intelligence matrix..."):
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
                    st.error(f"[ ERROR ] {str(e)}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    if st.session_state.doc_stats:
        st.success(f"[ ONLINE ] Document loaded successfully")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Pages", st.session_state.doc_stats["pages"])
        with c2:
            st.metric("Chunks", st.session_state.doc_stats["chunks"])

        st.markdown("---")
        st.markdown('<span class="sec-label">// Suggested Queries</span>', unsafe_allow_html=True)
        suggestions = [
            "Summarise this document",
            "What are the key points?",
            "List all skills mentioned",
            "What is the conclusion?",
        ]
        for s in suggestions:
            st.markdown(f'<span class="chip">{s}</span>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="margin-top:1rem; padding:1.2rem; background:rgba(0,245,255,0.02); border:1px solid rgba(0,245,255,0.1); clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));">
            <p style="font-family: Share Tech Mono, monospace; font-size:0.78rem; color:#6A7A9A; margin:0; line-height:1.9; letter-spacing:0.5px;">
                AWAITING INPUT...<br>
                Upload a PDF to initialise<br>
                the intelligence system.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT: Q&A ────────────────────────────────────────────────────────────────
with col2:
    st.markdown('<span class="sec-label">// Query Interface</span>', unsafe_allow_html=True)

    if st.session_state.qa_chain is not None:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-icon">◈</span>
                <div class="empty-title">System Ready</div>
                <div class="empty-sub">// Awaiting your query...</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(f'<span style="font-family:Rajdhani,sans-serif;font-size:0.95rem;color:#E0E8FF;">{chat["question"]}</span>', unsafe_allow_html=True)
                with st.chat_message("assistant"):
                    st.markdown(f'<span style="font-family:Rajdhani,sans-serif;font-size:0.95rem;color:#E0E8FF;line-height:1.7;">{chat["answer"]}</span>', unsafe_allow_html=True)

        user_question = st.chat_input("Enter query...")
        if user_question:
            with st.chat_message("user"):
                st.markdown(f'<span style="font-family:Rajdhani,sans-serif;font-size:0.95rem;color:#E0E8FF;">{user_question}</span>', unsafe_allow_html=True)
            with st.chat_message("assistant"):
                with st.spinner("Scanning intelligence matrix..."):
                    try:
                        answer = ask_question(st.session_state.qa_chain, user_question)
                        st.markdown(f'<span style="font-family:Rajdhani,sans-serif;font-size:0.95rem;color:#E0E8FF;line-height:1.7;">{answer}</span>', unsafe_allow_html=True)
                        st.session_state.chat_history.append({"question": user_question, "answer": answer})
                    except Exception as e:
                        st.error(f"[ ERROR ] {str(e)}")
    else:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">◈</span>
            <div class="empty-title">Awaiting Document</div>
            <div class="empty-sub">// Upload a PDF to activate query interface</div>
        </div>
        """, unsafe_allow_html=True)
