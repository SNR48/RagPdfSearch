import streamlit as st
import os
import time
import tempfile
from rag_engine import RAGEngine

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg: #f4f1ea;
    --panel: rgba(255, 252, 246, 0.78);
    --panel-strong: #fffdf8;
    --text: #1f2937;
    --muted: #6b7280;
    --line: rgba(120, 98, 73, 0.16);
    --accent: #c26d3a;
    --accent-2: #8f4e2a;
    --accent-soft: rgba(194, 109, 58, 0.12);
    --success: #1f8f63;
    --warning: #d18b17;
    --danger: #d14b4b;
    --shadow: 0 20px 60px rgba(73, 48, 28, 0.10);
    --radius: 22px;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(210, 166, 121, 0.25), transparent 28%),
        radial-gradient(circle at top right, rgba(193, 122, 73, 0.18), transparent 24%),
        linear-gradient(180deg, #f9f6f0 0%, #f2ece2 100%);
    color: var(--text);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1240px;
}

section[data-testid="stSidebar"] {
    background: rgba(255, 250, 242, 0.88);
    backdrop-filter: blur(14px);
    border-right: 1px solid var(--line);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
}

.main-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #1e1b16;
    margin-bottom: 0.25rem;
}

.subtitle {
    font-size: 0.92rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
    line-height: 1.6;
}

.brand-chip {
    display: inline-block;
    background: rgba(255,255,255,0.72);
    border: 1px solid var(--line);
    color: var(--accent-2);
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

.hero-card {
    background: linear-gradient(135deg, rgba(255,253,248,0.92), rgba(255,247,237,0.92));
    border: 1px solid rgba(194,109,58,0.14);
    box-shadow: var(--shadow);
    border-radius: 28px;
    padding: 1.6rem 1.6rem 1.3rem 1.6rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -90px;
    top: -110px;
    background: radial-gradient(circle, rgba(194,109,58,0.18), transparent 68%);
    pointer-events: none;
}

.hero-eyebrow {
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-2);
    margin-bottom: 0.5rem;
}

.hero-title {
    font-size: 2.25rem;
    line-height: 1.05;
    font-weight: 800;
    letter-spacing: -0.05em;
    color: #171411;
    margin-bottom: 0.65rem;
    max-width: 760px;
}

.hero-copy {
    font-size: 1rem;
    line-height: 1.75;
    color: #5f5a52;
    max-width: 760px;
    margin-bottom: 1rem;
}

.hero-stats {
    display: flex;
    gap: 0.9rem;
    flex-wrap: wrap;
    margin-top: 0.4rem;
}

.hero-stat {
    min-width: 140px;
    background: rgba(255,255,255,0.68);
    border: 1px solid rgba(194,109,58,0.12);
    border-radius: 18px;
    padding: 0.85rem 1rem;
}

.hero-stat-value {
    font-size: 1.2rem;
    font-weight: 800;
    color: #1d1a16;
}

.hero-stat-label {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.15rem;
}

.section-title {
    font-size: 0.95rem;
    font-weight: 800;
    color: #2d261f;
    margin: 1rem 0 0.8rem 0;
}

.status-card {
    background: var(--panel);
    backdrop-filter: blur(12px);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    box-shadow: 0 10px 30px rgba(73, 48, 28, 0.05);
}

.status-ok { border-left: 4px solid var(--success); }
.status-warn { border-left: 4px solid var(--warning); }
.status-err { border-left: 4px solid var(--danger); }

.metric-box {
    background: linear-gradient(180deg, #fffdf9, #f7f1e8);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 10px 25px rgba(73, 48, 28, 0.06);
}

.metric-val {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--accent-2);
}

.metric-lbl {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

.answer-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(250,244,236,0.92));
    border: 1px solid rgba(194,109,58,0.14);
    border-radius: 22px;
    padding: 18px 20px;
    color: var(--text);
    font-size: 0.98rem;
    line-height: 1.8;
    margin-top: 10px;
    box-shadow: 0 14px 34px rgba(73, 48, 28, 0.07);
}

.chunk-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: #4b5563;
}

.chunk-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8a8176;
    margin-bottom: 8px;
}

.empty-state {
    background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(250,244,236,0.88));
    border: 1px solid var(--line);
    border-radius: 26px;
    padding: 2rem;
    text-align: center;
    box-shadow: var(--shadow);
    color: var(--text);
}

.empty-title {
    font-size: 1.4rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    color: #1f1a15;
}

.empty-copy {
    color: var(--muted);
    max-width: 620px;
    margin: 0 auto;
    line-height: 1.75;
}

div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7) !important;
    border: 1.5px dashed rgba(194,109,58,0.35) !important;
    border-radius: 18px !important;
    padding: 0.35rem !important;
}

div[data-testid="stFileUploader"] section {
    padding: 1rem 0.6rem !important;
}

.stTextInput > div > div > input,
.stTextArea textarea {
    background: rgba(255,255,255,0.86) !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    font-family: 'Manrope', sans-serif !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #fffaf5 !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 0.75rem 1.1rem !important;
    box-shadow: 0 14px 30px rgba(143, 78, 42, 0.22) !important;
}

.stButton > button:hover {
    filter: brightness(1.03) !important;
    transform: translateY(-1px);
}

div[data-testid="stChatMessage"] {
    background: transparent !important;
}

[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.7);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 0.2rem;
    box-shadow: 0 12px 30px rgba(73, 48, 28, 0.06);
}

.stAlert {
    border-radius: 18px !important;
}

hr {
    border-color: rgba(120, 98, 73, 0.12);
}

@media (max-width: 768px) {
    .hero-title {
        font-size: 1.75rem;
    }

    .hero-card,
    .empty-state {
        padding: 1.25rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine()
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

rag: RAGEngine = st.session_state.rag

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-chip">Document Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Ask Your PDF</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload a PDF, index it locally, and get grounded answers with a polished assistant-style experience.</div>',
        unsafe_allow_html=True)
    st.markdown("### Upload PDF")
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    chunk_size   = st.slider("Chunk Size (tokens)", 200, 800, 400, 50)
    chunk_overlap = st.slider("Chunk Overlap", 20, 200, 80, 10)
    top_k        = st.slider("Top-K Retrieval", 1, 8, 3)

    if uploaded_file and st.button("⚡ Index PDF"):
        save_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Parsing & embedding…"):
            count = rag.load_pdf(save_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        st.session_state.pdf_loaded = True
        st.session_state.pdf_name   = uploaded_file.name
        st.session_state.chunks_count = count
        st.session_state.chat_history = []
        st.success(f"✓ Indexed {count} chunks")

    st.divider()

    # System status
    st.markdown("### System Status")
    ollama_ok = rag.check_ollama()
    st.markdown(
        f'<div class="status-card {"status-ok" if ollama_ok else "status-err"}">'
        f'{"✓" if ollama_ok else "✗"} Ollama LLM: {"online" if ollama_ok else "offline"}'
        f'</div>', unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="status-card status-ok">✓ Embeddings: sentence-transformers</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="status-card {"status-ok" if st.session_state.pdf_loaded else "status-warn"}">'
        f'{"✓" if st.session_state.pdf_loaded else "○"} '
        f'{"PDF: " + st.session_state.pdf_name if st.session_state.pdf_loaded else "No PDF loaded"}'
        f'</div>', unsafe_allow_html=True
    )

    if st.session_state.chunks_count:
        st.markdown(
            f'<div class="metric-box"><div class="metric-val">{st.session_state.chunks_count}</div>'
            f'<div class="metric-lbl">chunks indexed</div></div>',
            unsafe_allow_html=True
        )

    if not ollama_ok:
        st.warning("Ollama not running. See README for setup.")

# ─── Main Panel ────────────────────────────────────────────────────────────────
st.markdown(f'''
<div class="hero-card">
    <div class="hero-eyebrow">Private • Local • Retrieval-Augmented</div>
    <div class="hero-title">Turn long PDFs into a clean, premium Q&A experience.</div>
    <div class="hero-copy">
        Search contracts, reports, manuals, and research papers with grounded answers sourced from your own document.
    </div>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-value">{st.session_state.chunks_count if st.session_state.chunks_count else "0"}</div>
            <div class="hero-stat-label">Chunks indexed</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">{len(st.session_state.chat_history)}</div>
            <div class="hero-stat-label">Questions asked</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-value">Local</div>
            <div class="hero-stat-label">Processing mode</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

if not st.session_state.pdf_loaded:
    st.markdown("""
<div class="empty-state">
    <div class="empty-title">Start with a document</div>
    <div class="empty-copy">
        Upload a PDF from the left panel, click <b>Index PDF</b>, and this workspace will transform into a document chat assistant with source-grounded answers.
    </div>
</div>
""", unsafe_allow_html=True)

else:
    # Chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-box">{entry["answer"]}</div>', unsafe_allow_html=True)
            with st.expander("📎 Retrieved Chunks"):
                for i, chunk in enumerate(entry["chunks"]):
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<div class="chunk-meta">chunk #{chunk["id"]} · score {chunk["score"]:.3f} · page {chunk["page"]}</div>'
                        f'{chunk["text"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Query input
    question = st.chat_input("Ask a question about your document…")
    if question:
        if not rag.check_ollama():
            st.error("Ollama is not running. Please start Ollama and load a model.")
        else:
            with st.spinner("Retrieving & generating…"):
                result = rag.query(question, top_k=top_k)
            st.session_state.chat_history.append({
                "question": question,
                "answer":   result["answer"],
                "chunks":   result["chunks"]
            })
            st.rerun()
