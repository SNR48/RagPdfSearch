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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background-color: #0d0d0d;
    color: #e8e8e8;
}

section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #2a2a2a;
}

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: #00ff88;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 0.85rem;
    color: #555;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 2rem;
}

.status-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
}

.status-ok { border-left: 3px solid #00ff88; }
.status-warn { border-left: 3px solid #ffaa00; }
.status-err { border-left: 3px solid #ff4444; }

.chunk-card {
    background: #0f0f0f;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.82rem;
    color: #aaa;
}

.chunk-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #444;
    margin-bottom: 6px;
}

.answer-box {
    background: #0a1a12;
    border: 1px solid #00ff8833;
    border-radius: 8px;
    padding: 18px 22px;
    color: #d4f5e4;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-top: 10px;
}

.stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #161616 !important;
    color: #e8e8e8 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

.stButton > button {
    background-color: #00ff88 !important;
    color: #000 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    letter-spacing: 0.3px !important;
}

.stButton > button:hover {
    background-color: #00cc6a !important;
}

div[data-testid="stFileUploader"] {
    background-color: #111 !important;
    border: 1px dashed #2a2a2a !important;
    border-radius: 8px !important;
}

.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
}

.metric-box {
    flex: 1;
    background: #111;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}

.metric-val {
    font-size: 1.5rem;
    font-weight: 600;
    color: #00ff88;
}

.metric-lbl {
    font-size: 0.7rem;
    color: #555;
    margin-top: 2px;
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
    st.markdown('<div class="main-title">⬡ PDF RAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">// local · free · grounded</div>', unsafe_allow_html=True)

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
st.markdown('<div class="main-title" style="font-size:1.4rem">Ask Your PDF</div>', unsafe_allow_html=True)

if not st.session_state.pdf_loaded:
    st.info("👈  Upload and index a PDF using the sidebar to get started.")
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
