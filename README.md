# PDF RAG Assistant 📄
> Local, free, grounded — no API keys, no cloud, runs entirely on your machine.

## Stack
| Layer | Tool | Why |
|-------|------|-----|
| LLM | Ollama + `llama3.2:3b` | Free, 2 GB RAM, fast on Ryzen 5 |
| Embeddings | `all-MiniLM-L6-v2` | 90 MB, CPU-fast, great quality |
| Vector DB | ChromaDB | In-process, no server needed |
| PDF Parser | PyMuPDF | Fast, accurate text extraction |
| UI | Streamlit | Zero-config web UI |

---

## Setup (Windows 11)

### 1 — Install Ollama
Download from https://ollama.com/download/windows and run the installer.

### 2 — Pull the LLM model
Open PowerShell or CMD:
```powershell
ollama pull llama3.2:3b
```
> 💡 **Alternative models for 8 GB RAM:**
> - `ollama pull phi3:mini` — Microsoft Phi-3, very smart for its size
> - `ollama pull gemma2:2b` — Google Gemma 2B, excellent for Q&A
> - `ollama pull mistral:7b-q4` — Mistral 7B quantised (needs ~5 GB VRAM/RAM)

### 3 — Create Python environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 4 — Install dependencies (CPU-only PyTorch saves ~2 GB)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5 — Run Ollama server (keep this terminal open)
```powershell
ollama serve
```

### 6 — Launch the app (new terminal, venv activated)
```powershell
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## How It Works

```
PDF Upload
    │
    ▼
PyMuPDF → raw text per page
    │
    ▼
Sliding Window Chunker  (chunk_size=400 words, overlap=80)
    │
    ▼
SentenceTransformer → 384-dim embeddings (CPU)
    │
    ▼
ChromaDB collection (cosine similarity index)

──── At Query Time ────────────────────────────────

User Question
    │
    ▼
Embed question  →  ChromaDB cosine search  →  Top-K chunks
    │
    ▼
Prompt template:  [System] + [Context chunks] + [Question]
    │
    ▼
Ollama llama3.2:3b  →  Grounded answer
```

---

## Tips for 8 GB RAM

- Keep Ollama model at `3b` or `2b` quantised versions.
- Set `num_ctx=2048` in `rag_engine.py` if you run out of RAM during generation.
- Close Chrome tabs before running the LLM — Ollama needs ~2–3 GB headroom.
- `torch` CPU-only wheel saves ~1.5 GB vs the CUDA wheel.

---

## Changing the LLM model
Edit `rag_engine.py`, line 20:
```python
OLLAMA_MODEL = "phi3:mini"   # or gemma2:2b, mistral:7b-q4, etc.
```
