"""
RAG Engine
----------
- PDF Parsing   : PyMuPDF (fitz)
- Chunking      : Sliding window with token-based overlap
- Embeddings    : sentence-transformers/all-MiniLM-L6-v2  (free, local, ~90 MB)
- Vector Store  : ChromaDB  (persistent, local)
- LLM           : Ollama  (llama3.2:3b recommended for 8 GB RAM)
"""

import re
import os
try:
    import fitz  # PyMuPDF
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "PyMuPDF is not installed in the active environment. "
        "Activate OWN\\venv and run: pip install pymupdf "
        "or pip install -r requirements.txt"
    ) from exc
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
from typing import List, Dict, Any


OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"          # good balance for 8 GB RAM
EMBED_MODEL  = "all-MiniLM-L6-v2"     # 384-dim, ~90 MB, very fast on CPU
CHROMA_DIR   = "./chroma_db"


class RAGEngine:
    def __init__(self):
        print("[RAGEngine] Loading embedding model…")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        print("[RAGEngine] Ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # PDF  →  Chunks
    # ──────────────────────────────────────────────────────────────────────────
    def load_pdf(self, path: str, chunk_size: int = 400, chunk_overlap: int = 80) -> int:
        """Parse PDF, chunk text, embed, store in ChromaDB. Returns chunk count."""
        # Reset collection
        try:
            self.chroma_client.delete_collection("pdf_chunks")
        except Exception:
            pass
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )

        # Extract text per page
        doc   = fitz.open(path)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append({"page": page_num + 1, "text": text})
        doc.close()

        # Chunk
        chunks = self._chunk_pages(pages, chunk_size, chunk_overlap)
        if not chunks:
            return 0

        # Embed & upsert
        texts      = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress_bar=False).tolist()

        self.collection.add(
            ids        = [f"chunk_{i}" for i in range(len(chunks))],
            documents  = texts,
            embeddings = embeddings,
            metadatas  = [{"page": c["page"], "chunk_id": i} for i, c in enumerate(chunks)]
        )
        return len(chunks)

    def _chunk_pages(self, pages: List[Dict], size: int, overlap: int) -> List[Dict]:
        """Sliding-window word chunking across pages."""
        chunks = []
        current_words = []
        current_page  = 1

        for page_data in pages:
            words = page_data["text"].split()
            for word in words:
                current_words.append(word)
                if len(current_words) >= size:
                    chunks.append({
                        "text": " ".join(current_words),
                        "page": current_page
                    })
                    current_words = current_words[size - overlap:]
            current_page = page_data["page"]

        if current_words:
            chunks.append({"text": " ".join(current_words), "page": current_page})

        return chunks

    # ──────────────────────────────────────────────────────────────────────────
    # Query  →  Retrieve  →  Generate
    # ──────────────────────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Full RAG pipeline: embed question → retrieve → generate grounded answer."""

        # 1. Embed question
        q_embedding = self.embedder.encode([question]).tolist()

        # 2. Retrieve top-k chunks
        results = self.collection.query(
            query_embeddings=q_embedding,
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        chunks_info = []
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            chunks_info.append({
                "id":    meta.get("chunk_id", i),
                "page":  meta.get("page", "?"),
                "score": 1 - dist,          # cosine similarity
                "text":  doc[:400] + ("…" if len(doc) > 400 else "")
            })

        context = "\n\n---\n\n".join(docs)

        # 3. Build prompt
        prompt = self._build_prompt(question, context)

        # 4. Call Ollama
        answer = self._call_ollama(prompt)

        return {"answer": answer, "chunks": chunks_info}

    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a precise document assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I couldn't find this in the document."
Do not hallucinate. Be concise and cite relevant details.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API."""
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512,
                        "num_ctx":     4096
                    }
                },
                timeout=120
            )
            resp.raise_for_status()
            return resp.json().get("response", "No response from model.").strip()
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Make sure it is running: `ollama serve`"
        except Exception as e:
            return f"❌ Ollama error: {str(e)}"

    def check_ollama(self) -> bool:
        """Ping Ollama health endpoint."""
        try:
            r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False
