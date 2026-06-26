# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites

```bash
ollama serve
ollama pull llama3.2          # LLM used by both apps
pip install -r requirements.txt
```

## Commands

```bash
# FastAPI backend (pairs with a React frontend)
cd src && uvicorn backend:app --reload --port 8000

# Standalone Streamlit app
streamlit run src/apply-knowledge.py
```

## Architecture

Both apps share the same embedding model (`keepitreal/vietnamese-sbert`) and LLM (`Ollama/llama3.2`), but differ in structure and state management.

### FastAPI Backend (`src/backend.py`)

Vietnamese-language RAG API with three endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{status, document_loaded}` |
| `/upload` | POST | Accepts a PDF; chunks, embeds, and indexes it into ChromaDB |
| `/chat` | POST `{query}` | RAG query; returns answer + TTS audio + scored sources |

**Module-level singletons**: `_embeddings` and `_llm` are initialized once at import time. `_vector_db` is a global `Chroma | None` replaced on each `/upload`.

**`/upload`** creates the ChromaDB collection with `collection_metadata={"hnsw:space": "cosine"}`. This is required — without it, `/chat` cannot compute a valid cosine similarity score.

**`/chat`** implements three features:

1. **Cosine Similarity Visualization** — calls `vector_db.similarity_search_with_score(query, k=4)`. With `hnsw:space=cosine`, `distance = 1 − cosine_similarity`, so each source's score is `round(1.0 − distance, 4)` (range 0–1, 1 = identical).

2. **Text-to-Speech** — `gTTS(text=answer, lang="vi")` encodes MP3 as `"data:audio/mp3;base64,..."`. Requires internet; falls back to `""` on failure.

3. **Markdown Export-Ready JSON** — response schema maps 1-to-1 onto a `.md` file:
   ```
   # Answer
   {answer}

   ## Sources
   - **[{cosine_score}]** {text}
   ```

**`/chat` response shape:**
```json
{
  "answer": "...",
  "audio_base64": "data:audio/mp3;base64,...",
  "sources": [{ "text": "...", "cosine_score": 0.895 }]
}
```

### Streamlit App (`src/apply-knowledge.py`)

Single-file standalone with the same embedding/LLM stack but no API layer. State lives in `st.session_state.vector_db`; uploading a new PDF replaces it entirely (no multi-document support).

**Key difference from backend**: The Streamlit app does NOT set `hnsw:space=cosine` on its ChromaDB collection, so similarity scores are not exposed. It also re-instantiates `HuggingFaceEmbeddings` and `Ollama` on each PDF upload and each chat turn (no singletons).

**RAG Chat** (main area): PDF → `RecursiveCharacterTextSplitter` (600 chars / 75 overlap) → embeddings → ChromaDB → top-4 chunks → strict Vietnamese prompt → `RetrievalQA`. Prompt forbids the LLM from answering outside context.

**Quiz Generator** (sidebar): Retrieves 8 chunks for a topic, calls Ollama with `format="json"` to produce multiple-choice questions, validates with `json.loads`, then offers a download button. LLM calls are at lines 79 (quiz) and 165 (chat) — update both to change the model.
