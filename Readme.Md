# RAG Backend API

A Retrieval-Augmented Generation (RAG) API built with FastAPI and Google Gemini. Upload a PDF and ask questions about its content. Includes two parallel implementations — a manual pipeline and a LangChain-powered pipeline.

---

## Tech Stack

- **FastAPI** — REST API framework
- **Google Gemini** — Embeddings (`gemini-embedding-001`) and LLM (`gemini-2.5-flash`)
- **FAISS** — Vector similarity search
- **Unstructured** — PDF text extraction
- **LangChain** — LCEL-based RAG chain (alternative pipeline)

---

## Project Structure

```
backend/
├── app/
│   ├── ai/
│   │   └── gemini_llm.py         # Gemini LLM wrapper
│   ├── core/
│   │   └── config.py             # Environment config
│   ├── models/
│   │   └── question_model.py     # Pydantic request model
│   ├── routers/
│   │   ├── ai_router.py          # Manual RAG endpoints
│   │   └── langchain_router.py   # LangChain RAG endpoints
│   ├── services/
│   │   ├── pdf_sevice.py         # PDF text extraction
│   │   ├── chunk_service.py      # Text chunking
│   │   ├── embedding_service.py  # Gemini embeddings
│   │   ├── vector_service.py     # FAISS vector store
│   │   ├── rag_service.py        # Manual RAG pipeline
│   │   └── langchain_service.py  # LangChain RAG pipeline
│   └── main.py                   # FastAPI app entry
├── uploads/                      # Uploaded PDFs (git-ignored)
├── .env                          # Environment variables (git-ignored)
├── requirements.txt
└── server.py
```

---

## Setup

### 1. Clone & create virtual environment

```bash
git clone <repo-url>
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the root of `backend/`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run the server

```bash
uvicorn app.main:app --reload
```

API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## API Endpoints

### Manual RAG Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload a PDF and index it |
| `POST` | `/ask`    | Ask a question about the PDF |

### LangChain RAG Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/langchain/upload` | Upload a PDF and index it via LangChain |
| `POST` | `/langchain/ask`    | Ask a question via LangChain LCEL chain |

---

## Usage Examples

### Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "message": "File 'your_document.pdf' processed successfully.",
  "chunks": 42
}
```

### Ask a Question

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

**Response:**
```json
{
  "answer": "This document is about..."
}
```

> Same request body works for `/langchain/ask`

---

## How It Works

```
PDF Upload
   │
   ├── Extract text (Unstructured)
   ├── Split into overlapping chunks (800 chars, 150 overlap)
   ├── Generate embeddings (Gemini gemini-embedding-001)
   └── Store in FAISS vector index

Ask Question
   │
   ├── Embed the question
   ├── Search FAISS for top-3 relevant chunks
   ├── Build prompt with context + question
   └── Generate answer (Gemini gemini-2.5-flash)
```

---

## Manual vs LangChain Pipeline

| Feature | Manual (`rag_service`) | LangChain (`langchain_service`) |
|---|---|---|
| Embeddings | `google-generativeai` directly | `GoogleGenerativeAIEmbeddings` |
| Vector Store | `faiss-cpu` + `numpy` manually | `FAISS.from_texts()` |
| LLM Call | `google-generativeai` directly | `ChatGoogleGenerativeAI` |
| Chain | Manual prompt + call | LCEL pipe (`\|` operator) |
