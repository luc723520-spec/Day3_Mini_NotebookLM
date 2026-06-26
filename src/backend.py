import io
import json
import os
import base64
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG API with Cosine Similarity, TTS, and Markdown Export")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons — loaded once at startup to avoid reloading on every request
# ---------------------------------------------------------------------------

_embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
_llm = Ollama(model="llama3.2", temperature=0.0)
_vector_db: Chroma | None = None

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """ROLE: Bạn là trợ lý AI nghiêm túc, chỉ trả lời dựa trên TÀI LIỆU ĐƯỢC CUNG CẤP.

QUY TẮC TUYỆT ĐỐI:
1. Chỉ dùng thông tin trong phần Ngữ cảnh bên dưới.
2. Nếu không tìm thấy câu trả lời, phải nói: "Tôi không biết thông tin này vì không có trong tài liệu."
3. Không bịa, không đoán mò, không mở rộng ngoài phạm vi văn bản.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Câu trả lời tiếng Việt ngắn gọn, chính xác:"""

_PROMPT = PromptTemplate(template=_PROMPT_TEMPLATE, input_variables=["context", "question"])

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str


class GenerateQuizRequest(BaseModel):
    topic: str
    num_questions: int = 5


class SourceDoc(BaseModel):
    text: str
    cosine_score: float


class ChatResponse(BaseModel):
    answer: str
    audio_base64: str
    sources: list[SourceDoc]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "document_loaded": _vector_db is not None}


@app.post("/upload", summary="Upload and index a PDF document")
async def upload_pdf(file: UploadFile = File(...)):
    global _vector_db

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        raw_text = "".join(page.extract_text() or "" for page in reader.pages)
    finally:
        os.remove(tmp_path)

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=75, separators=["\n\n", "\n", " ", ""]
    ).split_text(raw_text)

    # hnsw:space=cosine makes similarity_search_with_score return cosine distance
    # so we can invert it to a true similarity score in /chat
    _vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=_embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return {"message": "Document indexed successfully.", "chunk_count": len(chunks)}


@app.post("/chat", response_model=ChatResponse, summary="RAG chat with cosine scores and TTS")
async def chat(request: ChatRequest):
    if _vector_db is None:
        raise HTTPException(status_code=400, detail="No document uploaded. POST to /upload first.")

    # ── Feature 1: Cosine Similarity Visualization ──────────────────────────
    # similarity_search_with_score returns (Document, distance).
    # With hnsw:space=cosine the distance = 1 − cosine_similarity, so we invert
    # to surface a human-readable score where 1.0 = perfect match.
    docs_and_scores = _vector_db.similarity_search_with_score(request.query, k=4)

    context = "\n\n".join(doc.page_content for doc, _ in docs_and_scores)
    sources = [
        SourceDoc(
            text=doc.page_content,
            cosine_score=round(1.0 - float(score), 4),
        )
        for doc, score in docs_and_scores
    ]

    # ── LLM Generation ───────────────────────────────────────────────────────
    answer: str = _llm.invoke(_PROMPT.format(context=context, question=request.query))

    # ── Feature 2: Text-to-Speech → Base64 ──────────────────────────────────
    # gTTS calls Google's TTS API; requires internet access.
    # Falls back to empty string so the rest of the response still works.
    try:
        tts = gTTS(text=answer, lang="vi", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_base64 = "data:audio/mp3;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        audio_base64 = ""

    # ── Feature 3: Markdown Export-Ready JSON ────────────────────────────────
    # The response maps directly to .md:
    #   # Answer\n{answer}\n\n## Sources\n- **[{cosine_score}]** {text}
    return ChatResponse(answer=answer, audio_base64=audio_base64, sources=sources)


@app.post("/generate-quiz", summary="Generate a multiple-choice quiz from indexed document")
async def generate_quiz(request: GenerateQuizRequest):
    if _vector_db is None:
        raise HTTPException(status_code=400, detail="No document uploaded. POST to /upload first.")

    docs = _vector_db.similarity_search(request.topic, k=8)
    context = "\n\n".join(doc.page_content for doc in docs)

    # The schema mirrors the frontend's expected quiz JSON so the response can be
    # loaded directly into app state without any client-side transformation.
    prompt = f"""You are a JSON-only AI. Based on the CONTEXT below, generate {request.num_questions} multiple-choice quiz questions about "{request.topic}".

STRICT RULES:
- Output ONLY valid JSON. No text outside the JSON structure.
- Use exactly this schema:

{{
  "title": "Quiz: {request.topic}",
  "questions": [
    {{
      "question": "Question text here?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctAnswer": 0,
      "explanation": "Explanation of why the answer is correct."
    }}
  ]
}}

CONTEXT:
{context}"""

    quiz_llm = Ollama(model="llama3.2", temperature=0.1, format="json")
    raw = quiz_llm.invoke(prompt)

    try:
        quiz_data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="LLM produced invalid JSON. Please try again.")

    questions = quiz_data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise HTTPException(status_code=422, detail="LLM response missing valid 'questions' array. Please try again.")

    return quiz_data
