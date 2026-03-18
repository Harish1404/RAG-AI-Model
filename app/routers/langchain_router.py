import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.pdf_sevice import extract_elements_from_pdf
from app.services.chunk_service import split_text_into_chunks
from app.models.question_model import QuestionRequest
import app.services.langchain_service as langchain_service

router = APIRouter(prefix="/langchain", tags=["LangChain AI"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    texts = extract_elements_from_pdf(file_path)
    chunks = split_text_into_chunks("\n".join(texts))
    langchain_service.initialize_vector_store(chunks)

    return {"message": f"File '{file.filename}' processed successfully.", "chunks": len(chunks)}


@router.post("/ask")
def ask_ai(request: QuestionRequest):
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        answer = langchain_service.ask_question(request.question)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
