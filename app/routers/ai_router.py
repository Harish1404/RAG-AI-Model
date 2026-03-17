import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.services.pdf_sevice import extract_elements_from_pdf
from app.services.chunk_service import split_text_into_chunks
from app.services.embedding_service import generate_embedding
import app.services.rag_service as rag_service

router = APIRouter(tags=["Gemini AI"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    texts = extract_elements_from_pdf(file_path)
    full_text = "\n".join(texts)
    chunks = split_text_into_chunks(full_text)
    embeddings = [generate_embedding(chunk) for chunk in chunks]
    rag_service.initialize_vector_store(embeddings, chunks)

    return {"message": f"File '{file.filename}' processed successfully.", "chunks": len(chunks)}


@router.post("/ask")
def ask_ai(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")
    
    if not rag_service.vector_store:
        raise HTTPException(status_code=400, detail="No PDF file has been uploaded and processed.")
    
    try:
        answer = rag_service.ask_question(request.question)
        return {"answer": answer}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
