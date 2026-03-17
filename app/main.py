from fastapi import FastAPI
from app.routers import ai_router

app = FastAPI()

@app.get("/")
def home():
    return {"message": "RAG API Running"}

app.include_router(ai_router.router)