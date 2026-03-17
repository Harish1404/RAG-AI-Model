import google.generativeai as genai
from app.core.config import settings


# configure API key
genai.configure(api_key=settings.GEMINI_API_KEY)


def generate_embedding(text: str):
    """
    Converts text into vector embedding
    """

    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text
    )

    return result["embedding"]