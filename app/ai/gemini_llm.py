import google.generativeai as genai
from app.core.config import settings

# configure gemini with our api key
genai.configure(api_key=settings.GEMINI_API_KEY)

# create model instance
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")

def generate_response(prompt: str) -> str:
    response = gemini_model.generate_content(prompt)
    return response.text


# alias used by rag_service
ask_gemini = generate_response