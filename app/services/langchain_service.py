from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import pinecone
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.core.config import settings

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=settings.GEMINI_API_KEY
)

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=settings.GEMINI_API_KEY
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a friendly and intelligent assistant.

Answer the question using ONLY the provided context.

If the answer is not found, say:
"I couldn't find this in the document. Please upload more relevant data."

---------------------
Context:
{context}
---------------------

Question:
{question}

Instructions:
- Start with a short direct answer
- Then explain in detail
- Use headings and bullet points
- Keep it simple and easy to understand
- Add examples if helpful
- Be concise but informative

Answer:
"""
)

vector_store = None
qa_chain = None


def initialize_vector_store(chunks: list[str]):
    global vector_store, qa_chain

    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(ssearch_kwargs={"k": 3, "score_threshold": 0.7})

    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )


def ask_question(question: str) -> str:
    
    if qa_chain is None:
        raise ValueError("No document uploaded yet. Please upload a file first.")

    return qa_chain.invoke(question)
