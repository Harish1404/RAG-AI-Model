from app.services.embedding_service import generate_embedding
from app.services.vector_service import VectorStore
from app.ai.gemini_llm import ask_gemini


# global vector store (temporary for now)
vector_store = None


def initialize_vector_store(embeddings, chunks):

    global vector_store

    dimension = len(embeddings[0])

    vector_store = VectorStore(dimension)

    vector_store.add_vectors(embeddings, chunks)


def ask_question(question: str):

    if vector_store is None:
        raise ValueError("No document uploaded yet. Please upload a file first.")

    # convert question to embedding
    query_embedding = generate_embedding(question)

    # search similar chunks
    relevant_chunks = vector_store.search(query_embedding)

    # combine chunks as context
    context = "\n".join(relevant_chunks)

    prompt = f"""
        Use the following context to answer the question.

        Context: {context}

        Question: {question}

        Answer clearly.
        """

    response = ask_gemini(prompt)

    return response