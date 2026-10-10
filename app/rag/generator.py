from langchain_groq import ChatGroq

from app.core.config import get_settings


def get_llm() -> ChatGroq:
    settings = get_settings()

    return ChatGroq(
        model=settings.groq_generation_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def generate_answer(question: str, context: str) -> str:
    llm = get_llm()

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    # Return only the assistant's answer text, not provider metadata.
    return response.text


def generate_general_answer(question: str) -> str:
    llm = get_llm()
    response = llm.invoke(
        f"""You are a helpful assistant. Answer the user's question clearly and
concisely. If the question requires current information that you cannot verify,
say so.

Question:
{question}

Answer:
"""
    )
    return response.text
