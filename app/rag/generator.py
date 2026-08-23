from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import get_settings


def get_llm() -> ChatGoogleGenerativeAI:
    settings = get_settings()

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
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

    # Gemini 3 may return structured content blocks rather than a bare string.
    # AIMessage.text normalizes both response formats to plain text.
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
