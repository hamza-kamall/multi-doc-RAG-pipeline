from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from functools import lru_cache

from app.core.config import get_settings

@lru_cache
def get_reranker() -> CrossEncoderReranker:
    settings = get_settings()
    model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2"
    )
    return CrossEncoderReranker(
        model=model,
        top_n=3,
    )

def query_indexer(query: str) -> list[str]:
    template = """You are an AI assistant. Generate five alternative search
    questions for the user's question. The alternatives will be used to retrieve
    relevant chunks from a vector store. Return only the five questions, with one
    question per line. Do not number them."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "The user question is: {question}"),
        ]
    )

    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.7,
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": query})
    res = [line.strip() for line in response.splitlines() if line.strip()]
    return res[:3]


def RRF(scores: list[list[Document]], k: int = 60) -> list[Document]:
    fused_scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    for docs in scores:
        for index, d in enumerate(docs):
            chunk_id = str(d.metadata.get("chunk_id"))
            documents[chunk_id] = d

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0

            fused_scores[chunk_id] += 1 / (index + k)

    ranked_chunk_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

                
    return [documents[chunk_id] for chunk_id in ranked_chunk_ids]
            
def reranker(query: str, docs: list[Document]) -> dict[str, list[Document]]:
    compressor = get_reranker()
    compressed_docs = compressor.compress_documents(docs, query)

    return {"documents": compressed_docs}
