from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag.pipeline import app as build_graph
from app.rag.retriever import get_retriever
router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    document_id: str
    query: str


@router.post("")
def chat_with_document(
    request: ChatRequest,
) -> dict[str, str | list[dict[str, str | int]]]:
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question is required.")

    retriever = get_retriever(document_id=request.document_id)
    graph = build_graph(retriever)
    result = graph.invoke(
        {
            "question": query,
            "document_id": request.document_id,
            "retry_count": 0,
        }
    )
    result_documents = result.get("documents", [])


    context = [
        {
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "document_id": doc.metadata.get("document_id", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "text": doc.page_content,
        }
        for doc in result_documents
    ]

    return {
        "answer": result["generation"],
        "context": context,
    }


