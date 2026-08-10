import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/chat", tags=["chat"])

CHUNKS_DIR = Path("uploads") / "chunks"

class ChatRequest(BaseModel):
    document_id: str
    query: str




@router.post("")
def chat_with_document(request: ChatRequest) -> dict[str, str | list[dict[str, str | int]]]:
    chunks_path = CHUNKS_DIR / f"{request.document_id}.json"

    if not chunks_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Chunks not found. Index the document first.",
        )

    question = request.query.strip().lower()

    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    results = []

    for chunk in chunks:
        text = str(chunk["text"])
        score = text.lower().count(question)

        if score > 0:
            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "chunk_index": chunk["chunk_index"],
                    "score": score,
                    "text": text,
                }
            )

    results.sort(key=lambda item: item["score"], reverse=True)

    return {
        "answer": "LLM generation is not connected yet. Here are the retrieved chunks.",
        "context": results[:3],
    }