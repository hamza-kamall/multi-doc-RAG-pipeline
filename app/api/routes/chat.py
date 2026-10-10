from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.database import get_db
from app.models import User
from app.rag.pipeline import app as build_graph
from app.rag.retriever import get_retriever
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.documents import ChunkResponse
from app.services.documents import get_user_document


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat_with_document(
    request: ChatRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> ChatResponse:
    document = await get_user_document(
        session,
        document_id=request.document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.status != "indexed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document has not been indexed.",
        )

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required.",
        )

    document_id = str(document.id)
    retriever = get_retriever(document_id=document_id)
    graph = build_graph(retriever)

    result = await run_in_threadpool(
        graph.invoke,
        {
            "question": query,
            "document_id": document_id,
            "retry_count": 0,
        },
    )

    context = [
        ChunkResponse(
            chunk_id=str(item.metadata.get("chunk_id", "unknown")),
            document_id=str(
                item.metadata.get("document_id", document.id)
            ),
            chunk_index=int(item.metadata.get("chunk_index", -1)),
            text=item.page_content,
        )
        for item in result.get("documents", [])
    ]

    return ChatResponse(
        answer=str(result["generation"]),
        context=context,
    )