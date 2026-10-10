import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from fastapi.concurrency import run_in_threadpool

from app.api.dependencies.auth import get_current_user
from app.db.database import get_db
from app.models import Document, User
from app.schemas.documents import ChunkResponse, DocumentResponse, DocumentDeleteResponse, SearchRequest
from app.services.documents import (
    create_document,
    get_user_document,
    list_user_documents,
    update_document_status,
    delete_document
)
from app.services.text_extraction import (
    TextExtractionError,
    extract_document_text,
)
from app.rag.vectorstore import add_chunks_to_vectorstore,delete_document_from_vectorstore

UPLOAD_DIR = Path("uploads")
CHUNKS_DIR = UPLOAD_DIR / "chunks"
allowed_types = {"application/pdf", "text/csv", "text/plain"}
router = APIRouter(prefix="/documents", tags=["documents"])


# FastAPI receives UploadFile object
# Endpoint validates file type
# Endpoint reads bytes
# Endpoint saves bytes to uploads/
# Endpoint returns metadata
@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: Annotated[UploadFile, File()],
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Document:
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF, CSV, and TXT files are allowed.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    document_id = uuid.uuid4()
    original_filename = Path(file.filename or "upload").name
    stored_filename = f"{document_id}-{original_filename}"
    file_path = UPLOAD_DIR / stored_filename

    contents = await file.read()
    file_path.write_bytes(contents)

    try:
        return await create_document(
            session,
            document_id=document_id,
            user_id=current_user.id,
            original_filename=original_filename,
            stored_filename=stored_filename,
            content_type=file.content_type,
            size_bytes=len(contents),
        )
    except SQLAlchemyError:
        await session.rollback()
        file_path.unlink(missing_ok=True)
        raise


@router.get(
    "",
    response_model=list[DocumentResponse],
)
async def list_docs(
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[Document]:
    return await list_user_documents(
        session,
        current_user.id,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_doc(
    document_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> Document:
    document = await get_user_document(
        session,
        document_id=document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document


@router.post("/{document_id}/index")
async def index_doc(
    document_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> dict[str, str | int]:
    document = await get_user_document(
        session,
        document_id=document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    file_path = UPLOAD_DIR / document.stored_filename

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found on disk.",
        )

    if document.content_type not in {
        "text/plain",
        "application/pdf",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only plain text and PDF files can be indexed.",
        )

    try:
        text = await run_in_threadpool(
            extract_document_text,
            file_path,
            document.content_type,
        )
    except TextExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if not text.strip():
        detail = (
            "PDF contains no extractable text."
            if document.content_type == "application/pdf"
            else "File is empty."
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)

    document_id_text = str(document.id)

    chunk_records = [
        {
            "chunk_id": f"{document_id_text}-{index}",
            "document_id": document_id_text,
            "chunk_index": index,
            "text": chunk,
        }
        for index, chunk in enumerate(chunks)
    ]

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_path = CHUNKS_DIR / f"{document_id}.json"

    with chunk_path.open("w", encoding="utf-8") as file:
        json.dump(chunk_records, file, indent=4)

    embedded_chunks_count = await run_in_threadpool(
        add_chunks_to_vectorstore,
        chunk_records,
    )

    await update_document_status(
        session,
        document,
        "indexed",
    )

    return {
        "document_id": document_id_text,
        "original_filename": document.original_filename,
        "stored_filename": document.stored_filename,
        "content_type": document.content_type,
        "size_bytes": document.size_bytes,
        "status": document.status,
        "length": len(text),
        "num_chunks": len(chunks),
        "embedded_chunks": embedded_chunks_count,
    }


@router.get("/{document_id}/chunks",response_model=list[ChunkResponse])
async def get_doc_chunks(
    document_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)]
    ) -> list[ChunkResponse]:

    document = await get_user_document(
            session,
            document_id=document_id,
            user_id=current_user.id,
        )
    
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    
    chunk_path = CHUNKS_DIR / f"{document.id}.json"

    if not chunk_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chunks not found for the specified document.",
        )
    
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks

@router.post(
    "/{document_id}/semantic-search",
    response_model=list[ChunkResponse],
)
async def semantic_search_doc(
    document_id: uuid.UUID,
    request: SearchRequest,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[ChunkResponse]:
    from app.rag.retriever import get_retriever

    document = await get_user_document(
        session,
        document_id=document_id,
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
            detail="Query is required.",
        )

    retriever = get_retriever(
        document_id=str(document.id),
    )

    results = await run_in_threadpool(
        retriever.invoke,
        query,
    )

    return [
        ChunkResponse(
            chunk_id=str(result.metadata.get("chunk_id", "unknown")),
            document_id=str(result.metadata.get("document_id", document.id)),
            chunk_index=int(result.metadata.get("chunk_index", -1)),
            text=result.page_content,
        )
        for result in results
    ]

@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
)
async def delete_doc(
    document_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DocumentDeleteResponse:
    document = await get_user_document(
        session,
        document_id=document_id,
        user_id=current_user.id,
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    uploaded_file = UPLOAD_DIR / document.stored_filename
    chunk_file = CHUNKS_DIR / f"{document.id}.json"

    await run_in_threadpool(
        delete_document_from_vectorstore,
        str(document.id),
    )

    uploaded_file.unlink(missing_ok=True)
    chunk_file.unlink(missing_ok=True)

    await delete_document(
        session,
        document,
    )

    return DocumentDeleteResponse(
        document_id=document.id,
        detail="Document deleted successfully.",
    )
