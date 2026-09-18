import json
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.database import get_db
from app.models import Document, User
from app.schemas.documents import DocumentIndexResponse, DocumentResponse
from app.rag.vectorstore import add_chunks_to_vectorstore
from app.services.documents import (
    create_document,
    get_user_document,
    list_user_documents,
    update_document_status,
)

UPLOAD_DIR = Path("uploads")
CHUNKS_DIR = UPLOAD_DIR / "chunks"
METADATA_FILE = UPLOAD_DIR / "documents.json"
allowed_types = {"application/pdf", "text/csv", "text/plain"}
router = APIRouter(prefix="/documents", tags=["documents"])


def load_docs() -> list[dict[str, str | int]]:
    if not METADATA_FILE.exists():
        return []

    if METADATA_FILE.stat().st_size == 0:
        return []

    with open(METADATA_FILE, "r") as f:
        return json.load(f)


def upload_docs(docs: list[dict[str, str | int]]) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE, "w") as f:
        json.dump(docs, f, indent=4)


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


@router.post(
    "/{document_id}/index",
    response_model=DocumentIndexResponse,
)
async def index_doc(
    document_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
) -> DocumentIndexResponse:
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

    if document.content_type != "text/plain":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only plain text files can be indexed.",
        )

    text = file_path.read_text(encoding="utf-8")

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is empty.",
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

    return DocumentIndexResponse(
        document_id=document.id,
        original_filename=document.original_filename,
        stored_filename=document.stored_filename,
        content_type=document.content_type,
        size_bytes=document.size_bytes,
        status=document.status,
        created_at=document.created_at,
        length=len(text),
        num_chunks=len(chunks),
        embedded_chunks=embedded_chunks_count,
    )


@router.get("/{document_id}/chunks")
def get_doc_chunks(    document_id: uuid.UUID,
    session: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)]) -> list[dict[str, str | int]]:
    path = CHUNKS_DIR / f"{document_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Chunks not found for the specified document")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks

class request(BaseModel) :
    query : str

@router.post("/{document_id}/chunks/search")
def search_doc_chunks(document_id: str, request: request) -> list[dict[str, str | int]]:
    path = CHUNKS_DIR / f"{document_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Chunks not found for the specified document")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    query = request.query.lower()
    results =[]

    for chunk in chunks :
        text = str(chunk["text"]).lower()
        score = text.count(query)

        if score > 0 :
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "score": score
            })
    return results

@router.post("/{document_id}/semantic-search")
def semantic_search_doc(document_id: str, request: request) :
    from app.rag.retriever import get_retriever

    retriever = get_retriever(document_id=document_id)
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    results = retriever.invoke(query)

    return [
        {
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "document_id": doc.metadata.get("document_id", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "text": doc.page_content,
        }
        for doc in results
    ]
@router.delete("/{document_id}")
def delete_doc(document_id: str) -> dict[str, str]:
    docs = load_docs()

    document_to_delete = None

    for doc in docs:
        if doc["document_id"] == document_id:
            document_to_delete = doc
            break

    if document_to_delete is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = UPLOAD_DIR / str(document_to_delete["stored_filename"])

    if file_path.exists():
        file_path.unlink()

    docs.remove(document_to_delete)
    upload_docs(docs)

    return {
        "document_id": document_id,
        "detail": "Document deleted successfully.",
    }

