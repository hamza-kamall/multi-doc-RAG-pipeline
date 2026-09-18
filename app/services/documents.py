import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document


async def create_document(
    session: AsyncSession,
    *,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
    original_filename: str,
    stored_filename: str,
    content_type: str,
    size_bytes: int,
) -> Document:
    document = Document(
        id=document_id,
        user_id=user_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        content_type=content_type,
        size_bytes=size_bytes,
        status="uploaded",
    )

    session.add(document)
    await session.commit()
    await session.refresh(document)

    return document


async def list_user_documents(
    session: AsyncSession,
    user_id: uuid.UUID,
) -> list[Document]:
    statement = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    result = await session.execute(statement)

    return list(result.scalars().all())


async def get_user_document(
    session: AsyncSession,
    document_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Document | None:
    statement = select(Document).where(
        Document.id == document_id,
        Document.user_id == user_id,
    )
    result = await session.execute(statement)

    return result.scalar_one_or_none()


async def update_document_status(
    session: AsyncSession,
    document: Document,
    status: str,
) -> None:
    document.status = status
    await session.commit()
