import uuid

from pydantic import BaseModel, Field

from app.schemas.documents import ChunkResponse


class ChatRequest(BaseModel):
    document_id: uuid.UUID
    query: str = Field(
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    answer: str
    context: list[ChunkResponse]