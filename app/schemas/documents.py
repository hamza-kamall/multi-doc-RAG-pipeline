import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        validate_by_name=True,
    )

    document_id: uuid.UUID = Field(validation_alias="id")
    original_filename: str
    stored_filename: str
    content_type: str
    size_bytes: int
    status: str
    created_at: datetime


class DocumentIndexResponse(DocumentResponse):
    length: int
    num_chunks: int
    embedded_chunks: int
