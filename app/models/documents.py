import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.users import User   

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
            Uuid,
            primary_key=True,
            default=uuid.uuid4,
        )

    owner: Mapped["User"] = relationship(
        back_populates="documents",
    )
    
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),  
        nullable=False,
        index=True
        )

    

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable = False            
    )

    stored_filename: Mapped[str] = mapped_column(
        String(255),
        nullable = False 
    )

    content_type: Mapped[str] = mapped_column(
    String(100),
    nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )

    size_bytes: Mapped[int] = mapped_column(
        nullable=False
    )




    