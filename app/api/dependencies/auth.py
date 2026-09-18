import uuid
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.database import get_db
from app.models import User

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/auth/login",
)

async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    session: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )

        subject = payload.get("sub")
        
        if not isinstance(subject, str):
            raise credentials_error

        user_id = uuid.UUID(subject)

    except (InvalidTokenError, ValueError) as exc :
        raise credentials_error from exc 

    user = await session.get(User, user_id)

    if user is None:
        raise credentials_error

    return user