from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models import User
from app.schemas.users import TokenResponse, UserCreate, UserResponse
from app.services.users import create_user, get_user_by_username,authenticate_user

router = APIRouter(
    prefix="/users",tags=["users"]
    )


@router.post("",response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserCreate,
    session: Annotated[AsyncSession, Depends(get_db)]
):
    existing_user = await get_user_by_username(session,user_data.username) 
    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        )

    try :
        return await create_user(session,user_data)
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists"
        ) from exc

