from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password,verify_password
from app.models import User
from app.schemas.users import UserCreate

async def get_user_by_username(
        session : AsyncSession,
        username : str,
) -> User | None:

    statement = select(User).where(User.username == username)
    result = await session.execute(statement)

    return result.scalar_one_or_none()

async def create_user(
        session : AsyncSession,
        user_data : UserCreate) -> User:

    user = User(username=user_data.username, password_hash=hash_password(user_data.password))
    session.add(user)
    await session.commit()
    await session.refresh(user)

    return user

async def authenticate_user(
    session: AsyncSession,
    username: str,
    password: str,
) -> User | None:
    user = await get_user_by_username(session, username)

    if user is None:
        return None

    if not verify_password(password, user.password_hash):
        return None

    return user