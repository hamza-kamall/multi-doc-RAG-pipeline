import os
import asyncio
import shutil
import tempfile
from pathlib import Path

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault(
    "DATABASE_URL",
    "sqlite+aiosqlite:///:memory:",
)
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-only-secret-key-that-is-at-least-32-bytes",
)
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "30")

from app.db.base import Base
from app.db.database import get_db
from app.main import create_app
from app.models import Document, User


class ASGITestClient:
    def __init__(self, app):
        self.app = app
        self.headers: dict[str, str] = {}

    def request(self, method: str, url: str, **kwargs):
        async def send_request():
            transport = httpx.ASGITransport(app=self.app)
            request_headers = {
                **self.headers,
                **kwargs.pop("headers", {}),
            }

            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as async_client:
                return await async_client.request(
                    method,
                    url,
                    headers=request_headers,
                    **kwargs,
                )

        return asyncio.run(send_request())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)


@pytest.fixture
def client(tmp_path):
    database_path = tmp_path / "test.db"

    test_engine = create_async_engine(
        f"sqlite+aiosqlite:///{database_path.as_posix()}",
    )

    TestSessionFactory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def prepare_database() -> None:
        async with test_engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_get_db():
        async with TestSessionFactory() as session:
            yield session

    asyncio.run(prepare_database())

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db

    yield ASGITestClient(app)

    app.dependency_overrides.clear()
    asyncio.run(test_engine.dispose())


def authenticate_client(client, username: str):
    password = "test_password_123"

    registration = client.post(
        "/users",
        json={
            "username": username,
            "password": password,
        },
    )
    assert registration.status_code == 201

    login = client.post(
        "/auth/login",
        data={
            "username": username,
            "password": password,
        },
    )
    assert login.status_code == 200

    token = login.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"

    return client


@pytest.fixture
def authenticated_client(client):
    return authenticate_client(client, "test_user")


@pytest.fixture
def authenticated_client_factory(client):
    def create(username: str):
        user_client = ASGITestClient(client.app)
        return authenticate_client(user_client, username)

    return create


@pytest.fixture
def isolated_document_storage(monkeypatch):
    from app.api.routes import documents

    test_root = Path(tempfile.mkdtemp(prefix="rag-backend-tests-"))
    upload_dir = test_root / "uploads"
    chunks_dir = upload_dir / "chunks"

    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "CHUNKS_DIR", chunks_dir)

    yield

    shutil.rmtree(test_root, ignore_errors=True)
