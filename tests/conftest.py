import os
import asyncio
import shutil
import tempfile
from pathlib import Path

import httpx
import pytest


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")

from app.main import create_app


class ASGITestClient:
    def __init__(self):
        self.app = create_app()

    def request(self, method: str, url: str, **kwargs):
        async def send_request():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as async_client:
                return await async_client.request(method, url, **kwargs)

        return asyncio.run(send_request())

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)


@pytest.fixture
def client():
    return ASGITestClient()


@pytest.fixture
def isolated_document_storage(monkeypatch):
    from app.api.routes import documents

    test_root = Path(tempfile.mkdtemp(prefix="rag-backend-tests-"))
    upload_dir = test_root / "uploads"
    chunks_dir = upload_dir / "chunks"
    metadata_file = upload_dir / "documents.json"

    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "CHUNKS_DIR", chunks_dir)
    monkeypatch.setattr(documents, "METADATA_FILE", metadata_file)

    yield

    shutil.rmtree(test_root, ignore_errors=True)
