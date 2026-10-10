from langchain_core.documents import Document as LangChainDocument
import httpx
import pytest
from groq import APIConnectionError

from app.api.routes import chat, documents


def upload_text(client, text: bytes = b"Supporting text"):
    return client.post(
        "/documents/upload",
        files={"file": ("notes.txt", text, "text/plain")},
    )


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout, APIConnectionError])
def test_chat_returns_503_on_provider_transport_failure(
    authenticated_client,
    monkeypatch,
    isolated_document_storage,
    error_type,
):
    uploaded = upload_and_index(authenticated_client, monkeypatch)

    class FailingGraph:
        def invoke(self, state):
            if error_type is APIConnectionError:
                raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))
            raise error_type("SSL connection failed; private provider details")

    monkeypatch.setattr(chat, "get_retriever", lambda document_id: object())
    monkeypatch.setattr(chat, "build_graph", lambda retriever: FailingGraph())

    response = authenticated_client.post(
        "/chat",
        json={"document_id": uploaded["document_id"], "query": "What is an LLM?"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "AI provider temporarily unavailable. Please try again shortly."
    }


def upload_and_index(client, monkeypatch):
    uploaded = upload_text(client).json()
    monkeypatch.setattr(
        documents,
        "add_chunks_to_vectorstore",
        lambda chunks: len(chunks),
    )
    response = client.post(
        f"/documents/{uploaded['document_id']}/index"
    )
    assert response.status_code == 200
    return uploaded


def test_chat_rejects_unindexed_document(
    authenticated_client,
    isolated_document_storage,
):
    uploaded = upload_text(authenticated_client).json()

    response = authenticated_client.post(
        "/chat",
        json={
            "document_id": uploaded["document_id"],
            "query": "What is supported?",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Document has not been indexed."


def test_chat_rejects_another_users_document(
    authenticated_client_factory,
    isolated_document_storage,
):
    owner = authenticated_client_factory("chat_owner")
    intruder = authenticated_client_factory("chat_intruder")
    uploaded = upload_text(owner).json()

    response = intruder.post(
        "/chat",
        json={
            "document_id": uploaded["document_id"],
            "query": "Reveal this document",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_chat_rejects_blank_question(
    authenticated_client,
    monkeypatch,
    isolated_document_storage,
):
    uploaded = upload_and_index(authenticated_client, monkeypatch)

    response = authenticated_client.post(
        "/chat",
        json={
            "document_id": uploaded["document_id"],
            "query": "   ",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question is required."


def test_chat_returns_graph_answer_and_context(
    authenticated_client,
    monkeypatch,
    isolated_document_storage,
):
    uploaded = upload_and_index(authenticated_client, monkeypatch)
    document_id = uploaded["document_id"]
    context_document = LangChainDocument(
        page_content="Supporting text",
        metadata={
            "chunk_id": f"{document_id}-0",
            "document_id": document_id,
            "chunk_index": 0,
        },
    )

    class FakeGraph:
        def invoke(self, state):
            assert state == {
                "question": "What is supported?",
                "document_id": document_id,
                "retry_count": 0,
            }
            return {
                "generation": "Supported answer",
                "documents": [context_document],
            }

    fake_retriever = object()

    def fake_get_retriever(document_id):
        assert document_id == uploaded["document_id"]
        return fake_retriever

    monkeypatch.setattr(
        chat,
        "get_retriever",
        fake_get_retriever,
    )
    monkeypatch.setattr(
        chat,
        "build_graph",
        lambda retriever: FakeGraph(),
    )

    response = authenticated_client.post(
        "/chat",
        json={
            "document_id": document_id,
            "query": "What is supported?",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Supported answer",
        "context": [
            {
                "chunk_id": f"{document_id}-0",
                "document_id": document_id,
                "chunk_index": 0,
                "text": "Supporting text",
            }
        ],
    }
