from langchain_core.documents import Document

from app.api.routes import chat


def test_chat_rejects_blank_question(client):
    response = client.post(
        "/chat",
        json={"document_id": "doc-1", "query": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question is required."


def test_chat_returns_graph_answer_and_context(client, monkeypatch):
    document = Document(
        page_content="Supporting text",
        metadata={
            "chunk_id": "doc-1-0",
            "document_id": "doc-1",
            "chunk_index": 0,
        },
    )

    class FakeGraph:
        def invoke(self, state):
            assert state == {
                "question": "What is supported?",
                "document_id": "doc-1",
                "retry_count": 0,
            }
            return {"generation": "Supported answer", "documents": [document]}

    fake_retriever = object()
    monkeypatch.setattr(chat, "get_retriever", lambda document_id: fake_retriever)
    monkeypatch.setattr(
        chat,
        "build_graph",
        lambda retriever: FakeGraph(),
    )

    response = client.post(
        "/chat",
        json={"document_id": "doc-1", "query": "What is supported?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Supported answer",
        "context": [{
            "chunk_id": "doc-1-0",
            "document_id": "doc-1",
            "chunk_index": 0,
            "text": "Supporting text",
        }],
    }
