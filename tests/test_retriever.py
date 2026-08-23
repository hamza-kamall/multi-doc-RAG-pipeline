from app.rag import retriever


def test_retriever_filters_results_by_document_id(monkeypatch):
    captured = {}

    class FakeVectorStore:
        def as_retriever(self, **kwargs):
            captured.update(kwargs)
            return "retriever"

    monkeypatch.setattr(retriever, "get_vectorstore", lambda: FakeVectorStore())
    result = retriever.get_retriever("doc-123")

    assert result == "retriever"
    assert captured == {
        "search_kwargs": {"k": 3, "filter": {"document_id": "doc-123"}}
    }
