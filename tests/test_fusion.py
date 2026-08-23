from langchain_core.documents import Document

from app.rag import fusion


def make_document(chunk_id: str) -> Document:
    return Document(page_content=f"content for {chunk_id}", metadata={"chunk_id": chunk_id})


def test_rrf_rewards_document_returned_by_multiple_queries():
    repeated = make_document("repeated")
    first_once = make_document("first-once")
    second_once = make_document("second-once")

    ranked = fusion.RRF([
        [first_once, repeated],
        [second_once, repeated],
        [repeated],
    ])

    assert [doc.metadata["chunk_id"] for doc in ranked] == [
        "repeated", "first-once", "second-once"
    ]


def test_reranker_passes_query_and_documents_to_cached_compressor(monkeypatch):
    docs = [make_document("a"), make_document("b")]

    class FakeCompressor:
        def compress_documents(self, documents, query):
            assert documents == docs
            assert query == "question"
            return [documents[1]]

    monkeypatch.setattr(fusion, "get_reranker", lambda: FakeCompressor())
    assert fusion.reranker("question", docs) == {"documents": [docs[1]]}
