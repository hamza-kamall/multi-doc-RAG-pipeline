from langchain_core.documents import Document

from app.api.routes import documents


def upload_text(client, text: bytes = b"Python is useful for backend APIs."):
    return client.post(
        "/documents/upload",
        files={"file": ("notes.txt", text, "text/plain")},
    )


def test_upload_list_get_and_delete_document(client, isolated_document_storage):
    upload_response = upload_text(client)
    assert upload_response.status_code == 200
    uploaded = upload_response.json()
    document_id = uploaded["document_id"]
    assert uploaded["status"] == "uploaded"
    assert uploaded["size_bytes"] > 0

    assert client.get("/documents").json() == [uploaded]
    assert client.get(f"/documents/{document_id}").json() == uploaded

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 200
    assert delete_response.json()["document_id"] == document_id
    assert client.get(f"/documents/{document_id}").status_code == 404


def test_upload_rejects_unsupported_file_type(client, isolated_document_storage):
    response = client.post(
        "/documents/upload",
        files={"file": ("image.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_index_text_document_creates_chunks_without_real_embeddings(
    client, monkeypatch, isolated_document_storage
):
    uploaded = upload_text(client, (b"FastAPI and RAG. " * 150)).json()
    monkeypatch.setattr(documents, "add_chunks_to_vectorstore", lambda chunks: len(chunks))

    response = client.post(f"/documents/{uploaded['document_id']}/index")

    assert response.status_code == 200
    indexed = response.json()
    assert indexed["status"] == "indexed"
    assert indexed["num_chunks"] >= 2
    assert indexed["embedded_chunks"] == indexed["num_chunks"]

    chunks_response = client.get(f"/documents/{uploaded['document_id']}/chunks")
    chunks = chunks_response.json()
    assert chunks_response.status_code == 200
    assert len(chunks) == indexed["num_chunks"]
    assert all(chunk["document_id"] == uploaded["document_id"] for chunk in chunks)


def test_index_rejects_empty_text_document(client, isolated_document_storage):
    uploaded = upload_text(client, b"   ").json()
    response = client.post(f"/documents/{uploaded['document_id']}/index")

    assert response.status_code == 400
    assert response.json()["detail"] == "File is empty"


def test_keyword_chunk_search(client, monkeypatch, isolated_document_storage):
    uploaded = upload_text(client, b"Python builds APIs. Java builds APIs.").json()
    monkeypatch.setattr(documents, "add_chunks_to_vectorstore", lambda chunks: len(chunks))
    client.post(f"/documents/{uploaded['document_id']}/index")

    response = client.post(
        f"/documents/{uploaded['document_id']}/chunks/search",
        json={"query": "python"},
    )

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["score"] == 1


def test_semantic_search_uses_document_retriever(
    client, monkeypatch, isolated_document_storage
):
    class FakeRetriever:
        def invoke(self, query):
            assert query == "backend"
            return [Document(
                page_content="Backend content",
                metadata={"chunk_id": "doc-1-0", "document_id": "doc-1", "chunk_index": 0},
            )]

    from app.rag import retriever as retriever_module
    monkeypatch.setattr(retriever_module, "get_retriever", lambda document_id: FakeRetriever())

    response = client.post(
        "/documents/doc-1/semantic-search",
        json={"query": "backend"},
    )

    assert response.status_code == 200
    assert response.json()[0]["chunk_id"] == "doc-1-0"
