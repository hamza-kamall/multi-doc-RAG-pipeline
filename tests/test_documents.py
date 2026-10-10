from io import BytesIO

from langchain_core.documents import Document as LangChainDocument
from pypdf import PdfWriter

from app.api.routes import documents


def upload_text(
    client,
    text: bytes = b"Python is useful for backend APIs.",
):
    return client.post(
        "/documents/upload",
        files={"file": ("notes.txt", text, "text/plain")},
    )


def upload_pdf(client, contents: bytes):
    return client.post(
        "/documents/upload",
        files={"file": ("notes.pdf", contents, "application/pdf")},
    )


def test_upload_list_and_get_owned_document(
    authenticated_client,
    isolated_document_storage,
):
    upload_response = upload_text(authenticated_client)

    assert upload_response.status_code == 201
    uploaded = upload_response.json()
    document_id = uploaded["document_id"]
    assert uploaded["status"] == "uploaded"
    assert uploaded["size_bytes"] > 0

    list_response = authenticated_client.get("/documents")
    assert list_response.status_code == 200
    assert list_response.json() == [uploaded]

    get_response = authenticated_client.get(f"/documents/{document_id}")
    assert get_response.status_code == 200
    assert get_response.json() == uploaded


def test_upload_rejects_unsupported_file_type(
    authenticated_client,
    isolated_document_storage,
):
    response = authenticated_client.post(
        "/documents/upload",
        files={"file": ("image.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_user_cannot_access_another_users_document(
    authenticated_client_factory,
    isolated_document_storage,
):
    owner = authenticated_client_factory("owner_user")
    intruder = authenticated_client_factory("intruder_user")

    uploaded = upload_text(owner).json()
    response = intruder.get(f"/documents/{uploaded['document_id']}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


def test_index_creates_chunks_without_real_embeddings(
    authenticated_client,
    monkeypatch,
    isolated_document_storage,
):
    uploaded = upload_text(
        authenticated_client,
        b"FastAPI and RAG. " * 150,
    ).json()
    monkeypatch.setattr(
        documents,
        "add_chunks_to_vectorstore",
        lambda chunks: len(chunks),
    )

    response = authenticated_client.post(
        f"/documents/{uploaded['document_id']}/index"
    )

    assert response.status_code == 200
    indexed = response.json()
    assert indexed["status"] == "indexed"
    assert indexed["num_chunks"] >= 2
    assert indexed["embedded_chunks"] == indexed["num_chunks"]

    chunks_response = authenticated_client.get(
        f"/documents/{uploaded['document_id']}/chunks"
    )
    chunks = chunks_response.json()
    assert chunks_response.status_code == 200
    assert len(chunks) == indexed["num_chunks"]
    assert all(
        chunk["document_id"] == uploaded["document_id"]
        for chunk in chunks
    )


def test_index_rejects_empty_document(
    authenticated_client,
    isolated_document_storage,
):
    uploaded = upload_text(authenticated_client, b"   ").json()

    response = authenticated_client.post(
        f"/documents/{uploaded['document_id']}/index"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "File is empty."


def test_index_pdf_extracts_and_embeds_text(
    authenticated_client,
    monkeypatch,
    isolated_document_storage,
):
    uploaded = upload_pdf(
        authenticated_client,
        b"placeholder PDF bytes",
    ).json()
    extracted_calls: list[tuple[str, str]] = []

    def fake_extract(file_path, content_type):
        extracted_calls.append((file_path.name, content_type))
        return "PDF text about FastAPI and retrieval."

    monkeypatch.setattr(
        documents,
        "extract_document_text",
        fake_extract,
    )
    monkeypatch.setattr(
        documents,
        "add_chunks_to_vectorstore",
        lambda chunks: len(chunks),
    )

    response = authenticated_client.post(
        f"/documents/{uploaded['document_id']}/index"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    assert response.json()["embedded_chunks"] == 1
    assert extracted_calls == [
        (uploaded["stored_filename"], "application/pdf")
    ]


def test_index_pdf_rejects_document_without_extractable_text(
    authenticated_client,
    isolated_document_storage,
):
    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(buffer)
    uploaded = upload_pdf(authenticated_client, buffer.getvalue()).json()

    response = authenticated_client.post(
        f"/documents/{uploaded['document_id']}/index"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "PDF contains no extractable text."
    )


def test_semantic_search_uses_owned_document_retriever(
    authenticated_client,
    monkeypatch,
    isolated_document_storage,
):
    uploaded = upload_text(authenticated_client, b"Backend content").json()
    document_id = uploaded["document_id"]
    monkeypatch.setattr(
        documents,
        "add_chunks_to_vectorstore",
        lambda chunks: len(chunks),
    )
    index_response = authenticated_client.post(
        f"/documents/{document_id}/index"
    )
    assert index_response.status_code == 200

    class FakeRetriever:
        def invoke(self, query):
            assert query == "backend"
            return [
                LangChainDocument(
                    page_content="Backend content",
                    metadata={
                        "chunk_id": f"{document_id}-0",
                        "document_id": document_id,
                        "chunk_index": 0,
                    },
                )
            ]

    from app.rag import retriever as retriever_module

    monkeypatch.setattr(
        retriever_module,
        "get_retriever",
        lambda document_id: FakeRetriever(),
    )

    response = authenticated_client.post(
        f"/documents/{document_id}/semantic-search",
        json={"query": "backend"},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "chunk_id": f"{document_id}-0",
            "document_id": document_id,
            "chunk_index": 0,
            "text": "Backend content",
        }
    ]


def test_delete_removes_database_row_and_files(
    authenticated_client,
    monkeypatch,
    isolated_document_storage,
):
    uploaded = upload_text(authenticated_client).json()
    document_id = uploaded["document_id"]
    uploaded_file = documents.UPLOAD_DIR / uploaded["stored_filename"]
    chunk_file = documents.CHUNKS_DIR / f"{document_id}.json"
    chunk_file.parent.mkdir(parents=True, exist_ok=True)
    chunk_file.write_text("[]", encoding="utf-8")

    deleted_vector_ids: list[str] = []
    monkeypatch.setattr(
        documents,
        "delete_document_from_vectorstore",
        deleted_vector_ids.append,
    )

    response = authenticated_client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["document_id"] == document_id
    assert deleted_vector_ids == [document_id]
    assert not uploaded_file.exists()
    assert not chunk_file.exists()
    assert authenticated_client.get(
        f"/documents/{document_id}"
    ).status_code == 404
