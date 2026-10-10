from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


from app.core.config import get_settings

def get_vectorstore() -> Chroma:
    settings = get_settings()

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name=settings.chroma_collection,
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )

def add_chunks_to_vectorstore(chunks: list[dict[str, str | int]]) -> int:
    vectorstore = get_vectorstore()

    documents = [
        Document(
            page_content=str(chunk["text"]),
            metadata={
                "chunk_id": str(chunk["chunk_id"]),
                "document_id": str(chunk["document_id"]),
                "chunk_index": chunk["chunk_index"],
            },
        )
        for chunk in chunks
    ]

    ids = [str(chunk["chunk_id"]) for chunk in chunks]

    vectorstore.add_documents(documents,ids=ids)


    return len(documents)

def delete_document_from_vectorstore(
    document_id: str,
) -> None:
    vectorstore = get_vectorstore()
    vectorstore.delete(
        where={"document_id": document_id},
    )