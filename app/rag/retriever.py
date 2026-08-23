from app.rag.vectorstore import get_vectorstore

def get_retriever(document_id: str):
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": 3, "filter": {"document_id": document_id}}
    )
