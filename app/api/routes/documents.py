
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from  app.rag.vectorstore import add_chunks_to_vectorstore
import json



from pydantic import BaseModel

UPLOAD_DIR = Path("uploads")
CHUNKS_DIR = UPLOAD_DIR / "chunks"
METADATA_FILE = UPLOAD_DIR / "documents.json"
allowed_types = {"application/pdf", "text/csv", "text/plain"}
router = APIRouter(prefix = "/documents",tags=["documents"])

def load_docs() -> list[dict[str, str | int]]:
    if not METADATA_FILE.exists():
        return []

    if METADATA_FILE.stat().st_size == 0:
        return []

    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def upload_docs(docs: list[dict[str, str | int]]) -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with open(METADATA_FILE, "w") as f:
        json.dump(docs, f, indent=4)
    
# FastAPI receives UploadFile object
# Endpoint validates file type
# Endpoint reads bytes
# Endpoint saves bytes to uploads/
# Endpoint returns metadata
@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF, CSV, and TXT files are allowed.")

    UPLOAD_DIR.mkdir(exist_ok=True)

    id = str(uuid.uuid4())
    
    uploaded_filename = Path(file.filename).name
    safe_filename = f"{id}-{uploaded_filename}"
    file_path = UPLOAD_DIR / safe_filename
    contents = await file.read()
    file_path.write_bytes(contents)


    Document = {
        "document_id": str(id),
        "original_filename": file.filename or "unknown",
        "stored_filename": safe_filename,
        "content_type": file.content_type or "unknown",
        "size_bytes": len(contents),
        "status": "uploaded",
    }
    # Loads all documents from the metadata file, appends the new document, and saves it back to the metadata file
    docs = load_docs()
    docs.append(Document)
    upload_docs(docs)

    return Document

@router.get("") 
async def list_docs() -> list[dict[str, str | int]]:
    return load_docs()

@router.get("/{document_id}") 
async def get_doc(document_id: str) -> dict[str, str | int]:
    docs = load_docs()
    for doc in docs:
        if doc["document_id"] == document_id:
            return doc
    raise HTTPException(status_code=404, detail="Document not found")

@router.post("/{document_id}/index")
def index_doc(document_id: str) -> dict[str, str | int]:
    docs = load_docs()
    doc_to_index = None 

    for doc in docs :
        if doc["document_id"] == document_id:
            doc_to_index = doc
            break

    if doc_to_index is None :
        raise HTTPException(status_code=404, detail="Document not found")

    path = UPLOAD_DIR / str(doc_to_index["stored_filename"])

    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    if doc_to_index.get("content_type") != "text/plain":
        raise HTTPException(status_code=400, detail="Only plain text files can be indexed")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    chunk_record = []

    for index, chunk in enumerate(chunks):
        chunk_record.append({
            "chunk_id": f"{document_id}-{index}",
            "document_id": document_id,
            "chunk_index": index,
            "text": chunk
        })
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    path = CHUNKS_DIR / f"{document_id}.json"
    with open(path,"w",encoding="utf-8") as f :
        json.dump(chunk_record,f,indent=4)

    embedded_chunks_count = add_chunks_to_vectorstore(chunk_record)

    doc_to_index["status"] = "indexed"
    doc_to_index["length"] = len(text)
    doc_to_index["num_chunks"] = len(chunks)
    doc_to_index["embedded_chunks"] = embedded_chunks_count

    upload_docs(docs)

    return doc_to_index

@router.get("/{document_id}/chunks")
def get_doc_chunks(document_id: str) -> list[dict[str, str | int]]:
    path = CHUNKS_DIR / f"{document_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Chunks not found for the specified document")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return chunks

class request(BaseModel) :
    query : str

@router.post("/{document_id}/chunks/search")
def search_doc_chunks(document_id: str, request: request) -> list[dict[str, str | int]]:
    path = CHUNKS_DIR / f"{document_id}.json"

    if not path.exists():
        raise HTTPException(status_code=404, detail="Chunks not found for the specified document")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    query = request.query.lower()
    results =[]

    for chunk in chunks :
        text = str(chunk["text"]).lower()
        score = text.count(query)

        if score > 0 :
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "score": score
            })
    return results

@router.post("/{document_id}/semantic-search")
def semantic_search_doc(document_id: str, request: request) :
    from app.rag.retriever import get_retriever

    retriever = get_retriever(document_id=document_id)
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    results = retriever.invoke(query)

    return [
        {
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "document_id": doc.metadata.get("document_id", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", -1),
            "text": doc.page_content,
        }
        for doc in results
    ]
@router.delete("/{document_id}")
def delete_doc(document_id: str) -> dict[str, str]:
    docs = load_docs()

    document_to_delete = None

    for doc in docs:
        if doc["document_id"] == document_id:
            document_to_delete = doc
            break

    if document_to_delete is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = UPLOAD_DIR / str(document_to_delete["stored_filename"])

    if file_path.exists():
        file_path.unlink()

    docs.remove(document_to_delete)
    upload_docs(docs)

    return {
        "document_id": document_id,
        "detail": "Document deleted successfully.",
    }

