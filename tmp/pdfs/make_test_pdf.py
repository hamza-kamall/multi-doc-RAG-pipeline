from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tmp' / 'pdf-tools'))
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT
import pymupdf
from pypdf import PdfReader

output = ROOT / 'output' / 'pdf' / 'rag_chat_test.pdf'
output.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Lead', fontSize=11, leading=16, textColor=colors.HexColor('#475569'), spaceAfter=16))
styles['Title'].alignment = TA_LEFT
styles['Title'].textColor = colors.HexColor('#15324F')
styles['BodyText'].fontSize = 11
styles['BodyText'].leading = 16
styles['BodyText'].spaceAfter = 12
styles['Heading2'].textColor = colors.HexColor('#15324F')
story = []

def paragraph(text, style='BodyText'):
    story.append(Paragraph(text, styles[style]))

paragraph('Atlas RAG: System Reference', 'Title')
paragraph('A fictional, text-based reference document for testing PDF ingestion, retrieval and grounded chat answers. These facts describe a test system, not your actual application.', 'Lead')
paragraph('1. Overview and ownership', 'Heading2')
paragraph('Atlas RAG is a document question-answering service owned by the fictional Northstar Labs. Its project lead is Mira Patel. The internal project code is ATLAS-731. The backend uses FastAPI, PostgreSQL and a shared Chroma vector collection named atlas_documents.')
paragraph('Users register with a unique username and a password. Passwords are stored as hashes, never as plain text. A user ID is a UUID. Each document has its own UUID and a user_id foreign key identifying its owner. Every document read, search and delete request must check ownership before accessing content.')
paragraph('2. Upload and indexing', 'Heading2')
paragraph('Atlas accepts TXT files and text-based PDF files up to 12 MB. Scanned PDFs without selectable text are not supported because Atlas does not perform OCR. Original uploads are stored on disk, while document metadata is stored in PostgreSQL.')
paragraph('During indexing, text is extracted, split into chunks of 800 characters with an overlap of 120 characters, embedded and added to Chroma. Each chunk includes document_id, user_id and chunk_index metadata. The embedding model is all-MiniLM-L6-v2. Indexing changes the document status from uploaded to indexing and then to indexed; errors change it to failed.')
paragraph('3. Retrieval isolation', 'Heading2')
paragraph('Atlas does not rebuild the vector collection for each user. It reuses the shared collection. Before retrieval, the API validates that the requested document belongs to the authenticated user. Vector retrieval then filters by that document_id. Sharing a collection does not authorize sharing documents.')
story.append(PageBreak())
paragraph('Atlas RAG: Answering and Cleanup', 'Title')
paragraph('4. Retrieval and answer generation', 'Heading2')
paragraph('For document questions, Atlas creates three alternate queries, performs semantic retrieval, combines results with Reciprocal Rank Fusion (RRF), and reranks the combined candidates with a cross-encoder. It passes the top four chunks to the answer generator. A smaller language model handles routing and grading; a larger language model generates the final answer.')
paragraph('A relevance grader checks whether retrieved chunks relate to the question. A grounding grader checks whether the generated answer is supported by those chunks. A usefulness grader checks whether the answer addresses the question. Graders return structured JSON fields containing lowercase yes or no.')
paragraph('If an answer fails the grounding check, the self-healing pipeline rewrites the query and repeats retrieval before generating another answer. It allows at most two retries after the initial attempt, for three generation attempts in total. If those attempts are exhausted, it returns: I could not produce a grounded answer from this document.')
paragraph('5. Deletion and sessions', 'Heading2')
paragraph('Deleting a document removes its Chroma vectors, its chunk JSON file, its original uploaded file and its PostgreSQL document row. Deleting a user also removes that user\'s documents and their corresponding files and vectors. A database cascade deletes related database rows only; it does not delete disk files or Chroma vectors. Application cleanup handles those external resources.')
paragraph('Chat messages belong to a login session and are deleted when the user explicitly logs out. Closing the browser is not treated as a confirmed logout. Session expiration also triggers cleanup in Atlas.')
paragraph('6. Boundaries of this reference', 'Heading2')
paragraph('This document does not specify a monthly subscription price, a production deployment region, a JWT signing secret, a database password or any real user credentials. An answer claiming specific values for those items is not grounded in this reference. No web lookup is needed to answer questions about the fictional Atlas system.')

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#CBD5E1'))
    canvas.line(48, 43, 547, 43)
    canvas.setFont('Helvetica', 9)
    canvas.setFillColor(colors.HexColor('#64748B'))
    canvas.drawString(48, 28, 'FICTIONAL TEST DATA | Atlas RAG')
    canvas.drawRightString(547, 28, f'Page {doc.page}')

SimpleDocTemplate(str(output), pagesize=(595, 842), rightMargin=48, leftMargin=48, topMargin=45, bottomMargin=58).build(story, onFirstPage=footer, onLaterPages=footer)
reader = PdfReader(output)
assert len(reader.pages) == 2, f'Unexpected page count: {len(reader.pages)}'
text = ' '.join(' '.join(page.extract_text() for page in reader.pages).split())
for expected in ['ATLAS-731', '800 characters', '120 characters', 'two retries', '12 MB']:
    assert expected in text, f'Missing extracted text: {expected}'
render_dir = ROOT / 'tmp' / 'pdfs'
with pymupdf.open(output) as pdf:
    for i, page in enumerate(pdf):
        page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(str(render_dir / f'rag_chat_test_page_{i + 1}.png'))
print(f'Created {output}; verified 2 pages and extractable text.')
