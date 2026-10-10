from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class TextExtractionError(ValueError):
    """Raised when text cannot be extracted from a supported document."""


def extract_document_text(
    file_path: Path,
    content_type: str,
) -> str:
    if content_type == "text/plain":
        try:
            return file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise TextExtractionError(
                "Unable to read the text file."
            ) from exc

    if content_type == "application/pdf":
        try:
            reader = PdfReader(file_path)
            return "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
        except (OSError, PdfReadError) as exc:
            raise TextExtractionError(
                "Unable to read the PDF file."
            ) from exc

    raise TextExtractionError(
        "Only plain text and PDF files can be indexed."
    )
