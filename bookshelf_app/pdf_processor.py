import logging
import tempfile
from pathlib import Path
from typing import Tuple, Optional

from pypdf import PdfReader

from .stirling_client import StirlingClient
from .models import Metadata

logger = logging.getLogger(__name__)


def _extract_text_from_pdf(path: Path, max_pages: int) -> str:
    reader = PdfReader(path)
    text_parts = []
    num_pages = min(len(reader.pages), max_pages)
    for i in range(num_pages):
        page = reader.pages[i]
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            logger.warning("Error extracting text from page %s of %s: %s", i, path, e)
            page_text = ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_pdf_text_with_ocr_if_needed(
    path: Path,
    stirling: StirlingClient,
    max_pages: int = 10,
    min_chars_threshold: int = 500,
) -> Tuple[str, Path]:
    """
    Extract text from the first max_pages of the PDF.
    If too little text is found, run OCR via StirlingPDF and retry.
    Returns (text_excerpt, working_pdf_path).
    working_pdf_path may be a temp OCR'd copy.
    """
    logger.info("Extracting PDF text from %s", path)
    text = _extract_text_from_pdf(path, max_pages=max_pages)
    if len(text.strip()) >= min_chars_threshold:
        return text, path

    logger.info("Low text detected in %s (len=%d); running OCR via StirlingPDF", path, len(text.strip()))
    ocr_bytes = stirling.ocr_pdf(path)
    tmp_file = Path(tempfile.mkstemp(suffix=".ocr.pdf")[1])
    tmp_file.write_bytes(ocr_bytes)

    # Now extract text from OCR'd version
    ocr_text = _extract_text_from_pdf(tmp_file, max_pages=max_pages)
    return ocr_text, tmp_file


def apply_pdf_metadata_update(
    output_pdf_path: Path,
    stirling: StirlingClient,
    metadata: Metadata,
):
    """
    Call StirlingPDF to update PDF metadata in-place at output_pdf_path.
    """
    title = metadata.get("title")
    author = metadata.get("author")
    subject = metadata.get("short_description")

    logger.info("Updating metadata for %s (title=%r, author=%r)", output_pdf_path, title, author)

    updated_bytes = stirling.update_metadata(
        output_pdf_path,
        title=title,
        author=author,
        subject=subject,
        delete_all=True,
    )
    output_pdf_path.write_bytes(updated_bytes)

