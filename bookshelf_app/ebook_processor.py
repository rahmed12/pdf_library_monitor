import logging
from pathlib import Path
from typing import Tuple

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_ebook_text(path: Path, max_chars: int = 15000) -> str:
    """
    Extracts text from EPUB by aggregating document items.
    """
    logger.info("Extracting EPUB text from %s", path)
    book = epub.read_epub(str(path))
    texts = []

    for item in book.get_items():
        try:
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content()
                html = content.decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                if text:
                    texts.append(text)
                # stop if we exceed max_chars
                if sum(len(t) for t in texts) > max_chars:
                    break
        except Exception as e:
            logger.warning("Error extracting from EPUB item in %s: %s", path, e)

    combined = "\n\n".join(texts)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined

