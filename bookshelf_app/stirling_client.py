import os
from pathlib import Path
from typing import Optional, Dict, Any

import requests


class StirlingClient:
    """
    Thin wrapper around StirlingPDF endpoints used by this POC:
    - /api/v1/misc/ocr-pdf
    - /api/v1/misc/update-metadata
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("STIRLING_BASE_URL", "http://localhost:8080")

    def ocr_pdf(
        self,
        pdf_path: Path,
        languages: Optional[list[str]] = None,
        ocr_type: str = "Normal",
        ocr_render_type: str = "sandwich",
    ) -> bytes:
        """
        Calls /api/v1/misc/ocr-pdf and returns the OCR'd PDF bytes.
        """
        languages = languages or ["eng"]
        url = f"{self.base_url}/api/v1/misc/ocr-pdf"

        files = {
            "fileInput": (pdf_path.name, open(pdf_path, "rb"), "application/pdf"),
        }

        data: Dict[str, Any] = {
            "languages": languages,
            "ocrType": ocr_type,         # e.g. "Normal"
            "ocrRenderType": ocr_render_type,  # e.g. "sandwich"
            "sidecar": False,
            "deskew": False,
            "clean": False,
            "cleanFinal": False,
            "removeImagesAfter": False,
        }

        resp = requests.post(url, files=files, data=data, timeout=600)
        resp.raise_for_status()
        return resp.content

    def update_metadata(
        self,
        pdf_path: Path,
        title: Optional[str],
        author: Optional[str],
        subject: Optional[str] = None,
        delete_all: bool = True,
    ) -> bytes:
        """
        Fix: StirlingPDF requires TWO requests when delete_all=True:
        1. Delete all metadata
        2. Update metadata
        This version also removes the temporary cleaned file after use.
        """

        url = f"{self.base_url}/api/v1/misc/update-metadata"
        temp_path = None  # track temporary cleaned file for cleanup

        # ---------------------------------------------------------
        # STEP 1 — DELETE ALL METADATA
        # ---------------------------------------------------------
        if delete_all:
            files = {
                "fileInput": (pdf_path.name, open(pdf_path, "rb"), "application/pdf"),
            }
            data = {"deleteAll": "true"}

            resp = requests.post(url, files=files, data=data, timeout=600)
            resp.raise_for_status()

            cleaned_pdf_bytes = resp.content

            # Create temporary cleaned PDF
            temp_path = pdf_path.with_suffix(".cleaned.pdf")
            temp_path.write_bytes(cleaned_pdf_bytes)

            # Use cleaned version for the next request
            pdf_path = temp_path

        # ---------------------------------------------------------
        # STEP 2 — APPLY METADATA
        # ---------------------------------------------------------
        try:
            files = {
                "fileInput": (pdf_path.name, open(pdf_path, "rb"), "application/pdf"),
            }

            data = {"deleteAll": "false"}  # Do NOT delete again

            if title:
                data["title"] = title
            if author:
                data["author"] = author
            if subject:
                data["subject"] = subject

            resp = requests.post(url, files=files, data=data, timeout=600)
            resp.raise_for_status()

            updated_bytes = resp.content

        finally:
            # ---------------------------------------------------------
            # ALWAYS CLEAN UP TEMPORARY FILE
            # ---------------------------------------------------------
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass  # silent cleanup failure

        return updated_bytes
