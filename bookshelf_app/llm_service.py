import json
import os
from typing import Dict, List, Optional

import requests

from .models import Metadata, Classification


class LLMService:
    """
    Simple wrapper around Ollama's /api/chat endpoint for JSON-style responses.
    """

    def __init__(self, base_url: Optional[str] = None, default_model: str = "llama3"):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.default_model = default_model

    def _chat(self, model: str, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns: {"message":{"role":"assistant","content":"..."}...}
        return data["message"]["content"]

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """
        Extracts JSON object from model output. Tries to be robust against extra text.
        """
        first = text.find("{")
        last = text.rfind("}")
        if first == -1 or last == -1:
            raise ValueError("No JSON object found in model output")
        snippet = text[first : last + 1]
        return json.loads(snippet)

    def infer_metadata(self, model: str, text_excerpt: str) -> Metadata:
        system_prompt = (
            "You are an expert librarian. "
            "Given an excerpt from a book (PDF or EPUB), you infer clean metadata.\n"
            "You MUST respond with a single valid JSON object only.\n"
            "Do not include backticks, markdown, or explanations.\n"
        )

        user_prompt = f"""
You are given an excerpt from a book.

Excerpt (may be first pages, preface, or partial content):
---
{text_excerpt[:8000]}
---

Infer the following fields:

- title: the best guess of the book's title.
- author: the best guess of the main author(s).
- subtitle: optional subtitle if you can infer one.
- short_description: 1-3 sentence description of what the book is about.

Return your answer as JSON with this exact schema:

{{
  "title": "string or null",
  "author": "string or null",
  "subtitle": "string or null",
  "short_description": "string or null"
}}
"""

        raw = self._chat(model, system_prompt, user_prompt)
        data = self._extract_json(raw)
        return {
            "title": data.get("title"),
            "author": data.get("author"),
            "subtitle": data.get("subtitle"),
            "short_description": data.get("short_description"),
        }

    def classify_document(
        self,
        model: str,
        text_excerpt: str,
        metadata: Metadata,
        existing_labels: List[str],
    ) -> Classification:
        existing_labels_str = ", ".join(sorted(existing_labels)) if existing_labels else "none"

        system_prompt = (
            "You are a book classifier. "
            "You decide which high-level category a document belongs to.\n"
            "You MUST respond with a single valid JSON object only.\n"
            "Do not include backticks, markdown, or explanations.\n"
        )

        user_prompt = f"""
You are classifying a book (PDF or EPUB).

Existing category labels (folder names) under the user's library are:
{existing_labels_str if existing_labels else "[no existing labels yet]"}

Book metadata (you may ignore if low quality):
- title: {metadata.get("title")}
- author: {metadata.get("author")}
- short_description: {metadata.get("short_description")}

Excerpt:
---
{text_excerpt[:8000]}
---

Hybrid classification rules:

1. If the book clearly fits one of the existing labels, choose that label exactly.
2. If not, invent a concise new label (e.g., "Software", "Math", "Psychology", "Business", "Marketing"), but avoid very narrow or weird phrases.
3. Keep labels short (1-3 words) and human-friendly (Capitalized).
4. Return:

{{
  "label": "one of the existing labels or a new concise label",
  "confidence": 0.0 to 1.0,
  "reason": "short explanation"
}}

Remember: respond with JSON ONLY.
"""

        raw = self._chat(model, system_prompt, user_prompt)
        data = self._extract_json(raw)
        return {
            "label": data.get("label"),
            "confidence": data.get("confidence"),
            "reason": data.get("reason"),
        }

