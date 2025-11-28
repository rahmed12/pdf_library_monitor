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
            "You are a universal book classifier.\n"
            "Your goal is to assign a single best high-level subject/genre label to the book.\n"
            "You MUST respond with a single valid JSON object only.\n"
            "Do not include backticks, markdown, or explanations outside of JSON.\n"
        )

        user_prompt = f"""
    You are classifying a book (PDF or EPUB).

    Existing category labels (folder names) in the user's library:
    {existing_labels_str if existing_labels else "[no existing labels]"}

    Book metadata (may be noisy or incomplete):
    - title: {metadata.get("title")}
    - author: {metadata.get("author")}
    - short_description: {metadata.get("short_description")}

    Excerpt:
    ---
    {text_excerpt[:8000]}
    ---

    Your job:

    1. First, imagine there are NO existing folders at all.
    - Decide the single best high-level subject/genre label for this book,
        based purely on its real topic.
    - Examples of valid high-level labels (for inspiration, not limitation):
        "Business", "Marketing", "Software", "Programming", "Math", "Science",
        "Psychology", "Self-Help", "History", "Fiction", "Technology", "Design",
        "Philosophy", "Education", "Finance", "Health", "Art", "Writing", "Sales".
    - The label must be:
        - 1–3 words
        - Capitalized
        - Broad and recognizable (not hyper-specific or weird).

    2. Then, compare that ideal label to the existing labels:
    - If one of the existing labels is essentially the SAME category
        (e.g., your ideal label is "Programming" and an existing label is "Programming"),
        then use that existing label.
    - If existing labels are only broader umbrellas (e.g., "Business" for a clearly
        Marketing-specific book, or "Science" for a clearly Biology-specific book),
        you SHOULD STILL use the more precise ideal label instead of the broad one.
    - Do NOT pick a broader existing label just because it partially fits.
        Always prefer the most accurate, specific high-level subject label.

    3. If no existing label matches your ideal label closely enough,
    create a NEW label using that ideal label.

    Return JSON with this exact schema:

    {{
    "label": "string",                // final chosen label (either existing or new)
    "confidence": 0.0 to 1.0,         // your confidence in this label
    "reason": "short explanation"     // 1–3 sentences explaining why this label fits best
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