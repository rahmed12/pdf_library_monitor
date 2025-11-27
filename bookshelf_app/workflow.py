import logging
import os
from pathlib import Path
from typing import Dict, Any, List

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.func import entrypoint, task

from .models import DocumentState, Config, Metadata, Classification
from .llm_service import LLMService
from .stirling_client import StirlingClient
from .pdf_processor import extract_pdf_text_with_ocr_if_needed, apply_pdf_metadata_update
from .ebook_processor import extract_ebook_text

logger = logging.getLogger(__name__)


def _get_existing_labels(root_dir: Path) -> List[str]:
    if not root_dir.exists():
        return []
    labels = []
    for child in root_dir.iterdir():
        if child.is_dir():
            labels.append(child.name)
    return labels


def _safe_label(label: str) -> str:
    import re

    normalized = re.sub(r"[^A-Za-z0-9 _-]", "", label or "").strip()
    return normalized or "Uncategorized"


@task
def extract_text_task(state: DocumentState) -> DocumentState:
    config: Config = state["config"]
    doc_type = state["doc_type"]
    path = Path(state["path"])
    errors = state.get("errors", [])

    try:
        if doc_type == "pdf":
            stirling = StirlingClient()
            text, working_path = extract_pdf_text_with_ocr_if_needed(
                path,
                stirling=stirling,
                max_pages=config.get("max_pages", 10),
            )
            state["raw_text_excerpt"] = text
            state["working_pdf_path"] = str(working_path)
        else:
            text = extract_ebook_text(path)
            state["raw_text_excerpt"] = text
    except Exception as e:
        msg = f"extract_text_task error for {path}: {e}"
        logger.exception(msg)
        errors.append(msg)

    state["errors"] = errors
    return state


@task
def infer_metadata_and_classify_task(state: DocumentState) -> DocumentState:
    config: Config = state["config"]
    errors = state.get("errors", [])
    text_excerpt = state.get("raw_text_excerpt") or ""

    default_model = config.get("default_model", "llama3")
    metadata_model = config.get("metadata_model") or default_model
    classification_model = config.get("classification_model") or default_model

    llm = LLMService(default_model=default_model)

    try:
        metadata: Metadata = llm.infer_metadata(metadata_model, text_excerpt)
        state["inferred_metadata"] = metadata
    except Exception as e:
        msg = f"infer_metadata error for {state['path']}: {e}"
        logger.exception(msg)
        errors.append(msg)
        state["inferred_metadata"] = {}

    # classification
    try:
        doc_type = state["doc_type"]
        if doc_type == "pdf":
            root_dir = Path(config["pdf_output_dir"])
        else:
            root_dir = Path(config["ebook_output_dir"])

        existing_labels = _get_existing_labels(root_dir)

        classification: Classification = llm.classify_document(
            classification_model,
            state.get("raw_text_excerpt") or "",
            state.get("inferred_metadata") or {},
            existing_labels=existing_labels,
        )

        label = _safe_label(classification.get("label") or "Uncategorized")
        classification["label"] = label

        # enforce confidence default
        if classification.get("confidence") is None:
            classification["confidence"] = 0.5

        state["classification"] = classification
    except Exception as e:
        msg = f"classify_document error for {state['path']}: {e}"
        logger.exception(msg)
        errors.append(msg)
        state["classification"] = {"label": "Uncategorized", "confidence": 0.0, "reason": "classification failed"}

    state["errors"] = errors
    return state


@task
def finalize_file_task(state: DocumentState) -> DocumentState:
    import shutil

    config: Config = state["config"]
    errors = state.get("errors", [])
    path = Path(state["path"])
    doc_type = state["doc_type"]

    if doc_type == "pdf":
        root_dir = Path(config["pdf_output_dir"])
    else:
        root_dir = Path(config["ebook_output_dir"])

    classification = state.get("classification") or {}
    label = _safe_label(classification.get("label") or "Uncategorized")
    category_dir = root_dir / label
    category_dir.mkdir(parents=True, exist_ok=True)

    dest_path = category_dir / path.name

    # Copy to destination
    try:
        shutil.copy2(path, dest_path)
        logger.info("Copied %s -> %s", path, dest_path)
    except Exception as e:
        msg = f"Failed to copy {path} to {dest_path}: {e}"
        logger.exception(msg)
        errors.append(msg)
        state["errors"] = errors
        return state  # bail out; don't move original

    # For PDFs: update metadata in the destination file
    if doc_type == "pdf":
        try:
            stirling = StirlingClient()
            metadata = state.get("inferred_metadata") or {}
            apply_pdf_metadata_update(dest_path, stirling, metadata)
        except Exception as e:
            msg = f"Failed to update metadata for {dest_path}: {e}"
            logger.exception(msg)
            errors.append(msg)

    # Move original to input_dir/processed/
    try:
        input_dir = Path(config["input_dir"])
        processed_dir = input_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / path.name
        shutil.move(str(path), str(processed_path))
        logger.info("Moved original %s -> %s", path, processed_path)
    except Exception as e:
        msg = f"Failed to move original {path} to processed/: {e}"
        logger.exception(msg)
        errors.append(msg)

    state["destination_path"] = str(dest_path)
    state["errors"] = errors
    return state

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

conn = sqlite3.connect(str(checkpoint_dir / "checkpoints.sqlite"), check_same_thread=False)
checkpointer = SqliteSaver(conn)

@entrypoint(checkpointer=checkpointer)
def process_document(state: DocumentState) -> DocumentState:
    state = extract_text_task(state).result()
    state = infer_metadata_and_classify_task(state).result()
    state = finalize_file_task(state).result()
    return state


