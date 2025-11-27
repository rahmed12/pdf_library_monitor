from typing import TypedDict, Literal, Optional, List, Dict, Any


DocType = Literal["pdf", "ebook"]


class Metadata(TypedDict, total=False):
    title: Optional[str]
    author: Optional[str]
    subtitle: Optional[str]
    short_description: Optional[str]


class Classification(TypedDict, total=False):
    label: Optional[str]
    confidence: Optional[float]
    reason: Optional[str]


class Config(TypedDict, total=False):
    input_dir: str
    pdf_output_dir: str
    ebook_output_dir: str
    default_model: str
    metadata_model: Optional[str]
    classification_model: Optional[str]
    max_pages: int
    log_level: str


class DocumentState(TypedDict, total=False):
    path: str
    doc_type: DocType
    config: Config
    raw_text_excerpt: Optional[str]
    inferred_metadata: Metadata
    classification: Classification
    destination_path: Optional[str]
    working_pdf_path: Optional[str]  # for OCR'd temp path if used
    errors: List[str]
    thread_id: Optional[str]



def make_initial_state(path: str, doc_type: DocType, config: Config) -> DocumentState:
    return {
        "path": path,
        "doc_type": doc_type,
        "config": config,
        "raw_text_excerpt": None,
        "inferred_metadata": {},
        "classification": {},
        "destination_path": None,
        "working_pdf_path": None,
        "errors": [],
        "thread_id": None,

    }

