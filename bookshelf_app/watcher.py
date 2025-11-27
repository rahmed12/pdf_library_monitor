import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from .models import Config, make_initial_state
from .workflow import process_document

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".epub"}


def _doc_type_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext == ".epub":
        return "ebook"
    else:
        raise ValueError(f"Unsupported file type for {path}")


def _run_pipeline_for_file(path: Path, config: Config):
    try:
        doc_type = _doc_type_for_path(path)
    except ValueError:
        logger.info("Skipping unsupported file: %s", path)
        return

    logger.info("Processing %s (%s)", path, doc_type)

    
    state = make_initial_state(str(path), doc_type, config)
    thread_id = f"file-{path.name}"
    state["thread_id"] = thread_id

    # LangGraph Functional API expects thread_id in config["configurable"]
    lg_config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = process_document.invoke(
        state,
        lg_config
    )

    logger.info(
        "Done %s -> label=%r dest=%r errors=%s",
        path,
        (result.get("classification") or {}).get("label"),
        result.get("destination_path"),
        result.get("errors"),
    )


def process_existing_files(input_dir: Path, config: Config):
    logger.info("Batch-processing existing files in %s", input_dir)
    processed_dir = input_dir / "processed"
    for item in input_dir.iterdir():
        if item.is_dir():
            # skip processed/ and other dirs
            if item == processed_dir:
                continue
            continue
        if item.suffix.lower() in SUPPORTED_EXTS:
            _run_pipeline_for_file(item, config)


class NewFileHandler(FileSystemEventHandler):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)
        if path.suffix.lower() not in SUPPORTED_EXTS:
            return

        # Wait until file is fully written
        logger.info("Detected new file %s; waiting for stabilization", path)
        stable = False
        last_size = -1
        for _ in range(10):
            size = path.stat().st_size
            if size == last_size:
                stable = True
                break
            last_size = size
            time.sleep(0.5)

        if not stable:
            logger.warning("File %s did not stabilize; processing anyway", path)

        _run_pipeline_for_file(path, self.config)


def run_watch(input_dir: Path, config: Config):
    logger.info("Starting watcher on %s", input_dir)
    event_handler = NewFileHandler(config)
    observer = Observer()
    observer.schedule(event_handler, str(input_dir), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Stopping watcher...")
        observer.stop()
    observer.join()

