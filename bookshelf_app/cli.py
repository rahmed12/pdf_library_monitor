import argparse
import logging
import os
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from .models import Config
from .watcher import process_existing_files, run_watch


def build_config(args) -> Config:
    return cast(
        Config,
        {
            "input_dir": str(Path(args.input_dir).resolve()),
            "pdf_output_dir": str(Path(args.pdf_output_dir).resolve()),
            "ebook_output_dir": str(Path(args.ebook_output_dir).resolve()),
            "default_model": args.default_model,
            "metadata_model": args.metadata_model,
            "classification_model": args.classification_model,
            "max_pages": args.max_pages,
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        },
    )


def setup_logging(log_level: str):
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Shelf Tamer - PDF/EPUB organizer (POC)")

    parser.add_argument("--input-dir", required=True, help="Inbox directory with PDFs and EPUBs")
    parser.add_argument("--pdf-output-dir", required=True, help="Output root for categorized PDFs")
    parser.add_argument("--ebook-output-dir", required=True, help="Output root for categorized EPUBs")

    parser.add_argument(
        "--default-model",
        default="llama3",
        help="Default Ollama model to use (e.g., llama3, qwen2, etc.)",
    )
    parser.add_argument(
        "--metadata-model",
        default=None,
        help="Ollama model for metadata inference (defaults to default-model if not set)",
    )
    parser.add_argument(
        "--classification-model",
        default=None,
        help="Ollama model for classification (defaults to default-model if not set)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Max number of pages to read from PDFs for analysis",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--once",
        action="store_true",
        help="Run once on existing files and exit (default if neither --once nor --watch given)",
    )
    mode_group.add_argument(
        "--watch",
        action="store_true",
        help="Watch input directory for new files and process continuously",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    config = build_config(args)
    setup_logging(config["log_level"])

    input_dir = Path(config["input_dir"])
    input_dir.mkdir(parents=True, exist_ok=True)

    pdf_output_dir = Path(config["pdf_output_dir"])
    ebook_output_dir = Path(config["ebook_output_dir"])
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    ebook_output_dir.mkdir(parents=True, exist_ok=True)

    # Default behavior: run once if neither flag specified
    if not args.once and not args.watch:
        args.once = True

    if args.once:
        process_existing_files(input_dir, config)

    if args.watch:
        run_watch(input_dir, config)


if __name__ == "__main__":
    main()

