"""Load unstructured documents (TXT, email, PDF) as text for LLM extraction."""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from src.schemas import SourceFormat

_EMAIL_HEADER = re.compile(r"\A\s*(?:(?:From|To|Subject|Date|Cc):[^\n]*\n){2,}", re.IGNORECASE)


class EmptyDocument(ValueError):
    """No extractable text (e.g. a scanned PDF with no text layer)."""


def load_document(path: Path) -> tuple[str, SourceFormat]:
    """Return (text, source_format) for a TXT, email, or PDF file."""
    if path.suffix.lower() == ".pdf":
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        fmt: SourceFormat = "pdf"
    else:
        text = path.read_text(errors="replace")
        fmt = "email" if _EMAIL_HEADER.match(text) else "txt"
    if not text.strip():
        raise EmptyDocument(f"{path.name} has no extractable text")
    return text, fmt
