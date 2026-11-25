"""Deterministic parsers for structured formats, plus text loading for the rest (D1).

Structured parsers raise ParseError on anything they don't understand, so the
router can fall back to LLM extraction instead of guessing.
"""

from src.parsers.common import ParseError
from src.parsers.csv_parser import parse_csv
from src.parsers.documents import EmptyDocument, load_document
from src.parsers.json_parser import parse_json
from src.parsers.xml_parser import parse_xml

STRUCTURED_PARSERS = {".json": parse_json, ".csv": parse_csv, ".xml": parse_xml}

__all__ = [
    "STRUCTURED_PARSERS",
    "EmptyDocument",
    "ParseError",
    "load_document",
    "parse_csv",
    "parse_json",
    "parse_xml",
]
