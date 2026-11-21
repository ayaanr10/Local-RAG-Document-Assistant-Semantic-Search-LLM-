"""Read UTF-8 TXT and Markdown notes as source-aware text."""
from pathlib import Path
from rag.chunking import PageText

MAX_TEXT_BYTES = 5 * 1024 * 1024


def extract_text_file(filename: str, raw: bytes) -> list[PageText]:
    suffix = Path(filename).suffix.lower()
    if suffix not in ('.txt', '.md', '.markdown'):
        raise ValueError('Only .txt and .md text documents are supported here.')
    if not raw or len(raw) > MAX_TEXT_BYTES:
        raise ValueError('Text document must be non-empty and at most 5 MB.')
    try:
        text = raw.decode('utf-8-sig').strip()
    except UnicodeDecodeError as exc:
        raise ValueError('Please save your text document using UTF-8 encoding.') from exc
    if not text:
        raise ValueError('The uploaded text document is empty.')
    return [PageText(1, text)]
