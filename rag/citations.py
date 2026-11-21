"""Safe display formatting for LLM-provided inline document citations.

Display is deliberately separate from retrieval and generation: we preserve the
answer and replace long repeated citation strings with numbered reference chips.
These references are *model-provided*, not independently verified citations.
"""
from __future__ import annotations

from html import escape
import re

_CITATION = re.compile(
    r"\[(?P<filename>[^\[\]\r\n]{1,180}?\.(?:pdf|md|txt))\s*,\s*"
    r"p(?:age)?\.?\s*(?P<page>\d+(?:\s*[-–]\s*\d+)?)\]",
    re.IGNORECASE,
)


def format_answer_citations(answer: str) -> tuple[str, list[dict[str, str | int]]]:
    """Return safe Markdown/HTML and the numbered source key.

    Repeated citations to the same filename and page share the same number.
    Plain text is HTML escaped so LLM output cannot insert arbitrary HTML.
    Markdown emphasis/line breaks are preserved for Streamlit's Markdown parser.
    """
    chunks: list[str] = []
    key_to_number: dict[tuple[str, str], int] = {}
    sources: list[dict[str, str | int]] = []
    cursor = 0
    for m in _CITATION.finditer(answer):
        chunks.append(escape(answer[cursor:m.start()]))
        name = m.group('filename').strip()
        page = m.group('page').strip()
        key = (name, page)
        if key not in key_to_number:
            n = len(sources) + 1
            key_to_number[key] = n
            sources.append({'number': n, 'filename': name, 'page': page})
        n = key_to_number[key]
        tooltip = escape(f'{name}, page {page}', quote=True)
        chunks.append(f'<sup class="atlas-citation" title="{tooltip}">[{n}]</sup>')
        cursor = m.end()
    chunks.append(escape(answer[cursor:]))
    return ''.join(chunks), sources


def render_reference_key(sources: list[dict[str, str | int]]) -> str:
    """Render a compact source key; caller must use with HTML-enabled Markdown."""
    if not sources:
        return ''
    items = []
    for src in sources:
        n = int(src['number'])
        name = escape(str(src['filename']))
        page = escape(str(src['page']))
        items.append(
            f'<span class="atlas-source-item"><b>[{n}]</b> '
            f'{name} <span class="atlas-source-page">· p. {page}</span></span>'
        )
    return (
        '<div class="atlas-source-key">'
        '<div class="atlas-source-heading">Sources mentioned in answer</div>'
        + ''.join(items) + '</div>'
    )
