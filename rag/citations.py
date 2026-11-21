"""Render compact, escaped citations without exposing long file names inline.

New responses cite [S1], [S2] etc. against the provided retrieved matches.
Legacy Gemini responses that cite [filename, p. 8] are supported too,
including old indexes where the 90-character filename lost its extension.
Citation formatting is not a substitute for factual source verification.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import re

_CITATION = re.compile(
    r"\[(?:[Ss](?P<source_id>\d+)|(?P<filename>[^\[\]\r\n]{1,260}?)\s*,\s*"
    r"p(?:age)?\.?\s*(?P<page>\d+(?:\s*[-–]\s*\d+)?))\]",
    re.IGNORECASE,
)


def _short_name(filename: str, number: int) -> str:
    """Use readable labels; retain the full source name in the hover tooltip."""
    if len(filename) > 64:
        # Some uploaded filenames are opaque Google Drive IDs, not human labels.
        if re.match(r'^[A-Za-z0-9_+=-]{64,}(?:\.(?:pdf|txt|md))?$', filename):
            return f'Uploaded document {number}'
        suffix = Path(filename).suffix
        allowed_suffix = suffix if suffix.lower() in ('.pdf', '.md', '.txt') else ''
        return filename[:38] + '…' + allowed_suffix
    return filename


def format_answer_citations(
    answer: str, context_matches: list[dict] | tuple[dict, ...] | None = None
) -> tuple[str, list[dict[str, str | int]]]:
    """Replace recognized citations with short reference chips and source key.

    [S1] is resolved only against a passed context passage. Legacy text-style
    citations are supported even if old source names have no file extension.
    Unknown source IDs are kept as text; they are never fabricated as citations.
    """
    parts: list[str] = []
    key_to_number: dict[tuple[str, str], int] = {}
    sources: list[dict[str, str | int]] = []
    cursor = 0
    for match in _CITATION.finditer(answer):
        parts.append(escape(answer[cursor:match.start()]))
        source_id = match.group('source_id')
        if source_id is not None:
            index = int(source_id) - 1
            if context_matches is None or not (0 <= index < len(context_matches)):
                parts.append(escape(match.group(0)))
                cursor = match.end()
                continue
            source = context_matches[index]
            name = str(source['source'])
            page = str(source['page'])
        else:
            name = match.group('filename').strip()
            page = match.group('page').strip()
        key = (name, page)
        if key not in key_to_number:
            number = len(sources) + 1
            key_to_number[key] = number
            sources.append({'number': number, 'filename': name, 'page': page})
        number = key_to_number[key]
        tooltip = escape(f'{name}, page {page}', quote=True)
        parts.append(f'<sup class="atlas-citation" title="{tooltip}">[{number}]</sup>')
        cursor = match.end()
    parts.append(escape(answer[cursor:]))
    return ''.join(parts), sources


def render_reference_key(sources: list[dict[str, str | int]]) -> str:
    """Render small, readable source labels with original names in tooltips."""
    if not sources:
        return ''
    items: list[str] = []
    for source in sources:
        number = int(source['number'])
        full_name = str(source['filename'])
        display_name = escape(_short_name(full_name, number))
        tooltip = escape(full_name, quote=True)
        page = escape(str(source['page']))
        items.append(
            f'<span class="atlas-source-item" title="{tooltip}"><b>[{number}]</b> '
            f'{display_name} <span class="atlas-source-page">· p. {page}</span></span>'
        )
    return (
        '<div class="atlas-source-key">'
        '<div class="atlas-source-heading">Sources mentioned in answer</div>'
        + ''.join(items) + '</div>'
    )
