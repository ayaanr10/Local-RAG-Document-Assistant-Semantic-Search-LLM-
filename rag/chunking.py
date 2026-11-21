"""Split page text into overlapping, source-aware chunks."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-based for humans
    text: str


@dataclass(frozen=True)
class Chunk:
    page_number: int
    chunk_number: int
    text: str


def chunk_pages(pages: list[PageText], chunk_size: int = 220, overlap: int = 50) -> list[Chunk]:
    """Return word-based chunks, never crossing a PDF page boundary.

    `overlap` words repeat between neighboring chunks so context at a
    boundary is less likely to get lost.
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be > 0 and 0 <= overlap < chunk_size")

    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for page in pages:
        words = page.text.split()
        for chunk_index, start in enumerate(range(0, len(words), step)):
            part = words[start:start + chunk_size]
            if not part:
                break
            chunks.append(Chunk(page.page_number, chunk_index, " ".join(part)))
            if start + chunk_size >= len(words):
                break
    return chunks
