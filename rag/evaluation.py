"""Retrieval evaluation and token-budget calculations (no LLM required)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalMetrics:
    count: int
    hit_rate: float
    mrr: float
    average_first_relevant_rank: float | None
    results: list[dict]


def evaluate_retrieval(cases: list[dict], search_fn, top_k: int = 4) -> RetrievalMetrics:
    """Each case: question, expected_source and optional expected_page.

    Relevant means an exact match on the labelled source/page. This is a small,
    transparent benchmark, not an automated answer-correctness assessment.
    """
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if not cases:
        return RetrievalMetrics(0, 0.0, 0.0, None, [])
    found, reciprocals, ranks, records = 0, 0.0, [], []
    for case in cases:
        question = (case.get('question') or '').strip()
        source = (case.get('expected_source') or '').strip()
        if not question or not source:
            raise ValueError('Benchmark rows need question and expected_source')
        page_value = case.get('expected_page')
        page = None if page_value in (None, '') else int(page_value)
        matches = search_fn(question, top_k=top_k)
        rank = next((i for i, match in enumerate(matches, 1)
                     if match['source'] == source and
                     (page is None or int(match['page']) == page)), None)
        if rank is not None:
            found += 1
            reciprocals += 1 / rank
            ranks.append(rank)
        records.append({'question': question, 'expected_source': source,
                        'expected_page': page, 'rank': rank,
                        'hit': rank is not None})
    return RetrievalMetrics(len(cases), found / len(cases), reciprocals / len(cases),
                            sum(ranks) / len(ranks) if ranks else None, records)


def estimate_tokens(text: str) -> int:
    """Rough UI estimate, not model-specific tokenization or billing."""
    return max(1, (len(text) + 3) // 4) if text else 0


def select_context(matches: list[dict], max_context_chars: int = 8000) -> list[dict]:
    """Keep complete chunks within a character budget; do not cut sources mid-sentence."""
    if max_context_chars < 1:
        return []
    accepted, used = [], 0
    for item in matches:
        size = len(item['text']) + 100  # metadata and prompt overhead
        if used + size <= max_context_chars:
            accepted.append(item)
            used += size
    return accepted
