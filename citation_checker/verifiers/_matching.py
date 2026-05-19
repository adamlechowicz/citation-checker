"""Shared matching helpers for API verifier search results."""

from __future__ import annotations

from collections.abc import Iterable

from ..fuzzy import TITLE_THRESHOLD, _score_title
from ..models import RemoteRecord


def best_title_match(
    query_title: str,
    records: Iterable[RemoteRecord],
) -> tuple[RemoteRecord | None, float]:
    """Return the best record whose title clears the global title threshold."""
    best_record: RemoteRecord | None = None
    best_score = -1.0

    for record in records:
        if record.title is None:
            continue
        score = _score_title(query_title, record.title)
        if score > best_score:
            best_score = score
            best_record = record

    if best_record is not None and best_score >= TITLE_THRESHOLD:
        return best_record, best_score
    return None, best_score
