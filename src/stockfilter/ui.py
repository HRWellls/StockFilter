from __future__ import annotations


def symbol_matches_query(symbol: str, query: str) -> bool:
    """Return whether a symbol contains a normalized, case-insensitive query."""

    normalized_query = query.strip().casefold()
    return not normalized_query or normalized_query in symbol.casefold()
