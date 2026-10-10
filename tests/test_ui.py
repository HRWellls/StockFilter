from stockfilter.ui import symbol_matches_query


def test_symbol_search_is_case_insensitive_and_partial() -> None:
    assert symbol_matches_query("GOOGL", "goog") is True
    assert symbol_matches_query("600519.SS", "0519") is True


def test_symbol_search_ignores_surrounding_whitespace() -> None:
    assert symbol_matches_query("GOOG", "  goog  ") is True


def test_empty_symbol_search_matches_everything() -> None:
    assert symbol_matches_query("MCD", "  ") is True


def test_symbol_search_rejects_non_match() -> None:
    assert symbol_matches_query("MSFT", "goog") is False
