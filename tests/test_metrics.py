from stockfilter.metrics import cagr, linear_slope_positive, max_drawdown, percentile_rank


def test_max_drawdown_uses_peak_to_trough() -> None:
    assert max_drawdown([100, 120, 90, 108]) == 0.25


def test_percentile_ignores_invalid_pe() -> None:
    assert percentile_rank(15, [10, 20, -3, float("nan")]) == 0.5


def test_cagr_and_slope() -> None:
    assert round(cagr(100, 121, 2), 6) == 0.1
    assert linear_slope_positive([1, 2, 3]) is True
    assert linear_slope_positive([3, 2, 1]) is False

