from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence


def cagr(first: float, last: float, years: float) -> Optional[float]:
    if years <= 0 or first <= 0 or last < 0:
        return None
    return (last / first) ** (1.0 / years) - 1.0


def max_drawdown(total_return_prices: Iterable[float]) -> Optional[float]:
    peak: Optional[float] = None
    worst = 0.0
    seen = False
    for raw in total_return_prices:
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            continue
        seen = True
        peak = value if peak is None else max(peak, value)
        worst = max(worst, 1.0 - value / peak)
    return worst if seen else None


def percentile_rank(current: float, history: Iterable[float]) -> Optional[float]:
    valid = [float(v) for v in history if v is not None and math.isfinite(float(v)) and v > 0]
    if not valid or current <= 0 or not math.isfinite(current):
        return None
    return sum(value <= current for value in valid) / len(valid)


def linear_slope_positive(values: Sequence[float], log_values: bool = False) -> Optional[bool]:
    cleaned = [float(value) for value in values]
    if len(cleaned) < 2 or any(not math.isfinite(value) for value in cleaned):
        return None
    if log_values:
        if any(value <= 0 for value in cleaned):
            return None
        cleaned = [math.log(value) for value in cleaned]
    n = len(cleaned)
    mean_x = (n - 1) / 2
    mean_y = sum(cleaned) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(cleaned))
    denominator = sum((x - mean_x) ** 2 for x in range(n))
    return numerator / denominator > 0 if denominator else None

