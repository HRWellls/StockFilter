from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Market(str, Enum):
    US = "US"
    CN = "CN"


class StockSnapshot(BaseModel):
    """One point-in-time, normalized stock record consumed by the rule engine."""

    model_config = ConfigDict(extra="forbid")

    market: Market
    symbol: str
    name: str
    industry: str
    as_of_date: date
    source: str = "unknown"
    is_bank: bool = False
    security_type: str = "common_stock"
    is_active: bool = True
    special_treatment: bool = False
    listing_years: Optional[float] = None
    market_cap: Optional[float] = None
    avg_turnover_20d: Optional[float] = None

    pe_ttm: Optional[float] = None
    pe_percentile_10y: Optional[float] = None
    pe_history_samples: Optional[int] = None
    pb: Optional[float] = None
    pb_percentile_10y: Optional[float] = None
    pb_history_samples: Optional[int] = None
    positive_profit_years_10y: Optional[int] = None
    net_income_cagr_5y: Optional[float] = None
    net_income_slope_positive: Optional[bool] = None
    latest_net_income_vs_3y_median: Optional[float] = None
    book_value_per_share_cagr_5y: Optional[float] = None
    book_value_per_share_slope_positive: Optional[bool] = None
    operating_income_cagr_5y: Optional[float] = None
    operating_income_slope_positive: Optional[bool] = None
    latest_operating_income_vs_3y_median: Optional[float] = None
    revenue_trend_positive_5y: Optional[bool] = None
    operating_income_trend_positive_5y: Optional[bool] = None
    fcf_trend_positive_5y: Optional[bool] = None
    cash_conversion_5y: Optional[float] = None

    average_equity: Optional[float] = None
    average_assets: Optional[float] = None
    equity_asset_ratio: Optional[float] = None
    roe: Optional[float] = None
    roic: Optional[float] = None
    roa: Optional[float] = None
    fcf_ttm: Optional[float] = None
    fcf_margin_ttm: Optional[float] = None
    net_debt_ebitda: Optional[float] = None
    interest_coverage: Optional[float] = None
    diluted_share_cagr_5y: Optional[float] = None
    loan_to_deposit_ratio: Optional[float] = None
    allowance_to_loans: Optional[float] = None

    total_return_cagr_10y: Optional[float] = None
    total_return_slope_positive: Optional[bool] = None
    positive_return_year_ratio_10y: Optional[float] = None
    gross_margin_std_5y: Optional[float] = None
    max_drawdown_10y: Optional[float] = None

    data_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    price_age_days: Optional[int] = None
    latest_financial_period_available: bool = False
    warning: Optional[str] = None

    @property
    def positive_trends_count(self) -> int:
        values = (
            self.revenue_trend_positive_5y,
            self.operating_income_trend_positive_5y,
            self.fcf_trend_positive_5y,
        )
        return sum(value is True for value in values)

    @property
    def roe_is_meaningful(self) -> bool:
        return (
            self.average_equity is not None
            and self.average_equity > 0
            and self.equity_asset_ratio is not None
        )


class ConditionResult(BaseModel):
    key: str
    label: str
    passed: bool
    actual: Any = None
    operator: str
    threshold: Any
    detail: str = ""


class ScreenResult(BaseModel):
    market: Market
    symbol: str
    name: str
    industry: str
    as_of_date: date
    all_conditions_passed: bool
    screening_branch: str = "ordinary"
    pe_percentile_10y: Optional[float] = None
    roe_status: str
    condition_results: List[ConditionResult]
    fail_reasons: List[str]
    warnings: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def summary(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "branch": self.screening_branch,
            "passed": self.all_conditions_passed,
            "failed_conditions": len(self.fail_reasons),
        }
