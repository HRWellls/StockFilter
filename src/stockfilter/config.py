from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from .models import Market


class UniverseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_listing_years: float
    min_market_cap: float
    min_avg_turnover_20d: float
    excluded_industries: List[str]
    allowed_security_types: List[str]


class QualityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_data_completeness: float
    max_price_age_days: int
    min_positive_profit_years_10y: int
    min_operating_income_cagr_5y: float
    require_positive_operating_income_slope: bool
    min_latest_operating_income_vs_3y_median: float
    min_positive_trends_5y: int
    min_cash_conversion_5y: float
    min_roe: float
    min_equity_asset_ratio_for_roe: float
    min_roic_fallback: float
    min_roa_fallback: float
    min_fcf_margin_fallback: float
    require_positive_fcf_fallback: bool
    max_net_debt_ebitda: float
    min_interest_coverage: float
    max_diluted_share_cagr_5y: float
    min_total_return_cagr_10y: float
    require_positive_total_return_slope: bool
    min_positive_return_year_ratio_10y: float
    max_gross_margin_std_5y: float
    max_drawdown_10y: float


class ValuationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_pe_exclusive: float
    max_pe_inclusive: float
    max_pe_percentile_10y: float
    min_pe_history_samples: int


class BankConfig(BaseModel):
    """Hard-condition preset for deposit-taking banks."""

    model_config = ConfigDict(extra="forbid")
    max_pe_inclusive: float
    max_pe_percentile_10y: float
    min_pe_history_samples: int
    max_pb_inclusive: float
    max_pb_percentile_10y: float
    min_pb_history_samples: int
    min_positive_profit_years_10y: int
    min_net_income_cagr_5y: float
    require_positive_net_income_slope: bool
    min_latest_net_income_vs_3y_median: float
    min_book_value_per_share_cagr_5y: float
    require_positive_book_value_per_share_slope: bool
    min_roe: float
    min_roa: float
    min_equity_asset_ratio: float
    max_loan_to_deposit_ratio: float
    min_allowance_to_loans: float
    max_diluted_share_cagr_5y: float
    min_total_return_cagr_10y: float
    min_positive_return_year_ratio_10y: float
    max_drawdown_10y: float


class ScreenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    market: Market
    disabled_conditions: List[str] = Field(default_factory=list)
    universe: UniverseConfig
    quality: QualityConfig
    valuation: ValuationConfig
    bank: BankConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


load_dotenv(project_root() / ".env")


def default_config_path(market: Market) -> Path:
    suffix = "us" if market == Market.US else "cn"
    return project_root() / "config" / f"default_{suffix}.yaml"


def load_config(market: Market, path: Optional[Path] = None) -> ScreenConfig:
    config_path = path or default_config_path(market)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = ScreenConfig.model_validate(raw)
    if config.market != market:
        raise ValueError(f"配置市场 {config.market} 与请求市场 {market} 不一致")
    return config
