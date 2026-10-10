from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Protocol

import pandas as pd

from ..models import Market, StockSnapshot


class DataProvider(Protocol):
    name: str

    def fetch(self, market: Market) -> List[StockSnapshot]: ...


def _base(market: Market, symbol: str, name: str, industry: str) -> Dict[str, Any]:
    return {
        "market": market,
        "symbol": symbol,
        "name": name,
        "industry": industry,
        "as_of_date": date.today(),
        "source": "demo_v1",
        "security_type": "common_stock",
        "is_active": True,
        "special_treatment": False,
        "listing_years": 20,
        "pe_history_samples": 120,
        "positive_profit_years_10y": 10,
        "operating_income_cagr_5y": 0.06,
        "operating_income_slope_positive": True,
        "latest_operating_income_vs_3y_median": 1.06,
        "revenue_trend_positive_5y": True,
        "operating_income_trend_positive_5y": True,
        "fcf_trend_positive_5y": True,
        "cash_conversion_5y": 1.08,
        "average_assets": 50_000_000_000,
        "roic": 0.18,
        "roa": 0.11,
        "fcf_ttm": 5_000_000_000,
        "fcf_margin_ttm": 0.16,
        "net_debt_ebitda": 2.6,
        "interest_coverage": 8.0,
        "diluted_share_cagr_5y": -0.01,
        "total_return_cagr_10y": 0.11,
        "total_return_slope_positive": True,
        "positive_return_year_ratio_10y": 0.70,
        "gross_margin_std_5y": 0.018,
        "max_drawdown_10y": 0.36,
        "data_completeness": 1.0,
        "price_age_days": 0,
        "latest_financial_period_available": True,
        "warning": "内置演示快照，仅用于验证系统流程，不代表当前真实数据。",
    }


class DemoProvider:
    name = "demo"

    def fetch(self, market: Market) -> List[StockSnapshot]:
        if market == Market.US:
            mcd = _base(Market.US, "MCD", "McDonald's", "Restaurants")
            mcd.update(
                {
                    "market_cap": 210_000_000_000,
                    "avg_turnover_20d": 650_000_000,
                    "pe_ttm": 24.0,
                    "pe_percentile_10y": 0.22,
                    "average_equity": -4_000_000_000,
                    "equity_asset_ratio": -0.08,
                    "roe": None,
                    "roic": 0.19,
                    "roa": 0.12,
                    "fcf_margin_ttm": 0.19,
                    "net_debt_ebitda": 3.1,
                    "interest_coverage": 7.2,
                    "max_drawdown_10y": 0.38,
                }
            )
            expensive = _base(Market.US, "DEMOX", "Expensive Demo Corp", "Consumer Products")
            expensive.update(
                {
                    "market_cap": 45_000_000_000,
                    "avg_turnover_20d": 90_000_000,
                    "pe_ttm": 48.0,
                    "pe_percentile_10y": 0.82,
                    "average_equity": 15_000_000_000,
                    "equity_asset_ratio": 0.30,
                    "roe": 0.16,
                }
            )
            small = _base(Market.US, "DEMOS", "Small Demo Corp", "Industrials")
            small.update(
                {
                    "market_cap": 2_000_000_000,
                    "avg_turnover_20d": 5_000_000,
                    "pe_ttm": 16.0,
                    "pe_percentile_10y": 0.15,
                    "average_equity": 800_000_000,
                    "equity_asset_ratio": 0.35,
                    "roe": 0.14,
                }
            )
            return [StockSnapshot.model_validate(item) for item in (mcd, expensive, small)]

        quality = _base(Market.CN, "600000.SH", "质量演示股份", "食品饮料")
        quality.update(
            {
                "market_cap": 120_000_000_000,
                "avg_turnover_20d": 420_000_000,
                "pe_ttm": 21.0,
                "pe_percentile_10y": 0.24,
                "average_equity": 40_000_000_000,
                "equity_asset_ratio": 0.48,
                "roe": 0.17,
            }
        )
        declining = _base(Market.CN, "000001.SZ", "下滑演示股份", "制造业")
        declining.update(
            {
                "market_cap": 85_000_000_000,
                "avg_turnover_20d": 260_000_000,
                "pe_ttm": 18.0,
                "pe_percentile_10y": 0.20,
                "average_equity": 25_000_000_000,
                "equity_asset_ratio": 0.42,
                "roe": 0.13,
                "operating_income_cagr_5y": -0.04,
                "operating_income_slope_positive": False,
                "operating_income_trend_positive_5y": False,
                "fcf_trend_positive_5y": False,
                "latest_operating_income_vs_3y_median": 0.72,
            }
        )
        return [StockSnapshot.model_validate(item) for item in (quality, declining)]


class CsvProvider:
    name = "csv"

    def __init__(self, path: Path):
        self.path = path

    def fetch(self, market: Market) -> List[StockSnapshot]:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV 不存在：{self.path}")
        frame = pd.read_csv(self.path)
        required = {"market", "symbol", "name", "industry", "as_of_date"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CSV 缺少必要列：{', '.join(sorted(missing))}")
        records: List[StockSnapshot] = []
        for row_number, raw in enumerate(frame.to_dict(orient="records"), start=2):
            cleaned = {key: (None if pd.isna(value) else value) for key, value in raw.items()}
            cleaned["market"] = str(cleaned["market"]).upper()
            for key in (
                "is_active",
                "special_treatment",
                "operating_income_slope_positive",
                "revenue_trend_positive_5y",
                "operating_income_trend_positive_5y",
                "fcf_trend_positive_5y",
                "total_return_slope_positive",
                "latest_financial_period_available",
            ):
                if key in cleaned and isinstance(cleaned[key], str):
                    cleaned[key] = cleaned[key].strip().lower() in {"true", "1", "yes", "y", "是"}
            try:
                record = StockSnapshot.model_validate(cleaned)
            except Exception as exc:
                raise ValueError(f"CSV 第 {row_number} 行校验失败：{exc}") from exc
            if record.market != market:
                raise ValueError(f"CSV 第 {row_number} 行市场为 {record.market}，预期 {market}")
            records.append(record)
        if not records:
            raise ValueError("CSV 中没有数据")
        return records


def csv_template(market: Market) -> bytes:
    sample = DemoProvider().fetch(market)[0].model_dump(mode="json")
    sample["source"] = "csv_import"
    sample["warning"] = ""
    return pd.DataFrame([sample]).to_csv(index=False).encode("utf-8-sig")
