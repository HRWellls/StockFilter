from __future__ import annotations

import json
import math
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

import akshare as ak
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from ..config import ScreenConfig, project_root
from ..metrics import cagr, linear_slope_positive, max_drawdown, percentile_rank
from ..models import Market, StockSnapshot

ProgressCallback = Callable[[int, int, str], None]


ALLOWED_SECTORS = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Healthcare",
    "Industrials",
    "Technology",
    "Utilities",
]


def _number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_div(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _latest_values(frame: pd.DataFrame, column: str, count: int) -> list[float]:
    if column not in frame:
        return []
    return [float(value) for value in frame[column].dropna().tail(count).tolist()]


def _series_cagr(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    return cagr(values[0], values[-1], len(values) - 1)


def _history_metrics(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {}
    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    price_column = "Adj Close" if "Adj Close" in frame and frame["Adj Close"].notna().any() else "Close"
    prices = pd.to_numeric(frame[price_column], errors="coerce").dropna()
    prices = prices[prices > 0]
    if prices.empty:
        return {}
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    first_date = prices.index[0]
    last_date = prices.index[-1]
    # A ten-year provider window usually starts on the next trading day, a few days
    # short of the calendar anniversary. One week prevents that from becoming 9.99.
    years = max(((last_date - first_date).days + 7) / 365.25, 0)
    annual = prices.resample("YE").last()
    annual_returns = annual.pct_change().dropna()
    return {
        "prices": prices,
        "pe_prices": pd.to_numeric(frame["Close"], errors="coerce").dropna(),
        "last_date": last_date.date(),
        "listing_years": years,
        "total_return_cagr_10y": cagr(float(prices.iloc[0]), float(prices.iloc[-1]), years),
        "total_return_slope_positive": linear_slope_positive(prices.resample("ME").last().tolist(), True),
        "positive_return_year_ratio_10y": (
            float((annual_returns > 0).mean()) if not annual_returns.empty else None
        ),
        "max_drawdown_10y": max_drawdown(prices.tolist()),
    }


def _historical_pe(
    prices: pd.Series, quarterly_eps: pd.DataFrame, current_pe: Optional[float]
) -> tuple[Optional[float], int]:
    """Rank current PE against one point-in-time TTM PE per quarterly filing."""
    required = {"filed_date", "ttm_eps"}
    if (
        current_pe is None
        or current_pe <= 0
        or prices.empty
        or quarterly_eps.empty
        or not required.issubset(quarterly_eps.columns)
    ):
        return None, 0
    reports = quarterly_eps.dropna(subset=["filed_date", "ttm_eps"]).copy()
    reports["filed_date"] = pd.to_datetime(reports["filed_date"]).dt.tz_localize(None)
    reports = reports[reports["ttm_eps"] > 0]
    if reports.empty:
        return None, 0
    monthly = prices.resample("ME").last().rename("price").to_frame()
    monthly["filing_month"] = monthly.index.to_period("M")
    reports["filing_month"] = reports["filed_date"].dt.to_period("M")
    reports = reports.sort_values("filed_date").drop_duplicates("filing_month", keep="last")
    merged = reports.merge(monthly.reset_index(drop=True), on="filing_month", how="inner")
    merged["pe"] = merged["price"] / merged["ttm_eps"]
    valid = merged.loc[(merged["pe"] > 0) & np.isfinite(merged["pe"]), "pe"].tolist()
    return percentile_rank(current_pe, valid), len(valid)


def _historical_pb(
    prices: pd.Series, annuals: pd.DataFrame, current_pb: Optional[float]
) -> tuple[Optional[float], int]:
    if current_pb is None or current_pb <= 0 or annuals.empty:
        return None, 0
    if "equity" not in annuals or "shares" not in annuals:
        return None, 0
    reports = annuals.copy()
    reports["book_value_per_share"] = reports["equity"] / reports["shares"]
    reports = reports.replace([np.inf, -np.inf], np.nan).dropna(subset=["book_value_per_share"])
    reports = reports[reports["book_value_per_share"] > 0]
    if reports.empty:
        return None, 0
    if "filed_date" not in reports:
        reports["filed_date"] = reports.index + pd.to_timedelta(120, unit="D")
    monthly = prices.resample("ME").last().to_frame("price")
    monthly.index.name = "price_date"
    monthly = monthly.reset_index()
    reports["filed_date"] = pd.to_datetime(reports["filed_date"])
    merged = pd.merge_asof(
        monthly.sort_values("price_date"),
        reports[["filed_date", "book_value_per_share"]].sort_values("filed_date"),
        left_on="price_date",
        right_on="filed_date",
        direction="backward",
    )
    merged["pb"] = merged["price"] / merged["book_value_per_share"]
    valid = merged.loc[(merged["pb"] > 0) & np.isfinite(merged["pb"]), "pb"].tolist()
    return percentile_rank(current_pb, valid), len(valid)


def _calculate_snapshot(
    market: Market,
    quote: dict[str, Any],
    history: pd.DataFrame,
    annuals: pd.DataFrame,
    quarterly_eps: pd.DataFrame,
    source: str,
    warning: str,
) -> StockSnapshot:
    symbol = str(quote.get("symbol", ""))
    name = str(quote.get("shortName") or quote.get("longName") or symbol)
    history_data = _history_metrics(history)
    history_data.pop("prices", None)
    pe_prices = history_data.pop("pe_prices", pd.Series(dtype=float))
    if not pe_prices.empty:
        pe_prices.index = pd.to_datetime(pe_prices.index).tz_localize(None)
    current_pe = _number(quote.get("trailingPE"))
    pe_percentile, pe_samples = _historical_pe(pe_prices, quarterly_eps, current_pe)
    current_pb = _number(quote.get("priceToBook"))
    pb_percentile, pb_samples = _historical_pb(pe_prices, annuals, current_pb)

    annuals = annuals.sort_index()
    recent_10 = annuals.tail(10)
    recent_6 = annuals.tail(6)
    recent_5 = annuals.tail(5)
    net_values = _latest_values(recent_10, "net_income", 10)
    net_values_6 = _latest_values(recent_6, "net_income", 6)
    op_values = _latest_values(recent_6, "operating_income", 6)
    revenue_values = _latest_values(recent_6, "revenue", 6)
    fcf_values = _latest_values(recent_6, "fcf", 6)
    gross_margins = _latest_values(recent_5, "gross_margin", 5)
    share_values = _latest_values(recent_6, "shares", 6)
    if "equity" in recent_6 and "shares" in recent_6:
        bvps_series = (recent_6["equity"] / recent_6["shares"]).replace([np.inf, -np.inf], np.nan)
        book_values_per_share = [float(value) for value in bvps_series.dropna() if value > 0]
    else:
        book_values_per_share = []

    latest = annuals.iloc[-1] if not annuals.empty else pd.Series(dtype=float)
    previous = annuals.iloc[-2] if len(annuals) >= 2 else pd.Series(dtype=float)
    equity = _number(latest.get("equity"))
    previous_equity = _number(previous.get("equity"))
    assets = _number(latest.get("assets"))
    previous_assets = _number(previous.get("assets"))
    average_equity = (
        (equity + previous_equity) / 2 if equity is not None and previous_equity is not None else equity
    )
    average_assets = (
        (assets + previous_assets) / 2 if assets is not None and previous_assets is not None else assets
    )
    net_income = _number(latest.get("net_income"))
    operating_income = _number(latest.get("operating_income"))
    revenue = _number(latest.get("revenue"))
    fcf = _number(latest.get("fcf"))
    debt = _number(latest.get("debt"))
    cash = _number(latest.get("cash"))
    depreciation = _number(latest.get("depreciation")) or 0.0
    ebitda = operating_income + depreciation if operating_income is not None else None
    net_debt = debt - cash if debt is not None and cash is not None else None
    interest = _number(latest.get("interest"))
    tax = _number(latest.get("tax"))
    pretax = _number(latest.get("pretax"))
    tax_rate = min(max(_safe_div(tax, pretax) or 0.21, 0), 0.5)
    invested_capital = (
        debt + equity - cash
        if debt is not None and equity is not None and cash is not None
        else None
    )

    previous_three_op = _latest_values(annuals.iloc[:-1], "operating_income", 3)
    op_median = float(np.median(previous_three_op)) if previous_three_op else None
    previous_three_net = _latest_values(annuals.iloc[:-1], "net_income", 3)
    net_median = float(np.median(previous_three_net)) if previous_three_net else None
    cfo_5 = _latest_values(recent_5, "cfo", 5)
    net_5 = _latest_values(recent_5, "net_income", 5)
    positive_trends = {
        "revenue": linear_slope_positive(revenue_values),
        "operating": linear_slope_positive(op_values),
        "fcf": linear_slope_positive(fcf_values),
    }
    market_cap = _number(quote.get("marketCap"))
    price = _number(quote.get("regularMarketPrice"))
    volume = _number(quote.get("averageDailyVolume3Month"))
    as_of = history_data.get("last_date", date.today())
    price_age = max((date.today() - as_of).days, 0)

    values_for_completeness = [
        current_pe,
        pe_percentile,
        market_cap,
        history_data.get("listing_years"),
        history_data.get("total_return_cagr_10y"),
        history_data.get("max_drawdown_10y"),
        _series_cagr(op_values),
        _safe_div(sum(cfo_5), sum(net_5)) if net_5 and sum(net_5) else None,
        _safe_div(net_income, average_equity),
        _safe_div(net_debt, ebitda),
    ]
    completeness = sum(value is not None for value in values_for_completeness) / len(values_for_completeness)
    is_bank = bool(quote.get("_is_bank", False))
    loans = _number(latest.get("loans"))
    deposits = _number(latest.get("deposits"))
    allowance = _number(latest.get("allowance_for_loan_losses"))
    if is_bank:
        bank_values_for_completeness = [
            current_pe,
            pe_percentile,
            current_pb,
            pb_percentile,
            market_cap,
            history_data.get("listing_years"),
            history_data.get("total_return_cagr_10y"),
            _series_cagr(net_values_6),
            _series_cagr(book_values_per_share),
            _safe_div(net_income, average_equity),
            _safe_div(net_income, average_assets),
            _safe_div(loans, deposits),
            _safe_div(allowance, loans),
        ]
        completeness = sum(value is not None for value in bank_values_for_completeness) / len(
            bank_values_for_completeness
        )

    return StockSnapshot(
        market=market,
        symbol=symbol,
        name=name,
        industry="银行" if is_bank else str(quote.get("sector") or "非金融/非房地产（Yahoo 预筛）"),
        as_of_date=as_of,
        source=source,
        is_bank=is_bank,
        security_type="common_stock",
        is_active=True,
        special_treatment=name.upper().startswith(("ST ", "*ST")),
        listing_years=history_data.get("listing_years"),
        market_cap=market_cap,
        avg_turnover_20d=(price * volume if price is not None and volume is not None else None),
        pe_ttm=current_pe,
        pe_percentile_10y=pe_percentile,
        pe_history_samples=pe_samples,
        pb=current_pb,
        pb_percentile_10y=pb_percentile,
        pb_history_samples=pb_samples,
        positive_profit_years_10y=sum(value > 0 for value in net_values) if net_values else None,
        net_income_cagr_5y=_series_cagr(net_values_6),
        net_income_slope_positive=linear_slope_positive(net_values_6),
        latest_net_income_vs_3y_median=_safe_div(net_income, net_median),
        book_value_per_share_cagr_5y=_series_cagr(book_values_per_share),
        book_value_per_share_slope_positive=linear_slope_positive(book_values_per_share),
        operating_income_cagr_5y=_series_cagr(op_values),
        operating_income_slope_positive=linear_slope_positive(op_values),
        latest_operating_income_vs_3y_median=_safe_div(operating_income, op_median),
        revenue_trend_positive_5y=positive_trends["revenue"],
        operating_income_trend_positive_5y=positive_trends["operating"],
        fcf_trend_positive_5y=positive_trends["fcf"],
        cash_conversion_5y=_safe_div(sum(cfo_5), sum(net_5)) if net_5 and sum(net_5) else None,
        average_equity=average_equity,
        average_assets=average_assets,
        equity_asset_ratio=_safe_div(average_equity, average_assets),
        roe=_safe_div(net_income, average_equity),
        roic=_safe_div(
            operating_income * (1 - tax_rate) if operating_income is not None else None,
            invested_capital,
        ),
        roa=_safe_div(net_income, average_assets),
        fcf_ttm=fcf,
        fcf_margin_ttm=_safe_div(fcf, revenue),
        net_debt_ebitda=_safe_div(net_debt, ebitda),
        interest_coverage=_safe_div(operating_income, abs(interest) if interest else None),
        diluted_share_cagr_5y=_series_cagr(share_values),
        loan_to_deposit_ratio=_safe_div(loans, deposits),
        allowance_to_loans=_safe_div(allowance, loans),
        total_return_cagr_10y=history_data.get("total_return_cagr_10y"),
        total_return_slope_positive=history_data.get("total_return_slope_positive"),
        positive_return_year_ratio_10y=history_data.get("positive_return_year_ratio_10y"),
        gross_margin_std_5y=(float(np.std(gross_margins)) if len(gross_margins) >= 3 else None),
        max_drawdown_10y=history_data.get("max_drawdown_10y"),
        data_completeness=completeness,
        price_age_days=price_age,
        latest_financial_period_available=not annuals.empty,
        warning=warning,
    )


class FreeMarketProvider:
    """Free, personal-research data pipeline for US and mainland China stocks."""

    name = "free_online"

    def __init__(
        self,
        config: ScreenConfig,
        limit: int = 0,
        symbols: Optional[list[str]] = None,
        progress: Optional[ProgressCallback] = None,
    ):
        self.config = config
        self.limit = max(limit, 0)
        self.symbols = {symbol.strip().upper() for symbol in (symbols or []) if symbol.strip()}
        self.progress = progress or (lambda current, total, message: None)
        self.cache_root = project_root() / "data" / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._sec_tickers: Optional[dict[str, int]] = None

    def fetch(self, market: Market) -> list[StockSnapshot]:
        quotes = self._screen_quotes(market)
        if self.symbols:
            quotes = [quote for quote in quotes if str(quote.get("symbol", "")).upper() in self.symbols]
        if self.limit:
            quotes = quotes[: self.limit]
        total = len(quotes)
        if not quotes:
            raise RuntimeError("免费在线筛选器没有返回候选股票")
        self.progress(0, total, f"已取得 {total} 只基础候选，开始下载十年数据")
        records: list[StockSnapshot] = []
        for index, quote in enumerate(quotes, start=1):
            symbol = str(quote.get("symbol"))
            self.progress(index - 1, total, f"{symbol}：下载行情与财报")
            try:
                history = self._price_history(symbol)
                history_metrics = _history_metrics(history)
                if self._fails_price_prefilter(quote, history_metrics):
                    annuals = pd.DataFrame()
                    quarterly_eps = pd.DataFrame()
                    warning = "免费在线数据；基础价格硬条件未通过，已跳过财报下载。"
                elif market == Market.US:
                    annuals, quarterly_eps = self._sec_financials(symbol, history)
                    warning = "SEC 官方财报 + Yahoo 个人研究行情；历史 PE 每份季报披露月取一个 TTM 样本。"
                else:
                    annuals, quarterly_eps = self._china_financials(symbol)
                    warning = (
                        "新浪银行财报 + Yahoo 个人研究行情；历史 PE 每份季报披露月取一个 TTM 样本。"
                        if quote.get("_is_bank")
                        else "新浪财报 + Yahoo 个人研究行情；历史 PE 采用季度 TTM 样本，A 股 EBITDA 在缺少折旧时以 EBIT 保守近似。"
                    )
                records.append(
                    _calculate_snapshot(
                        market, quote, history, annuals, quarterly_eps, self.name, warning
                    )
                )
            except Exception as exc:
                records.append(self._failed_snapshot(market, quote, str(exc)))
            self.progress(index, total, f"已处理 {index}/{total}：{symbol}")
        return records

    def _screen_quotes(self, market: Market) -> list[dict[str, Any]]:
        exchanges = ["NMS", "NYQ", "ASE"] if market == Market.US else ["SHH", "SHZ"]
        region = "us" if market == Market.US else "cn"
        universe = self.config.universe
        valuation = self.config.valuation
        base_clauses = [
            yf.EquityQuery("eq", ["region", region]),
            yf.EquityQuery("is-in", ["exchange", *exchanges]),
        ]
        disabled = set(self.config.disabled_conditions)
        ordinary_clauses = list(base_clauses)
        if "industry" not in disabled:
            ordinary_clauses.append(yf.EquityQuery("is-in", ["sector", *ALLOWED_SECTORS]))
        if "market_cap" not in disabled:
            ordinary_clauses.append(
                yf.EquityQuery("gte", ["intradaymarketcap", universe.min_market_cap])
            )
        if "pe_ttm" not in disabled:
            ordinary_clauses.extend(
                [
                    yf.EquityQuery("gt", ["peratio.lasttwelvemonths", valuation.min_pe_exclusive]),
                    yf.EquityQuery("lte", ["peratio.lasttwelvemonths", valuation.max_pe_inclusive]),
                ]
            )

        bank_clauses = list(base_clauses)
        if market == Market.US:
            bank_clauses.append(yf.EquityQuery("eq", ["peer_group", "Banks"]))
        else:
            bank_clauses.append(yf.EquityQuery("eq", ["sector", "Financial Services"]))
        if "market_cap" not in disabled:
            bank_clauses.append(yf.EquityQuery("gte", ["intradaymarketcap", universe.min_market_cap]))
        if "bank_pe_ttm" not in disabled:
            bank_clauses.extend(
                [
                    yf.EquityQuery("gt", ["peratio.lasttwelvemonths", 0]),
                    yf.EquityQuery(
                        "lte", ["peratio.lasttwelvemonths", self.config.bank.max_pe_inclusive]
                    ),
                ]
            )

        ordinary = self._run_yahoo_screen(yf.EquityQuery("and", ordinary_clauses))
        banks = self._run_yahoo_screen(yf.EquityQuery("and", bank_clauses))
        if market == Market.CN:
            banks = [quote for quote in banks if self._is_china_bank_quote(quote)]
        for quote in ordinary:
            quote["_is_bank"] = False
        for quote in banks:
            quote["_is_bank"] = True
        combined = {str(quote.get("symbol")): quote for quote in ordinary}
        combined.update({str(quote.get("symbol")): quote for quote in banks})
        return sorted(combined.values(), key=lambda quote: str(quote.get("symbol", "")))

    @staticmethod
    def _run_yahoo_screen(query: yf.EquityQuery) -> list[dict[str, Any]]:
        quotes: list[dict[str, Any]] = []
        offset = 0
        while True:
            response = yf.screen(query, offset=offset, size=250, sortField="ticker", sortAsc=True)
            page = response.get("quotes", [])
            quotes.extend(page)
            offset += len(page)
            if not page or offset >= int(response.get("total", offset)):
                break
        return [quote for quote in quotes if quote.get("quoteType") in (None, "EQUITY")]

    @staticmethod
    def _is_china_bank_quote(quote: dict[str, Any]) -> bool:
        symbol = str(quote.get("symbol", "")).upper()
        name = str(quote.get("shortName") or quote.get("longName") or "").upper()
        return symbol.startswith("000001.") or any(
            marker in f" {name} " for marker in (" BANK ", " BANKING ", " BK ")
        )

    def _price_history(self, symbol: str) -> pd.DataFrame:
        path = self.cache_root / "prices" / f"{symbol.replace('/', '_')}.json"
        cached = self._read_frame_cache(path, max_age_hours=18)
        if cached is not None:
            return cached
        frame = yf.Ticker(symbol).history(period="10y", interval="1d", auto_adjust=False, actions=True)
        if frame.empty:
            raise RuntimeError("无十年行情")
        self._write_frame_cache(path, frame)
        return frame

    def _fails_price_prefilter(self, quote: dict[str, Any], metrics: dict[str, Any]) -> bool:
        q = self.config.quality
        b = self.config.bank
        u = self.config.universe
        is_bank = bool(quote.get("_is_bank", False))
        price = _number(quote.get("regularMarketPrice"))
        volume = _number(quote.get("averageDailyVolume3Month"))
        turnover = price * volume if price is not None and volume is not None else None
        disabled = set(self.config.disabled_conditions)
        checks = []
        if "listing_years" not in disabled:
            checks.append(metrics.get("listing_years") is not None and metrics["listing_years"] >= u.min_listing_years)
        if "avg_turnover_20d" not in disabled:
            checks.append(turnover is not None and turnover >= u.min_avg_turnover_20d)
        return_key = "bank_total_return_cagr_10y" if is_bank else "total_return_cagr_10y"
        min_return = b.min_total_return_cagr_10y if is_bank else q.min_total_return_cagr_10y
        if return_key not in disabled:
            checks.append(
                metrics.get("total_return_cagr_10y") is not None
                and metrics["total_return_cagr_10y"] >= min_return
            )
        if "total_return_slope_positive" not in disabled:
            checks.append(metrics.get("total_return_slope_positive") is True)
        positive_years_key = (
            "bank_positive_return_year_ratio_10y" if is_bank else "positive_return_year_ratio_10y"
        )
        min_positive_years = (
            b.min_positive_return_year_ratio_10y
            if is_bank
            else q.min_positive_return_year_ratio_10y
        )
        if positive_years_key not in disabled:
            checks.append(
                metrics.get("positive_return_year_ratio_10y") is not None
                and metrics["positive_return_year_ratio_10y"] >= min_positive_years
            )
        drawdown_key = "bank_max_drawdown_10y" if is_bank else "max_drawdown_10y"
        max_allowed_drawdown = b.max_drawdown_10y if is_bank else q.max_drawdown_10y
        if drawdown_key not in disabled:
            checks.append(
                metrics.get("max_drawdown_10y") is not None
                and metrics["max_drawdown_10y"] <= max_allowed_drawdown
            )
        return not all(checks)

    def _sec_mapping(self) -> dict[str, int]:
        if self._sec_tickers is not None:
            return self._sec_tickers
        path = self.cache_root / "sec" / "company_tickers.json"
        if path.exists() and datetime.fromtimestamp(path.stat().st_mtime) > datetime.now() - timedelta(days=1):
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            headers = {"User-Agent": os.getenv("STOCKFILTER_SEC_USER_AGENT", "StockFilter personal research contact@example.invalid")}
            response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=30)
            response.raise_for_status()
            raw = response.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw), encoding="utf-8")
        self._sec_tickers = {
            str(value["ticker"]).upper(): int(value["cik_str"]) for value in raw.values()
        }
        return self._sec_tickers

    def _sec_financials(
        self, symbol: str, history: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        mapping = self._sec_mapping()
        lookup = symbol.replace("-", ".").upper()
        cik = mapping.get(symbol.upper()) or mapping.get(lookup)
        if cik is None:
            raise RuntimeError("SEC CIK 映射缺失")
        path = self.cache_root / "sec" / f"CIK{cik:010d}.json"
        if path.exists() and datetime.fromtimestamp(path.stat().st_mtime) > datetime.now() - timedelta(days=1):
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            headers = {"User-Agent": os.getenv("STOCKFILTER_SEC_USER_AGENT", "StockFilter personal research contact@example.invalid")}
            response = requests.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                headers=headers,
                timeout=45,
            )
            response.raise_for_status()
            raw = response.json()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(raw), encoding="utf-8")
            time.sleep(0.12)
        stock_splits = (
            history["Stock Splits"]
            if "Stock Splits" in history
            else pd.Series(dtype=float)
        )
        return self._parse_sec_facts(raw), self._parse_sec_quarterly_eps(raw, stock_splits)

    @staticmethod
    def _parse_sec_facts(raw: dict[str, Any]) -> pd.DataFrame:
        gaap = raw.get("facts", {}).get("us-gaap", {})

        def fact(tags: list[str], unit: str, duration: bool) -> pd.DataFrame:
            combined = pd.DataFrame()
            for tag in tags:
                entries = gaap.get(tag, {}).get("units", {}).get(unit, [])
                rows = []
                for entry in entries:
                    if entry.get("form") != "10-K" or entry.get("fp") != "FY":
                        continue
                    if not entry.get("end") or entry.get("val") is None:
                        continue
                    if duration:
                        if not entry.get("start"):
                            continue
                        days = (pd.Timestamp(entry["end"]) - pd.Timestamp(entry["start"])).days
                        if not 250 <= days <= 450:
                            continue
                    rows.append(entry)
                if not rows:
                    continue
                frame = pd.DataFrame(rows)
                frame["end"] = pd.to_datetime(frame["end"])
                frame["filed"] = pd.to_datetime(frame["filed"])
                frame = frame.sort_values("filed").drop_duplicates("end", keep="last").set_index("end")
                combined = frame if combined.empty else combined.combine_first(frame)
            return combined.sort_index()

        specs = {
            "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"], "USD", True),
            "operating_income": (["OperatingIncomeLoss"], "USD", True),
            "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD", True),
            "cfo": (["NetCashProvidedByUsedInOperatingActivities"], "USD", True),
            "capex": (["PaymentsToAcquirePropertyPlantAndEquipment"], "USD", True),
            "gross_profit": (["GrossProfit"], "USD", True),
            "cost_of_revenue": (["CostOfGoodsAndServicesSold", "CostOfRevenue"], "USD", True),
            "interest": ([
                "InterestExpenseNonoperating",
                "InterestExpenseNonOperating",
                "InterestExpense",
            ], "USD", True),
            "tax": (["IncomeTaxExpenseBenefit"], "USD", True),
            "pretax": (["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"], "USD", True),
            "depreciation": (["DepreciationDepletionAndAmortization", "DepreciationDepletionAndAmortizationPropertyPlantAndEquipment"], "USD", True),
            "shares": (["WeightedAverageNumberOfDilutedSharesOutstanding"], "shares", True),
            "eps": (["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"], "USD/shares", True),
            "assets": (["Assets"], "USD", False),
            "equity": (["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "USD", False),
            "cash": (["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "USD", False),
            "debt_current": (["LongTermDebtCurrent", "ShortTermBorrowings"], "USD", False),
            "debt_long": (["LongTermDebtNoncurrent", "LongTermDebt"], "USD", False),
            "loans": ([
                "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
                "LoansAndLeasesReceivableNetOfDeferredIncome",
                "LoansAndLeasesReceivableNetReportedAmount",
                "LoansReceivableNet",
            ], "USD", False),
            "deposits": (["Deposits"], "USD", False),
            "allowance_for_loan_losses": ([
                "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
                "FinancingReceivableAllowanceForCreditLosses",
                "LoansAndLeasesReceivableAllowance",
            ], "USD", False),
            "net_interest_income": (["InterestIncomeExpenseNet"], "USD", True),
            "noninterest_expense": (["NoninterestExpense"], "USD", True),
        }
        output = pd.DataFrame()
        filed_dates: list[pd.Series] = []
        for name, (tags, unit, duration) in specs.items():
            frame = fact(tags, unit, duration)
            if frame.empty:
                continue
            output[name] = pd.to_numeric(frame["val"], errors="coerce")
            filed_dates.append(frame["filed"].rename(name))
        if output.empty:
            return output
        if filed_dates:
            output["filed_date"] = pd.concat(filed_dates, axis=1).max(axis=1)
        capex = output["capex"].fillna(0) if "capex" in output else pd.Series(0.0, index=output.index)
        cfo = output["cfo"] if "cfo" in output else pd.Series(np.nan, index=output.index)
        debt_current = output["debt_current"].fillna(0) if "debt_current" in output else pd.Series(0.0, index=output.index)
        debt_long = output["debt_long"].fillna(0) if "debt_long" in output else pd.Series(0.0, index=output.index)
        if "gross_profit" in output:
            gross_profit = output["gross_profit"]
        elif "revenue" in output and "cost_of_revenue" in output:
            gross_profit = output["revenue"] - output["cost_of_revenue"]
        else:
            gross_profit = pd.Series(np.nan, index=output.index)
        revenue = output["revenue"] if "revenue" in output else pd.Series(np.nan, index=output.index)
        output["capex"] = capex
        output["fcf"] = cfo - capex
        output["debt"] = debt_current + debt_long
        output["gross_margin"] = gross_profit / revenue
        if "operating_income" in output:
            output["gross_margin"] = output["gross_margin"].fillna(output["operating_income"] / revenue)
        return output.sort_index().tail(12)

    @staticmethod
    def _parse_sec_quarterly_eps(
        raw: dict[str, Any], stock_splits: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Build point-in-time TTM EPS observations from SEC 10-Q/10-K facts.

        SEC 10-Q EPS facts generally contain a stand-alone quarter as well as a
        year-to-date value. We retain 60-120 day facts, then derive fiscal Q4
        from the full-year 10-K less the preceding three stand-alone quarters.
        Yahoo historical closes are split-adjusted, so every SEC EPS fact is
        normalized to the latest share basis before quarters are combined.
        """
        gaap = raw.get("facts", {}).get("us-gaap", {})
        entries: list[dict[str, Any]] = []
        for priority, tag in enumerate(
            ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")
        ):
            tag_entries = gaap.get(tag, {}).get("units", {}).get("USD/shares", [])
            entries.extend({**entry, "_tag_priority": priority} for entry in tag_entries)
        rows = []
        for entry in entries:
            if (
                entry.get("form") not in {"10-Q", "10-K"}
                or not entry.get("start")
                or not entry.get("end")
                or not entry.get("filed")
                or entry.get("val") is None
            ):
                continue
            start = pd.Timestamp(entry["start"])
            end = pd.Timestamp(entry["end"])
            rows.append(
                {
                    "period_start": start,
                    "period_end": end,
                    "filed_date": pd.Timestamp(entry["filed"]),
                    "eps": float(entry["val"]),
                    "days": (end - start).days,
                    "form": entry["form"],
                    "tag_priority": entry["_tag_priority"],
                }
            )
        if not rows:
            return pd.DataFrame(columns=["filed_date", "quarter_eps", "ttm_eps"])

        facts = pd.DataFrame(rows)
        if stock_splits is not None and not stock_splits.empty:
            splits = pd.to_numeric(stock_splits, errors="coerce").dropna()
            splits = splits[splits > 0]
            splits.index = pd.to_datetime(splits.index).tz_localize(None)
            if not splits.empty:
                facts["eps"] = [
                    eps / float(splits.loc[splits.index > filed_date].prod())
                    for eps, filed_date in zip(facts["eps"], facts["filed_date"])
                ]
        quarter_facts = facts[
            facts["days"].between(60, 120) & facts["form"].isin(["10-Q", "10-K"])
        ]
        quarter_facts = (
            quarter_facts.sort_values(["period_end", "tag_priority", "filed_date"])
            .drop_duplicates("period_end", keep="first")
            .set_index("period_end")
        )
        annual_facts = facts[
            facts["days"].between(250, 450) & facts["form"].eq("10-K")
        ]
        annual_facts = (
            annual_facts.sort_values(["period_end", "tag_priority", "filed_date"])
            .drop_duplicates("period_end", keep="first")
            .set_index("period_end")
        )

        quarters = quarter_facts[["filed_date", "eps"]].rename(columns={"eps": "quarter_eps"})
        derived = []
        for period_end, annual in annual_facts.iterrows():
            fiscal_quarters = quarter_facts.loc[
                (quarter_facts.index > annual["period_start"])
                & (quarter_facts.index < period_end)
            ].tail(3)
            if len(fiscal_quarters) != 3:
                continue
            derived.append(
                {
                    "period_end": period_end,
                    "filed_date": annual["filed_date"],
                    "quarter_eps": annual["eps"] - fiscal_quarters["eps"].sum(),
                }
            )
        if derived:
            q4 = pd.DataFrame(derived).set_index("period_end")
            quarters = pd.concat([quarters, q4])
        if quarters.empty:
            return pd.DataFrame(columns=["filed_date", "quarter_eps", "ttm_eps"])
        quarters = quarters.sort_index().loc[lambda frame: ~frame.index.duplicated(keep="first")]
        quarters["ttm_eps"] = quarters["quarter_eps"].rolling(4, min_periods=4).sum()
        return quarters.dropna(subset=["ttm_eps"]).tail(44)

    def _china_financials(self, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        code, exchange = symbol.split(".")
        sina_code = ("sh" if exchange == "SS" else "sz") + code
        frames: dict[str, pd.DataFrame] = {}
        for report in ("资产负债表", "利润表", "现金流量表"):
            path = self.cache_root / "cn_financials" / f"{sina_code}_{report}.json"
            cached = self._read_frame_cache(path, max_age_hours=24 * 7)
            if cached is None:
                cached = ak.stock_financial_report_sina(stock=sina_code, symbol=report)
                self._write_frame_cache(path, cached)
                time.sleep(0.15)
            frames[report] = cached
        return self._parse_china_statements(frames), self._parse_china_quarterly_eps(frames)

    @staticmethod
    def _parse_china_quarterly_eps(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Convert cumulative A-share EPS reports into stand-alone quarters and TTM EPS."""
        income = frames["利润表"].copy()
        income["period_end"] = pd.to_datetime(
            income["报告日"].astype(str), format="%Y%m%d", errors="coerce"
        )
        eps_column = next(
            (name for name in ("稀释每股收益", "基本每股收益") if name in income), None
        )
        if eps_column is None:
            return pd.DataFrame(columns=["filed_date", "quarter_eps", "ttm_eps"])
        income["cumulative_eps"] = pd.to_numeric(income[eps_column], errors="coerce")
        income = income.dropna(subset=["period_end", "cumulative_eps"]).sort_values("period_end")
        income = income.drop_duplicates("period_end", keep="last")
        if income.empty:
            return pd.DataFrame(columns=["filed_date", "quarter_eps", "ttm_eps"])

        filed_column = next(
            (
                name
                for name in ("公告日期", "披露日期", "发布日期", "更新日期")
                if name in income
            ),
            None,
        )
        if filed_column:
            filed = pd.to_datetime(income[filed_column].astype(str), errors="coerce")
        else:
            offsets = income["period_end"].dt.month.map({3: 30, 6: 60, 9: 30, 12: 120})
            filed = income["period_end"] + pd.to_timedelta(offsets, unit="D")
        income["filed_date"] = filed
        income["year"] = income["period_end"].dt.year
        income["quarter_eps"] = income.groupby("year")["cumulative_eps"].diff()
        first_in_year = income.groupby("year").cumcount().eq(0)
        income.loc[first_in_year, "quarter_eps"] = income.loc[first_in_year, "cumulative_eps"]
        income = income.set_index("period_end")
        income["ttm_eps"] = income["quarter_eps"].rolling(4, min_periods=4).sum()
        return income[["filed_date", "quarter_eps", "ttm_eps"]].dropna(subset=["ttm_eps"]).tail(44)

    @staticmethod
    def _parse_china_statements(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        def yearly(frame: pd.DataFrame) -> pd.DataFrame:
            result = frame.copy()
            column = "报告日"
            result[column] = pd.to_datetime(result[column].astype(str), format="%Y%m%d", errors="coerce")
            result = result[result[column].dt.month.eq(12) & result[column].dt.day.eq(31)]
            return result.set_index(column).sort_index()

        balance = yearly(frames["资产负债表"])
        income = yearly(frames["利润表"])
        cashflow = yearly(frames["现金流量表"])
        output = pd.DataFrame(index=income.index.union(balance.index).union(cashflow.index))

        def column(frame: pd.DataFrame, names: list[str]) -> pd.Series:
            for name in names:
                if name in frame:
                    return pd.to_numeric(frame[name], errors="coerce")
            return pd.Series(index=frame.index, dtype=float)

        output["revenue"] = column(income, ["营业收入", "营业总收入"])
        output["operating_income"] = column(income, ["营业利润"])
        output["net_income"] = column(income, ["归属于母公司所有者的净利润", "净利润"])
        output["interest"] = column(income, ["利息费用", "财务费用"])
        output["tax"] = column(income, ["所得税费用"])
        output["pretax"] = output["net_income"] + output["tax"]
        output["eps"] = column(income, ["稀释每股收益", "基本每股收益"])
        cost = column(income, ["营业成本"])
        output["gross_profit"] = output["revenue"] - cost
        output["gross_margin"] = output["gross_profit"] / output["revenue"]
        output["assets"] = column(balance, ["资产总计"])
        output["equity"] = column(balance, [
            "归属于母公司股东权益合计",
            "归属于母公司股东的权益",
            "所有者权益(或股东权益)合计",
            "所有者权益",
            "股东权益",
        ])
        output["cash"] = column(balance, ["货币资金"])
        output["shares"] = column(balance, ["实收资本(或股本)", "股本"])
        output["loans"] = column(balance, ["发放贷款及垫款", "发放贷款及垫款净额"])
        output["deposits"] = column(balance, ["客户存款(吸收存款)", "吸收存款"])
        output["allowance_for_loan_losses"] = column(balance, ["减:贷款损失准备", "贷款损失准备"]).abs()
        output["net_interest_income"] = column(income, ["净利息收入"])
        output["noninterest_expense"] = column(income, ["业务及管理费用", "营业支出"]).abs()
        debt_parts = [column(balance, [name]).fillna(0) for name in ("短期借款", "长期借款", "应付债券")]
        output["debt"] = sum(debt_parts)
        accumulated = column(balance, ["累计折旧"])
        output["depreciation"] = accumulated.diff().clip(lower=0)
        output["cfo"] = column(cashflow, ["经营活动产生的现金流量净额"])
        output["capex"] = column(cashflow, ["购建固定资产、无形资产和其他长期资产所支付的现金"]).abs()
        output["fcf"] = output["cfo"] - output["capex"]
        output["filed_date"] = output.index + pd.to_timedelta(120, unit="D")
        return output.sort_index().tail(12)

    def _failed_snapshot(self, market: Market, quote: dict[str, Any], error: str) -> StockSnapshot:
        is_bank = bool(quote.get("_is_bank", False))
        return StockSnapshot(
            market=market,
            symbol=str(quote.get("symbol", "UNKNOWN")),
            name=str(quote.get("shortName") or quote.get("symbol", "UNKNOWN")),
            industry="银行（数据获取失败）" if is_bank else "数据获取失败",
            as_of_date=date.today(),
            source=self.name,
            is_bank=is_bank,
            market_cap=_number(quote.get("marketCap")),
            pe_ttm=_number(quote.get("trailingPE")),
            pb=_number(quote.get("priceToBook")),
            data_completeness=0,
            price_age_days=0,
            latest_financial_period_available=False,
            warning=f"免费数据获取失败：{error}",
        )

    @staticmethod
    def _read_frame_cache(path: Path, max_age_hours: int) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified < datetime.now() - timedelta(hours=max_age_hours):
            return None
        try:
            return pd.read_json(path, orient="table")
        except (ValueError, OSError):
            return None

    @staticmethod
    def _write_frame_cache(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_json(path, orient="table", date_format="iso")
