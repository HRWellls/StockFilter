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


def _historical_pe(prices: pd.Series, annuals: pd.DataFrame, current_pe: Optional[float]) -> tuple[Optional[float], int]:
    if current_pe is None or current_pe <= 0 or annuals.empty or "eps" not in annuals:
        return None, 0
    reports = annuals.dropna(subset=["eps"]).copy()
    if reports.empty:
        return None, 0
    if "filed_date" not in reports:
        reports["filed_date"] = reports.index + pd.to_timedelta(120, unit="D")
    reports = reports.sort_values("filed_date")
    monthly = prices.resample("ME").last().to_frame("price")
    monthly.index.name = "price_date"
    monthly = monthly.reset_index()
    reports.index.name = "period_end"
    reports_reset = reports.reset_index()
    reports_reset["filed_date"] = pd.to_datetime(reports_reset["filed_date"])
    merged = pd.merge_asof(
        monthly.sort_values("price_date"),
        reports_reset[["filed_date", "eps"]].sort_values("filed_date"),
        left_on="price_date",
        right_on="filed_date",
        direction="backward",
    )
    merged["pe"] = merged["price"] / merged["eps"]
    valid = merged.loc[(merged["pe"] > 0) & np.isfinite(merged["pe"]), "pe"].tolist()
    return percentile_rank(current_pe, valid), len(valid)


def _calculate_snapshot(
    market: Market,
    quote: dict[str, Any],
    history: pd.DataFrame,
    annuals: pd.DataFrame,
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
    pe_percentile, pe_samples = _historical_pe(pe_prices, annuals, current_pe)

    annuals = annuals.sort_index()
    recent_10 = annuals.tail(10)
    recent_6 = annuals.tail(6)
    recent_5 = annuals.tail(5)
    net_values = _latest_values(recent_10, "net_income", 10)
    op_values = _latest_values(recent_6, "operating_income", 6)
    revenue_values = _latest_values(recent_6, "revenue", 6)
    fcf_values = _latest_values(recent_6, "fcf", 6)
    gross_margins = _latest_values(recent_5, "gross_margin", 5)
    share_values = _latest_values(recent_6, "shares", 6)

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

    return StockSnapshot(
        market=market,
        symbol=symbol,
        name=name,
        industry=str(quote.get("sector") or "非金融/非房地产（Yahoo 预筛）"),
        as_of_date=as_of,
        source=source,
        security_type="common_stock",
        is_active=True,
        special_treatment=name.upper().startswith(("ST ", "*ST")),
        listing_years=history_data.get("listing_years"),
        market_cap=market_cap,
        avg_turnover_20d=(price * volume if price is not None and volume is not None else None),
        pe_ttm=current_pe,
        pe_percentile_10y=pe_percentile,
        pe_history_samples=pe_samples,
        positive_profit_years_10y=sum(value > 0 for value in net_values) if net_values else None,
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
                    warning = "免费在线数据；基础价格硬条件未通过，已跳过财报下载。"
                elif market == Market.US:
                    annuals = self._sec_annuals(symbol)
                    warning = "SEC 官方财报 + Yahoo 个人研究行情；历史 PE 采用年报 EPS 在披露后延用。"
                else:
                    annuals = self._china_annuals(symbol)
                    warning = "新浪财报 + Yahoo 个人研究行情；A 股 EBITDA 在缺少折旧时以 EBIT 保守近似。"
                records.append(_calculate_snapshot(market, quote, history, annuals, self.name, warning))
            except Exception as exc:
                records.append(self._failed_snapshot(market, quote, str(exc)))
            self.progress(index, total, f"已处理 {index}/{total}：{symbol}")
        return records

    def _screen_quotes(self, market: Market) -> list[dict[str, Any]]:
        exchanges = ["NMS", "NYQ", "ASE"] if market == Market.US else ["SHH", "SHZ"]
        region = "us" if market == Market.US else "cn"
        universe = self.config.universe
        valuation = self.config.valuation
        clauses = [
            yf.EquityQuery("eq", ["region", region]),
            yf.EquityQuery("is-in", ["exchange", *exchanges]),
        ]
        disabled = set(self.config.disabled_conditions)
        if "industry" not in disabled:
            clauses.append(yf.EquityQuery("is-in", ["sector", *ALLOWED_SECTORS]))
        if "market_cap" not in disabled:
            clauses.append(yf.EquityQuery("gte", ["intradaymarketcap", universe.min_market_cap]))
        if "pe_ttm" not in disabled:
            clauses.extend(
                [
                    yf.EquityQuery("gt", ["peratio.lasttwelvemonths", valuation.min_pe_exclusive]),
                    yf.EquityQuery("lte", ["peratio.lasttwelvemonths", valuation.max_pe_inclusive]),
                ]
            )
        query = yf.EquityQuery("and", clauses)
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
        u = self.config.universe
        price = _number(quote.get("regularMarketPrice"))
        volume = _number(quote.get("averageDailyVolume3Month"))
        turnover = price * volume if price is not None and volume is not None else None
        disabled = set(self.config.disabled_conditions)
        checks = []
        if "listing_years" not in disabled:
            checks.append(metrics.get("listing_years") is not None and metrics["listing_years"] >= u.min_listing_years)
        if "avg_turnover_20d" not in disabled:
            checks.append(turnover is not None and turnover >= u.min_avg_turnover_20d)
        if "total_return_cagr_10y" not in disabled:
            checks.append(
                metrics.get("total_return_cagr_10y") is not None
                and metrics["total_return_cagr_10y"] >= q.min_total_return_cagr_10y
            )
        if "total_return_slope_positive" not in disabled:
            checks.append(metrics.get("total_return_slope_positive") is True)
        if "positive_return_year_ratio_10y" not in disabled:
            checks.append(
                metrics.get("positive_return_year_ratio_10y") is not None
                and metrics["positive_return_year_ratio_10y"] >= q.min_positive_return_year_ratio_10y
            )
        if "max_drawdown_10y" not in disabled:
            checks.append(
                metrics.get("max_drawdown_10y") is not None
                and metrics["max_drawdown_10y"] <= q.max_drawdown_10y
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

    def _sec_annuals(self, symbol: str) -> pd.DataFrame:
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
        return self._parse_sec_facts(raw)

    @staticmethod
    def _parse_sec_facts(raw: dict[str, Any]) -> pd.DataFrame:
        gaap = raw.get("facts", {}).get("us-gaap", {})

        def fact(tags: list[str], unit: str, duration: bool) -> pd.DataFrame:
            best = pd.DataFrame()
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
                if len(frame) > len(best):
                    best = frame
            return best

        specs = {
            "revenue": (["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"], "USD", True),
            "operating_income": (["OperatingIncomeLoss"], "USD", True),
            "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD", True),
            "cfo": (["NetCashProvidedByUsedInOperatingActivities"], "USD", True),
            "capex": (["PaymentsToAcquirePropertyPlantAndEquipment"], "USD", True),
            "gross_profit": (["GrossProfit"], "USD", True),
            "cost_of_revenue": (["CostOfGoodsAndServicesSold", "CostOfRevenue"], "USD", True),
            "interest": (["InterestExpenseNonOperating", "InterestExpense"], "USD", True),
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

    def _china_annuals(self, symbol: str) -> pd.DataFrame:
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
        return self._parse_china_statements(frames)

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
        output["equity"] = column(balance, ["归属于母公司股东权益合计", "所有者权益(或股东权益)合计", "所有者权益"])
        output["cash"] = column(balance, ["货币资金"])
        output["shares"] = column(balance, ["实收资本(或股本)"])
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
        return StockSnapshot(
            market=market,
            symbol=str(quote.get("symbol", "UNKNOWN")),
            name=str(quote.get("shortName") or quote.get("symbol", "UNKNOWN")),
            industry="数据获取失败",
            as_of_date=date.today(),
            source=self.name,
            market_cap=_number(quote.get("marketCap")),
            pe_ttm=_number(quote.get("trailingPE")),
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
