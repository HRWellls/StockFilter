from __future__ import annotations

import math
from typing import Any, List, Optional

from .config import ScreenConfig
from .models import ConditionResult, ScreenResult, StockSnapshot


def _display(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return round(value, 6)
    return value


class ScreeningEngine:
    """Deterministic AND-only rule engine. Missing values always fail."""

    def __init__(self, config: ScreenConfig):
        self.config = config

    def screen(self, stock: StockSnapshot) -> ScreenResult:
        if stock.market != self.config.market:
            raise ValueError(f"股票市场 {stock.market} 与配置市场 {self.config.market} 不一致")

        conditions: List[ConditionResult] = []

        def add(
            key: str,
            label: str,
            actual: Any,
            operator: str,
            threshold: Any,
            passed: bool,
            detail: str = "",
        ) -> None:
            if key in self.config.disabled_conditions:
                return
            conditions.append(
                ConditionResult(
                    key=key,
                    label=label,
                    actual=_display(actual),
                    operator=operator,
                    threshold=_display(threshold),
                    passed=bool(passed),
                    detail=detail,
                )
            )

        def ge(key: str, label: str, actual: Optional[float], threshold: float) -> None:
            add(key, label, actual, ">=", threshold, actual is not None and actual >= threshold)

        def le(key: str, label: str, actual: Optional[float], threshold: float) -> None:
            add(key, label, actual, "<=", threshold, actual is not None and actual <= threshold)

        universe = self.config.universe
        quality = self.config.quality
        valuation = self.config.valuation

        add(
            "security_type",
            "证券类型",
            stock.security_type,
            "in",
            universe.allowed_security_types,
            stock.security_type in universe.allowed_security_types,
        )
        add("is_active", "正常上市", stock.is_active, "==", True, stock.is_active is True)
        add(
            "special_treatment",
            "无特殊处理/退市风险",
            stock.special_treatment,
            "==",
            False,
            stock.special_treatment is False,
        )
        industry_lower = stock.industry.casefold()
        excluded = [item for item in universe.excluded_industries if item.casefold() in industry_lower]
        add(
            "industry",
            "行业未被排除",
            stock.industry,
            "not contains",
            universe.excluded_industries,
            not excluded,
            f"命中排除项：{', '.join(excluded)}" if excluded else "",
        )
        ge("listing_years", "上市年限", stock.listing_years, universe.min_listing_years)
        ge("market_cap", "总市值", stock.market_cap, universe.min_market_cap)
        ge(
            "avg_turnover_20d",
            "20 日平均成交额",
            stock.avg_turnover_20d,
            universe.min_avg_turnover_20d,
        )
        ge(
            "data_completeness",
            "核心字段完整度",
            stock.data_completeness,
            quality.min_data_completeness,
        )
        le("price_age_days", "行情数据年龄（天）", stock.price_age_days, quality.max_price_age_days)
        add(
            "latest_financial_period_available",
            "财务数据为最新可得期间",
            stock.latest_financial_period_available,
            "==",
            True,
            stock.latest_financial_period_available is True,
        )

        add(
            "pe_ttm",
            "TTM PE",
            stock.pe_ttm,
            f"> {valuation.min_pe_exclusive} and <=",
            valuation.max_pe_inclusive,
            stock.pe_ttm is not None
            and stock.pe_ttm > valuation.min_pe_exclusive
            and stock.pe_ttm <= valuation.max_pe_inclusive,
        )
        le(
            "pe_percentile_10y",
            "十年 PE 百分位",
            stock.pe_percentile_10y,
            valuation.max_pe_percentile_10y,
        )
        ge(
            "pe_history_samples",
            "PE 历史有效样本数",
            stock.pe_history_samples,
            valuation.min_pe_history_samples,
        )
        ge(
            "positive_profit_years_10y",
            "十年正利润年份数",
            stock.positive_profit_years_10y,
            quality.min_positive_profit_years_10y,
        )
        ge(
            "operating_income_cagr_5y",
            "五年营业利润 CAGR",
            stock.operating_income_cagr_5y,
            quality.min_operating_income_cagr_5y,
        )
        if quality.require_positive_operating_income_slope:
            add(
                "operating_income_slope_positive",
                "五年营业利润回归斜率为正",
                stock.operating_income_slope_positive,
                "==",
                True,
                stock.operating_income_slope_positive is True,
            )
        ge(
            "latest_operating_income_vs_3y_median",
            "最新营业利润/三年中位数",
            stock.latest_operating_income_vs_3y_median,
            quality.min_latest_operating_income_vs_3y_median,
        )
        ge(
            "positive_trends_5y",
            "收入、营业利润、FCF 五年正趋势数量",
            stock.positive_trends_count,
            quality.min_positive_trends_5y,
        )
        ge(
            "cash_conversion_5y",
            "五年累计经营现金流/累计净利润",
            stock.cash_conversion_5y,
            quality.min_cash_conversion_5y,
        )

        equity_ratio = stock.equity_asset_ratio
        roe_meaningful = (
            stock.average_equity is not None
            and stock.average_equity > 0
            and equity_ratio is not None
            and abs(equity_ratio) >= quality.min_equity_asset_ratio_for_roe
        )
        if roe_meaningful:
            roe_status = "ROE_MEANINGFUL"
            add(
                "capital_return_path",
                "资本回报（ROE 路径）",
                stock.roe,
                ">=",
                quality.min_roe,
                stock.roe is not None and stock.roe >= quality.min_roe,
            )
        else:
            roe_status = "ROE_NOT_MEANINGFUL"
            fallback_checks = {
                "ROIC": stock.roic is not None and stock.roic >= quality.min_roic_fallback,
                "ROA": stock.roa is not None and stock.roa >= quality.min_roa_fallback,
                "FCF": (not quality.require_positive_fcf_fallback)
                or (stock.fcf_ttm is not None and stock.fcf_ttm > 0),
                "FCF margin": stock.fcf_margin_ttm is not None
                and stock.fcf_margin_ttm >= quality.min_fcf_margin_fallback,
                "净债务/EBITDA": stock.net_debt_ebitda is not None
                and stock.net_debt_ebitda <= quality.max_net_debt_ebitda,
                "利息保障": stock.interest_coverage is not None
                and stock.interest_coverage >= quality.min_interest_coverage,
                "股数增速": stock.diluted_share_cagr_5y is not None
                and stock.diluted_share_cagr_5y <= quality.max_diluted_share_cagr_5y,
            }
            failed = [name for name, passed in fallback_checks.items() if not passed]
            add(
                "capital_return_path",
                "资本回报（ROE 不适用替代路径）",
                "全部通过" if not failed else f"失败：{', '.join(failed)}",
                "all",
                "ROIC/ROA/FCF/杠杆/利息/股数",
                not failed,
                "平均权益非正或权益/资产过小，未使用 ROE",
            )

        le(
            "net_debt_ebitda",
            "净债务/EBITDA",
            stock.net_debt_ebitda,
            quality.max_net_debt_ebitda,
        )
        ge(
            "interest_coverage",
            "利息保障倍数",
            stock.interest_coverage,
            quality.min_interest_coverage,
        )
        le(
            "diluted_share_cagr_5y",
            "五年稀释后股数 CAGR",
            stock.diluted_share_cagr_5y,
            quality.max_diluted_share_cagr_5y,
        )
        ge(
            "total_return_cagr_10y",
            "十年总回报 CAGR",
            stock.total_return_cagr_10y,
            quality.min_total_return_cagr_10y,
        )
        if quality.require_positive_total_return_slope:
            add(
                "total_return_slope_positive",
                "总回报价格对数回归斜率为正",
                stock.total_return_slope_positive,
                "==",
                True,
                stock.total_return_slope_positive is True,
            )
        ge(
            "positive_return_year_ratio_10y",
            "十年正收益年份占比",
            stock.positive_return_year_ratio_10y,
            quality.min_positive_return_year_ratio_10y,
        )
        le(
            "gross_margin_std_5y",
            "五年毛利率标准差",
            stock.gross_margin_std_5y,
            quality.max_gross_margin_std_5y,
        )
        le(
            "max_drawdown_10y",
            "十年最大回撤",
            stock.max_drawdown_10y,
            quality.max_drawdown_10y,
        )

        failures = [f"{item.label}：实际值 {item.actual}，要求 {item.operator} {item.threshold}" for item in conditions if not item.passed]
        warnings = [stock.warning] if stock.warning else []
        if not roe_meaningful:
            warnings.append("ROE 不适用，已执行回购型公司替代路径")
        return ScreenResult(
            market=stock.market,
            symbol=stock.symbol,
            name=stock.name,
            industry=stock.industry,
            as_of_date=stock.as_of_date,
            all_conditions_passed=not failures,
            roe_status=roe_status,
            condition_results=conditions,
            fail_reasons=failures,
            warnings=warnings,
        )

    def screen_many(self, stocks: List[StockSnapshot]) -> List[ScreenResult]:
        return [self.screen(stock) for stock in stocks]
