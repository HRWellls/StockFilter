from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from stockfilter.config import ScreenConfig, load_config
from stockfilter.data.providers import csv_template
from stockfilter.models import Market, ScreenResult
from stockfilter.services import StockFilterService
from stockfilter.storage import DuckDBStore

ROOT = Path(__file__).resolve().parent


st.set_page_config(page_title="StockFilter V1", page_icon="📈", layout="wide")


def db_path() -> Path:
    return Path(os.getenv("STOCKFILTER_DB_PATH", str(ROOT / "data" / "stockfilter.duckdb")))


@st.cache_resource
def service() -> StockFilterService:
    return StockFilterService(DuckDBStore(db_path()))


def percent_input(label: str, value: float, key: str, step: float = 0.5) -> float:
    return st.number_input(label, value=float(value * 100), step=step, key=key) / 100


def edit_config(market: Market) -> ScreenConfig:
    base = load_config(market)
    data = base.model_dump(mode="python")
    u = data["universe"]
    q = data["quality"]
    v = data["valuation"]
    b = data["bank"]
    prefix = market.value.lower()

    st.sidebar.markdown(f"### {'美股' if market == Market.US else 'A 股'}参数")
    condition_labels = {
        "security_type": "证券类型",
        "is_active": "正常上市",
        "special_treatment": "特殊处理/退市风险",
        "industry": "排除行业",
        "listing_years": "上市年限",
        "market_cap": "总市值",
        "avg_turnover_20d": "20 日平均成交额",
        "data_completeness": "数据完整度",
        "price_age_days": "行情新鲜度",
        "latest_financial_period_available": "财务数据期间",
        "pe_ttm": "TTM PE",
        "pe_percentile_10y": "十年 PE 百分位",
        "pe_history_samples": "PE 历史样本数",
        "positive_profit_years_10y": "正利润年份",
        "operating_income_cagr_5y": "营业利润 CAGR",
        "operating_income_slope_positive": "营业利润斜率",
        "latest_operating_income_vs_3y_median": "最新营业利润水平",
        "positive_trends_5y": "三项增长趋势",
        "cash_conversion_5y": "现金含量",
        "capital_return_path": "ROE/替代资本回报路径",
        "net_debt_ebitda": "净债务/EBITDA",
        "interest_coverage": "利息保障倍数",
        "diluted_share_cagr_5y": "股数增速",
        "total_return_cagr_10y": "十年总回报",
        "total_return_slope_positive": "总回报趋势",
        "positive_return_year_ratio_10y": "正收益年份占比",
        "gross_margin_std_5y": "毛利率稳定性",
        "max_drawdown_10y": "十年最大回撤",
        "bank_pe_ttm": "银行 TTM PE",
        "bank_pe_percentile_10y": "银行十年 PE 百分位",
        "bank_pe_history_samples": "银行 PE 历史样本数",
        "bank_pb": "银行 PB",
        "bank_pb_percentile_10y": "银行十年 PB 百分位",
        "bank_pb_history_samples": "银行 PB 历史样本数",
        "bank_positive_profit_years_10y": "银行正利润年份",
        "bank_net_income_cagr_5y": "银行净利润 CAGR",
        "bank_net_income_slope_positive": "银行净利润斜率",
        "bank_latest_net_income_vs_3y_median": "银行最新净利润水平",
        "bank_book_value_per_share_cagr_5y": "银行每股净资产 CAGR",
        "bank_book_value_per_share_slope_positive": "银行每股净资产斜率",
        "bank_roe": "银行 ROE",
        "bank_roa": "银行 ROA",
        "bank_equity_asset_ratio": "银行权益/资产",
        "bank_loan_to_deposit_ratio": "银行贷款/存款",
        "bank_allowance_to_loans": "银行贷款损失准备/贷款",
        "bank_diluted_share_cagr_5y": "银行股数增速",
        "bank_total_return_cagr_10y": "银行十年总回报",
        "bank_total_return_slope_positive": "银行总回报趋势",
        "bank_positive_return_year_ratio_10y": "银行正收益年份占比",
        "bank_max_drawdown_10y": "银行十年最大回撤",
    }
    with st.sidebar.expander("条件启停", expanded=False):
        enabled = st.multiselect(
            "已启用条件（移除即关闭）",
            options=list(condition_labels),
            default=[key for key in condition_labels if key not in data["disabled_conditions"]],
            format_func=lambda key: condition_labels[key],
            key=f"{prefix}_enabled_conditions",
        )
        data["disabled_conditions"] = [key for key in condition_labels if key not in enabled]
    with st.sidebar.expander("股票池", expanded=False):
        u["min_listing_years"] = st.number_input(
            "最短上市年限", min_value=1.0, value=float(u["min_listing_years"]), key=f"{prefix}_years"
        )
        unit = "美元" if market == Market.US else "人民币"
        u["min_market_cap"] = st.number_input(
            f"最低总市值（{unit}）",
            min_value=0.0,
            value=float(u["min_market_cap"]),
            format="%.0f",
            key=f"{prefix}_cap",
        )
        u["min_avg_turnover_20d"] = st.number_input(
            f"最低 20 日平均成交额（{unit}）",
            min_value=0.0,
            value=float(u["min_avg_turnover_20d"]),
            format="%.0f",
            key=f"{prefix}_turnover",
        )
        raw_industries = st.text_input(
            "排除行业（逗号分隔）",
            value=",".join(u["excluded_industries"]),
            key=f"{prefix}_industries",
        )
        u["excluded_industries"] = [item.strip() for item in raw_industries.split(",") if item.strip()]

    with st.sidebar.expander("估值", expanded=True):
        v["min_pe_exclusive"] = st.number_input(
            "PE 下限（不含）", value=float(v["min_pe_exclusive"]), key=f"{prefix}_pe_min"
        )
        v["max_pe_inclusive"] = st.number_input(
            "PE 上限（含）", value=float(v["max_pe_inclusive"]), key=f"{prefix}_pe_max"
        )
        v["max_pe_percentile_10y"] = percent_input(
            "十年 PE 百分位上限", v["max_pe_percentile_10y"], f"{prefix}_pe_pct"
        )
        v["min_pe_history_samples"] = st.number_input(
            "PE 历史最少季报样本",
            min_value=1,
            value=int(v["min_pe_history_samples"]),
            key=f"{prefix}_pe_samples",
        )

    with st.sidebar.expander("增长和现金流", expanded=False):
        q["min_positive_profit_years_10y"] = st.number_input(
            "十年最少正利润年份", min_value=1, max_value=10,
            value=int(q["min_positive_profit_years_10y"]), key=f"{prefix}_profit_years"
        )
        q["min_operating_income_cagr_5y"] = percent_input(
            "五年营业利润 CAGR 下限", q["min_operating_income_cagr_5y"], f"{prefix}_op_cagr"
        )
        q["min_latest_operating_income_vs_3y_median"] = percent_input(
            "最新营业利润/三年中位数下限",
            q["min_latest_operating_income_vs_3y_median"], f"{prefix}_op_median"
        )
        q["min_positive_trends_5y"] = st.number_input(
            "三项趋势最少通过数", min_value=1, max_value=3,
            value=int(q["min_positive_trends_5y"]), key=f"{prefix}_trends"
        )
        q["min_cash_conversion_5y"] = st.number_input(
            "经营现金流/净利润下限", min_value=0.0,
            value=float(q["min_cash_conversion_5y"]), step=0.05, key=f"{prefix}_cash_conversion"
        )

    with st.sidebar.expander("资本回报与负债", expanded=False):
        q["min_roe"] = percent_input("ROE 下限", q["min_roe"], f"{prefix}_roe")
        q["min_equity_asset_ratio_for_roe"] = percent_input(
            "ROE 有效所需权益/资产下限",
            q["min_equity_asset_ratio_for_roe"], f"{prefix}_equity_ratio"
        )
        q["min_roic_fallback"] = percent_input(
            "替代路径 ROIC 下限", q["min_roic_fallback"], f"{prefix}_roic"
        )
        q["min_roa_fallback"] = percent_input(
            "替代路径 ROA 下限", q["min_roa_fallback"], f"{prefix}_roa"
        )
        q["min_fcf_margin_fallback"] = percent_input(
            "替代路径 FCF 利润率下限", q["min_fcf_margin_fallback"], f"{prefix}_fcf_margin"
        )
        q["max_net_debt_ebitda"] = st.number_input(
            "净债务/EBITDA 上限", value=float(q["max_net_debt_ebitda"]),
            step=0.1, key=f"{prefix}_debt"
        )
        q["min_interest_coverage"] = st.number_input(
            "利息保障倍数下限", value=float(q["min_interest_coverage"]),
            step=0.5, key=f"{prefix}_interest"
        )
        q["max_diluted_share_cagr_5y"] = percent_input(
            "五年稀释股数 CAGR 上限", q["max_diluted_share_cagr_5y"], f"{prefix}_shares"
        )

    with st.sidebar.expander("银行专用硬条件", expanded=False):
        st.caption("仅作用于银行；房地产仍排除。银行不使用 FCF、毛利率或净债务/EBITDA。")
        b["max_pe_inclusive"] = st.number_input(
            "银行 PE 上限", value=float(b["max_pe_inclusive"]), key=f"{prefix}_bank_pe"
        )
        b["max_pe_percentile_10y"] = percent_input(
            "银行十年 PE 百分位上限", b["max_pe_percentile_10y"], f"{prefix}_bank_pe_pct"
        )
        b["min_pe_history_samples"] = st.number_input(
            "银行 PE 历史最少季报样本", min_value=1,
            value=int(b["min_pe_history_samples"]), key=f"{prefix}_bank_pe_samples"
        )
        b["max_pb_inclusive"] = st.number_input(
            "银行 PB 上限", value=float(b["max_pb_inclusive"]), step=0.1, key=f"{prefix}_bank_pb"
        )
        b["max_pb_percentile_10y"] = percent_input(
            "银行十年 PB 百分位上限", b["max_pb_percentile_10y"], f"{prefix}_bank_pb_pct"
        )
        b["min_pb_history_samples"] = st.number_input(
            "银行 PB 历史最少月样本", min_value=1,
            value=int(b["min_pb_history_samples"]), key=f"{prefix}_bank_pb_samples"
        )
        b["min_positive_profit_years_10y"] = st.number_input(
            "银行十年最少正利润年份", min_value=1, max_value=10,
            value=int(b["min_positive_profit_years_10y"]), key=f"{prefix}_bank_profit_years"
        )
        b["min_net_income_cagr_5y"] = percent_input(
            "银行五年净利润 CAGR 下限", b["min_net_income_cagr_5y"], f"{prefix}_bank_income_cagr"
        )
        b["min_latest_net_income_vs_3y_median"] = percent_input(
            "银行最新净利润/三年中位数下限",
            b["min_latest_net_income_vs_3y_median"], f"{prefix}_bank_income_median"
        )
        b["min_book_value_per_share_cagr_5y"] = percent_input(
            "银行每股净资产 CAGR 下限",
            b["min_book_value_per_share_cagr_5y"], f"{prefix}_bank_bvps_cagr"
        )
        b["min_roe"] = percent_input("银行 ROE 下限", b["min_roe"], f"{prefix}_bank_roe")
        b["min_roa"] = percent_input("银行 ROA 下限", b["min_roa"], f"{prefix}_bank_roa", step=0.1)
        b["min_equity_asset_ratio"] = percent_input(
            "银行权益/资产下限", b["min_equity_asset_ratio"], f"{prefix}_bank_equity_assets"
        )
        b["max_loan_to_deposit_ratio"] = percent_input(
            "银行贷款/存款上限", b["max_loan_to_deposit_ratio"], f"{prefix}_bank_loan_deposit"
        )
        b["min_allowance_to_loans"] = percent_input(
            "银行贷款损失准备/贷款下限",
            b["min_allowance_to_loans"], f"{prefix}_bank_allowance", step=0.1
        )
        b["max_diluted_share_cagr_5y"] = percent_input(
            "银行五年稀释股数 CAGR 上限",
            b["max_diluted_share_cagr_5y"], f"{prefix}_bank_shares"
        )
        b["min_total_return_cagr_10y"] = percent_input(
            "银行十年总回报 CAGR 下限",
            b["min_total_return_cagr_10y"], f"{prefix}_bank_return"
        )
        b["min_positive_return_year_ratio_10y"] = percent_input(
            "银行十年正收益年份占比下限",
            b["min_positive_return_year_ratio_10y"], f"{prefix}_bank_positive_years"
        )
        b["max_drawdown_10y"] = percent_input(
            "银行十年最大回撤上限", b["max_drawdown_10y"], f"{prefix}_bank_drawdown"
        )

    with st.sidebar.expander("长期表现和数据质量", expanded=False):
        q["min_total_return_cagr_10y"] = percent_input(
            "十年总回报 CAGR 下限", q["min_total_return_cagr_10y"], f"{prefix}_return"
        )
        q["min_positive_return_year_ratio_10y"] = percent_input(
            "十年正收益年份占比下限",
            q["min_positive_return_year_ratio_10y"], f"{prefix}_positive_years"
        )
        q["max_gross_margin_std_5y"] = percent_input(
            "五年毛利率标准差上限", q["max_gross_margin_std_5y"], f"{prefix}_margin_std"
        )
        q["max_drawdown_10y"] = percent_input(
            "十年最大回撤上限", q["max_drawdown_10y"], f"{prefix}_drawdown"
        )
        q["min_data_completeness"] = percent_input(
            "核心字段完整度下限", q["min_data_completeness"], f"{prefix}_complete"
        )
        q["max_price_age_days"] = st.number_input(
            "行情最大年龄（自然日）", min_value=0,
            value=int(q["max_price_age_days"]), key=f"{prefix}_age"
        )

    return ScreenConfig.model_validate(data)


def source_controls(market: Market) -> tuple[str, Path, int, List[str]]:
    label = "美股" if market == Market.US else "A 股"
    env_name = "STOCKFILTER_US_CSV" if market == Market.US else "STOCKFILTER_CN_CSV"
    default_path = ROOT / "data" / "imports" / ("us_metrics.csv" if market == Market.US else "cn_metrics.csv")
    source_label = st.radio(
        f"{label}数据源",
        ["免费在线数据", "内置演示数据", "规范化 CSV"],
        horizontal=True,
        key=f"source_{market.value}",
    )
    path = Path(
        st.text_input(
            f"{label} CSV 路径",
            value=os.getenv(env_name, str(default_path)),
            disabled=source_label != "规范化 CSV",
            key=f"path_{market.value}",
        )
    )
    st.download_button(
        f"下载{label} CSV 模板",
        csv_template(market),
        file_name=f"{market.value.lower()}_metrics_template.csv",
        mime="text/csv",
        key=f"template_{market.value}",
    )
    limit = st.number_input(
        "本次在线同步上限（0 表示全部）",
        min_value=0,
        value=0,
        step=1,
        disabled=source_label != "免费在线数据",
        help="首次调试可填 5；正式筛选使用 0。在线数据会缓存，第二次更快。",
        key=f"limit_{market.value}",
    )
    symbols_raw = st.text_input(
        "指定在线代码（逗号分隔，留空为全部）",
        value="",
        disabled=source_label != "免费在线数据",
        help="美股例如 MCD；A 股例如 600519.SS、000333.SZ。指定代码仍须先通过市值和 PE 预筛。",
        key=f"symbols_{market.value}",
    )
    source_map = {
        "免费在线数据": "free_online",
        "内置演示数据": "demo",
        "规范化 CSV": "csv",
    }
    if source_label == "免费在线数据":
        st.info("首次全量同步可能需要数分钟。免费接口偶尔会限速，已成功的数据会保留在本地缓存。")
    symbols = [symbol.strip().upper() for symbol in symbols_raw.split(",") if symbol.strip()]
    return source_map[source_label], path, int(limit), symbols


def sync_with_progress(
    market: Market,
    source: str,
    path: Path,
    limit: int,
    symbols: List[str],
    config: ScreenConfig,
) -> tuple[str, int]:
    bar = st.progress(0.0)
    status = st.empty()

    def update(current: int, total: int, message: str) -> None:
        bar.progress(current / total if total else 0.0)
        status.caption(message)

    try:
        return service().sync(
            market,
            source,
            path,
            config=config,
            limit=limit,
            symbols=symbols,
            progress=update,
        )
    finally:
        bar.empty()
        status.empty()


def batch_status(market: Market) -> None:
    batch = service().store.latest_batch(market)
    if batch:
        _, source, synced_at, count = batch
        st.caption(f"最近同步：{synced_at:%Y-%m-%d %H:%M:%S} · 来源 {source} · {count} 只")
    else:
        st.caption("尚未同步数据")


def show_results(market: Market, run_id: str, results: List[ScreenResult]) -> None:
    label = "美股" if market == Market.US else "A 股"
    target_limit = 30 if market == Market.US else 15
    passed = sum(item.all_conditions_passed for item in results)
    st.subheader(f"{label}筛选结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("基础候选数（非结果数）", len(results))
    c2.metric("最终入选数", passed, help=f"当前目标上限：{target_limit} 只")
    c3.metric("硬条件淘汰数", len(results) - passed)
    st.caption(
        f"运行 ID：{run_id}。目标为最终入选不超过 {target_limit} 只；"
        "基础候选只是进入逐项检查的股票，不等于筛选结果。"
    )
    if passed > target_limit:
        st.warning(
            f"本次最终入选 {passed} 只，超过目标上限 {target_limit} 只。"
            "请根据逐项结果收紧硬条件；系统不会用评分或 Top-N 强行截断。"
        )

    show_rejected = st.checkbox(
        "显示未通过股票（阈值诊断）",
        value=False,
        help="默认结果表只显示通过全部硬条件的股票；开启后可检查淘汰原因。",
        key=f"show_rejected_{market.value}",
    )
    visible = [item for item in results if item.all_conditions_passed or show_rejected]
    summary = pd.DataFrame([item.summary() for item in visible])
    if not summary.empty:
        # 同一失败条件数量内先分支、再行业分组；同行业按 PE 百分位升序。
        summary = (
            summary.assign(
                _branch_order=summary["branch"].ne("bank").astype(int),
                _industry=[item.industry.casefold() for item in visible],
                _pe_percentile=[item.pe_percentile_10y for item in visible],
            )
            .sort_values(
                [
                    "failed_conditions",
                    "_branch_order",
                    "_industry",
                    "_pe_percentile",
                    "symbol",
                ],
                ascending=[True, True, True, True, True],
                kind="stable",
                na_position="last",
            )
            .drop(columns=["_branch_order", "_industry", "_pe_percentile"])
            .reset_index(drop=True)
        )
    if summary.empty:
        st.info("本次没有股票通过全部硬条件。可开启上方诊断开关查看淘汰原因。")
    else:
        st.dataframe(summary, width="stretch", hide_index=True)
    st.download_button(
        "导出当前表格 CSV",
        summary.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"screen_{market.value.lower()}_{run_id[:8]}.csv",
        mime="text/csv",
        key=f"export_{market.value}",
    )
    if not visible:
        return

    selected_symbol = st.selectbox(
        "查看逐项条件",
        [item.symbol for item in visible],
        key=f"detail_{market.value}",
    )
    selected = next(item for item in visible if item.symbol == selected_symbol)
    if selected.roe_status == "ROE_NOT_MEANINGFUL":
        st.warning("ROE 不适用：系统已执行 ROIC、ROA、FCF、杠杆、利息和股数替代路径。")
    for warning in selected.warnings:
        st.info(warning)
    details = pd.DataFrame([item.model_dump() for item in selected.condition_results])
    details["actual"] = details["actual"].map(str)
    details["threshold"] = details["threshold"].map(str)
    st.dataframe(details, width="stretch", hide_index=True)


st.title("长牛价值硬条件选股 · V3")
st.caption("所有已配置条件必须全部通过；不使用评分、权重或自动交易。")

with st.sidebar:
    st.header("参数设置")
    st.caption("预设：default_v1。修改仅作用于本次筛选。")

us_config = edit_config(Market.US)
cn_config = edit_config(Market.CN)

tab_us, tab_cn = st.tabs(["美股", "A 股"])
with tab_us:
    us_source, us_path, us_limit, us_symbols = source_controls(Market.US)
    batch_status(Market.US)
    col1, col2 = st.columns(2)
    if col1.button("同步美股数据", type="primary", width="stretch"):
        try:
            batch_id, count = sync_with_progress(
                Market.US, us_source, us_path, us_limit, us_symbols, us_config
            )
            st.success(f"同步成功：{count} 只，批次 {batch_id[:8]}")
        except Exception as exc:
            st.error(f"同步失败：{exc}")
    if col2.button("开始筛选美股", width="stretch"):
        try:
            run_id, result = service().screen(Market.US, us_config)
            st.session_state["result_US"] = (run_id, result)
        except Exception as exc:
            st.error(f"筛选失败：{exc}")
    if "result_US" in st.session_state:
        show_results(Market.US, *st.session_state["result_US"])

with tab_cn:
    cn_source, cn_path, cn_limit, cn_symbols = source_controls(Market.CN)
    batch_status(Market.CN)
    col1, col2 = st.columns(2)
    if col1.button("同步 A 股数据", type="primary", width="stretch"):
        try:
            batch_id, count = sync_with_progress(
                Market.CN, cn_source, cn_path, cn_limit, cn_symbols, cn_config
            )
            st.success(f"同步成功：{count} 只，批次 {batch_id[:8]}")
        except Exception as exc:
            st.error(f"同步失败：{exc}")
    if col2.button("开始筛选 A 股", width="stretch"):
        try:
            run_id, result = service().screen(Market.CN, cn_config)
            st.session_state["result_CN"] = (run_id, result)
        except Exception as exc:
            st.error(f"筛选失败：{exc}")
    if "result_CN" in st.session_state:
        show_results(Market.CN, *st.session_state["result_CN"])
