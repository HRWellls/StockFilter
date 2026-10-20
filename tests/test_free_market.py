import pandas as pd

from stockfilter.data.free_market import FreeMarketProvider, _historical_pe, _history_metrics


def test_history_metrics_separate_total_return_and_pe_prices() -> None:
    index = pd.DatetimeIndex(
        [pd.Timestamp(year=year, month=12, day=31) for year in range(2015, 2027)]
    ).as_unit("ns")
    frame = pd.DataFrame(
        {
            "Close": [100 + value for value in range(12)],
            "Adj Close": [80 + value * 2 for value in range(12)],
        },
        index=index,
    )
    result = _history_metrics(frame)
    assert result["pe_prices"].iloc[0] == 100
    assert result["prices"].iloc[0] == 80
    assert result["max_drawdown_10y"] == 0


def test_parse_china_statements_builds_fcf_and_debt() -> None:
    dates = ["20231231", "20241231", "20251231"]
    balance = pd.DataFrame(
        {
            "报告日": dates,
            "资产总计": [1000, 1100, 1200],
            "归属于母公司股东权益合计": [500, 550, 600],
            "货币资金": [100, 120, 140],
            "实收资本(或股本)": [10, 10, 10],
            "短期借款": [30, 35, 40],
            "长期借款": [40, 45, 50],
            "应付债券": [0, 0, 0],
            "累计折旧": [10, 15, 21],
        }
    )
    income = pd.DataFrame(
        {
            "报告日": dates,
            "营业收入": [500, 550, 620],
            "营业成本": [200, 220, 240],
            "营业利润": [100, 115, 140],
            "归属于母公司所有者的净利润": [80, 90, 110],
            "利息费用": [5, 5, 6],
            "所得税费用": [20, 22, 27],
            "稀释每股收益": [8, 9, 11],
        }
    )
    cashflow = pd.DataFrame(
        {
            "报告日": dates,
            "经营活动产生的现金流量净额": [90, 110, 130],
            "购建固定资产、无形资产和其他长期资产所支付的现金": [20, 25, 30],
        }
    )
    result = FreeMarketProvider._parse_china_statements(
        {"资产负债表": balance, "利润表": income, "现金流量表": cashflow}
    )
    assert result.iloc[-1]["fcf"] == 100
    assert result.iloc[-1]["debt"] == 90
    assert result.iloc[-1]["gross_margin"] > 0


def test_parse_china_bank_statement_fields() -> None:
    dates = ["20231231", "20241231", "20251231"]
    balance = pd.DataFrame(
        {
            "报告日": dates,
            "资产总计": [1000, 1100, 1200],
            "股东权益": [80, 90, 100],
            "股本": [10, 10, 10],
            "发放贷款及垫款": [600, 650, 700],
            "客户存款(吸收存款)": [800, 850, 900],
            "减:贷款损失准备": [12, 13, 14],
        }
    )
    income = pd.DataFrame(
        {
            "报告日": dates,
            "营业收入": [50, 55, 60],
            "营业利润": [20, 22, 24],
            "净利润": [15, 16, 18],
            "净利息收入": [30, 32, 35],
            "业务及管理费用": [10, 11, 12],
            "基本每股收益": [1.5, 1.6, 1.8],
        }
    )
    cashflow = pd.DataFrame({"报告日": dates})

    result = FreeMarketProvider._parse_china_statements(
        {"资产负债表": balance, "利润表": income, "现金流量表": cashflow}
    )

    assert result.iloc[-1]["equity"] == 100
    assert result.iloc[-1]["shares"] == 10
    assert result.iloc[-1]["loans"] == 700
    assert result.iloc[-1]["deposits"] == 900
    assert result.iloc[-1]["allowance_for_loan_losses"] == 14


def test_sec_parser_accepts_current_nonoperating_interest_tag() -> None:
    entry = {
        "start": "2025-01-01",
        "end": "2025-12-31",
        "val": 292_000_000,
        "filed": "2026-02-13",
        "form": "10-K",
        "fp": "FY",
    }
    raw = {
        "facts": {
            "us-gaap": {
                "InterestExpenseNonoperating": {"units": {"USD": [entry]}},
            }
        }
    }

    result = FreeMarketProvider._parse_sec_facts(raw)

    assert result.loc[pd.Timestamp("2025-12-31"), "interest"] == 292_000_000


def test_historical_pe_uses_one_sample_per_quarterly_filing_month() -> None:
    prices = pd.Series(
        [100.0, 200.0, 300.0, 400.0],
        index=pd.to_datetime(["2025-01-31", "2025-04-30", "2025-07-31", "2025-10-31"]),
    )
    quarterly_eps = pd.DataFrame(
        {
            "filed_date": pd.to_datetime(
                ["2025-01-20", "2025-04-20", "2025-07-20", "2025-10-20"]
            ),
            "ttm_eps": [10.0, 10.0, 10.0, 10.0],
        }
    )

    percentile, samples = _historical_pe(prices, quarterly_eps, current_pe=25.0)

    assert percentile == 0.5
    assert samples == 4


def test_historical_pe_can_limit_samples_to_a_date_window() -> None:
    prices = pd.Series(
        [100.0, 200.0, 300.0, 400.0],
        index=pd.to_datetime(["2022-01-31", "2023-01-31", "2024-01-31", "2025-01-31"]),
    )
    quarterly_eps = pd.DataFrame(
        {
            "filed_date": pd.to_datetime(
                ["2022-01-20", "2023-01-20", "2024-01-20", "2025-01-20"]
            ),
            "ttm_eps": [10.0, 10.0, 10.0, 10.0],
        }
    )

    percentile, samples = _historical_pe(
        prices,
        quarterly_eps,
        current_pe=25.0,
        start_date=pd.Timestamp("2023-01-01"),
    )

    assert percentile == 1 / 3
    assert samples == 3


def test_sec_quarterly_eps_derives_q4_and_builds_ttm() -> None:
    entries = []
    for year, values in ((2024, [1.0, 2.0, 3.0, 10.0]), (2025, [2.0, 3.0, 4.0, 14.0])):
        for month, value in zip((3, 6, 9), values[:3]):
            start_month = month - 2
            entries.append(
                {
                    "start": f"{year}-{start_month:02d}-01",
                    "end": f"{year}-{month:02d}-30",
                    "val": value,
                    "filed": f"{year}-{month + 1:02d}-20",
                    "form": "10-Q",
                }
            )
        entries.append(
            {
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
                "val": values[3],
                "filed": f"{year + 1}-02-15",
                "form": "10-K",
            }
        )
    raw = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {"units": {"USD/shares": entries}},
            }
        }
    }

    result = FreeMarketProvider._parse_sec_quarterly_eps(raw)

    assert result.loc[pd.Timestamp("2024-12-31"), "quarter_eps"] == 4.0
    assert result.loc[pd.Timestamp("2024-12-31"), "ttm_eps"] == 10.0
    assert result.loc[pd.Timestamp("2025-12-31"), "quarter_eps"] == 5.0
    assert result.loc[pd.Timestamp("2025-12-31"), "ttm_eps"] == 14.0


def test_sec_quarterly_eps_normalizes_facts_before_and_after_stock_split() -> None:
    entries = [
        {
            "start": "2021-01-01",
            "end": "2021-03-31",
            "val": 20.0,
            "filed": "2021-04-20",
            "form": "10-Q",
        },
        {
            "start": "2021-04-01",
            "end": "2021-06-30",
            "val": 25.0,
            "filed": "2021-07-20",
            "form": "10-Q",
        },
        {
            "start": "2021-07-01",
            "end": "2021-09-30",
            "val": 30.0,
            "filed": "2021-10-20",
            "form": "10-Q",
        },
        {
            "start": "2021-01-01",
            "end": "2021-12-31",
            "val": 100.0,
            "filed": "2022-02-15",
            "form": "10-K",
        },
        {
            "start": "2022-01-01",
            "end": "2022-03-31",
            "val": 24.0,
            "filed": "2022-04-20",
            "form": "10-Q",
        },
        {
            "start": "2022-04-01",
            "end": "2022-06-30",
            "val": 1.3,
            "filed": "2022-07-26",
            "form": "10-Q",
        },
    ]
    raw = {
        "facts": {
            "us-gaap": {
                "EarningsPerShareDiluted": {"units": {"USD/shares": entries}},
            }
        }
    }
    splits = pd.Series([20.0], index=pd.to_datetime(["2022-07-18"]))

    result = FreeMarketProvider._parse_sec_quarterly_eps(raw, splits)

    assert result.loc[pd.Timestamp("2021-12-31"), "quarter_eps"] == 1.25
    assert result.loc[pd.Timestamp("2021-12-31"), "ttm_eps"] == 5.0
    assert result.loc[pd.Timestamp("2022-06-30"), "ttm_eps"] == 5.25

    prices = pd.Series(
        [100.0, 105.0],
        index=pd.to_datetime(["2022-02-28", "2022-07-31"]),
    )
    percentile, samples = _historical_pe(prices, result, current_pe=15.0)

    assert percentile == 0.0
    assert samples == 2


def test_china_quarterly_eps_converts_cumulative_values_to_ttm() -> None:
    income = pd.DataFrame(
        {
            "报告日": [
                "20240331", "20240630", "20240930", "20241231",
                "20250331", "20250630", "20250930", "20251231",
            ],
            "稀释每股收益": [1.0, 3.0, 6.0, 10.0, 2.0, 5.0, 9.0, 14.0],
        }
    )

    result = FreeMarketProvider._parse_china_quarterly_eps({"利润表": income})

    assert result.loc[pd.Timestamp("2024-12-31"), "ttm_eps"] == 10.0
    assert result.loc[pd.Timestamp("2025-12-31"), "quarter_eps"] == 5.0
    assert result.loc[pd.Timestamp("2025-12-31"), "ttm_eps"] == 14.0
