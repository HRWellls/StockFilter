import pandas as pd

from stockfilter.data.free_market import FreeMarketProvider, _history_metrics


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
