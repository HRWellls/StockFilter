from stockfilter.config import load_config, load_ui_config, reset_ui_config, save_ui_config
from stockfilter.data.providers import DemoProvider
from stockfilter.models import Market
from stockfilter.screening import ScreeningEngine


def test_mcd_passes_via_negative_equity_fallback() -> None:
    config = load_config(Market.US)
    mcd = DemoProvider().fetch(Market.US)[0]
    result = ScreeningEngine(config).screen(mcd)

    assert result.symbol == "MCD"
    assert result.all_conditions_passed is True
    assert result.roe_status == "ROE_NOT_MEANINGFUL"
    capital = next(item for item in result.condition_results if item.key == "capital_return_path")
    assert capital.passed is True
    assert "替代路径" in capital.label


def test_all_enabled_conditions_are_hard() -> None:
    config = load_config(Market.US)
    stock = DemoProvider().fetch(Market.US)[0].model_copy(update={"pe_ttm": 31.0})
    result = ScreeningEngine(config).screen(stock)

    assert result.all_conditions_passed is False
    assert any(item.key == "pe_ttm" and not item.passed for item in result.condition_results)


def test_disabled_condition_is_not_evaluated() -> None:
    config = load_config(Market.US).model_copy(update={"disabled_conditions": ["pe_ttm"]})
    stock = DemoProvider().fetch(Market.US)[0].model_copy(update={"pe_ttm": 31.0})
    result = ScreeningEngine(config).screen(stock)

    assert result.all_conditions_passed is True
    assert all(item.key != "pe_ttm" for item in result.condition_results)


def test_missing_value_fails_instead_of_becoming_zero() -> None:
    config = load_config(Market.US)
    stock = DemoProvider().fetch(Market.US)[0].model_copy(update={"max_drawdown_10y": None})
    result = ScreeningEngine(config).screen(stock)

    assert result.all_conditions_passed is False
    condition = next(item for item in result.condition_results if item.key == "max_drawdown_10y")
    assert condition.actual is None
    assert condition.passed is False


def test_normal_equity_uses_roe_path() -> None:
    config = load_config(Market.CN)
    stock = DemoProvider().fetch(Market.CN)[0]
    result = ScreeningEngine(config).screen(stock)

    assert result.all_conditions_passed is True
    assert result.roe_status == "ROE_MEANINGFUL"


def _bank_snapshot(market: Market):
    stock = DemoProvider().fetch(market)[0]
    return stock.model_copy(
        update={
            "is_bank": True,
            "industry": "银行",
            "pe_ttm": 10.0,
            "pe_percentile_10y": 0.20,
            "pe_history_samples": 120,
            "pb": 1.2,
            "pb_percentile_10y": 0.80,
            "pb_history_samples": 120,
            "positive_profit_years_10y": 10,
            "net_income_cagr_5y": 0.05,
            "net_income_slope_positive": True,
            "latest_net_income_vs_3y_median": 1.05,
            "book_value_per_share_cagr_5y": 0.04,
            "book_value_per_share_slope_positive": True,
            "average_equity": 80_000_000_000,
            "average_assets": 1_000_000_000_000,
            "equity_asset_ratio": 0.08,
            "roe": 0.12,
            "roa": 0.01,
            "loan_to_deposit_ratio": 0.75,
            "allowance_to_loans": 0.02,
            "diluted_share_cagr_5y": 0.00,
            "total_return_cagr_10y": 0.08,
            "total_return_slope_positive": True,
            "positive_return_year_ratio_10y": 0.60,
            "max_drawdown_10y": 0.50,
            "data_completeness": 1.0,
        }
    )


def test_bank_uses_dedicated_hard_condition_branch() -> None:
    result = ScreeningEngine(load_config(Market.US)).screen(_bank_snapshot(Market.US))

    keys = {item.key for item in result.condition_results}
    assert result.all_conditions_passed is True
    assert result.roe_status == "BANK_ROE"
    assert "bank_loan_to_deposit_ratio" in keys
    assert "net_debt_ebitda" not in keys
    assert "cash_conversion_5y" not in keys
    assert "bank_pb_percentile_10y" not in keys


def test_china_bank_has_lower_absolute_pe_cap() -> None:
    stock = _bank_snapshot(Market.CN).model_copy(update={"pe_ttm": 13.0})
    result = ScreeningEngine(load_config(Market.CN)).screen(stock)

    condition = next(item for item in result.condition_results if item.key == "bank_pe_ttm")
    assert condition.threshold == 12.0
    assert condition.passed is False


def test_result_summary_contains_only_table_columns() -> None:
    result = ScreeningEngine(load_config(Market.US)).screen(_bank_snapshot(Market.US))

    assert result.pe_percentile_10y == 0.20
    assert list(result.summary()) == [
        "symbol",
        "name",
        "branch",
        "passed",
        "failed_conditions",
    ]


def test_ui_config_persists_and_can_be_reset(tmp_path) -> None:
    path = tmp_path / "ui_config_us.yaml"
    original = load_config(Market.US)
    changed = original.model_copy(
        update={
            "valuation": original.valuation.model_copy(update={"max_pe_inclusive": 27.5})
        }
    )

    save_ui_config(changed, path)

    assert load_ui_config(Market.US, path).valuation.max_pe_inclusive == 27.5
    reset_ui_config(Market.US, path)
    assert not path.exists()
