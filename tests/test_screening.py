from stockfilter.config import load_config
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
