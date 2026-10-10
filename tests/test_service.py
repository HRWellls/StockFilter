from pathlib import Path

from stockfilter.config import load_config
from stockfilter.models import Market
from stockfilter.services import StockFilterService
from stockfilter.storage import DuckDBStore


def test_sync_screen_and_reload(tmp_path: Path) -> None:
    store = DuckDBStore(tmp_path / "test.duckdb")
    service = StockFilterService(store)

    batch_id, count = service.sync(Market.US, "demo", tmp_path / "unused.csv")
    assert batch_id
    assert count == 3

    run_id, results = service.screen(Market.US, load_config(Market.US))
    assert run_id
    assert [item.symbol for item in results if item.all_conditions_passed] == ["MCD"]
    reloaded = store.load_run_results(run_id)
    assert len(reloaded) == 3
    assert next(item for item in reloaded if item.symbol == "MCD").all_conditions_passed

