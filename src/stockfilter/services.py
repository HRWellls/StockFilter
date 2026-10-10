from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .config import ScreenConfig
from .data.free_market import FreeMarketProvider
from .data.providers import CsvProvider, DataProvider, DemoProvider
from .models import Market, ScreenResult
from .screening import ScreeningEngine
from .storage import DuckDBStore


class StockFilterService:
    def __init__(self, store: DuckDBStore):
        self.store = store

    def sync(
        self,
        market: Market,
        source: str,
        csv_path: Path,
        config: Optional[ScreenConfig] = None,
        limit: int = 0,
        symbols: Optional[List[str]] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[str, int]:
        provider: DataProvider
        if source == "demo":
            provider = DemoProvider()
        elif source == "csv":
            provider = CsvProvider(csv_path)
        elif source == "free_online":
            if config is None:
                raise ValueError("免费在线数据同步需要筛选配置")
            provider = FreeMarketProvider(config, limit=limit, symbols=symbols, progress=progress)
        else:
            raise ValueError(f"不支持的数据源：{source}")
        records = provider.fetch(market)
        batch_id = self.store.save_snapshots(market, provider.name, records)
        return batch_id, len(records)

    def screen(self, market: Market, config: ScreenConfig) -> Tuple[str, List[ScreenResult]]:
        batch_id, records = self.store.load_latest_snapshots(market)
        results = ScreeningEngine(config).screen_many(records)
        run_id = self.store.save_run(market, batch_id, config, results)
        return run_id, results
