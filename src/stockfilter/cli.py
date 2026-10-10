from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config, project_root
from .models import Market
from .services import StockFilterService
from .storage import DuckDBStore


def main() -> None:
    parser = argparse.ArgumentParser(description="StockFilter V1")
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync")
    sync.add_argument("--market", choices=["US", "CN"], required=True)
    sync.add_argument("--source", choices=["free_online", "demo", "csv"], default="free_online")
    sync.add_argument("--csv", type=Path)
    sync.add_argument("--limit", type=int, default=0, help="在线同步上限，0 表示全部")
    sync.add_argument("--symbols", default="", help="逗号分隔的在线测试代码，例如 MCD")
    screen = sub.add_parser("screen")
    screen.add_argument("--market", choices=["US", "CN"], required=True)

    args = parser.parse_args()
    market = Market(args.market)
    path = Path(os.getenv("STOCKFILTER_DB_PATH", project_root() / "data" / "stockfilter.duckdb"))
    service = StockFilterService(DuckDBStore(path))
    if args.command == "sync":
        default_csv = project_root() / "data" / "imports" / f"{market.value.lower()}_metrics.csv"
        batch_id, count = service.sync(
            market,
            args.source,
            args.csv or default_csv,
            config=load_config(market),
            limit=args.limit,
            symbols=[symbol for symbol in args.symbols.split(",") if symbol],
            progress=lambda current, total, message: print(f"[{current}/{total}] {message}"),
        )
        print(f"synced market={market.value} records={count} batch={batch_id}")
        return
    run_id, results = service.screen(market, load_config(market))
    passed = [result.symbol for result in results if result.all_conditions_passed]
    print(f"run={run_id} total={len(results)} passed={len(passed)} symbols={','.join(passed)}")


if __name__ == "__main__":
    main()
