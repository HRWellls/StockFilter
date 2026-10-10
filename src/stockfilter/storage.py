from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import duckdb

from .config import ScreenConfig
from .models import Market, ScreenResult, StockSnapshot


class DuckDBStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def _initialize(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_batches (
                    batch_id VARCHAR PRIMARY KEY,
                    market VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    synced_at TIMESTAMP NOT NULL,
                    record_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    batch_id VARCHAR NOT NULL,
                    market VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    as_of_date DATE NOT NULL,
                    payload_json VARCHAR NOT NULL
                );
                CREATE TABLE IF NOT EXISTS screen_runs (
                    run_id VARCHAR PRIMARY KEY,
                    market VARCHAR NOT NULL,
                    batch_id VARCHAR NOT NULL,
                    config_json VARCHAR NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    total_count INTEGER NOT NULL,
                    passed_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS screen_results (
                    run_id VARCHAR NOT NULL,
                    symbol VARCHAR NOT NULL,
                    passed BOOLEAN NOT NULL,
                    payload_json VARCHAR NOT NULL
                );
                """
            )

    def save_snapshots(self, market: Market, source: str, records: List[StockSnapshot]) -> str:
        batch_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with self._connect() as con:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                "INSERT INTO sync_batches VALUES (?, ?, ?, ?, ?)",
                [batch_id, market.value, source, now, len(records)],
            )
            con.executemany(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?)",
                [
                    [
                        batch_id,
                        record.market.value,
                        record.symbol,
                        record.as_of_date,
                        record.model_dump_json(),
                    ]
                    for record in records
                ],
            )
            con.execute("COMMIT")
        return batch_id

    def latest_batch(self, market: Market) -> Optional[Tuple[str, str, datetime, int]]:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT batch_id, source, synced_at, record_count
                FROM sync_batches WHERE market = ?
                ORDER BY synced_at DESC LIMIT 1
                """,
                [market.value],
            ).fetchone()
        return tuple(row) if row else None

    def load_latest_snapshots(self, market: Market) -> Tuple[str, List[StockSnapshot]]:
        batch = self.latest_batch(market)
        if batch is None:
            raise RuntimeError(f"{market.value} 尚无同步数据")
        batch_id = batch[0]
        with self._connect() as con:
            rows = con.execute(
                "SELECT payload_json FROM snapshots WHERE batch_id = ? ORDER BY symbol", [batch_id]
            ).fetchall()
        return batch_id, [StockSnapshot.model_validate_json(row[0]) for row in rows]

    def save_run(
        self,
        market: Market,
        batch_id: str,
        config: ScreenConfig,
        results: List[ScreenResult],
    ) -> str:
        run_id = str(uuid.uuid4())
        passed = sum(result.all_conditions_passed for result in results)
        with self._connect() as con:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                "INSERT INTO screen_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    market.value,
                    batch_id,
                    config.model_dump_json(),
                    datetime.now(timezone.utc),
                    len(results),
                    passed,
                ],
            )
            con.executemany(
                "INSERT INTO screen_results VALUES (?, ?, ?, ?)",
                [
                    [run_id, result.symbol, result.all_conditions_passed, result.model_dump_json()]
                    for result in results
                ],
            )
            con.execute("COMMIT")
        return run_id

    def load_run_results(self, run_id: str) -> List[ScreenResult]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT payload_json FROM screen_results WHERE run_id = ? ORDER BY symbol", [run_id]
            ).fetchall()
        return [ScreenResult.model_validate_json(row[0]) for row in rows]
