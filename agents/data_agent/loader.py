import polars as pl
import duckdb
import redis
import json
import os
from pathlib import Path

class StaticLoader:
    """Loads CSV, JSON, or Parquet files via DuckDB + Polars."""

    SUPPORTED = {".csv", ".json", ".parquet", ".jsonl"}

    def load(self, file_path: str) -> pl.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if path.suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        con = duckdb.connect()
        df  = con.execute(f"SELECT * FROM {self._reader(path)}('{file_path}')").pl()
        con.close()
        return df

    def load_sample(self, file_path: str, n: int = 10_000) -> pl.DataFrame:
        path = Path(file_path)
        con  = duckdb.connect()
        df   = con.execute(
            f"SELECT * FROM {self._reader(path)}('{file_path}') USING SAMPLE {n} ROWS"
        ).pl()
        con.close()
        return df

    def _reader(self, path: Path) -> str:
        return {
            ".csv":     "read_csv_auto",
            ".json":    "read_json_auto",
            ".jsonl":   "read_json_auto",
            ".parquet": "read_parquet",
        }.get(path.suffix.lower(), "read_csv_auto")

class LiveStreamLoader:
    """
    Consumes events from a Redis LIST (dev mode).
    Swap redis_client → kafka_consumer for production.
    """

    def __init__(self, topic: str = "nexus:live:orders",
                 host: str = "localhost", port: int = 6379):
        self.topic  = topic
        self.client = redis.Redis(host=host, port=port, decode_responses=True)

    def consume_batch(self, batch_size: int = 100) -> pl.DataFrame:
        """
        Pops up to batch_size events from the queue.
        Returns a DataFrame (empty if no events).
        """
        raw_events = []
        for _ in range(batch_size):
            item = self.client.lpop(self.topic)
            if item is None:
                break
            raw_events.append(json.loads(item))

        if not raw_events:
            return pl.DataFrame()  # empty — caller handles this

        return pl.DataFrame(raw_events)

    def queue_length(self) -> int:
        return self.client.llen(self.topic)

    def consume_blocking(self, timeout: int = 5) -> dict | None:
        """
        Blocks until one event arrives (or timeout).
        Use this in the Monitor Agent's main loop.
        """
        result = self.client.blpop(self.topic, timeout=timeout)
        if result is None:
            return None
        _, raw = result
        return json.loads(raw)