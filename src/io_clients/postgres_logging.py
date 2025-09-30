"""Registro de logs para monitoreo de Postgres"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Optional, Sequence

import psycopg2
from psycopg2.extensions import connection
from psycopg2.extras import execute_batch

from src.config import PostgresSettings, get_settings

logger = logging.getLogger(__name__)


CREATE_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS monitoring;
"""

CREATE_SCORING_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS monitoring.scoring_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(128) NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP NOT NULL,
    rows_processed INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    error_rate DOUBLE PRECISION NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SEGMENT_METRICS_SQL = """
CREATE TABLE IF NOT EXISTS monitoring.segment_metrics (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(128) NOT NULL,
    segment VARCHAR(128) NOT NULL,
    metric VARCHAR(128) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class PostgresLogger:
    settings: PostgresSettings

    @classmethod
    def from_settings(cls, settings: Optional[PostgresSettings] = None) -> "PostgresLogger":
        return cls(settings=settings or get_settings().postgres)

    @contextmanager
    def connect(self) -> Iterator[connection]:
        conn = psycopg2.connect(self.settings.monitoring_uri())
        try:
            yield conn
        finally:
            conn.close()

    def ensure_tables(self) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            logger.debug("Ensuring monitoring tables exist")
            cur.execute(CREATE_SCHEMA_SQL)
            cur.execute(CREATE_SCORING_RUNS_SQL)
            cur.execute(CREATE_SEGMENT_METRICS_SQL)
            conn.commit()

    def log_scoring_run(self, run_id: str, started_at: datetime, finished_at: datetime, rows_processed: int, latency_ms: int, error_rate: float, notes: Optional[str] = None) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO monitoring.scoring_runs(
                    run_id, started_at, finished_at, rows_processed, latency_ms, error_rate, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (run_id, started_at, finished_at, rows_processed, latency_ms, error_rate, notes),
            )
            conn.commit()
            logger.info("Logged scoring run %s", run_id)

    def log_segment_metrics(self, run_id: str, rows: Iterable[tuple[str, str, float]]) -> None:
        payload: Sequence[tuple[str, str, float]] = list(rows)
        if not payload:
            return
        with self.connect() as conn, conn.cursor() as cur:
            execute_batch(
                cur,
                """
                INSERT INTO monitoring.segment_metrics(run_id, segment, metric, value)
                VALUES (%s, %s, %s, %s)
                """,
                [(run_id, segment, metric, value) for segment, metric, value in payload],
            )
            conn.commit()
            logger.info("Logged %d segment metrics for run %s", len(payload), run_id)


__all__ = ["PostgresLogger", "CREATE_SCHEMA_SQL", "CREATE_SCORING_RUNS_SQL", "CREATE_SEGMENT_METRICS_SQL"]
