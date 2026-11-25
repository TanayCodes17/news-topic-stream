"""SQLite sink for every headline the service has scored.

A tiny stdlib store is enough: the dashboard needs recent items and aggregate
counts, and having history on disk means a demo can be screenshotted after the
stream has been running for a while.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.config import LIVE_DB

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    object_id    TEXT    NOT NULL UNIQUE,
    created_at   TEXT,
    ingested_at  TEXT    NOT NULL,
    title        TEXT    NOT NULL,
    url          TEXT,
    topic_id     INTEGER,
    topic_label  TEXT,
    distance     REAL,
    is_novel     INTEGER NOT NULL DEFAULT 0,
    source       TEXT    NOT NULL DEFAULT 'live'
);
CREATE INDEX IF NOT EXISTS idx_predictions_topic ON predictions(topic_id);
"""

_COLUMNS = (
    "object_id",
    "created_at",
    "ingested_at",
    "title",
    "url",
    "topic_id",
    "topic_label",
    "distance",
    "is_novel",
    "source",
)


def connect(path: Path | str = LIVE_DB) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def session(path: Path | str = LIVE_DB):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | str = LIVE_DB) -> None:
    with session(path) as conn:
        conn.executescript(SCHEMA)


def insert_predictions(rows: Iterable[dict], path: Path | str = LIVE_DB) -> int:
    """Insert scored headlines, ignoring ones already stored. Returns new rows."""
    payload = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row in rows:
        payload.append(
            (
                row["object_id"],
                row.get("created_at") or now,
                row.get("ingested_at") or now,
                row["title"],
                row.get("url") or "",
                row.get("topic_id"),
                row.get("topic_label"),
                row.get("distance"),
                1 if row.get("is_novel") else 0,
                row.get("source") or "live",
            )
        )
    if not payload:
        return 0

    placeholders = ",".join("?" * len(_COLUMNS))
    statement = (
        f"INSERT OR IGNORE INTO predictions ({','.join(_COLUMNS)}) VALUES ({placeholders})"
    )
    try:
        with session(path) as conn:
            before = conn.total_changes
            conn.executemany(statement, payload)
            return conn.total_changes - before
    except sqlite3.OperationalError as exc:
        # The database file can be deleted or replaced while the service runs
        # (a fresh demo, a cleanup script). Recreate the schema and retry once.
        if "no such table" not in str(exc):
            raise
        init_db(path)
        with session(path) as conn:
            before = conn.total_changes
            conn.executemany(statement, payload)
            return conn.total_changes - before


def recent(limit: int = 50, path: Path | str = LIVE_DB) -> list[dict]:
    with session(path) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def existing_ids(object_ids: Iterable[str], path: Path | str = LIVE_DB) -> set[str]:
    """Which of these ids are already stored (SQLite caps parameters at ~999)."""
    ids = [str(i) for i in object_ids if i]
    found: set[str] = set()
    with session(path) as conn:
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT object_id FROM predictions WHERE object_id IN ({placeholders})",
                chunk,
            ).fetchall()
            found.update(r["object_id"] for r in rows)
    return found


def stats(path: Path | str = LIVE_DB) -> dict:
    with session(path) as conn:
        totals = conn.execute(
            "SELECT COUNT(*) AS total, "
            "       COALESCE(SUM(is_novel), 0) AS novel, "
            "       MAX(ingested_at) AS last_ingest, "
            "       AVG(distance) AS avg_distance "
            "FROM predictions"
        ).fetchone()
        by_topic = conn.execute(
            "SELECT topic_id, topic_label, COUNT(*) AS n "
            "FROM predictions GROUP BY topic_id, topic_label ORDER BY n DESC"
        ).fetchall()

    total = int(totals["total"] or 0)
    novel = int(totals["novel"] or 0)
    return {
        "total": total,
        "novel": novel,
        "novel_rate": round(novel / total, 4) if total else 0.0,
        "avg_distance": round(float(totals["avg_distance"]), 4)
        if totals["avg_distance"] is not None
        else None,
        "last_ingest": totals["last_ingest"],
        "by_topic": [
            {
                "topic_id": row["topic_id"],
                "label": row["topic_label"] or "unassigned",
                "count": int(row["n"]),
            }
            for row in by_topic
        ],
    }
