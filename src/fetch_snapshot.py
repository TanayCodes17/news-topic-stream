"""Download the FROZEN training snapshot and the replay holdout.

Run this once. The CSV it produces is committed to the repository, so training
is reproducible and never touches the network again. The hash in
``data/snapshot_meta.json`` lets anyone verify the frozen data is unchanged.

Usage:
    python -m src.fetch_snapshot            # ~13k headlines, ~1 minute
    python -m src.fetch_snapshot --rows 5000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    FETCH_PAGE_SIZE,
    FETCH_SLEEP_SECONDS,
    HN_API,
    HOLDOUT_ROWS,
    REPLAY_CSV,
    SNAPSHOT_CSV,
    SNAPSHOT_META,
    SNAPSHOT_TARGET_ROWS,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_batch(before_ts: int | None, hits: int) -> list[dict]:
    """One page of stories, newest first, optionally before a unix timestamp.

    The Algolia API caps a single query at 1000 records (``paginationLimitedTo``),
    so deep history is collected with a timestamp cursor rather than ``page``.
    """
    params: dict = {"tags": "story", "hitsPerPage": hits, "page": 0}
    if before_ts is not None:
        params["numericFilters"] = f"created_at_i<{before_ts}"
    resp = requests.get(HN_API, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("hits", [])


def fetch_rows(target_rows: int, page_size: int, sleep: float) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_titles: set[str] = set()
    cursor: int | None = None
    batch = 0

    while len(rows) < target_rows:
        hits = fetch_batch(cursor, page_size)
        if not hits:
            break
        stamps = [int(h["created_at_i"]) for h in hits if h.get("created_at_i")]
        if not stamps:
            break
        next_cursor = min(stamps)
        if cursor is not None and next_cursor >= cursor:
            break  # cursor is not advancing -- stop instead of looping forever
        cursor = next_cursor

        for hit in hits:
            object_id = str(hit.get("objectID") or "").strip()
            title = (hit.get("title") or "").strip()
            if not object_id or not title or object_id in seen_ids:
                continue
            # Near-duplicate reposts would inflate cluster density, so keep the
            # first (newest) occurrence of each normalised title.
            key = " ".join(title.lower().split())
            if key in seen_titles:
                continue
            seen_ids.add(object_id)
            seen_titles.add(key)
            rows.append(
                {
                    "object_id": object_id,
                    "created_at": hit.get("created_at"),
                    "title": title,
                    "url": hit.get("url") or "",
                    "points": hit.get("points") or 0,
                    "num_comments": hit.get("num_comments") or 0,
                }
            )
        batch += 1
        print(f"  batch {batch:>2}: {len(rows):>6} unique headlines")
        time.sleep(sleep)

    return rows


def split_chronological(df: pd.DataFrame, holdout_rows: int):
    """Oldest 90-ish% -> training, newest tail -> replay holdout.

    The holdout is never seen by the model, which is what makes it usable as a
    stand-in live stream when the network is unavailable.
    """
    df = df.sort_values("created_at").reset_index(drop=True)
    if len(df) <= holdout_rows:
        return df, df.iloc[0:0]
    return df.iloc[:-holdout_rows].reset_index(drop=True), df.iloc[-holdout_rows:].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a Hacker News headline snapshot.")
    parser.add_argument("--rows", type=int, default=SNAPSHOT_TARGET_ROWS)
    parser.add_argument("--page-size", type=int, default=FETCH_PAGE_SIZE)
    parser.add_argument("--sleep", type=float, default=FETCH_SLEEP_SECONDS)
    args = parser.parse_args()

    print(f"Fetching ~{args.rows} headlines from {HN_API} ...")
    rows = fetch_rows(args.rows, args.page_size, args.sleep)
    if not rows:
        raise SystemExit("No data returned by the API -- aborting without writing files.")

    df = pd.DataFrame(rows)
    train_df, holdout_df = split_chronological(df, HOLDOUT_ROWS)

    SNAPSHOT_CSV.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SNAPSHOT_CSV, index=False)
    holdout_df.to_csv(REPLAY_CSV, index=False)

    meta = {
        "source": HN_API,
        "query": {"tags": "story"},
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_train": int(len(train_df)),
        "rows_holdout": int(len(holdout_df)),
        "created_at_min": str(train_df["created_at"].min()),
        "created_at_max": str(train_df["created_at"].max()),
        "sha256_train_csv": _sha256(SNAPSHOT_CSV),
        "sha256_holdout_csv": _sha256(REPLAY_CSV),
    }
    SNAPSHOT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nFrozen snapshot written")
    print(f"  training rows : {len(train_df):>6}  -> {SNAPSHOT_CSV}")
    print(f"  holdout rows  : {len(holdout_df):>6}  -> {REPLAY_CSV}")
    print(f"  sha256        : {meta['sha256_train_csv']}")


if __name__ == "__main__":
    main()
