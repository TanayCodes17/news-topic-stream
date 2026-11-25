"""Shared path for anything that produces new headlines.

Both the live poller and the replay source hand items to ``score_and_store``,
so there is exactly one place where "new data meets the frozen model".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src import db
from src.config import LIVE_DB
from src.predict import ArtifactBundle


def _stable_id(title: str) -> str:
    return "t-" + hashlib.sha1(title.strip().lower().encode("utf-8")).hexdigest()[:16]


def score_and_store(
    bundle: ArtifactBundle,
    items: list[dict],
    source: str = "live",
    db_path: Path | str = LIVE_DB,
) -> dict:
    """Score items with the frozen artifacts and persist the results.

    ``items`` need only ``title``; ``object_id``, ``url`` and ``created_at`` are
    optional. Duplicate ids are ignored by the database, so repeated polls are
    harmless.
    """
    usable = [i for i in items if str(i.get("title") or "").strip()]
    if not usable:
        return {"scored": 0, "inserted": 0, "rows": []}

    predictions = bundle.predict_many([i["title"] for i in usable])

    rows = []
    for item, pred in zip(usable, predictions):
        rows.append(
            {
                "object_id": str(item.get("object_id") or _stable_id(item["title"])),
                "created_at": item.get("created_at"),
                "title": item["title"],
                "url": item.get("url") or "",
                "topic_id": pred["topic_id"],
                "topic_label": pred["label"],
                "distance": pred["distance"],
                "is_novel": pred["is_novel"],
                "source": source,
            }
        )

    already = db.existing_ids([r["object_id"] for r in rows], path=db_path)
    inserted = db.insert_predictions(rows, path=db_path)
    new_rows = [r for r in rows if r["object_id"] not in already]
    return {
        "scored": len(rows),
        "inserted": inserted,
        "new_rows": new_rows,
        "rows": rows,
    }
