"""Live source: newest Hacker News stories, polled from the Algolia API.

Same endpoint the frozen snapshot came from, so live headlines share the
training distribution. No API key, no rate limit worth worrying about.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import requests

from src.config import HN_API, LIVE_DB, POLL_BATCH, POLL_INTERVAL_SECONDS
from src.ingest import score_and_store
from src.predict import ArtifactBundle

log = logging.getLogger(__name__)


class HNLiveSource:
    def __init__(
        self,
        bundle: ArtifactBundle,
        db_path: Path | str = LIVE_DB,
        batch: int = POLL_BATCH,
        interval: float = POLL_INTERVAL_SECONDS,
        timeout: float = 20.0,
    ) -> None:
        self.bundle = bundle
        self.db_path = db_path
        self.batch = batch
        self.interval = interval
        self.timeout = timeout

    def fetch(self) -> list[dict]:
        resp = requests.get(
            HN_API,
            params={"tags": "story", "hitsPerPage": self.batch, "page": 0},
            timeout=self.timeout,
        )
        resp.raise_for_status()

        items: list[dict] = []
        for hit in resp.json().get("hits", []):
            title = (hit.get("title") or "").strip()
            object_id = str(hit.get("objectID") or "").strip()
            if not title or not object_id:
                continue
            items.append(
                {
                    "object_id": object_id,
                    "title": title,
                    "url": hit.get("url") or "",
                    "created_at": hit.get("created_at"),
                }
            )
        return items

    def poll_once(self) -> dict:
        """Fetch, score and store one batch. Never raises on network failure."""
        try:
            items = self.fetch()
        except requests.RequestException as exc:
            log.warning("live poll failed: %s", exc)
            return {"scored": 0, "inserted": 0, "error": str(exc)}

        result = score_and_store(self.bundle, items, source="live", db_path=self.db_path)
        log.info(
            "live poll: %s fetched, %s new",
            result["scored"],
            result["inserted"],
        )
        return result

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        while not (stop_event and stop_event.is_set()):
            try:
                self.poll_once()
            except Exception:  # never let one bad poll kill the stream
                log.exception("unexpected error during live poll; continuing")
            # Sleep in slices so shutdown stays responsive.
            deadline = time.monotonic() + self.interval
            while time.monotonic() < deadline:
                if stop_event and stop_event.is_set():
                    return
                time.sleep(0.1)
