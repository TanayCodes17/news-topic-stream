"""Replay source: a held-out slice of the snapshot, played back as a stream.

Every row here was excluded from training, so it behaves like genuinely unseen
data. This is the offline fallback: the demo works even with no network.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pandas as pd

from src.config import LIVE_DB, REPLAY_CSV, REPLAY_INTERVAL_SECONDS
from src.ingest import score_and_store
from src.predict import ArtifactBundle

log = logging.getLogger(__name__)


class ReplaySource:
    def __init__(
        self,
        bundle: ArtifactBundle,
        csv_path: Path | str = REPLAY_CSV,
        db_path: Path | str = LIVE_DB,
        interval: float = REPLAY_INTERVAL_SECONDS,
        loop: bool = False,
    ) -> None:
        self.bundle = bundle
        self.csv_path = Path(csv_path)
        self.db_path = db_path
        self.interval = interval
        self.loop = loop
        self._position = 0
        self._frame: pd.DataFrame | None = None

    @property
    def frame(self) -> pd.DataFrame:
        if self._frame is None:
            if not self.csv_path.exists():
                raise FileNotFoundError(
                    f"{self.csv_path} not found. Run `python -m src.fetch_snapshot` first."
                )
            df = pd.read_csv(self.csv_path)
            self._frame = df.sort_values("created_at").reset_index(drop=True)
        return self._frame

    @property
    def exhausted(self) -> bool:
        return self._position >= len(self.frame)

    def poll_once(self) -> dict:
        """Emit the next held-out headline."""
        if self.exhausted:
            if not self.loop:
                return {"scored": 0, "inserted": 0, "done": True}
            self._position = 0

        row = self.frame.iloc[self._position]
        self._position += 1

        item = {
            "object_id": str(row["object_id"]),
            "title": str(row["title"]),
            "url": str(row.get("url") or ""),
            "created_at": str(row.get("created_at") or ""),
        }
        return score_and_store(self.bundle, [item], source="replay", db_path=self.db_path)

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        while not (stop_event and stop_event.is_set()):
            try:
                result = self.poll_once()
            except Exception:  # keep replaying even if one row fails
                log.exception("unexpected error during replay; continuing")
                result = {}
            if result.get("done"):
                log.info("replay finished")
                return
            time.sleep(self.interval)
