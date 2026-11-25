"""Load the frozen artifacts and score incoming text.

This module is the *only* bridge between the offline stage and the online
stage. It never fits anything: the vectorizer, SVD and K-Means it uses were
fitted once by ``src/train.py`` on the committed snapshot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from src.config import ARTIFACT_DIR
from src.preprocess import clean_titles

UNASSIGNED = {
    "topic_id": None,
    "label": "unassigned",
    "keywords": [],
    "distance": None,
    "is_novel": True,
}


@dataclass
class ArtifactBundle:
    vectorizer: object
    svd: object
    kmeans: object
    topics: list[dict]
    threshold: float
    metrics: dict

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, artifact_dir: Path | str = ARTIFACT_DIR) -> "ArtifactBundle":
        d = Path(artifact_dir)
        required = ["vectorizer.pkl", "svd.pkl", "kmeans.pkl", "topics.json", "metrics.json"]
        missing = [f for f in required if not (d / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing artifacts in {d}: {missing}. Run `python -m src.train` first."
            )
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        topics = json.loads((d / "topics.json").read_text(encoding="utf-8"))
        return cls(
            vectorizer=joblib.load(d / "vectorizer.pkl"),
            svd=joblib.load(d / "svd.pkl"),
            kmeans=joblib.load(d / "kmeans.pkl"),
            topics=topics,
            threshold=float(metrics["novelty_threshold"]),
            metrics=metrics,
        )

    # -------------------------------------------------------------- inference
    def predict_many(self, texts: Sequence[object]) -> list[dict]:
        """Assign each text to a discovered topic.

        Returns one dict per input, in order. Texts that clean down to nothing,
        or that contain no known vocabulary, are returned as ``unassigned``
        instead of being forced into an arbitrary cluster.
        """
        cleaned = clean_titles(texts)
        results: list[dict] = [dict(UNASSIGNED) for _ in texts]
        if not cleaned:
            return results

        matrix = self.vectorizer.transform(cleaned)
        has_terms = matrix.getnnz(axis=1) > 0
        if not has_terms.any():
            return results

        rows = np.flatnonzero(has_terms)
        # Training clustered L2-normalised SVD vectors, so serving must project
        # into exactly that space. Skipping this silently sends almost every
        # headline to whichever centroid happens to sit nearest the origin.
        projected = normalize(self.svd.transform(matrix[rows]))
        distances = self.kmeans.transform(projected)  # (n, k)
        labels = distances.argmin(axis=1)
        nearest = distances[np.arange(len(rows)), labels]

        for row, label, dist in zip(rows, labels, nearest):
            topic = self.topics[int(label)]
            results[int(row)] = {
                "topic_id": int(label),
                "label": topic["label"],
                "keywords": topic["keywords"],
                "distance": round(float(dist), 4),
                "is_novel": bool(dist > self.threshold),
            }
        return results

    def predict_one(self, text: object) -> dict:
        return self.predict_many([text])[0]

    # ------------------------------------------------------------------ views
    def topic_table(self) -> list[dict]:
        return self.topics
