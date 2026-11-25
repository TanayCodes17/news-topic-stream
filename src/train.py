"""Offline training stage: frozen snapshot -> saved artifacts.

    python -m src.train

Everything here runs exactly once and writes to ``artifacts/``. The serving
process loads those files read-only and never re-fits anything.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from src.config import (
    ARTIFACT_DIR,
    K_RANGE,
    NOVELTY_PERCENTILE,
    REPLAY_CSV,
    SEED,
    SILHOUETTE_SAMPLE,
    SNAPSHOT_CSV,
    SNAPSHOT_META,
    SVD_COMPONENTS,
    TOP_KEYWORDS,
)
from src.predict import ArtifactBundle
from src.preprocess import build_count_vectorizer, build_vectorizer, clean_titles


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_snapshot(path: Path = SNAPSHOT_CSV) -> str | None:
    """Return a warning string if the committed CSV no longer matches its hash."""
    if not SNAPSHOT_META.exists():
        return "snapshot_meta.json not found; cannot verify the frozen data hash."
    meta = json.loads(SNAPSHOT_META.read_text(encoding="utf-8"))
    expected = meta.get("sha256_train_csv")
    actual = sha256_of(path)
    if expected and expected != actual:
        return (
            "WARNING: train_snapshot.csv hash does not match snapshot_meta.json.\n"
            f"  expected {expected}\n  actual   {actual}"
        )
    return None


def name_topics(counts, labels: np.ndarray, feature_names: np.ndarray, k: int) -> list[dict]:
    """Label each cluster with c-TF-IDF keywords.

    Plain mean TF-IDF ranks a term highly just because it is rare in the corpus,
    so a single niche story can name a 1,500-document cluster. c-TF-IDF scores
    ``term frequency inside the cluster x how exclusive that term is to the
    cluster``, which is what actually describes the theme.
    """
    sizes = np.bincount(labels, minlength=k)
    per_cluster = np.vstack(
        [np.asarray(counts[labels == c].sum(axis=0)).ravel() for c in range(k)]
    )
    totals = per_cluster.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    tf = per_cluster / totals
    corpus_freq = per_cluster.sum(axis=0)
    avg_words = per_cluster.sum(axis=1).mean()
    scores = tf * np.log(1.0 + avg_words / (corpus_freq + 1e-9))

    topics: list[dict] = []
    for cluster in range(k):
        order = scores[cluster].argsort()[::-1]
        keywords: list[str] = []
        used_words: set[str] = set()
        for idx in order:
            if scores[cluster][idx] <= 0:
                break
            term = str(feature_names[idx])
            words = set(term.split())
            # Skip near-duplicates ("openai" vs "openai model") so the label
            # stays informative.
            if words & used_words:
                continue
            keywords.append(term)
            used_words |= words
            if len(keywords) >= TOP_KEYWORDS:
                break

        topics.append(
            {
                "topic_id": cluster,
                "label": " · ".join(keywords[:3]) if keywords else f"topic {cluster}",
                "keywords": keywords,
                "size": int(sizes[cluster]),
            }
        )
    return topics


def choose_k(projected: np.ndarray, k_range=K_RANGE) -> tuple[int, dict[int, float]]:
    """Pick k by silhouette score; return the winner and the full sweep."""
    scores: dict[int, float] = {}
    n = len(projected)
    sample = min(SILHOUETTE_SAMPLE, n - 1)
    for k in k_range:
        if k >= n:
            continue
        model = KMeans(n_clusters=k, n_init=4, random_state=SEED)
        labels = model.fit_predict(projected)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(
            silhouette_score(projected, labels, sample_size=sample, random_state=SEED)
        )
        print(f"  k={k:>2}  silhouette={scores[k]:.4f}")
    if not scores:
        raise RuntimeError("Could not evaluate any k -- dataset too small.")
    best = max(scores, key=scores.get)
    return best, scores


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    df: pd.DataFrame,
    out_dir: Path = ARTIFACT_DIR,
    svd_components: int = SVD_COMPONENTS,
    k_range=K_RANGE,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = clean_titles(df["title"].tolist())
    kept = [t for t in texts if t]
    if len(kept) < 50:
        raise ValueError(f"Need at least 50 usable documents, got {len(kept)}.")

    vectorizer = build_vectorizer()
    matrix = vectorizer.fit_transform(kept)
    vocab_size = len(vectorizer.vocabulary_)

    # L2-normalise after SVD: Euclidean distance then behaves like cosine, which
    # stops the first component (overall "genericness") from swallowing most
    # documents into one giant cluster.
    components = max(2, min(svd_components, matrix.shape[1] - 1, len(kept) - 1))
    svd = TruncatedSVD(n_components=components, random_state=SEED)
    projected = normalize(svd.fit_transform(matrix))

    if verbose:
        print(f"documents      : {len(kept)}")
        print(f"vocabulary     : {vocab_size} terms")
        print(f"SVD components : {components} "
              f"({svd.explained_variance_ratio_.sum():.1%} variance kept)")
        print("selecting k ...")

    best_k, sweep = choose_k(projected, k_range)

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=SEED)
    labels = kmeans.fit_predict(projected)

    distances = kmeans.transform(projected)
    nearest = distances[np.arange(len(projected)), labels]
    threshold = float(np.percentile(nearest, NOVELTY_PERCENTILE))

    counts = build_count_vectorizer(vectorizer.vocabulary_).fit_transform(kept)
    topics = name_topics(counts, labels, vectorizer.get_feature_names_out(), best_k)

    sample = min(SILHOUETTE_SAMPLE, len(projected) - 1)
    metrics = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "n_documents": int(len(kept)),
        "n_features": int(vocab_size),
        "svd_components": int(components),
        "svd_variance_kept": round(float(svd.explained_variance_ratio_.sum()), 4),
        "k": int(best_k),
        "silhouette": round(
            float(silhouette_score(projected, labels, sample_size=sample, random_state=SEED)), 4
        ),
        "davies_bouldin": round(float(davies_bouldin_score(projected, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(projected, labels)), 2),
        "novelty_percentile": NOVELTY_PERCENTILE,
        "novelty_threshold": round(threshold, 4),
        "silhouette_sweep": {str(k): round(v, 4) for k, v in sorted(sweep.items())},
        "cluster_sizes": sorted(
            [int(s) for s in np.bincount(labels, minlength=best_k)], reverse=True
        ),
        "snapshot_sha256": sha256_of(SNAPSHOT_CSV) if SNAPSHOT_CSV.exists() else None,
    }

    joblib.dump(vectorizer, out_dir / "vectorizer.pkl")
    joblib.dump(svd, out_dir / "svd.pkl")
    joblib.dump(kmeans, out_dir / "kmeans.pkl")
    (out_dir / "topics.json").write_text(json.dumps(topics, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nchosen k       : {best_k}")
        print(f"silhouette     : {metrics['silhouette']:.4f}")
        print(f"novelty cutoff : {threshold:.4f} (p{NOVELTY_PERCENTILE} of training distances)")
        print("\ndiscovered topics")
        for topic in sorted(topics, key=lambda t: -t["size"]):
            print(f"  {topic['size']:>5} docs  |  {topic['label']}")
        print(f"\nartifacts -> {out_dir}")

    return metrics


# --------------------------------------------------------------------------- #
# End-to-end check on data the model has never seen
# --------------------------------------------------------------------------- #
def evaluate_holdout(out_dir: Path = ARTIFACT_DIR, holdout_path: Path = REPLAY_CSV) -> dict | None:
    if not holdout_path.exists():
        return None

    bundle = ArtifactBundle.load(out_dir)  # exercises the real serving path
    df = pd.read_csv(holdout_path)
    titles = df["title"].tolist()

    started = time.perf_counter()
    predictions = bundle.predict_many(titles)
    elapsed = time.perf_counter() - started

    assigned = [p for p in predictions if p["topic_id"] is not None]
    novel = sum(1 for p in predictions if p["is_novel"])
    top = Counter(p["label"] for p in assigned).most_common(5)

    result = {
        "n": len(titles),
        "assigned_rate": round(len(assigned) / max(len(titles), 1), 4),
        "novel_rate": round(novel / max(len(titles), 1), 4),
        "ms_per_item": round(elapsed / max(len(titles), 1) * 1000, 3),
        "top_topics": [{"label": label, "count": count} for label, count in top],
    }
    print("\nholdout check (unseen data, saved artifacts only)")
    print(f"  items            : {result['n']}")
    print(f"  assigned         : {result['assigned_rate']:.1%}")
    print(f"  flagged novel    : {result['novel_rate']:.1%}")
    print(f"  inference speed  : {result['ms_per_item']:.3f} ms/item")
    return result


def main() -> None:
    if not SNAPSHOT_CSV.exists():
        raise SystemExit(f"{SNAPSHOT_CSV} not found. Run `python -m src.fetch_snapshot` first.")

    warning = verify_snapshot()
    if warning:
        print(warning)

    df = pd.read_csv(SNAPSHOT_CSV)
    print(f"loaded frozen snapshot: {len(df)} rows from {SNAPSHOT_CSV.name}")
    metrics = train(df)

    holdout = evaluate_holdout()
    if holdout:
        metrics["holdout"] = holdout
        (ARTIFACT_DIR / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
