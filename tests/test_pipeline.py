"""Fast tests for the frozen-model pipeline.

These train a deliberately tiny model in a temp directory, so the whole suite
runs offline in a couple of seconds and never touches the committed artifacts.
"""

from __future__ import annotations

import json

import pytest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src import db
from src.ingest import score_and_store
from src.predict import ArtifactBundle
from src.preprocess import build_vectorizer, clean_title
from src.train import train

TEMPLATES = {
    "ai": [
        "openai releases a new language model",
        "llm benchmark shows reasoning gains",
        "neural network training tricks",
        "gpt model api pricing changes",
        "transformer architecture explained",
        "ai agent framework released",
    ],
    "systems": [
        "rust compiler gets faster incremental builds",
        "linux kernel patch review process",
        "c++ template metaprogramming tricks",
        "memory allocator performance tuning",
        "debugging segfaults with gdb",
        "distributed database consensus with raft",
    ],
    "space": [
        "nasa rocket launch to mars",
        "telescope captures distant galaxy image",
        "satellite orbit debris tracking",
        "astronaut spacewalk on the iss",
        "lunar lander mission update",
        "solar storm aurora forecast",
    ],
}


def make_corpus(per_topic: int = 24):
    rows = []
    for i in range(per_topic):
        for topic, templates in TEMPLATES.items():
            rows.append(
                {
                    "object_id": f"{topic}-{i}",
                    "created_at": f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
                    "title": templates[i % len(templates)],
                    "url": "",
                }
            )
    return rows


@pytest.fixture(scope="module")
def mini_model(tmp_path_factory):
    """Train a small model on synthetic data with lenient vectoriser settings."""
    import src.train as train_module

    original_tfidf = train_module.build_vectorizer
    original_count = train_module.build_count_vectorizer

    def lenient_tfidf():
        return TfidfVectorizer(
            min_df=1, max_df=1.0, ngram_range=(1, 1), sublinear_tf=True, stop_words="english"
        )

    def lenient_count(vocabulary):
        return CountVectorizer(vocabulary=vocabulary, stop_words="english")

    train_module.build_vectorizer = lenient_tfidf
    train_module.build_count_vectorizer = lenient_count
    try:
        import pandas as pd

        out_dir = tmp_path_factory.mktemp("artifacts")
        metrics = train(
            pd.DataFrame(make_corpus()),
            out_dir=out_dir,
            svd_components=5,
            k_range=range(2, 6),
            verbose=False,
        )
        yield out_dir, metrics
    finally:
        train_module.build_vectorizer = original_tfidf
        train_module.build_count_vectorizer = original_count


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def test_clean_title_removes_forum_noise_and_urls():
    cleaned = clean_title("Show HN: Rust is 10x faster (2026) https://example.com/x")
    assert "rust" in cleaned and "faster" in cleaned
    assert "show" not in cleaned.split()
    assert "hn" not in cleaned.split()
    assert "2026" not in cleaned
    assert "http" not in cleaned


def test_clean_title_handles_bad_input():
    assert clean_title(None) == ""
    assert clean_title(42) == ""
    assert clean_title("   ") == ""


def test_vectorizer_uses_the_agreed_settings():
    params = build_vectorizer().get_params()
    assert params["ngram_range"] == (1, 2)
    assert params["min_df"] == 3
    assert params["max_df"] == 0.05
    assert "hn" in set(params["stop_words"])
    assert "like" in set(params["stop_words"])


def test_unknown_vocabulary_produces_no_features():
    # Behaviour of the vectoriser itself; the production settings need a corpus
    # far larger than a unit test should carry.
    vectorizer = TfidfVectorizer(min_df=1)
    vectorizer.fit(["rust compiler builds", "linux kernel patch"])
    matrix = vectorizer.transform(["zzzqqq xxyyzz"])
    assert matrix.getnnz(axis=1)[0] == 0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_writes_every_artifact(mini_model):
    out_dir, _ = mini_model
    for name in ["vectorizer.pkl", "svd.pkl", "kmeans.pkl", "topics.json", "metrics.json"]:
        assert (out_dir / name).exists(), name


def test_metrics_are_sane(mini_model):
    out_dir, metrics = mini_model
    saved = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert saved["k"] == metrics["k"]
    assert 2 <= saved["k"] <= 5
    assert saved["n_documents"] > 50
    assert saved["novelty_threshold"] > 0
    assert -1.0 <= saved["silhouette"] <= 1.0


def test_every_topic_has_a_label_and_keywords(mini_model):
    out_dir, _ = mini_model
    topics = json.loads((out_dir / "topics.json").read_text(encoding="utf-8"))
    assert len(topics) >= 2
    for topic in topics:
        assert topic["label"].strip()
        assert len(topic["keywords"]) >= 1
        assert topic["size"] >= 0


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_bundle_loads_and_predicts(mini_model):
    out_dir, _ = mini_model
    bundle = ArtifactBundle.load(out_dir)
    prediction = bundle.predict_one("rust compiler builds are faster now")
    assert prediction["topic_id"] is not None
    assert isinstance(prediction["distance"], float)
    assert prediction["is_novel"] is False


def test_predictions_are_aligned_with_input(mini_model):
    out_dir, _ = mini_model
    bundle = ArtifactBundle.load(out_dir)
    texts = ["rust compiler builds", "", "nasa rocket launch to mars"]
    predictions = bundle.predict_many(texts)
    assert len(predictions) == len(texts)
    assert predictions[1]["topic_id"] is None
    assert predictions[1]["label"] == "unassigned"


def test_empty_text_is_unassigned_not_guessed(mini_model):
    out_dir, _ = mini_model
    bundle = ArtifactBundle.load(out_dir)
    prediction = bundle.predict_one("")
    assert prediction == {
        "topic_id": None,
        "label": "unassigned",
        "keywords": [],
        "distance": None,
        "is_novel": True,
    }


def test_same_text_gives_same_topic_after_reload(mini_model):
    out_dir, _ = mini_model
    first = ArtifactBundle.load(out_dir).predict_one("linux kernel patch review")
    second = ArtifactBundle.load(out_dir).predict_one("linux kernel patch review")
    assert first == second


def test_missing_artifacts_raise_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        ArtifactBundle.load(tmp_path)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def test_insert_is_idempotent(tmp_path):
    path = tmp_path / "live.db"
    db.init_db(path)
    row = {
        "object_id": "abc123",
        "title": "rust compiler builds",
        "topic_id": 1,
        "topic_label": "rust",
        "distance": 0.2,
        "is_novel": False,
    }
    assert db.insert_predictions([row], path) == 1
    assert db.insert_predictions([row], path) == 0
    assert db.stats(path)["total"] == 1
    assert db.existing_ids(["abc123", "nope"], path) == {"abc123"}


def test_insert_recovers_if_table_disappears(tmp_path):
    path = tmp_path / "live.db"
    db.init_db(path)
    path.unlink()  # simulate a deleted database while the service is running
    inserted = db.insert_predictions([{"object_id": "x1", "title": "hello world"}], path)
    assert inserted == 1


def test_score_and_store_persists_topic_and_reports_new_rows(mini_model, tmp_path):
    out_dir, _ = mini_model
    bundle = ArtifactBundle.load(out_dir)
    path = tmp_path / "live.db"
    db.init_db(path)

    items = [{"object_id": "n1", "title": "nasa rocket launch to mars"}]
    first = score_and_store(bundle, items, source="test", db_path=path)
    assert first["inserted"] == 1
    assert first["new_rows"][0]["topic_label"]

    second = score_and_store(bundle, items, source="test", db_path=path)
    assert second["inserted"] == 0
    assert second["new_rows"] == []
    assert db.stats(path)["total"] == 1


def test_stats_group_by_topic(mini_model, tmp_path):
    out_dir, _ = mini_model
    bundle = ArtifactBundle.load(out_dir)
    path = tmp_path / "live.db"
    db.init_db(path)
    score_and_store(
        bundle,
        [
            {"object_id": "s1", "title": "nasa rocket launch to mars"},
            {"object_id": "s2", "title": "telescope captures distant galaxy image"},
        ],
        source="test",
        db_path=path,
    )
    payload = db.stats(path)
    assert payload["total"] == 2
    assert sum(t["count"] for t in payload["by_topic"]) == 2
