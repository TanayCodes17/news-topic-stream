"""API surface tests.

The live poller is disabled here so the suite never depends on the network.
"""

from __future__ import annotations

import os

os.environ["AUTOSTART_POLLER"] = "0"

import pytest
from fastapi.testclient import TestClient

from src.serve import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["topics"] > 1
    assert payload["documents_trained_on"] > 1000
    assert payload["poller_running"] is False


def test_topics_expose_labels_and_keywords(client):
    payload = client.get("/topics").json()
    assert payload["k"] == len(payload["topics"])
    assert payload["novelty_threshold"] > 0
    for topic in payload["topics"]:
        assert topic["label"].strip()
        assert isinstance(topic["keywords"], list)


def test_predict_returns_one_result_per_text(client):
    texts = [
        "Rust compiler gets faster incremental builds",
        "Nvidia earnings beat expectations",
        "",
    ]
    response = client.post("/predict", json={"texts": texts})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(texts)
    assert len(body["predictions"]) == len(texts)
    assert body["predictions"][2]["label"] == "unassigned"


def test_predict_rejects_empty_payload(client):
    assert client.post("/predict", json={"texts": []}).status_code == 422


def test_recent_and_stats_are_consistent(client):
    recent = client.get("/recent", params={"limit": 5}).json()
    stats = client.get("/stats").json()
    assert isinstance(recent["items"], list)
    assert stats["total"] >= len(recent["items"])
    assert 0.0 <= stats["novel_rate"] <= 1.0
