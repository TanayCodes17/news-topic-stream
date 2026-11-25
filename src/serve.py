"""Serving stage: a small REST API around the frozen artifacts.

    uvicorn src.serve:app --port 8000

The model is loaded once at startup and only ever used for ``transform`` /
``predict``. Nothing in this process re-fits anything -- that is the whole point
of "fixed training data, dynamic execution".
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src import db
from src.config import API_HOST, API_PORT, LIVE_DB
from src.ingest import score_and_store
from src.predict import ArtifactBundle
from src.sources.hn_live import HNLiveSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("serve")


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


class IngestItem(BaseModel):
    title: str
    object_id: str | None = None
    url: str | None = None
    created_at: str | None = None


class IngestRequest(BaseModel):
    items: list[IngestItem] = Field(..., min_length=1, max_length=500)


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db(LIVE_DB)
    app.state.bundle = ArtifactBundle.load()
    app.state.stop_event = threading.Event()
    app.state.poller = None

    if os.environ.get("AUTOSTART_POLLER", "1") == "1":
        source = HNLiveSource(app.state.bundle, db_path=LIVE_DB)
        thread = threading.Thread(
            target=source.run_forever,
            args=(app.state.stop_event,),
            name="hn-poller",
            daemon=True,
        )
        thread.start()
        app.state.poller = thread
        log.info("live poller started (interval %.0fs)", source.interval)
    else:
        log.info("AUTOSTART_POLLER=0 -- poller disabled, use POST /poll or the replay source")

    log.info("model loaded: k=%s, %s documents", app.state.bundle.metrics["k"], app.state.bundle.metrics["n_documents"])
    try:
        yield
    finally:
        app.state.stop_event.set()


app = FastAPI(
    title="News Topic Stream",
    description="Fixed K-Means model over a live news feed.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    bundle: ArtifactBundle = app.state.bundle
    stats = db.stats(LIVE_DB)
    return {
        "status": "ok",
        "model_loaded": True,
        "topics": len(bundle.topics),
        "documents_trained_on": bundle.metrics["n_documents"],
        "trained_at": bundle.metrics["trained_at_utc"],
        "items_scored": stats["total"],
        "last_ingest": stats["last_ingest"],
        "poller_running": bool(app.state.poller and app.state.poller.is_alive()),
    }


@app.get("/topics")
def topics() -> dict:
    bundle: ArtifactBundle = app.state.bundle
    return {
        "k": bundle.metrics["k"],
        "novelty_threshold": bundle.threshold,
        "topics": bundle.topics,
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    bundle: ArtifactBundle = app.state.bundle
    predictions = bundle.predict_many(request.texts)
    return {"count": len(predictions), "predictions": predictions}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    """Accept new headlines from any external source and store them scored."""
    bundle: ArtifactBundle = app.state.bundle
    items = [item.model_dump() for item in request.items]
    result = score_and_store(bundle, items, source="api", db_path=LIVE_DB)
    return {"scored": result["scored"], "inserted": result["inserted"]}


@app.get("/recent")
def recent(limit: int = Query(50, ge=1, le=500)) -> dict:
    return {"items": db.recent(limit, LIVE_DB)}


@app.get("/stats")
def stats() -> dict:
    bundle: ArtifactBundle = app.state.bundle
    payload = db.stats(LIVE_DB)
    payload["k"] = bundle.metrics["k"]
    payload["novelty_threshold"] = bundle.threshold
    return payload


@app.post("/poll")
def poll() -> dict:
    """Fetch one live batch right now instead of waiting for the timer."""
    bundle: ArtifactBundle = app.state.bundle
    source = HNLiveSource(bundle, db_path=LIVE_DB)
    result = source.poll_once()
    if "error" in result:
        raise HTTPException(status_code=502, detail=f"live source unavailable: {result['error']}")
    return result


if __name__ == "__main__":
    uvicorn.run("src.serve:app", host=API_HOST, port=API_PORT, reload=False)
