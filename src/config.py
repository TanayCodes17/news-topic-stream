"""Single source of truth for paths, hyper-parameters and the random seed.

Everything the offline stage and the online stage agree on lives here, so the
two can never drift apart (e.g. a different TF-IDF setting at serving time).
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

SNAPSHOT_CSV = DATA_DIR / "train_snapshot.csv"
REPLAY_CSV = DATA_DIR / "replay_holdout.csv"
SNAPSHOT_META = DATA_DIR / "snapshot_meta.json"
LIVE_DB = DATA_DIR / "live.db"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Data collection (Hacker News via the Algolia API -- free, no key required)
# --------------------------------------------------------------------------- #
HN_API = "https://hn.algolia.com/api/v1/search_by_date"
SNAPSHOT_TARGET_ROWS = 13_000
HOLDOUT_ROWS = 1_000
FETCH_PAGE_SIZE = 1_000
FETCH_SLEEP_SECONDS = 0.3

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
TFIDF_PARAMS: dict = {
    "min_df": 3,
    "max_df": 0.05,
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "strip_accents": "unicode",
    # Tokens must start with a letter, so bare numbers ("10", "3d") never
    # become features.
    "token_pattern": r"(?u)\b[a-zA-Z][a-zA-Z0-9+#]{1,}\b",
}

# Words that are frequent in headlines but say nothing about the topic. sklearn's
# ENGLISH_STOP_WORDS already covers the function words; these are the remaining
# headline filler plus Hacker News artefacts.
EXTRA_STOPWORDS = frozenset(
    """
    hn show ask tell launch edit via update updates
    new year years day days week weeks month today tomorrow yesterday
    make makes made making get gets got getting go goes going went
    come comes came take takes took give gives gave
    see sees saw look looks looked want wants wanted
    need needs needed know knows knew think thinks thought
    say says said tells told asked
    why how what when where who which while about after before
    over under between than just still even now ever never again
    also back away yet already much many more most less
    best top good bad big small first last one two three
    people world thing things way ways something anything everything
    like use uses used using build builds built building
    pdf html www com org net
    don doesn isn aren wasn weren hasn haven hadn didn won wouldn couldn shouldn
    """.split()
)

SVD_COMPONENTS = 30
K_RANGE = tuple(range(12, 25))
SILHOUETTE_SAMPLE = 3_000

# A live headline whose distance to its nearest centroid exceeds this
# percentile of the *training* distances is reported as "novel".
NOVELTY_PERCENTILE = 95
TOP_KEYWORDS = 6

# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
API_HOST = "127.0.0.1"
API_PORT = 8010
POLL_INTERVAL_SECONDS = 30
POLL_BATCH = 40
REPLAY_INTERVAL_SECONDS = 2.0
