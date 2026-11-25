"""Live dashboard: new headlines scored by the frozen model, as they arrive.

    streamlit run dashboard/app.py

Talks to the API over plain HTTP and re-runs every few seconds, so the page
never holds a socket open and survives an API restart.
"""

from __future__ import annotations

import colorsys
import html
import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8010").rstrip("/")
REFRESH_SECONDS = float(os.environ.get("REFRESH_SECONDS", "5"))
FEED_LIMIT = int(os.environ.get("FEED_LIMIT", "40"))

st.set_page_config(page_title="News Topic Stream", page_icon="📰", layout="wide")

CSS = """
<style>
  .block-container {padding-top: 2rem; padding-bottom: 2rem;}
  .feed-row {
      display: flex; align-items: baseline; gap: 10px;
      padding: 7px 2px; border-bottom: 1px solid rgba(128,128,128,.16);
      font-size: 15px;
  }
  .dot {width: 9px; height: 9px; border-radius: 50%; flex: 0 0 9px; transform: translateY(-1px);}
  .chip {
      font-size: 11px; letter-spacing: .02em; text-transform: uppercase;
      border: 1px solid currentColor; border-radius: 999px;
      padding: 1px 8px; white-space: nowrap; opacity: .95;
  }
  .headline {flex: 1 1 auto; color: inherit; text-decoration: none;}
  .headline:hover {text-decoration: underline;}
  .meta {font-size: 11px; opacity: .55; white-space: nowrap; font-variant-numeric: tabular-nums;}
  .novel {font-size: 10px; font-weight: 600; color: #d97706; border: 1px solid #d97706;
          border-radius: 4px; padding: 0 4px; white-space: nowrap;}
  .muted {opacity: .6; font-size: 13px;}
</style>
"""


@st.cache_resource
def http() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "news-topic-stream-dashboard"})
    return session


def get(path: str, **params):
    try:
        response = http().get(f"{API_URL}{path}", params=params, timeout=4)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # API down, wrong port, timeout
        return {"_error": f"{type(exc).__name__}: {exc}"}


def topic_color(topic_id) -> str:
    if topic_id is None:
        return "#9aa0a6"
    hue = (int(topic_id) * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.55, 0.78)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def clock(value: str | None) -> str:
    if not value:
        return "--:--:--"
    return str(value)[11:19] or "--:--:--"


def render_feed(items: list[dict], only_novel: bool) -> None:
    rows = []
    for item in items:
        if only_novel and not item.get("is_novel"):
            continue
        colour = topic_color(item.get("topic_id"))
        title = html.escape(str(item.get("title", "")))
        label = html.escape(str(item.get("topic_label") or "unassigned"))
        url = item.get("url") or ""
        headline = (
            f'<a class="headline" href="{html.escape(url)}" target="_blank" rel="noopener">{title}</a>'
            if url
            else f'<span class="headline">{title}</span>'
        )
        distance = item.get("distance")
        dist_text = f"{distance:.3f}" if isinstance(distance, (int, float)) else "—"
        novel = '<span class="novel">NOVEL</span>' if item.get("is_novel") else ""
        rows.append(
            f'<div class="feed-row">'
            f'<span class="dot" style="background:{colour}"></span>'
            f'<span class="chip" style="color:{colour}">{label}</span>'
            f"{headline}{novel}"
            f'<span class="meta">{dist_text} · {html.escape(str(item.get("source", "")))} · {clock(item.get("ingested_at"))}</span>'
            f"</div>"
        )

    if not rows:
        st.markdown('<p class="muted">No new headlines yet — waiting for the next poll…</p>', unsafe_allow_html=True)
        return
    st.markdown("".join(rows), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.subheader("Controls")
    st.caption(f"API: {API_URL}")
    only_novel = st.toggle("Only novel headlines", value=False)
    st.divider()
    health = get("/health")
    if "_error" in health:
        st.error("API unreachable")
        st.code(health["_error"], language="text")
        st.caption("Start it with:  uvicorn src.serve:app --port 8010")
    else:
        st.success("API online")
        st.caption(f"Trained on {health['documents_trained_on']:,} headlines")
        st.caption(f"k = {health['topics']} topics")
        st.caption(f"Live poller: {'running' if health['poller_running'] else 'stopped'}")
    st.divider()
    if st.button("Fetch now", use_container_width=True):
        with st.spinner("Polling live source…"):
            result = get("/poll")
        if "_error" in result:
            st.warning(result["_error"])
        else:
            st.toast(f"{result.get('inserted', 0)} new headlines")
        st.rerun()


# --------------------------------------------------------------------------- #
# Live panels (re-run every few seconds without touching the sidebar)
# --------------------------------------------------------------------------- #
@st.fragment(run_every=REFRESH_SECONDS)
def live_panels() -> None:
    stats = get("/stats")
    recent = get("/recent", limit=FEED_LIMIT)

    if "_error" in stats:
        st.error(f"Cannot reach the API at {API_URL}")
        st.code(stats["_error"], language="text")
        return

    total = stats.get("total", 0)
    novel_rate = stats.get("novel_rate", 0.0) or 0.0
    avg_distance = stats.get("avg_distance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Headlines scored", f"{total:,}")
    col2.metric("Topics", stats.get("k", "—"))
    col3.metric("Flagged novel", f"{novel_rate:.1%}")
    col4.metric("Avg. distance", f"{avg_distance:.3f}" if avg_distance is not None else "—")

    left, right = st.columns([1.9, 1], gap="large")

    with left:
        st.markdown("#### Live feed")
        st.caption(
            f"Assigned by the frozen model · {stats.get('last_ingest') and clock(stats['last_ingest'])} · "
            f"refreshing every {REFRESH_SECONDS:.0f}s"
        )
        render_feed(recent.get("items", []), only_novel)

    with right:
        st.markdown("#### Topic distribution")
        by_topic = [t for t in stats.get("by_topic", []) if t["topic_id"] is not None]
        if by_topic:
            frame = pd.DataFrame(by_topic).sort_values("count", ascending=False).head(12)
            frame["label"] = frame["label"].str.slice(0, 34)
            try:
                st.bar_chart(frame, x="count", y="label", horizontal=True, height=320)
            except TypeError:  # older Streamlit without horizontal bars
                st.bar_chart(frame.set_index("label")["count"], height=320)
        else:
            st.caption("No data yet.")

        st.markdown("#### Discovered topics")
        topics = get("/topics")
        if "_error" not in topics:
            for topic in sorted(topics["topics"], key=lambda t: -t["size"])[:8]:
                colour = topic_color(topic["topic_id"])
                st.markdown(
                    f'<span class="dot" style="display:inline-block;background:{colour};margin-right:6px"></span>'
                    f'<b>{html.escape(topic["label"])}</b> '
                    f'<span class="muted">· {topic["size"]:,} docs</span>',
                    unsafe_allow_html=True,
                )


st.markdown(CSS, unsafe_allow_html=True)
st.title("📰 News Topic Stream")
st.markdown(
    '<p class="muted">Headlines are grouped by a K-Means model trained once on a frozen snapshot. '
    "The model never re-trains — it only assigns new articles to the topics it already knows.</p>",
    unsafe_allow_html=True,
)
live_panels()
