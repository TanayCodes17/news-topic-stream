"""Terminal live stream -- the demo that cannot break.

    python -m scripts.stream_cli --mode live
    python -m scripts.stream_cli --mode replay --interval 0.5

Runs the model in-process (no API needed) and prints each new headline with the
topic the frozen model assigned. Useful for a screen recording and as a fallback
if the dashboard misbehaves on the presentation machine.
"""

from __future__ import annotations

import argparse
import sys
import time

from src import db
from src.config import LIVE_DB, POLL_INTERVAL_SECONDS, REPLAY_INTERVAL_SECONDS
from src.predict import ArtifactBundle
from src.sources.hn_live import HNLiveSource
from src.sources.replay import ReplaySource


def colour(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"


def print_row(row: dict) -> None:
    topic = (row.get("topic_label") or "unassigned")[:34]
    distance = row.get("distance")
    dist_text = f"{distance:.3f}" if isinstance(distance, (int, float)) else "  —  "
    badge = colour(" NOVEL ", 33) if row.get("is_novel") else "       "
    print(
        f"{badge} {colour(f'{topic:<34}', 36)} {dist_text}  {row['title'][:88]}",
        flush=True,
    )


def main() -> int:
    # Windows consoles default to cp1252, which cannot print the "·" separator.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Stream headlines through the frozen topic model.")
    parser.add_argument("--mode", choices=["live", "replay"], default="live")
    parser.add_argument("--interval", type=float, default=None, help="seconds between polls")
    parser.add_argument("--polls", type=int, default=0, help="stop after N polls (0 = forever)")
    parser.add_argument("--db", default=str(LIVE_DB))
    args = parser.parse_args()

    bundle = ArtifactBundle.load()
    db.init_db(args.db)
    print(
        f"model: k={bundle.metrics['k']} · trained on {bundle.metrics['n_documents']:,} headlines · "
        f"novelty cutoff {bundle.threshold:.3f}"
    )
    print(f"mode: {args.mode}   (Ctrl+C to stop)\n")

    if args.mode == "live":
        source = HNLiveSource(
            bundle,
            db_path=args.db,
            interval=args.interval or POLL_INTERVAL_SECONDS,
        )
    else:
        source = ReplaySource(
            bundle,
            db_path=args.db,
            interval=args.interval or REPLAY_INTERVAL_SECONDS,
        )

    polls = 0
    try:
        while True:
            result = source.poll_once()
            for row in result.get("new_rows", []):
                print_row(row)
            if result.get("done"):
                print("replay finished")
                break
            polls += 1
            if args.polls and polls >= args.polls:
                break
            time.sleep(source.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    except Exception as exc:  # keep the demo alive on a transient network error
        print(f"source error: {exc}", file=sys.stderr)
        return 1

    total = db.stats(args.db)["total"]
    print(f"\nstored {total:,} scored headlines in {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
