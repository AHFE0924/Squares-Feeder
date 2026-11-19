"""Submit latest_public.csv to Kaggle when it is a real improvement.

Run by the GitHub workflow after the scraper has pushed. It:
  1. scores latest_public.csv with the official competition metric (kaggle_metric.py, an unchanged
     copy taken from Sergio Fernandez Rubio (GeiserX), https://github.com/GeiserX/squares-in-squares,
     Apache-2.0), reading the file through pandas the way Kaggle does;
  2. refuses to submit if the metric rejects even one puzzle;
  3. submits only if the official score beats the last automatic submission by MIN_GAIN, and at
     most MAX_PER_DAY times per UTC day (Kaggle allows about 10 a day, this leaves room for yours);
  4. records every submission in submit_state.json, which the workflow commits.

  python auto_submit.py              # score, and submit if the rules above allow it
  python auto_submit.py --dry-run    # score and decide, but never submit
  python auto_submit.py --dry-run --only 1-60     # quick local test on a few puzzles
"""
import os
import sys
import json
import math
import time
import argparse
import subprocess
import urllib.request
import multiprocessing as mp
from datetime import datetime, timezone

import pandas as pd

COMPETITION = "squares-in-squares"
CSV = "latest_public.csv"
STATE = "submit_state.json"
MIN_GAIN = 1e-7
MAX_PER_DAY = 5
METRIC_URL = ("https://raw.githubusercontent.com/GeiserX/squares-in-squares/"
              "f822b34456b29addec9b59478aac4eb2306f7566/kaggle_metric.py")
METRIC_DIR = "metric"

sys.stdout.reconfigure(line_buffering=True)
km = None


def load_metric():
    global km
    os.makedirs(METRIC_DIR, exist_ok=True)
    path = os.path.join(METRIC_DIR, "kaggle_metric.py")
    if not os.path.exists(path):
        urllib.request.urlretrieve(METRIC_URL, path)
    sys.path.insert(0, METRIC_DIR)
    import kaggle_metric
    km = kaggle_metric


def _eval(item):
    if km is None:
        load_metric()          # spawned workers (Windows) import it themselves
    n, rows = item
    try:
        return km.evaluate_single_puzzle(n, rows)
    except Exception as e:
        return (n, km.infeasible_score(n), None, f"error: {e}")


def official_score(path, only=None):
    df = pd.read_csv(path).drop(columns=["id"])          # default pandas floats, as Kaggle reads it
    pz = km._parse_submission_csv(df.to_csv(index=False))
    ns = range(1, 1001) if only is None else only
    items = [(n, pz[n]) for n in ns if n in pz]
    with mp.Pool(os.cpu_count() or 2) as pool:
        res = {r[0]: r for r in pool.imap_unordered(_eval, items, chunksize=4)}
    for n in ns:
        res.setdefault(n, (n, km.infeasible_score(n), None, "missing"))
    total = math.fsum(res[n][1] for n in ns)
    bad = [n for n in ns if res[n][3] != "ok"]
    return total, bad


def load_state():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"last_score": None, "submissions": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="e.g. 1-60, for a quick local test")
    a = ap.parse_args()
    only = None
    if a.only:
        lo, hi = (int(v) for v in a.only.split("-"))
        only = range(lo, hi + 1)

    load_metric()
    t = time.time()
    score, bad = official_score(CSV, only)
    print(f"official score of {CSV}: {score:.10f} | rejected puzzles: {bad} | {time.time() - t:.0f}s")
    if bad:
        print("NOT submitting: the metric rejects some puzzles.")
        return
    if only is not None:
        print("Partial check (--only): never submits.")
        return

    state = load_state()
    last = state.get("last_score")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n_today = sum(1 for s in state["submissions"] if s["time"].startswith(today))
    if last is not None and score > last - MIN_GAIN:
        print(f"NOT submitting: {score:.10f} does not beat the last auto-submission {last:.10f} by {MIN_GAIN}.")
        return
    if n_today >= MAX_PER_DAY:
        print(f"NOT submitting: already {n_today} auto-submissions today (limit {MAX_PER_DAY}).")
        return

    sha = os.environ.get("GITHUB_SHA", "local")[:7]
    msg = f"Squares-Feeder auto {score:.10f} ({sha})"
    cmd = ["kaggle", "competitions", "submit", "-c", COMPETITION, "-f", CSV, "-m", msg]
    if a.dry_run:
        print("DRY RUN, would run:", " ".join(cmd))
        return
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = (r.stdout + r.stderr).strip()
    print(out[-1000:])
    if r.returncode != 0 or "success" not in out.lower():
        print("Submission FAILED; state not changed.")
        sys.exit(1)
    state["last_score"] = score
    state["submissions"].append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "score": score, "commit": sha})
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)
    print(f"SUBMITTED {score:.10f}")


if __name__ == "__main__":
    main()
