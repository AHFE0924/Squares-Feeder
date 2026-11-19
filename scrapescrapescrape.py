"""Squares-Feeder: pull every public (and your own) Kaggle output for the Squares in Squares
competition and keep the best valid packing for each n in latest_public.csv.

  python scrapescrapescrape.py                               # download from Kaggle, then merge
  python scrapescrapescrape.py --merge-only                  # re-merge what is already on disk
  python scrapescrapescrape.py --merge-only --extra a.csv    # ... plus extra files (repeatable)

latest_public.csv is always safe to submit as it is:
  * columns id,n,square_id,x,y,angle; id runs 0..500499; angles in radians
  * all 1000 puzzles present (a plain grid for any n no source covers)
  * every pair of squares is at least CLEARANCE apart.  Packings whose squares touch exactly or
    overlap by round-off are scaled up by the smallest factor that opens that gap, so the Kaggle
    grader cannot reject a puzzle for it (a rejected puzzle is scored as a grid).
"""
import os
import sys
import glob
import math
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

COMPETITION = "squares-in-squares"
OUT = "latest_public.csv"
N_MAX = 1000
CLEARANCE = 1e-10            # minimum gap between any two squares in the output
DOWNLOAD_WORKERS = 4         # notebook outputs downloaded in parallel
DOWNLOAD_TIMEOUT = 90        # seconds per notebook output
SORTS = ("scoreAscending", "hotness", "dateRun", "voteCount")
ANGLE_COLS = ("angle", "deg", "degrees", "angle_deg", "theta", "rotation", "rot", "r")
DEGREE_COLS = ("deg", "degrees", "angle_deg")


# ------------------------------------------------------------------------------ Kaggle download

def download_outputs():
    from kaggle.api.kaggle_api_extended import KaggleApi    # only needed when downloading

    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    refs, seen = [], set()

    def add(kernels, tag):
        for k in kernels:
            if k.ref not in seen:
                seen.add(k.ref)
                refs.append((k.ref, tag))

    try:
        add(api.kernels_list(mine=True, competition=COMPETITION, page_size=25), "mine")
    except Exception as e:
        print(f"Listing your own notebooks failed: {e}")
    for sort_mode in SORTS:
        try:
            add(api.kernels_list(competition=COMPETITION, sort_by=sort_mode, page_size=25), sort_mode)
        except Exception as e:
            print(f"Listing notebooks by {sort_mode} failed: {e}")
    print(f"{len(refs)} notebooks to download ({DOWNLOAD_WORKERS} at a time)...")

    def fetch(item):
        ref, tag = item
        out_dir = os.path.join("downloads", ref.replace("/", "_"))
        os.makedirs(out_dir, exist_ok=True)
        try:
            r = subprocess.run(["kaggle", "kernels", "output", ref, "-p", out_dir, "-q"],
                               timeout=DOWNLOAD_TIMEOUT, check=False, capture_output=True)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        print(f"  {'ok     ' if ok else 'failed '} {ref} ({tag})")
        return ok

    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        n_ok = sum(pool.map(fetch, refs))
    print(f"Downloaded {n_ok}/{len(refs)} notebook outputs.")


# ------------------------------------------------------------------------------ geometry

def half_extent(a):
    return 0.5 * (np.abs(np.cos(a)) + np.abs(np.sin(a)))


def side_length(x, y, a):
    h = half_extent(a)
    return float(max((x + h).max() - (x - h).min(), (y + h).max() - (y - h).min()))


def _pair_gaps(x, y, a):
    """Separating-axis gap of every pair whose centres are close enough to touch."""
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    I, J = np.nonzero(np.triu(dx * dx + dy * dy < 2.01, 1))   # centres >= sqrt(2) apart never touch
    if len(I) == 0:
        return None, None
    c, s = np.cos(a), np.sin(a)
    rx, ry = x[J] - x[I], y[J] - y[I]
    ci, si, cj, sj = c[I], s[I], c[J], s[J]
    ext = 0.5 + 0.5 * (np.abs(ci * cj + si * sj) + np.abs(si * cj - ci * sj))
    proj = np.max(np.abs(np.stack([rx * ci + ry * si, -rx * si + ry * ci,
                                   rx * cj + ry * sj, -rx * sj + ry * cj])), axis=0)
    return proj, ext


def make_safe(x, y, a):
    """Scale the centres so that every gap is >= CLEARANCE. Returns (x, y, L) or None if impossible."""
    if len(x) == 1:
        return x, y, side_length(x, y, a)
    proj, ext = _pair_gaps(x, y, a)
    if proj is None:
        return x, y, side_length(x, y, a)
    if proj.min() < 1e-9:                       # two centres on top of each other
        return None
    need = float(((ext + CLEARANCE) / proj).max())
    if need > 1.0:
        f = need * (1.0 + 4e-16)
        cx, cy = x.mean(), y.mean()
        x = cx + (x - cx) * f
        y = cy + (y - cy) * f
        proj, ext = _pair_gaps(x, y, a)
        if proj is not None and (proj - ext).min() < 0.9 * CLEARANCE:
            return None
    return x, y, side_length(x, y, a)


# ------------------------------------------------------------------------------ merge

def puzzle_key(df, cols):
    if "n" in cols:
        return pd.to_numeric(df[cols["n"]], errors="coerce")
    if "puzzle" in cols:
        return pd.to_numeric(df[cols["puzzle"]], errors="coerce")
    if "id" in cols:
        sample = str(df[cols["id"]].iloc[0])
        if "_" in sample or "-" in sample:
            sep = "_" if "_" in sample else "-"
            return pd.to_numeric(df[cols["id"]].astype(str).str.split(sep).str[0], errors="coerce")
    return None


def merge(paths):
    best = {}            # n -> (L, x, y, angle_in_radians)

    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skipped {path}: {type(e).__name__}: {e}")
            continue
        cols = {str(c).lower().strip(): c for c in df.columns}
        a_key = next((k for k in ANGLE_COLS if k in cols), None)
        key = puzzle_key(df, cols)
        if "x" not in cols or "y" not in cols or a_key is None or key is None:
            continue
        df = df.assign(_n=key).dropna(subset=["_n"])
        ang = pd.to_numeric(df[cols[a_key]], errors="coerce")
        if ang.isna().all():
            continue
        amax = float(np.nanmax(np.abs(ang.to_numpy(dtype=np.float64))))
        # Unit is decided per file. Values up to 2*pi can be radians or small degree tilts: try both.
        if a_key in DEGREE_COLS or amax > 2 * math.pi + 1e-6:
            units = ("deg",)
        else:
            units = ("rad", "deg")

        upgrades = 0
        for n_val, g in df.groupby("_n", sort=False):
            n = int(n_val)
            if n < 1 or n > N_MAX or len(g) != n:
                continue
            if "square_id" in cols:
                g = g.sort_values(cols["square_id"])
            x = pd.to_numeric(g[cols["x"]], errors="coerce").to_numpy(dtype=np.float64)
            y = pd.to_numeric(g[cols["y"]], errors="coerce").to_numpy(dtype=np.float64)
            raw = pd.to_numeric(g[cols[a_key]], errors="coerce").to_numpy(dtype=np.float64)
            if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(raw).all()):
                continue
            for unit in units:
                a = np.deg2rad(raw) if unit == "deg" else raw
                cur = best.get(n, (math.inf,))[0]
                if side_length(x, y, a) >= cur - 1e-12:      # making it safe can only make it larger
                    continue
                safe = make_safe(x, y, a)
                if safe is not None and safe[2] < cur - 1e-12:
                    best[n] = (safe[2], safe[0], safe[1], a.copy())
                    upgrades += 1
        print(f"Parsed {path} -> {upgrades} puzzle upgrades")
    return best


def grid(n):
    k = math.ceil(math.sqrt(n))
    i = np.arange(n)
    return float(k), 0.5 + (i % k), 0.5 + (i // k), np.zeros(n)


def write_output(best):
    missing = [n for n in range(1, N_MAX + 1) if n not in best]
    for n in missing:
        best[n] = grid(n)
    ns, sids, xs, ys, angs = [], [], [], [], []
    for n in range(1, N_MAX + 1):
        _, x, y, a = best[n]
        ns.append(np.full(n, n)); sids.append(np.arange(n))
        xs.append(x); ys.append(y); angs.append(a)
    x = np.concatenate(xs); y = np.concatenate(ys); a = np.concatenate(angs)
    out = pd.DataFrame({
        "id": np.arange(len(x)),
        "n": np.concatenate(ns),
        "square_id": np.concatenate(sids),
        "x": [repr(float(v)) for v in x],          # shortest exact round-trip of each double
        "y": [repr(float(v)) for v in y],
        "angle": [repr(float(v)) for v in a],
    })
    tmp = OUT + ".tmp"
    out.to_csv(tmp, index=False)
    os.replace(tmp, OUT)
    score = sum(best[n][0] / math.sqrt(n) - 1.0 for n in range(1, N_MAX + 1))
    note = f" | {len(missing)} puzzles filled with a grid" if missing else ""
    print(f"SUCCESS: wrote {OUT} | Kaggle score {score:.6f}{note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true", help="skip the Kaggle download")
    ap.add_argument("--extra", action="append", default=[], help="additional CSV to merge (repeatable)")
    args = ap.parse_args()

    if not args.merge_only:
        download_outputs()

    # current file first: it already holds the best known packings, so most others are skipped fast
    paths = [OUT] + args.extra + sorted(glob.glob("downloads/**/*.csv", recursive=True))
    print(f"Merging {len(paths)} CSV files...")
    best = merge(paths)
    if not best:
        print(f"Nothing could be parsed; leaving {OUT} untouched.")
        return
    write_output(best)


if __name__ == "__main__":
    main()
