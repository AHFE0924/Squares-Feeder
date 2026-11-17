import os
import sys
import glob
import shutil
import subprocess
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Force live real-time logging in GitHub Actions
sys.stdout.reconfigure(line_buffering=True)

def calc_bounding_box(x, y, deg):
    rad = np.deg2rad(deg)
    r = 0.5 * (np.abs(np.cos(rad)) + np.abs(np.sin(rad)))
    lx = np.max(x + r) - np.min(x - r)
    ly = np.max(y + r) - np.min(y - r)
    return float(max(lx, ly))

def is_valid_packing(x, y, deg, tol=1e-7):
    n = len(x)
    if n <= 1:
        return True
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx * dx + dy * dy
    np.fill_diagonal(dist2, 999.0)
    if np.min(dist2) < 1.0 - tol:
        return False

    mask = np.triu(dist2 < 2.0 + 1e-5, k=1)
    i_idx, j_idx = np.where(mask)
    if len(i_idx) == 0:
        return True

    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    rx, ry = x[j_idx] - x[i_idx], y[j_idx] - y[i_idx]
    ci, si = c[i_idx], s[i_idx]
    cj, sj = c[j_idx], s[j_idx]

    thresh = 0.5 + 0.5 * (np.abs(ci * cj + si * sj) + np.abs(ci * sj - si * cj)) - tol
    p1 = np.abs(rx * ci + ry * si)
    p2 = np.abs(-rx * si + ry * ci)
    p3 = np.abs(rx * cj + ry * sj)
    p4 = np.abs(-rx * sj + ry * cj)

    max_sep = np.maximum(np.maximum(p1, p2), np.maximum(p3, p4))
    return bool(np.all(max_sep >= thresh))

def extract_puzzle_key(df):
    cols = {c.lower(): c for c in df.columns}
    if "n" in cols:
        return df[cols["n"]].astype(int)
    if "puzzle" in cols:
        return df[cols["puzzle"]].astype(int)
    if "id" in cols:
        sample = str(df[cols["id"]].iloc[0])
        if "_" in sample or "-" in sample:
            sep = "_" if "_" in sample else "-"
            return df[cols["id"]].astype(str).str.split(sep).str[0].astype(int)
    return None

def run_scraper():
    print("Authenticating with Kaggle API...", flush=True)
    api = KaggleApi()
    api.authenticate()
    os.makedirs("downloads", exist_ok=True)

    seen_refs = set()
    for sort_mode in ["hotness", "dateRun", "voteCount"]:
        print(f"Checking top notebooks by {sort_mode}...", flush=True)
        try:
            kernels = api.kernels_list(competition="squares-in-squares", sort_by=sort_mode, page_size=6)
            for k in kernels:
                if k.ref not in seen_refs:
                    seen_refs.add(k.ref)
                    out_dir = os.path.join("downloads", k.ref.replace("/", "_"))
                    os.makedirs(out_dir, exist_ok=True)
                    print(f"  -> Downloading {k.ref} (25s timeout)...", flush=True)
                    try:
                        subprocess.run(
                            ["kaggle", "kernels", "output", k.ref, "-p", out_dir, "-q"],
                            timeout=25,
                            check=False
                        )
                    except subprocess.TimeoutExpired:
                        print(f"  -> Timeout on {k.ref}, skipping!", flush=True)
        except Exception as e:
            print(f"Sort {sort_mode} error: {e}", flush=True)

    csv_files = glob.glob("downloads/**/*.csv", recursive=True)
    if os.path.exists("latest_public.csv"):
        csv_files.append("latest_public.csv")

    print(f"Found {len(csv_files)} CSV files. Starting puzzle-level ensemble...", flush=True)
    if not csv_files:
        return

    best_packings = {}
    best_scores = {}

    for path in csv_files:
        if os.path.getsize(path) < 100000:
            continue
        try:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            x_col, y_col = cols.get("x"), cols.get("y")
            a_col = cols.get("deg") or cols.get("angle") or cols.get("theta") or cols.get("r")
            if not (x_col and y_col and a_col):
                continue

            p_key = extract_puzzle_key(df)
            if p_key is None:
                continue

            df["_p_key"] = p_key
            improvements = 0
            for n_val, group in df.groupby("_p_key", sort=False):
                n_int = int(n_val)
                if len(group) != n_int:
                    continue
                x = group[x_col].to_numpy(dtype=np.float64)
                y = group[y_col].to_numpy(dtype=np.float64)
                deg = group[a_col].to_numpy(dtype=np.float64)
                L = calc_bounding_box(x, y, deg)
                if L < best_scores.get(n_int, float("inf")) - 1e-9:
                    if is_valid_packing(x, y, deg):
                        best_scores[n_int] = L
                        best_packings[n_int] = group.drop(columns=["_p_key"]).copy()
                        improvements += 1
            print(f"Parsed {path} -> {improvements} puzzle upgrades", flush=True)
        except Exception as e:
            print(f"Error parsing {path}: {e}", flush=True)

    if len(best_packings) >= 900:
        merged_df = pd.concat([best_packings[k] for k in sorted(best_packings.keys())], ignore_index=True)
        merged_df.to_csv("latest_public.csv", index=False)
        print(f"SUCCESS: Ensembled {len(best_packings)} puzzles! Total L: {sum(best_scores.values()):.5f}", flush=True)
    else:
        best_csv = max(csv_files, key=os.path.getsize)
        if best_csv != "latest_public.csv":
            shutil.copy(best_csv, "latest_public.csv")
        print(f"Fallback: copied {best_csv}", flush=True)

if __name__ == "__main__":
    run_scraper()
