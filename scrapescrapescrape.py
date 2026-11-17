import os
import glob
import shutil
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

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
    i_idx, j_idx = np.where((dist2 < 2.0 + 1e-5) & (np.triu(np.ones((n, n), dtype=bool), 1)))
    if len(i_idx) == 0:
        return True
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    for i, j in zip(i_idx, j_idx):
        rx, ry = x[j] - x[i], y[j] - y[i]
        for ux, uy in ((c[i], s[i]), (-s[i], c[i]), (c[j], s[j]), (-s[j], c[j])):
            proj_dist = abs(rx * ux + ry * uy)
            ri = 0.5 * (abs(ux * c[i] + uy * s[i]) + abs(-ux * s[i] + uy * c[i]))
            rj = 0.5 * (abs(ux * c[j] + uy * s[j]) + abs(-ux * s[j] + uy * c[j]))
            if proj_dist >= ri + rj - tol:
                break
        else:
            return False
    return True

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
    api = KaggleApi()
    api.authenticate()
    os.makedirs("downloads", exist_ok=True)

    seen_refs = set()
    for sort_mode in ["hotness", "dateRun", "voteCount"]:
        try:
            kernels = api.kernels_list(competition="squares-in-squares", sort_by=sort_mode, page_size=15)
            for k in kernels:
                if k.ref not in seen_refs:
                    seen_refs.add(k.ref)
                    out_dir = os.path.join("downloads", k.ref.replace("/", "_"))
                    os.makedirs(out_dir, exist_ok=True)
                    print(f"[{sort_mode}] Downloading: {k.ref}")
                    try:
                        api.kernels_output(k.ref, path=out_dir)
                    except Exception as e:
                        print(f"Skipping {k.ref}: {e}")
        except Exception as e:
            print(f"Sort {sort_mode} error: {e}")

    csv_files = glob.glob("downloads/**/*.csv", recursive=True)
    if os.path.exists("latest_public.csv"):
        csv_files.append("latest_public.csv")

    if not csv_files:
        print("No CSVs found.")
        return

    best_packings = {}
    best_scores = {}
    template_cols = None

    for path in csv_files:
        if os.path.getsize(path) < 100000:
            continue
        try:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            x_col = cols.get("x")
            y_col = cols.get("y")
            a_col = cols.get("deg") or cols.get("angle") or cols.get("theta") or cols.get("r")
            if not (x_col and y_col and a_col):
                continue

            p_key = extract_puzzle_key(df)
            if p_key is None:
                continue

            if template_cols is None:
                template_cols = list(df.columns)

            df["_p_key"] = p_key
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
        except Exception as e:
            print(f"Error parsing {path}: {e}")

    if len(best_packings) >= 900:
        merged_df = pd.concat([best_packings[k] for k in sorted(best_packings.keys())], ignore_index=True)
        merged_df.to_csv("latest_public.csv", index=False)
        print(f"Ensembled {len(best_packings)} puzzles! Total L score: {sum(best_scores.values()):.5f}")
    else:
        best_csv = max(csv_files, key=os.path.getsize)
        if best_csv != "latest_public.csv":
            shutil.copy(best_csv, "latest_public.csv")
        print(f"Fallback: copied {best_csv}")

if __name__ == "__main__":
    run_scraper()
