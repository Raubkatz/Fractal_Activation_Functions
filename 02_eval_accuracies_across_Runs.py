#!/usr/bin/env python3
# ------------------------------------------------------------
#  make_accuracy_reports.py  —  per-dataset txt summaries
# ------------------------------------------------------------
"""
From a tree of result-JSONs produce accuracy summaries with
mean, standard deviation, max and min over the individual runs.

Folder expectations
-------------------
<SOURCE_DIR>/<dataset>/*.json          (one JSON per config)

Output produced
---------------
<OUTPUT_DIR>/
    all/<dataset>/accuracy_summary_all.txt
    nonfractional/<dataset>/accuracy_summary_nonfractional.txt
"""
from __future__ import annotations
from pathlib import Path
from typing  import List
import json, numpy as np, pandas as pd

# ------------------------------------------------------------------
# 1) read every JSON and build a tidy DataFrame
# ------------------------------------------------------------------
def extract_data(root: Path, verbose: bool = False) -> pd.DataFrame:
    """Return one row per (dataset, optimiser, activation, vderiv)."""
    rows: List[dict] = []

    for file in root.rglob("*.json"):
        with open(file, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, list):        # tolerate list-wrapper format
            data = data[0]

        activation = data.get("activation", "relu")
        vderiv_entries = data.get("vderivs", data.get("results", []))

        for v in vderiv_entries:
            run_accs = [float(r["accuracy"]) for r in v.get("results", [])]
            # fall back to stored average if run list missing
            if run_accs:
                acc_mean = float(np.mean(run_accs))
                acc_std  = float(np.std (run_accs, ddof=0))  # population σ
                acc_max  = float(np.max (run_accs))
                acc_min  = float(np.min (run_accs))
                n_runs   = len(run_accs)
            else:
                acc_mean = float(v.get("avg accuracy", np.nan))
                acc_std  = np.nan
                acc_max  = acc_mean
                acc_min  = acc_mean
                n_runs   = 0

            rows.append(dict(
                dataset   = data["dataset"],
                optimizer = data["optimizer"],
                activation= activation,
                vderiv    = float(v["vderiv"]),
                runs      = n_runs,
                mean_acc  = acc_mean,
                std_acc   = acc_std,
                max_acc   = acc_max,
                min_acc   = acc_min,
            ))
        if verbose:
            print("loaded", file)

    if not rows:
        raise RuntimeError(f"No JSON files found under {root}")
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 2) helper to write ONE summary file
# ------------------------------------------------------------------
def _write_summary(df_cfg: pd.DataFrame, out_file: Path, tag: str) -> None:
    """df_cfg must already correspond to exactly one dataset."""
    df_sorted = df_cfg.sort_values("mean_acc", ascending=False)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        hdr  = (f"Accuracy summary ({tag}) — dataset: {df_cfg['dataset'].iloc[0]}\n"
                + "="*86 + "\n\n"
                + f"{'optimizer':12s} | {'activation':18s} | {'vderiv':7s} | "
                + f"{'runs':4s} | {'mean':7s} | {'std':7s} | "
                + f"{'max':7s} | {'min':7s}\n"
                + "-"*86 + "\n")
        f.write(hdr)

        for r in df_sorted.itertuples(index=False):
            f.write(f"{r.optimizer:12s} | {r.activation:18s} | "
                    f"{r.vderiv:7.2f} | {r.runs:4d} | "
                    f"{r.mean_acc:7.4f} | {r.std_acc:7.4f} | "
                    f"{r.max_acc:7.4f} | {r.min_acc:7.4f}\n")
    print("[REPORT]", out_file)

# ------------------------------------------------------------------
# 3) create summaries for all / fractional / non-fractional
# ------------------------------------------------------------------
def make_reports(df: pd.DataFrame, out_root: Path) -> None:
    frac_mask = df["optimizer"].str.startswith("F", na=False)

    for ds, df_ds in df.groupby("dataset"):
        _write_summary(df_ds,
                       out_root / "all" / ds / "accuracy_summary_all.txt",
                       "all")

        _write_summary(df_ds[~frac_mask],
                       out_root / "nonfractional" / ds / "accuracy_summary_nonfractional.txt",
                       "nonfractional")

# ------------------------------------------------------------------
# 4) main – edit SOURCE_DIR / OUTPUT_DIR here
# ------------------------------------------------------------------
def main() -> None:
    SOURCE_DIR = Path("results_cluster_40runs")
    OUTPUT_DIR = Path("reports_cluster_40runs")               # <— change as needed

    df_all = extract_data(SOURCE_DIR, verbose=False)
    make_reports(df_all, OUTPUT_DIR)
    print("\nDone ✔︎  Reports in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
