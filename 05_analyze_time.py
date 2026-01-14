#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time analysis + plotting for timing logs produced by 01_run_nn_classifications.py (timing add-on version).

What it does
------------
- Scans: <results_dir>/time_analysis/<dataset>/*_times.json
- Builds a per-run table (DataFrame) with:
    dataset, optimizer, activation, vderiv, run, t_* timings
- Produces per-dataset plots (PNG + EPS) and global plots:
    1) Heatmap: avg t_fit by (optimizer x activation)
    2) Heatmap: avg t_total_run by (optimizer x activation)
    3) Bar: activation means (avg over optimizers) for t_fit + t_total_run
    4) Bar: optimizer means (avg over activations) for t_fit + t_total_run
    5) Binary analysis: fractal vs non-fractal (boxplot + dataset-level bars)
- Saves summary CSVs.

NEW OUTPUTS (requested)
-----------------------
- Per dataset: activation-level mean and std for each timing key (CSV + TXT with "mean +- std")
- Global: activation-level mean and std for each timing key (CSV + TXT with "mean +- std")

Notes
-----
- "Non-fractal" is defined as: relu, sigmoid, tanh.
- "Fractal" is everything else.
- Uses the same Tailwind-inspired palette from 04_plot_activations.py.
"""

# =============================================================================
# SCRIPT CONFIGURATION (no argparse)
# =============================================================================
RESULTS_DIR = "results_jan11_40runs_seedval238974"
# =============================================================================

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Tailwind colour set (from 04_plot_activations.py) ------------------------
PALETTE = [
    "#a2faa3",  # Light green
    "#92c9b1",  # Cambridge blue
    "#4f759b",  # UCLA blue
    "#5d5179",  # Ultra-violet
    "#571f4e",  # Palatinate
]

# Non-fractal vs fractal definition (binary analysis)
NON_FRACTAL = {"relu", "sigmoid", "tanh"}

# Timing keys expected in per-run logs
TIME_KEYS = [
    "t_total_run",
    "t_split_scale",
    "t_compile",
    "t_fit",
    "t_evaluate",
    "t_predict",
    "t_metrics",
    "t_predict_and_metrics",
]

# -----------------------------------------------------------------------------
def _safe_read_json(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)

def _parse_filename(fname: str):
    """
    Expected: <optimizer>_<activation>_<dataset>_times.json

    Dataset names can contain hyphens; activation names can contain underscores.
    Parse optimizer as first token, dataset as last token before "_times.json",
    activation as the middle chunk.
    """
    if not fname.endswith("_times.json"):
        return None

    core = fname[:-10]  # strip "_times.json"
    parts = core.split("_")
    if len(parts) < 3:
        return None

    optimizer = parts[0]
    dataset = parts[-1]
    activation = "_".join(parts[1:-1])
    return optimizer, activation, dataset

def _tailwind_cmap():
    return LinearSegmentedColormap.from_list("tailwind_like", PALETTE)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _save_fig(fig, out_base: Path, dpi=300):
    fig.tight_layout()
    fig.savefig(str(out_base.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".eps")), bbox_inches="tight")
    plt.close(fig)

def _pretty_label(s: str) -> str:
    return s.replace("_", "\n")

def _strip_dataset_suffix_from_activation(activation: str, dataset: str) -> str:
    # If activation accidentally includes the dataset suffix (common if dataset names contain underscores),
    # remove a trailing "_<dataset>".
    suf = "_" + dataset
    if activation.endswith(suf):
        activation = activation[: -len(suf)]
    return activation

# -----------------------------------------------------------------------------
def load_timings(results_dir: Path) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per run and timing fields.
    """
    time_root = results_dir / "time_analysis"
    if not time_root.exists():
        raise FileNotFoundError(f"Missing folder: {time_root}")

    rows = []
    dataset_dirs = sorted([p for p in time_root.iterdir() if p.is_dir()])
    for dataset_dir in dataset_dirs:
        dataset_from_folder = dataset_dir.name

        for fp in sorted(dataset_dir.glob("*_times.json")):
            parsed = _parse_filename(fp.name)
            if parsed is None:
                continue

            optimizer, activation, dataset_from_name = parsed
            jd = _safe_read_json(fp)

            dataset = jd.get("dataset", dataset_from_folder)

            # Ensure activation does not carry dataset tokens (fixes x-tick label pollution)
            activation = _strip_dataset_suffix_from_activation(activation, dataset_from_folder)
            activation = _strip_dataset_suffix_from_activation(activation, dataset_from_name)

            timings = jd.get("timings", [])

            for block in timings:
                vderiv = block.get("vderiv", None)
                for run_entry in block.get("runs", []):
                    row = {
                        "dataset": dataset,
                        "optimizer": optimizer,
                        "activation": activation,
                        "vderiv": vderiv,
                        "run": run_entry.get("run #", None),
                        "source_file": str(fp),
                    }
                    for k in TIME_KEYS:
                        row[k] = run_entry.get(k, np.nan)
                    rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No timing data found in: {time_root}")
    return df

# -----------------------------------------------------------------------------
def per_dataset_plots(df: pd.DataFrame, out_root: Path):
    """
    Creates plots per dataset.
    """
    cmap = _tailwind_cmap()

    for dataset in sorted(df["dataset"].unique()):
        dfd = df[df["dataset"] == dataset].copy()

        dataset_out = out_root / "time_analysis_plots" / dataset
        _ensure_dir(dataset_out)

        print(f"[time-analysis] dataset='{dataset}' -> {dataset_out}")

        # Save per-run table
        dfd.to_csv(dataset_out / f"{dataset}_timings_per_run.csv", index=False)

        # ---------------------------------------------------------------------
        # NEW: Per-activation mean/std over ALL runs (includes all optimizers/vderivs)
        #      Separate tables for t_fit and t_predict (CSV + TXT)
        # ---------------------------------------------------------------------
        act_fit_stats = (
            dfd.groupby("activation", as_index=False)[["t_fit"]]
            .agg(["mean", "std"])
        )
        act_fit_stats.columns = [f"{k}_{stat}" for (k, stat) in act_fit_stats.columns]
        act_fit_stats = act_fit_stats.reset_index()
        if "t_fit_mean" in act_fit_stats.columns:
            act_fit_stats = act_fit_stats.sort_values("t_fit_mean", ascending=False)
        act_fit_stats.to_csv(dataset_out / f"{dataset}_activation_t_fit_mean_std.csv", index=False)

        txt_lines = []
        txt_lines.append(f"Dataset: {dataset}")
        txt_lines.append("Activation-level timing summary for t_fit (mean +- std over all runs; includes all optimizers/vderivs)")
        txt_lines.append("")
        for _, row in act_fit_stats.iterrows():
            a = row["activation"]
            m = row.get("t_fit_mean", np.nan)
            s = row.get("t_fit_std", np.nan)
            txt_lines.append(f"- {a}: t_fit: {m:.6f} +- {s:.6f}")
        (dataset_out / f"{dataset}_activation_t_fit_mean_std.txt").write_text("\n".join(txt_lines), encoding="utf-8")

        act_pred_stats = (
            dfd.groupby("activation", as_index=False)[["t_predict"]]
            .agg(["mean", "std"])
        )
        act_pred_stats.columns = [f"{k}_{stat}" for (k, stat) in act_pred_stats.columns]
        act_pred_stats = act_pred_stats.reset_index()
        if "t_predict_mean" in act_pred_stats.columns:
            act_pred_stats = act_pred_stats.sort_values("t_predict_mean", ascending=False)
        act_pred_stats.to_csv(dataset_out / f"{dataset}_activation_t_predict_mean_std.csv", index=False)

        txt_lines = []
        txt_lines.append(f"Dataset: {dataset}")
        txt_lines.append("Activation-level timing summary for t_predict (mean +- std over all runs; includes all optimizers/vderivs)")
        txt_lines.append("")
        for _, row in act_pred_stats.iterrows():
            a = row["activation"]
            m = row.get("t_predict_mean", np.nan)
            s = row.get("t_predict_std", np.nan)
            txt_lines.append(f"- {a}: t_predict: {m:.6f} +- {s:.6f}")
        (dataset_out / f"{dataset}_activation_t_predict_mean_std.txt").write_text("\n".join(txt_lines), encoding="utf-8")
        # ---------------------------------------------------------------------

        # Mean per (optimizer, activation)
        grp = dfd.groupby(["optimizer", "activation"], as_index=False)[TIME_KEYS].mean()
        optimizers = sorted(grp["optimizer"].unique())
        activations = sorted(grp["activation"].unique())

        # --------------------------
        # Heatmaps
        # --------------------------
        def make_heatmap(value_key: str, title: str, fname: str):
            pivot = grp.pivot(index="optimizer", columns="activation", values=value_key).reindex(
                index=optimizers, columns=activations
            )
            mat = pivot.to_numpy()

            fig, ax = plt.subplots(
                figsize=(max(10, 0.6 * len(activations)), max(4, 0.5 * len(optimizers))),
                dpi=200,
            )
            im = ax.imshow(mat, aspect="auto", cmap=cmap)

            ax.set_title(title)
            ax.set_xlabel("Activation")
            ax.set_ylabel("Optimizer")

            ax.set_xticks(np.arange(len(activations)))
            ax.set_yticks(np.arange(len(optimizers)))
            ax.set_xticklabels([_pretty_label(a) for a in activations], fontsize=8)
            ax.set_yticklabels(optimizers, fontsize=9)

            cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label("seconds")
            _save_fig(fig, dataset_out / fname)

        make_heatmap(
            "t_fit",
            f"{dataset}: avg training time t_fit (s) by optimizer × activation",
            f"{dataset}_heatmap_t_fit",
        )
        make_heatmap(
            "t_total_run",
            f"{dataset}: avg total run time t_total_run (s) by optimizer × activation",
            f"{dataset}_heatmap_t_total_run",
        )

        # --------------------------
        # Bar plots: activation means
        # --------------------------
        act_means = grp.groupby("activation", as_index=False)[["t_fit", "t_total_run"]].mean()
        act_means = act_means.sort_values("t_fit", ascending=False)

        x = np.arange(len(act_means))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(act_means))]

        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(act_means)), 6), dpi=200)
        ax.bar(x, act_means["t_fit"].to_numpy(), color=colors, label="t_fit (train)")
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_label(a) for a in act_means["activation"].tolist()], fontsize=8)
        ax.set_ylabel("seconds")
        ax.set_title(f"{dataset}: mean training time per activation (avg over optimizers)")
        #ax.legend()
        _save_fig(fig, dataset_out / f"{dataset}_bar_activation_t_fit")

        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(act_means)), 6), dpi=200)
        ax.bar(x, act_means["t_total_run"].to_numpy(), color=colors, label="t_total_run")
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_label(a) for a in act_means["activation"].tolist()], fontsize=8)
        ax.set_ylabel("seconds")
        ax.set_title(f"{dataset}: mean total run time per activation (avg over optimizers)")
        #ax.legend()
        _save_fig(fig, dataset_out / f"{dataset}_bar_activation_t_total_run")

        # ---------------------------------------------------------------------
        # NEW: Bar plots with error bars (mean +- std) for t_fit and t_predict
        # ---------------------------------------------------------------------
        if ("t_fit_mean" in act_fit_stats.columns) and ("t_fit_std" in act_fit_stats.columns):
            act_fit_plot = act_fit_stats.copy()
            x = np.arange(len(act_fit_plot))
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(act_fit_plot))]

            fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(act_fit_plot)), 6), dpi=200)
            ax.bar(
                x,
                act_fit_plot["t_fit_mean"].to_numpy(),
                yerr=act_fit_plot["t_fit_std"].to_numpy(),
                color=colors,
                capsize=3,
                label="t_fit (mean +- std)",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([_pretty_label(a) for a in act_fit_plot["activation"].tolist()], fontsize=8)
            ax.set_ylabel("seconds")
            ax.set_title(f"{dataset}: training time per activation (mean +- std; all runs)")
            #ax.legend()
            _save_fig(fig, dataset_out / f"{dataset}_bar_activation_t_fit_mean_std")

        if ("t_predict_mean" in act_pred_stats.columns) and ("t_predict_std" in act_pred_stats.columns):
            act_pred_plot = act_pred_stats.copy()
            x = np.arange(len(act_pred_plot))
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(act_pred_plot))]

            fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(act_pred_plot)), 6), dpi=200)
            ax.bar(
                x,
                act_pred_plot["t_predict_mean"].to_numpy(),
                yerr=act_pred_plot["t_predict_std"].to_numpy(),
                color=colors,
                capsize=3,
                label="t_predict (mean +- std)",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([_pretty_label(a) for a in act_pred_plot["activation"].tolist()], fontsize=8)
            ax.set_ylabel("seconds")
            ax.set_title(f"{dataset}: prediction time per activation (mean +- std; all runs)")
            #ax.legend()
            _save_fig(fig, dataset_out / f"{dataset}_bar_activation_t_predict_mean_std")
        # ---------------------------------------------------------------------

        # --------------------------
        # Bar plots: optimizer means
        # --------------------------
        opt_means = grp.groupby("optimizer", as_index=False)[["t_fit", "t_total_run"]].mean()
        opt_means = opt_means.sort_values("t_fit", ascending=False)

        x = np.arange(len(opt_means))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(opt_means))]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        ax.bar(x, opt_means["t_fit"].to_numpy(), color=colors, label="t_fit (train)")
        ax.set_xticks(x)
        ax.set_xticklabels(opt_means["optimizer"].tolist(), fontsize=10)
        ax.set_ylabel("seconds")
        ax.set_title(f"{dataset}: mean training time per optimizer (avg over activations)")
        #ax.legend()
        _save_fig(fig, dataset_out / f"{dataset}_bar_optimizer_t_fit")

        fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
        ax.bar(x, opt_means["t_total_run"].to_numpy(), color=colors, label="t_total_run")
        ax.set_xticks(x)
        ax.set_xticklabels(opt_means["optimizer"].tolist(), fontsize=10)
        ax.set_ylabel("seconds")
        ax.set_title(f"{dataset}: mean total run time per optimizer (avg over activations)")
        #ax.legend()
        _save_fig(fig, dataset_out / f"{dataset}_bar_optimizer_t_total_run")

        # --------------------------
        # Binary analysis: fractal vs non-fractal
        # --------------------------
        dfd["group"] = np.where(~dfd["activation"].isin(NON_FRACTAL), "fractal", "non-fractal")
        fractal_fit = dfd[dfd["group"] == "fractal"]["t_fit"].dropna().to_numpy()
        nonfr_fit = dfd[dfd["group"] == "non-fractal"]["t_fit"].dropna().to_numpy()

        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        bp = ax.boxplot(
            [fractal_fit, nonfr_fit],
            labels=["fractal", "non-fractal"],
            patch_artist=True,
            medianprops=dict(color=PALETTE[4]),
        )
        if len(bp["boxes"]) >= 1:
            bp["boxes"][0].set_facecolor(PALETTE[0])
        if len(bp["boxes"]) >= 2:
            bp["boxes"][1].set_facecolor(PALETTE[4])

        ax.set_ylabel("seconds (t_fit)")
        ax.set_title(f"{dataset}: training time distribution (fractal vs non-fractal)")
        _save_fig(fig, dataset_out / f"{dataset}_box_fractal_vs_nonfractal_t_fit")

        # Save small summary CSV
        summary = []
        if fractal_fit.size > 0:
            summary.append(("fractal", float(np.mean(fractal_fit)), float(np.std(fractal_fit))))
        if nonfr_fit.size > 0:
            summary.append(("non-fractal", float(np.mean(nonfr_fit)), float(np.std(nonfr_fit))))
        pd.DataFrame(summary, columns=["group", "mean_t_fit", "std_t_fit"]).to_csv(
            dataset_out / f"{dataset}_summary_fractal_vs_nonfractal.csv", index=False
        )

# -----------------------------------------------------------------------------
def global_plots(df: pd.DataFrame, out_root: Path):
    """
    Creates global (across datasets) summary plots.
    """
    global_out = out_root / "time_analysis_plots" / "_global"
    _ensure_dir(global_out)

    print(f"[time-analysis] global -> {global_out}")

    # Save full per-run table
    df.to_csv(global_out / "all_timings_per_run.csv", index=False)

    df2 = df.copy()
    df2["group"] = np.where(~df2["activation"].isin(NON_FRACTAL), "fractal", "non-fractal")

    # -------------------------------------------------------------------------
    # NEW: Global per-activation mean/std (CSV + TXT with "+- std")
    #      Separate tables for t_fit and t_predict (CSV + TXT)
    # -------------------------------------------------------------------------
    g_fit_stats = (
        df2.groupby("activation", as_index=False)[["t_fit"]]
        .agg(["mean", "std"])
    )
    g_fit_stats.columns = [f"{k}_{stat}" for (k, stat) in g_fit_stats.columns]
    g_fit_stats = g_fit_stats.reset_index()
    if "t_fit_mean" in g_fit_stats.columns:
        g_fit_stats = g_fit_stats.sort_values("t_fit_mean", ascending=False)
    g_fit_stats.to_csv(global_out / "global_activation_t_fit_mean_std.csv", index=False)

    txt_lines = []
    txt_lines.append("Global activation-level timing summary for t_fit (mean +- std over all runs; includes all datasets/optimizers/vderivs)")
    txt_lines.append("")
    for _, row in g_fit_stats.iterrows():
        a = row["activation"]
        m = row.get("t_fit_mean", np.nan)
        s = row.get("t_fit_std", np.nan)
        txt_lines.append(f"- {a}: t_fit: {m:.6f} +- {s:.6f}")
    (global_out / "global_activation_t_fit_mean_std.txt").write_text("\n".join(txt_lines), encoding="utf-8")

    g_pred_stats = (
        df2.groupby("activation", as_index=False)[["t_predict"]]
        .agg(["mean", "std"])
    )
    g_pred_stats.columns = [f"{k}_{stat}" for (k, stat) in g_pred_stats.columns]
    g_pred_stats = g_pred_stats.reset_index()
    if "t_predict_mean" in g_pred_stats.columns:
        g_pred_stats = g_pred_stats.sort_values("t_predict_mean", ascending=False)
    g_pred_stats.to_csv(global_out / "global_activation_t_predict_mean_std.csv", index=False)

    txt_lines = []
    txt_lines.append("Global activation-level timing summary for t_predict (mean +- std over all runs; includes all datasets/optimizers/vderivs)")
    txt_lines.append("")
    for _, row in g_pred_stats.iterrows():
        a = row["activation"]
        m = row.get("t_predict_mean", np.nan)
        s = row.get("t_predict_std", np.nan)
        txt_lines.append(f"- {a}: t_predict: {m:.6f} +- {s:.6f}")
    (global_out / "global_activation_t_predict_mean_std.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # NEW: Global bar plots with error bars for t_fit and t_predict (mean +- std)
    # -------------------------------------------------------------------------
    if ("t_fit_mean" in g_fit_stats.columns) and ("t_fit_std" in g_fit_stats.columns):
        x = np.arange(len(g_fit_stats))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(g_fit_stats))]
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(g_fit_stats)), 6), dpi=200)
        ax.bar(
            x,
            g_fit_stats["t_fit_mean"].to_numpy(),
            yerr=g_fit_stats["t_fit_std"].to_numpy(),
            color=colors,
            capsize=3,
            label="t_fit (mean +- std)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_label(a) for a in g_fit_stats["activation"].tolist()], fontsize=8)
        ax.set_ylabel("seconds")
        ax.set_title("Global: training time per activation (mean +- std; all runs)")
        #ax.legend()
        _save_fig(fig, global_out / "global_bar_activation_t_fit_mean_std")

    if ("t_predict_mean" in g_pred_stats.columns) and ("t_predict_std" in g_pred_stats.columns):
        x = np.arange(len(g_pred_stats))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(g_pred_stats))]
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(g_pred_stats)), 6), dpi=200)
        ax.bar(
            x,
            g_pred_stats["t_predict_mean"].to_numpy(),
            yerr=g_pred_stats["t_predict_std"].to_numpy(),
            color=colors,
            capsize=3,
            label="t_predict (mean +- std)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([_pretty_label(a) for a in g_pred_stats["activation"].tolist()], fontsize=8)
        ax.set_ylabel("seconds")
        ax.set_title("Global: prediction time per activation (mean +- std; all runs)")
        #ax.legend()
        _save_fig(fig, global_out / "global_bar_activation_t_predict_mean_std")
    # -------------------------------------------------------------------------

    # --------------------------
    # Dataset-level fractal vs non-fractal bars
    # --------------------------
    ds_grp = (
        df2.groupby(["dataset", "group"], as_index=False)["t_fit"]
        .mean()
        .pivot(index="dataset", columns="group", values="t_fit")
        .reset_index()
        .sort_values("dataset")
    )
    ds_grp.to_csv(global_out / "dataset_fractal_vs_nonfractal_mean_t_fit.csv", index=False)

    datasets = ds_grp["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.38
    fractal_vals = ds_grp.get("fractal", pd.Series([np.nan] * len(datasets))).to_numpy()
    nonfr_vals = ds_grp.get("non-fractal", pd.Series([np.nan] * len(datasets))).to_numpy()

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(datasets)), 6), dpi=200)
    ax.bar(x - width / 2, fractal_vals, width, label="fractal", color=PALETTE[0])
    ax.bar(x + width / 2, nonfr_vals, width, label="non-fractal", color=PALETTE[4])
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_label(d) for d in datasets], fontsize=9)
    ax.set_ylabel("seconds (mean t_fit)")
    ax.set_title("Training time by dataset: fractal vs non-fractal (mean t_fit)")
    #ax.legend()
    _save_fig(fig, global_out / "datasets_bar_fractal_vs_nonfractal_t_fit")

    # --------------------------
    # Global boxplot pooled
    # --------------------------
    fractal_all = df2[df2["group"] == "fractal"]["t_fit"].dropna().to_numpy()
    nonfr_all = df2[df2["group"] == "non-fractal"]["t_fit"].dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    bp = ax.boxplot(
        [fractal_all, nonfr_all],
        labels=["fractal", "non-fractal"],
        patch_artist=True,
        medianprops=dict(color=PALETTE[4]),
    )
    if len(bp["boxes"]) >= 1:
        bp["boxes"][0].set_facecolor(PALETTE[0])
    if len(bp["boxes"]) >= 2:
        bp["boxes"][1].set_facecolor(PALETTE[4])

    ax.set_ylabel("seconds (t_fit)")
    ax.set_title("Training time distribution (all datasets pooled)")
    _save_fig(fig, global_out / "global_box_fractal_vs_nonfractal_t_fit")

    # --------------------------
    # Global activation ranking
    # --------------------------
    act_mean = (
        df2.groupby("activation", as_index=False)["t_fit"]
        .mean()
        .sort_values("t_fit", ascending=False)
    )
    act_mean.to_csv(global_out / "global_activation_mean_t_fit.csv", index=False)

    x = np.arange(len(act_mean))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(act_mean))]
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(act_mean)), 6), dpi=200)
    ax.bar(x, act_mean["t_fit"].to_numpy(), color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([_pretty_label(a) for a in act_mean["activation"].tolist()], fontsize=8)
    ax.set_ylabel("seconds (mean t_fit)")
    ax.set_title("Global: mean training time per activation (all datasets pooled)")
    _save_fig(fig, global_out / "global_bar_activation_mean_t_fit")

# -----------------------------------------------------------------------------
def main():
    results_dir = Path(RESULTS_DIR).expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"RESULTS_DIR does not exist: {results_dir}")

    df = load_timings(results_dir)

    # Where plots go
    out_dir = results_dir / "time_analysis_plots"
    _ensure_dir(out_dir)

    per_dataset_plots(df, results_dir)
    global_plots(df, results_dir)

    print(f"Done. Plots + CSVs saved under: {out_dir}")

if __name__ == "__main__":
    main()
