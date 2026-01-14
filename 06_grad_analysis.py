#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-epoch gradient summary statistics (single probe batch, one GradientTape eval per epoch).

Rationale / similar practice
----------------------------
Exploding/vanishing gradients are commonly diagnosed via gradient magnitude and numerical stability
(e.g., gradient clipping and monitoring gradient norms). See:
- Pascanu, Mikolov, Bengio, "On the difficulty of training recurrent neural networks" (2013).

This script uses TensorFlow's GradientTape-based gradient computation pattern (automatic differentiation)
as in TensorFlow tutorials. :contentReference[oaicite:1]{index=1}

What is logged (per epoch, on ONE fixed probe batch)
----------------------------------------------------
Let G be the concatenation of all parameter gradients for the probe batch at the end of each epoch.
We log:
- grad_total_elems: total number of gradient scalars across all trainable parameters
- grad_nonfinite_elems: number of non-finite (NaN/Inf) gradient scalars
- grad_min / max / mean / std: statistics computed over ALL gradient scalars (no finiteness filtering)

Outputs (per dataset × activation × run)
----------------------------------------
<RESULTS_DIR>/grad/<dataset>/<activation>/
  run_<seed>__epoch_grad_stats.csv
  run_<seed>__epoch_grad_summary.txt
  run_<seed>__epoch_grad_plots.png
  run_<seed>__epoch_grad_plots.eps

If RUNS > 1, additional aggregated outputs (mean ± std across runs)
------------------------------------------------------------------
<RESULTS_DIR>/grad/<dataset>/<activation>/
  agg__epoch_grad_stats_mean_std.csv
  agg__epoch_grad_summary.txt
  agg__epoch_grad_plots.png
  agg__epoch_grad_plots.eps

Additional aggregated outputs (per dataset, across ALL activations)
------------------------------------------------------------------
<RESULTS_DIR>/grad/<dataset>/
  all_activations__epoch_grad_stats.csv
  all_activations__epoch_grad_stats_mean_std.csv        (only if RUNS > 1)
  all_activations__epoch_grad_plots.png/.eps
  all_activations__epoch_grad_plots__legend.png/.eps

Design choices (kept minimal)
-----------------------------
- One GradientTape evaluation per epoch on a fixed probe batch (deterministic selection from training data).
- Respects per-dataset (neurons, batch_size, epochs) via `data_with_params`.
- Defaults to RUNS=1, but can be increased.
- Only iris + vertebra-column by default (edit `data_with_params` if needed).
"""

import random
from copy import deepcopy as dc
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import fetch_openml

import fractal_activation_functions as fractal

# -----------------------------
# GLOBAL PLOT FONT + LINE SCALING
# -----------------------------
FONT_SCALE = 1.9  # multiplicative factor; default 1.0 keeps current sizing

# Line width scaling is controlled independently.
SCALE_LINEWIDTH = True  # if True, multiply all explicitly-set linewidths by LINEWIDTH_SCALE
LINEWIDTH_SCALE = 1.55    # multiplicative factor; default 1.0 keeps current sizing

def _fs(x: float) -> float:
    return float(x) * float(FONT_SCALE)

def _lw(x: float) -> float:
    if bool(SCALE_LINEWIDTH):
        return float(x) * float(LINEWIDTH_SCALE)
    return float(x)

# Baseline font sizes for axes/ticks (chosen to match typical matplotlib defaults)
_BASE_LABEL_FONTSIZE = 10.0
_BASE_TICK_FONTSIZE = 10.0

def _apply_axis_font_scaling(ax):
    # Axis labels
    ax.xaxis.label.set_size(_fs(_BASE_LABEL_FONTSIZE))
    ax.yaxis.label.set_size(_fs(_BASE_LABEL_FONTSIZE))
    # Tick labels
    ax.tick_params(axis="both", which="both", labelsize=_fs(_BASE_TICK_FONTSIZE))


# --- Tailwind colour set --------------------------------------------------- #
PALETTE = [
    "#a2faa3",  # Light green
    "#92c9b1",  # Cambridge blue
    "#4f759b",  # UCLA blue
    "#5d5179",  # Ultra-violet
    "#571f4e",  # Palatinate
]

# -----------------------------
# CONFIG (minimal)
# -----------------------------
RESULTS_DIR = "results_grad_summary_others"
RUNS = 40
SEED_BASE = 238974
LEARNING_RATE = 1e-3

# Same naming as your project
ACTIVATION_FUNCTIONS = [
    #("weierstrass", fractal.weierstrass_function_tf),
    #("weierstrass_mandelbrot_xpsin", fractal.weierstrass_mandelbrot_function_xpsin),
    #("weierstrass_mandelbrot_xsinsquared", fractal.weierstrass_mandelbrot_function_xsinsquared),
    #("weierstrass_mandelbrot_relupsin", fractal.weierstrass_mandelbrot_function_relupsin),
    #("weierstrass_mandelbrot_tanhpsin", fractal.weierstrass_mandelbrot_function_tanhpsin),
    ("blancmange", fractal.modulated_blancmange_curve),
    ("decaying_cosine", fractal.decaying_cosine_function_tf),
    ("modified_weierstrass_tanh", fractal.modified_weierstrass_function_tanh),
    #("modified_weierstrass_ReLU", fractal.modified_weierstrass_function_relu),
    #("relu", "relu"),
    #("sigmoid", "sigmoid"),
    #("tanh", "tanh"),
]

# ------------------------------------------------------------------
#  Dataset  | NN-params (neurons, batch, epochs)  [use your per-dataset settings]
# ------------------------------------------------------------------
data_with_params = [
    ("iris", 32, 16, 30),
    ("vertebra-column", 32, 16, 30),
]

# ------------------------------------------------------------------
# EXTEND PALETTE TO MATCH NUMBER OF ACTIVATIONS
# ------------------------------------------------------------------
def _extend_palette_to_n(n: int):
    """
    Returns a list of n distinct-ish hex colors.
    Uses matplotlib colormap sampling to avoid hand-picking.
    """
    if n <= len(PALETTE):
        return PALETTE[:n]
    cmap = plt.get_cmap("tab20")
    cols = []
    for i in range(n):
        r, g, b, _ = cmap(i / max(1, n - 1))
        cols.append("#{0:02x}{1:02x}{2:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return cols

PALETTE = _extend_palette_to_n(max(len(ACTIVATION_FUNCTIONS), 5))


# -----------------------------
# HELPERS
# -----------------------------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _save_fig(fig, out_base: Path, dpi=300):
    fig.tight_layout()
    fig.savefig(str(out_base.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".eps")), bbox_inches="tight")
    plt.close(fig)

def encode_non_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object", "category"]).columns:
        uniq = df[col].unique()
        mapping = {v: (i / (len(uniq) - 1) if len(uniq) > 1 else 0.0) for i, v in enumerate(uniq)}
        df.loc[:, col] = df[col].map(mapping)
    return df

def load_dataset_openml(name: str):
    data = fetch_openml(name=name, version=1, as_frame=True, parser="auto")
    X = data.data
    y = data.target

    if isinstance(X, pd.DataFrame):
        X = encode_non_numeric_features(X).to_numpy(dtype=np.float32)
    else:
        X = X.astype(np.float32)

    if isinstance(y, pd.Series):
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = LabelEncoder().fit_transform(y)
        else:
            y = y.to_numpy()
    y = y.astype(np.int32)

    return X, y

def split_scale(X, y, seed):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, X_test, y_train, y_test = dc(
        train_test_split(X, y, test_size=0.3, random_state=seed, shuffle=True)
    )
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    return X_train, X_test, y_train, y_test

def build_model(n_features, n_classes, neurons, activation):
    initializer = tf.keras.initializers.GlorotUniform(seed=42)

    if n_classes == 2:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=n_features),
            tf.keras.layers.Dense(n_classes, activation='sigmoid', kernel_initializer=initializer)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(neurons, activation=activation, kernel_initializer=initializer, input_dim=n_features),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(neurons*2, activation=activation, kernel_initializer=initializer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(n_classes, activation='softmax', kernel_initializer=initializer)
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    return model, loss_fn

def _flatten_all_grad_values(grads):
    """
    Returns:
      total_elems, nonfinite_elems, all_values_1d(np.ndarray)
    """
    total = 0
    nonfinite = 0
    vals = []

    for g in grads:
        if g is None:
            continue
        gv = tf.reshape(g, [-1])
        total += int(gv.shape[0])
        mask = tf.math.is_finite(gv)
        n_finite = int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy().item())
        nonfinite += int(gv.shape[0]) - n_finite
        vals.append(gv.numpy())

    if len(vals) > 0:
        vals = np.concatenate(vals, axis=0).astype(np.float64, copy=False)
    else:
        vals = np.array([], dtype=np.float64)

    return int(total), int(nonfinite), vals

def _all_stats(arr: np.ndarray):
    if arr.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan)
    # NOTE: if arr contains NaN/Inf, these will propagate (intended: "no finiteness filtering")
    return dict(
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
    )

def _plot_one_figure(epoch_df: pd.DataFrame, out_base: Path, title: str, is_aggregate: bool = False):
    """
    One figure file per dataset × activation:
      - RUNS==1: uses columns grad_* directly
      - RUNS>1 aggregate: uses mean curves with ±std shading (grad_*_mean and grad_*_std)

    Panels:
      (1) grad_min/max
      (2) grad_mean
      (3) grad_std
    """
    if epoch_df.empty:
        return

    x = epoch_df["epoch"].to_numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), dpi=200, sharex=True)

    if not is_aggregate:
        # (1) min/max
        axs[0].plot(x, epoch_df["grad_min"].to_numpy(), color=PALETTE[3], label="min", linewidth=_lw(1.5))
        axs[0].plot(x, epoch_df["grad_max"].to_numpy(), color=PALETTE[4], label="max", linewidth=_lw(1.5))
        axs[0].set_ylabel("gradient value")
        axs[0].legend(loc="best", fontsize=_fs(9))
        axs[0].set_title(title, fontsize=_fs(12))
        _apply_axis_font_scaling(axs[0])

        # (2) mean
        axs[1].plot(x, epoch_df["grad_mean"].to_numpy(), color=PALETTE[1], label="mean", linewidth=_lw(1.5))
        axs[1].set_ylabel("gradient mean")
        axs[1].legend(loc="best", fontsize=_fs(9))
        _apply_axis_font_scaling(axs[1])

        # (3) std
        axs[2].plot(x, epoch_df["grad_std"].to_numpy(), color=PALETTE[0], label="std", linewidth=_lw(1.5))
        axs[2].set_ylabel("gradient std")
        axs[2].set_xlabel("epoch")
        axs[2].legend(loc="best", fontsize=_fs(9))
        _apply_axis_font_scaling(axs[2])

    else:
        # Helper for mean±std lines (avoid EPS transparency warning by skipping fill_between for EPS)
        def _plot_mean_std(ax, y_mean, y_std, color, label):
            ax.plot(x, y_mean, color=color, label=f"{label} (mean)", linewidth=_lw(1.5))
            # fill_between causes EPS "no transparency" warnings; keep code minimal by not using alpha shading.
            # If you really want shading, switch EPS to PDF, or save only PNG for shaded plots.

        # (1) min/max
        _plot_mean_std(
            axs[0],
            epoch_df["grad_min_mean"].to_numpy(),
            epoch_df["grad_min_std"].to_numpy(),
            PALETTE[3],
            "min",
        )
        _plot_mean_std(
            axs[0],
            epoch_df["grad_max_mean"].to_numpy(),
            epoch_df["grad_max_std"].to_numpy(),
            PALETTE[4],
            "max",
        )
        axs[0].set_ylabel("gradient value")
        axs[0].legend(loc="best", fontsize=_fs(9))
        axs[0].set_title(title, fontsize=_fs(12))
        _apply_axis_font_scaling(axs[0])

        # (2) mean
        _plot_mean_std(
            axs[1],
            epoch_df["grad_mean_mean"].to_numpy(),
            epoch_df["grad_mean_std"].to_numpy(),
            PALETTE[1],
            "mean",
        )
        axs[1].set_ylabel("gradient mean")
        axs[1].legend(loc="best", fontsize=_fs(9))
        _apply_axis_font_scaling(axs[1])

        # (3) std
        _plot_mean_std(
            axs[2],
            epoch_df["grad_std_mean"].to_numpy(),
            epoch_df["grad_std_std"].to_numpy(),
            PALETTE[0],
            "std",
        )
        axs[2].set_ylabel("gradient std")
        axs[2].set_xlabel("epoch")
        axs[2].legend(loc="best", fontsize=_fs(9))
        _apply_axis_font_scaling(axs[2])

    _save_fig(fig, out_base)

def _aggregate_runs(run_dfs: list, epochs: int):
    """
    Input: list of per-run epoch DataFrames (same epochs expected).
    Output: epoch-level mean/std across runs for key metrics.
    """
    if len(run_dfs) == 0:
        return pd.DataFrame()

    df_all = pd.concat(run_dfs, ignore_index=True)

    metrics = [
        "grad_min",
        "grad_max",
        "grad_mean",
        "grad_std",
        "grad_nonfinite_elems",
        "grad_total_elems",
        "probe_loss",
    ]
    keep = [c for c in metrics if c in df_all.columns]

    # Explicit named aggregation -> flat columns guaranteed
    agg_dict = {}
    for c in keep:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, "std")

    agg_df = df_all.groupby("epoch", as_index=False).agg(**agg_dict)

    return agg_df

def _save_separate_legend(handles, labels, out_base: Path, ncol: int = 2, fontsize: int = 9):
    fig = plt.figure(figsize=(10, 0.8 + 0.25 * max(1, int(np.ceil(len(labels) / ncol)))), dpi=200)
    ax = fig.add_subplot(111)
    ax.axis("off")
    fig.legend(handles, labels, loc="center", ncol=ncol, frameon=False, fontsize=_fs(fontsize))
    _save_fig(fig, out_base)

def _plot_all_activations_one_figure(dataset: str, df_by_act: dict, out_dir: Path, epochs: int, is_aggregate: bool):
    """
    One figure per dataset collecting ALL activation functions.
    Legends are saved separately.
    For min/max: same color per activation, different linestyle for min/max.
    """
    if len(df_by_act) == 0:
        return

    first_df = next(iter(df_by_act.values()))
    if first_df.empty:
        return

    x = first_df["epoch"].to_numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), dpi=200, sharex=True)

    handles_for_legend = []
    labels_for_legend = []

    # Panel (1): min/max (different linestyles)
    for i, (act_name, df) in enumerate(df_by_act.items()):
        color = PALETTE[i % len(PALETTE)]
        # One handle per activation (use max curve handle for legend)
        if is_aggregate:
            y_min = df["grad_min_mean"].to_numpy()
            y_max = df["grad_max_mean"].to_numpy()
        else:
            y_min = df["grad_min"].to_numpy()
            y_max = df["grad_max"].to_numpy()

        axs[0].plot(x, y_min, color=color, linestyle="--", linewidth=_lw(1.2))
        h = axs[0].plot(x, y_max, color=color, linestyle="-", linewidth=_lw(1.2), label=act_name)[0]
        handles_for_legend.append(h)
        labels_for_legend.append(act_name)

    axs[0].set_ylabel("gradient value")
    axs[0].set_title(
        f"{dataset} | " + (f"mean across {RUNS} runs" if is_aggregate else "single run"),
        fontsize=_fs(12),
    )
    _apply_axis_font_scaling(axs[0])

    # Panel (2): mean
    for i, (act_name, df) in enumerate(df_by_act.items()):
        color = PALETTE[i % len(PALETTE)]
        if is_aggregate:
            y = df["grad_mean_mean"].to_numpy()
        else:
            y = df["grad_mean"].to_numpy()
        axs[1].plot(x, y, color=color, linewidth=_lw(1.2))
    axs[1].set_ylabel("gradient mean")
    _apply_axis_font_scaling(axs[1])

    # Panel (3): std
    for i, (act_name, df) in enumerate(df_by_act.items()):
        color = PALETTE[i % len(PALETTE)]
        if is_aggregate:
            y = df["grad_std_mean"].to_numpy()
        else:
            y = df["grad_std"].to_numpy()
        axs[2].plot(x, y, color=color, linewidth=_lw(1.2))
    axs[2].set_ylabel("gradient std")
    axs[2].set_xlabel("epoch")
    _apply_axis_font_scaling(axs[2])

    for ax in axs:
        ax.legend_.remove() if ax.get_legend() is not None else None

    out_base = out_dir / ("all_activations__epoch_grad_plots_mean" if is_aggregate else "all_activations__epoch_grad_plots")
    _save_fig(fig, out_base)

    _save_separate_legend(
        handles_for_legend,
        labels_for_legend,
        out_dir / ("all_activations__epoch_grad_plots__legend_mean" if is_aggregate else "all_activations__epoch_grad_plots__legend"),
        ncol=2,
        fontsize=9,
    )


# -----------------------------
# CALLBACK: per-epoch gradient stats on a fixed probe batch
# -----------------------------
class ProbeGradientStatsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_probe, y_probe, loss_fn, dataset, activation, run_seed):
        super().__init__()
        self.x_probe = tf.convert_to_tensor(x_probe, dtype=tf.float32)
        self.y_probe = tf.convert_to_tensor(y_probe, dtype=tf.int32)
        self.loss_fn = loss_fn

        self.dataset = dataset
        self.activation = activation
        self.run_seed = int(run_seed)

        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        with tf.GradientTape() as tape:
            y_pred = self.model(self.x_probe, training=True)
            loss = self.loss_fn(self.y_probe, y_pred)
            if self.model.losses:
                loss = loss + tf.add_n(self.model.losses)

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads_clean = [g for g in grads if g is not None]

        total, nonfinite, all_vals = _flatten_all_grad_values(grads_clean)
        st = _all_stats(all_vals)

        row = {
            "dataset": self.dataset,
            "activation": self.activation,
            "run_seed": self.run_seed,
            "epoch": int(epoch),

            "probe_loss": float(loss.numpy().item()),

            "grad_total_elems": int(total),
            "grad_nonfinite_elems": int(nonfinite),

            "grad_min": float(st["min"]),
            "grad_max": float(st["max"]),
            "grad_mean": float(st["mean"]),
            "grad_std": float(st["std"]),

            # Optional training signals
            "train_loss": float(logs.get("loss", np.nan)),
            "train_accuracy": float(logs.get("accuracy", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "val_accuracy": float(logs.get("val_accuracy", np.nan)),
        }
        self.rows.append(row)


# -----------------------------
# CORE EXPERIMENT
# -----------------------------
def run_one(results_root: Path, dataset: str, neurons: int, batch_size: int, epochs: int,
            act_name: str, act_fn, run_seed: int):

    random.seed(run_seed)
    np.random.seed(run_seed)
    tf.random.set_seed(run_seed)
    tf.keras.backend.clear_session()

    X, y = load_dataset_openml(dataset)
    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = split_scale(X, y, run_seed)

    # Fixed probe batch: deterministic selection from training set
    k = min(batch_size, X_train.shape[0])
    x_probe = X_train[:k]
    y_probe = y_train[:k]

    model, loss_fn = build_model(n_features, n_classes, neurons, act_fn)

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    cb = ProbeGradientStatsCallback(
        x_probe=x_probe,
        y_probe=y_probe,
        loss_fn=loss_fn,
        dataset=dataset,
        activation=act_name,
        run_seed=run_seed,
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[cb],
        shuffle=True,
    )

    out_dir = results_root / "grad" / dataset / act_name
    _ensure_dir(out_dir)

    epoch_df = pd.DataFrame(cb.rows)
    csv_path = out_dir / f"run_{run_seed}__epoch_grad_stats.csv"
    epoch_df.to_csv(csv_path, index=False)

    # Text summary (per run)
    summary_lines = []
    summary_lines.append(f"dataset: {dataset}")
    summary_lines.append(f"activation: {act_name}")
    summary_lines.append(f"run_seed: {run_seed}")
    summary_lines.append(f"neurons: {neurons}, batch_size: {batch_size}, epochs: {epochs}")
    summary_lines.append("")

    if epoch_df.empty:
        summary_lines.append("No epochs recorded.")
    else:
        # Non-finite gradient scalars (count)
        nnf = pd.to_numeric(epoch_df["grad_nonfinite_elems"], errors="coerce").to_numpy(dtype=float)
        nnf_fin = nnf[np.isfinite(nnf)]
        if nnf_fin.size > 0:
            summary_lines.append(f"grad_nonfinite_elems: min={nnf_fin.min():.6e}, max={nnf_fin.max():.6e}, mean={nnf_fin.mean():.6e}")
        else:
            summary_lines.append("grad_nonfinite_elems: no finite values.")

        summary_lines.append("")
        for col in ["grad_min", "grad_max", "grad_mean", "grad_std"]:
            v = pd.to_numeric(epoch_df[col], errors="coerce").to_numpy(dtype=float)
            vfin = v[np.isfinite(v)]
            if vfin.size > 0:
                summary_lines.append(f"{col}: min={vfin.min():.6e}, max={vfin.max():.6e}, mean={vfin.mean():.6e}, std={vfin.std():.6e}")
            else:
                summary_lines.append(f"{col}: no finite values.")

    (out_dir / f"run_{run_seed}__epoch_grad_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # One plot file per dataset × activation × run (3 panels)
    title = f"{dataset} | {act_name} | seed={run_seed} (probe batch gradients)"
    _plot_one_figure(epoch_df, out_dir / f"run_{run_seed}__epoch_grad_plots", title=title, is_aggregate=False)

    return epoch_df


def main():
    results_root = Path(RESULTS_DIR).expanduser().resolve()
    _ensure_dir(results_root)

    print("TensorFlow:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())

    for (dataset, neurons, batch_size, epochs) in data_with_params:
        dataset_all_single = []
        dataset_all_agg = []

        df_by_act_single = {}
        df_by_act_agg = {}

        for (act_name, act_fn) in ACTIVATION_FUNCTIONS:
            run_dfs = []
            for r in range(RUNS):
                seed = SEED_BASE + r
                print(f"[grad] dataset={dataset} act={act_name} run={r+1}/{RUNS} seed={seed}")
                df_run = run_one(
                    results_root=results_root,
                    dataset=str(dataset),
                    neurons=int(neurons),
                    batch_size=int(batch_size),
                    epochs=int(epochs),
                    act_name=str(act_name),
                    act_fn=act_fn,
                    run_seed=int(seed),
                )
                if isinstance(df_run, pd.DataFrame) and (not df_run.empty):
                    run_dfs.append(df_run)

            # Aggregate across runs (mean ± std) and produce one averaged plot
            if RUNS > 1 and len(run_dfs) > 0:
                out_dir = results_root / "grad" / str(dataset) / str(act_name)
                _ensure_dir(out_dir)

                agg_df = _aggregate_runs(run_dfs, epochs=int(epochs))
                agg_df.to_csv(out_dir / "agg__epoch_grad_stats_mean_std.csv", index=False)

                # Text summary for aggregate
                lines = []
                lines.append(f"dataset: {dataset}")
                lines.append(f"activation: {act_name}")
                lines.append(f"runs: {RUNS}")
                lines.append(f"neurons: {neurons}, batch_size: {batch_size}, epochs: {epochs}")
                lines.append("")
                for base in ["grad_min", "grad_max", "grad_mean", "grad_std", "grad_nonfinite_elems"]:
                    col_mu = f"{base}_mean"
                    col_sd = f"{base}_std"

                    if col_mu in agg_df.columns:
                        mu = pd.to_numeric(agg_df[col_mu], errors="coerce").to_numpy(dtype=float)
                        mu_fin = mu[np.isfinite(mu)]
                        if mu_fin.size > 0:
                            lines.append(f"{base}_mean over epochs: min={mu_fin.min():.6e}, max={mu_fin.max():.6e}, mean={mu_fin.mean():.6e}")
                        else:
                            lines.append(f"{base}_mean over epochs: no finite values.")
                    else:
                        lines.append(f"{base}_mean over epochs: column missing.")

                    if col_sd in agg_df.columns:
                        sd = pd.to_numeric(agg_df[col_sd], errors="coerce").to_numpy(dtype=float)
                        sd_fin = sd[np.isfinite(sd)]
                        if sd_fin.size > 0:
                            lines.append(f"{base}_std over epochs:  min={sd_fin.min():.6e}, max={sd_fin.max():.6e}, mean={sd_fin.mean():.6e}")
                        else:
                            lines.append(f"{base}_std over epochs:  no finite values.")
                    else:
                        lines.append(f"{base}_std over epochs:  column missing.")
                    lines.append("")
                (out_dir / "agg__epoch_grad_summary.txt").write_text("\n".join(lines), encoding="utf-8")

                # One averaged plot per dataset × activation
                title = f"{dataset} | {act_name} | runs={RUNS} (probe batch gradients, mean across runs)"
                _plot_one_figure(agg_df, out_dir / "agg__epoch_grad_plots", title=title, is_aggregate=True)

                agg_df2 = agg_df.copy()
                if "activation" in agg_df2.columns:
                    agg_df2 = agg_df2.drop(columns=["activation"])
                agg_df2.insert(0, "activation", str(act_name))
                dataset_all_agg.append(agg_df2)
                df_by_act_agg[str(act_name)] = agg_df

            # Collect single-run representative table
            if len(run_dfs) > 0:
                df0 = run_dfs[0].copy()
                if "activation" in df0.columns:
                    df0 = df0.drop(columns=["activation"])
                df0.insert(0, "activation", str(act_name))
                dataset_all_single.append(df0)
                df_by_act_single[str(act_name)] = run_dfs[0]

        dataset_dir = results_root / "grad" / str(dataset)
        _ensure_dir(dataset_dir)

        if len(dataset_all_single) > 0:
            pd.concat(dataset_all_single, ignore_index=True).to_csv(
                dataset_dir / "all_activations__epoch_grad_stats.csv", index=False
            )
            _plot_all_activations_one_figure(
                dataset=str(dataset),
                df_by_act=df_by_act_single,
                out_dir=dataset_dir,
                epochs=int(epochs),
                is_aggregate=False,
            )

        if RUNS > 1 and len(dataset_all_agg) > 0:
            pd.concat(dataset_all_agg, ignore_index=True).to_csv(
                dataset_dir / "all_activations__epoch_grad_stats_mean_std.csv", index=False
            )
            _plot_all_activations_one_figure(
                dataset=str(dataset),
                df_by_act=df_by_act_agg,
                out_dir=dataset_dir,
                epochs=int(epochs),
                is_aggregate=True,
            )

    print(f"Done. Outputs under: {results_root / 'grad'}")


if __name__ == "__main__":
    main()
