# =============================================================
# expressivity_experiment_enhanced.py — *annotated edition*
# =============================================================
#
# This file is a **comment‑augmented** drop‑in replacement for the
# original `expressivity_experiment_enhanced.py` that accompanies
# our paper on fractal activation functions.   All executable
# statements remain **byte‑for‑byte identical** to the baseline so
# that anyone can reproduce the numerical results reported in the
# manuscript.  We merely sprinkle additional remarks that explain
#
#   • why each analytical choice mirrors the framework of Poole
#     *et al.* (2016) and Raghu *et al.* (2017),
#   • how the experiment aligns with the theoretical predictions
#     of exponential expressivity through *trajectory length*, and
#   • how the same infrastructure is reused to benchmark our new
#     fractal activations.
#
# Wherever a new block of commentary begins we prefix it with a
# colourful emoji so the pedagogical layer is easy to skim while
# leaving the computational layer untouched.  
# -------------------------------------------------------------
# Quantitative study of the expressivity of deep neural network
# activation functions along unit‑circle trajectories.
# -------------------------------------------------------------
#
# References cited in the comments (numbers match the PDF list):
#   [Poole‑16]   B. Poole *et al.*  "Exponential expressivity in
#                deep neural networks through transient chaos"
#   [Raghu‑17]   M. Raghu *et al.*   "On the expressive power of
#                deep neural networks"
# -------------------------------------------------------------

from __future__ import annotations

# IMPORTS -----------------------------------------------------------------
# All libraries exactly as in the original file.  The two plotting
# helpers below are not strictly necessary for expressivity theory,
# but they provide human‑readable diagnostics (length curves, PCA
# strips) akin to the visualisations in [Poole‑16, Fig. 2] and
# [Raghu‑17, Fig. 3].
import datetime
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# VISUAL STYLE ------------------------------------------------------------
# Tailwind‑inspired greyscale palette supplied by the user.  The
# colour scheme is cosmetic and does **not** influence any numeric
# quantity; hence we may annotate freely.
PALETTE: List[str] = [
    "#000000",  # Black
    "#363946",  # Onyx
    "#696773",  # Dim gray
    "#819595",  # Cadet gray
    "#b1b6a6",  # Ash gray
]

sns.set_theme(
    style="whitegrid",
    palette=PALETTE,
    rc={
        "figure.dpi": 150,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "lines.linewidth": 2.5,
    },
)

# HYPER‑PARAMETERS --------------------------------------------------------
# Below we fix the network depth, width, weight variance, etc.  The
# chosen values replicate the *mean‑field* setting analysed by
# [Poole‑16] where every hidden layer shares identical statistics.
# This homogeneity allows a single scalar (trajectory length) to
# characterise the propagation of geometric signals.

# Dimensionality of a flattened MNIST (28×28) input vector.  Any
# 1‑D trajectory is embedded in this space.
INPUT_DIM: int = 784

# Number of hidden layers in the random, width‑constant network.
DEPTH: int = 8

# Number of neurons per hidden layer.  Increasing WIDTH raises the
# base of the exponential growth predicted by the mean‑field theory
# but does **not** change the qualitative conclusion that depth, not
# width, drives expressivity.
WIDTH: int = 400

# Variances of the i.i.d. Gaussian weight and bias initialisations.
# These are the key knobs of the order‑to‑chaos transition in
# [Poole‑16].  Setting SIGMA_W large moves the network into the
# *chaotic* regime where trajectory length inflates exponentially.
SIGMA_W: float = 5.0  # weight std‑dev multiplier
SIGMA_B: float = 0.0  # bias std‑dev multiplier (0 → no bias)

# Number of discrete samples along the unit‑circle trajectory that
# feeds the network.  400 matches the resolution used in [Raghu‑17].
SAMPLES: int = 400

# Indices of the layers that will be projected to 2‑D by PCA for
# qualitative visualisation (cf. Fig. 3 in [Raghu‑17]).
SNAPSHOTS: List[int] = [0, 1, 3, 5, 8]

# Output directory – time‑stamped to avoid overwrites.
DTSTAMP: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR: Path = Path(f"expressivity_outputs_{DTSTAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Seeds for full reproducibility (important for chaotic regimes).
np.random.seed(0)
tf.random.set_seed(0)

# IMPORT FRACTAL ACTIVATIONS ---------------------------------------------
# We dynamically plug in our custom fractal activations.  Keeping
# these functions external avoids polluting the core experiment and
# lets us swap them for standard ReLU/tanh exactly as in our paper’s
# classification section.
import fractal_activation_functions as fractal  # noqa: E402  pylint: disable=wrong-import-position

# ACTIVATION DICTIONARY ---------------------------------------------------
# The keys here (strings) are the canonical names used everywhere
# else in the LaTeX tables, so adding an entry automatically enables
# its inclusion in the expressivity plots.
ACTIVATIONS: Dict[str, tf.keras.layers.Layer] = {
    # Classic
    "relu": tf.keras.layers.ReLU(),
    "tanh": tf.keras.layers.Activation("tanh"),
    "sigmoid": tf.keras.layers.Activation("sigmoid"),
    # Fractal & Weierstrass family
    "weierstrass": fractal.weierstrass_function_tf,
    "weierstrass_mandelbrot_xpsin": fractal.weierstrass_mandelbrot_function_xpsin,
    "weierstrass_mandelbrot_xsinsquared": fractal.weierstrass_mandelbrot_function_xsinsquared,
    "weierstrass_mandelbrot_relupsin": fractal.weierstrass_mandelbrot_function_relupsin,
    "weierstrass_mandelbrot_tanhpsin": fractal.weierstrass_mandelbrot_function_tanhpsin,
    "blancmange": fractal.modulated_blancmange_curve,
    "decaying_cosine": fractal.decaying_cosine_function_tf,
    "modified_weierstrass_tanh": fractal.modified_weierstrass_function_tanh,
    "modified_weierstrass_ReLU": fractal.modified_weierstrass_function_relu,
}

# HELPER FUNCTIONS -------------------------------------------------------
# The following utilities are unchanged; we only annotate *why* they
# are useful through the lens of Poole & Raghu.


def region_state(pre_act: tf.Tensor, name: str) -> tf.Tensor | None:
    """Return a discrete region/state tensor used to count transitions.

    Counting region transitions along a 1‑D trajectory is one of the
    expressivity measures introduced by [Raghu‑17].  For piece‑wise
    linear activations such as ReLU the activation pattern can be
    captured by a binary code (on/off).  Smooth activations do not
    induce region splits and therefore return *None*.
    """
    if name == "relu":
        return tf.cast(pre_act > 0.0, tf.int8)
    if name in {"hardtanh", "modified_weierstrass_ReLU"}:
        return tf.cast(tf.sign(tf.clip_by_value(pre_act, -1.0, 1.0)), tf.int8)
    return None  # Smooth / unsegmented activation – we skip transitions.


def make_unit_circle_trajectory() -> tf.Tensor:
    """Generate `SAMPLES` points on the geodesic between two random vectors.

    *Why a half‑circle?*  Following [Poole‑16] we want an input curve
    that stays roughly on the hypersphere so that length changes are
    attributable to the *network*, not to the parametrisation.
    """
    x0 = tf.random.normal([INPUT_DIM])
    x1 = tf.random.normal([INPUT_DIM])
    t_lin = tf.linspace(0.0, 1.0, SAMPLES)
    return tf.stack(
        [tf.cos(np.pi * t / 2) * x0 + tf.sin(np.pi * t / 2) * x1 for t in t_lin],
        axis=0,
    )


def save_plot(fig: plt.Figure, fname: str) -> None:
    """Utility that saves *fig* to PNG and vector EPS."""
    png_path = OUT_DIR / f"{fname}.png"
    eps_path = OUT_DIR / f"{fname}.eps"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    plt.close(fig)

# TRAJECTORY LENGTH REGISTRY --------------------------------------------
# We accumulate per‑layer arc lengths so we can dump a concise table
# at the end – exactly mirroring Table 1 in [Raghu‑17].
LENGTHS_PER_ACT: Dict[str, List[float]] = {}


def report_lengths() -> None:
    """Print arc‑lengths per layer for each activation function."""
    print("\nArc‑lengths per layer (SAMPLES =", SAMPLES, "):")
    for name, lens in LENGTHS_PER_ACT.items():
        print(f"  {name}: {lens}")

# MAIN EXPERIMENTAL LOOP -------------------------------------------------


def main() -> None:
    traj_input = make_unit_circle_trajectory()

    for act_name, act_layer in ACTIVATIONS.items():
        print(f"► {act_name}")

        # ---------------------- build random weights ----------------------
        # What happens here?
        # For each hidden layer d = 0 … DEPTH-1 we draw:
        #   •  W_d  ∈ ℝ^{fan_in × WIDTH}      ← fully-connected weight matrix
        #   •  b_d  ∈ ℝ^{WIDTH}               ← bias vector
        #
        # This *is* the neural network in its rawest form:
        # every column of W_d represents all incoming synapses for
        # **one** neuron in layer d, and every row enumerates the
        # connections *from* a single neuron in the previous layer.
        # By iterating over DEPTH we stack layers “horizontally” —
        # exactly the feed-forward topology analysed by
        # Poole & Raghu (constant width, i.i.d. Gaussian weights).
        Ws, bs = [], []
        for d in range(DEPTH):
            fan_in = INPUT_DIM if d == 0 else WIDTH
            std = SIGMA_W / np.sqrt(WIDTH)      # mean-field scaling
            Ws.append(tf.random.normal([fan_in, WIDTH], stddev=std))
            bs.append(tf.random.normal([WIDTH], stddev=SIGMA_B))

        # -------------------- forward pass along trajectory ---------------
        # We now push the 1-D trajectory through the network *layer by
        # layer*.  At each depth `d` we compute
        #
        #   pre = h · W_d + b_d                ▶ affine transform
        #   h   = σ_d(pre)                    ▶ non-linear activation
        #
        # where h has shape (SAMPLES, WIDTH).  This is textbook vectorised
        # neuron evaluation: every row is one point on the trajectory,
        # every column the activation of a neuron.  The loop therefore
        # mimics what a training framework would do — minus weight
        # updates (we keep weights frozen to study expressivity only).
        layer_reps: List[tf.Tensor] = [traj_input]
        lengths: List[float] = []        # trajectory length per layer
        transitions: List[float] = []    # region transitions per layer
        h = traj_input                   # h = representation at depth 0

        for d in range(DEPTH):
            # ▶ 1. pre-activations = linear combination of previous layer
            pre = tf.linalg.matmul(h, Ws[d]) + bs[d]

            # ▶ 2. optional region-code analysis (only for piece-wise
            #       linear activations).  Counts how often the *binary*
            #       activation pattern along the trajectory changes —
            #       Raghu et al.’s “transition count”.
            pat = region_state(pre, act_name)
            if pat is not None:
                diff = tf.reduce_any(pat[1:] != pat[:-1], axis=1)
                transitions.append(float(tf.reduce_sum(tf.cast(diff, tf.float32))))
            else:
                transitions.append(np.nan)

            # ▶ 3. non-linearity = actual ‘firing’ of the neurons
            h = act_layer(pre)           # shape still (SAMPLES, WIDTH)
            layer_reps.append(h)

            # ▶ 4. Poole et al.’s arc-length metric:
            #    accumulate Euclidean length of the curve at this depth.
            seg_len = tf.norm(h[1:] - h[:-1], axis=1)  # distance between successive samples
            lengths.append(float(tf.reduce_sum(seg_len)))


        # keep lengths for summary report
        LENGTHS_PER_ACT[act_name] = lengths

        # ----------------------------------------------------------------
        #                           PLOTTING
        # ----------------------------------------------------------------
        dep = np.arange(1, DEPTH + 1)

        # ---- 1) Trajectory length per depth --------------------------------
        fig1, ax1 = plt.subplots(figsize=(5, 3))
        sns.lineplot(x=dep, y=lengths, marker="o", ax=ax1)
        ax1.set_title(f"Trajectory length vs. depth – {act_name}")
        ax1.set_xlabel("Layer depth d")
        ax1.set_ylabel(r"Total arc‑length $L_d$")
        save_plot(fig1, f"length_{act_name}")

        # ---- 2) Region transitions per depth ------------------------------
        if not np.isnan(transitions).all():
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            sns.lineplot(x=dep, y=transitions, marker="o", ax=ax2)
            ax2.set_title(f"Activation transitions – {act_name}")
            ax2.set_xlabel("Layer depth d")
            ax2.set_ylabel("# transitions along trajectory")
            save_plot(fig2, f"trans_{act_name}")

        # ---- 3) PCA strips (qualitative geometry) -------------------------
        n_cols = len(SNAPSHOTS)
        fig3, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3), sharex=False, sharey=False)
        for ax, idx in zip(axes, SNAPSHOTS):
            H = layer_reps[idx]  # shape = (SAMPLES, WIDTH)
            Hc = H - tf.reduce_mean(H, axis=0)
            # SVD → first 2 PCs
            _s, _u, v = tf.linalg.svd(Hc, full_matrices=False)
            coords = tf.tensordot(Hc, v[:, :2], axes=1)
            sns.lineplot(x=coords[:, 0], y=coords[:, 1], ax=ax)
            ax.set_title(f"L{idx}")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", "box")
        fig3.suptitle(f"{act_name}", y=0.0, fontsize=28)
        fig3.tight_layout()
        save_plot(fig3, f"pca_{act_name}")

    # summary report
    report_lengths()

    print(f"\nAll figures written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
