#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot every activation (classical + fractal) on [-2, 2].

Creates
    activation_all.(png|eps)          – all 12 curves
    activation_group1.(png|eps)       – first  ⌈n/2⌉ curves
    activation_group2.(png|eps)       – last   ⌊n/2⌋ curves
    activation_third1.(png|eps)       – 4 mixed curves
    activation_third2.(png|eps)       – 4 mixed curves
    activation_third3.(png|eps)       – 4 mixed curves
"""

# --------------------------------------------------------------------------- #
import time, pathlib, numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import fractal_activation_functions as fractal

# --- Tailwind colour set --------------------------------------------------- #
PALETTE = [
    "#a2faa3",  # Light green
    "#92c9b1",  # Cambridge blue
    "#4f759b",  # UCLA blue
    "#5d5179",  # Ultra-violet
    "#571f4e",  # Palatinate
]
LINE_STYLES = ["-", "--", "-.", ":"]

# --- activation registry ---------------------------------------------------- #
ACTIVATIONS = [
    ("weierstrass",                        fractal.weierstrass_function_tf),
    ("weierstrass_\nmandelbrot_\nxpsin",       fractal.weierstrass_mandelbrot_function_xpsin),
    ("weierstrass_\nmandelbrot_\nxsinsquared", fractal.weierstrass_mandelbrot_function_xsinsquared),
    ("weierstrass_\nmandelbrot_\nrelupsin",    fractal.weierstrass_mandelbrot_function_relupsin),
    ("weierstrass_\nmandelbrot_\ntanhpsin",    fractal.weierstrass_mandelbrot_function_tanhpsin),
    ("blancmange",                         fractal.modulated_blancmange_curve),
    ("decaying_cosine",                    fractal.decaying_cosine_function_tf),
    ("modified_\nweierstrass_\ntanh",          fractal.modified_weierstrass_function_tanh),
    ("modified_\nweierstrass_\nReLU",          fractal.modified_weierstrass_function_relu),
    ("relu",                               "relu"),
    ("sigmoid",                            "sigmoid"),
    ("tanh",                               "tanh"),
]

# hand-mixed order so every panel blends classical & fractal
MIXED_IDXS = [
     9,  0,  5, 10,   # relu, weierstrass, blancmange, sigmoid
     2,  6, 11,  7,   # xsinsq, dec_cos, tanh, mod_w_tanh
     1,  8,  4,  3    # xpsin, mod_w_ReLU, tanhpsin, relupsin
]

# --------------------------------------------------------------------------- #
def _eval(act, x):
    x_tf = tf.convert_to_tensor(x, tf.float32)
    return (tf.keras.activations.get(act)(x_tf) if isinstance(act, str) else act(x_tf)).numpy()

def _plot(idx_list, suffix, out_dir):
    x = np.linspace(-2., 2., 1_000, dtype=np.float32)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    for k, idx in enumerate(idx_list):
        name, act = ACTIVATIONS[idx]
        y     = _eval(act, x)
        color = PALETTE[k % len(PALETTE)]
        ls    = LINE_STYLES[(k // len(PALETTE)) % len(LINE_STYLES)]
        #ax.plot(x, y, ls, color=color, lw=1.0, label=name)
        ax.plot(x, y, ls, color=color, lw=1.25, label=name)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\sigma(x)$")
    #ax.set_title("Activation functions")
    #ax.legend(fontsize=11, ncol=2, loc="upper left")
    fig.tight_layout()

    for ext in ("png", "eps"):
        fig.savefig(out_dir / f"activation_{suffix}.{ext}",
                    dpi=300, bbox_inches="tight", format=ext)
    plt.close(fig)

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    out_dir = pathlib.Path(f"activation_plots_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n    = len(ACTIVATIONS)       # 12
    half = (n + 1) // 2           # 6 for group1 / 6 for group2

    # full list + simple halves (original order)
    _plot(list(range(n)),       "all",    out_dir)
    _plot(list(range(half)),    "group1", out_dir)
    _plot(list(range(half, n)), "group2", out_dir)

    # three equal-sized mixed panels (4 / 4 / 4)
    _plot(MIXED_IDXS[0:4],   "third1", out_dir)
    _plot(MIXED_IDXS[4:8],   "third2", out_dir)
    _plot(MIXED_IDXS[8:12],  "third3", out_dir)

    print("Saved figures in", out_dir.resolve())
