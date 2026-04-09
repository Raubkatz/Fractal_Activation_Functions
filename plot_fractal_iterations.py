import os
import math

import numpy as np
import tensorflow as tf
from tensorflow.experimental import numpy as tnp
import matplotlib.pyplot as plt


# ============================================================
# IMPORT YOUR FRACTAL ACTIVATION FUNCTION COLLECTION
# ============================================================
from fractal_activation_functions import (
    modulated_blancmange_curve,
    decaying_cosine_function_tf,
    modified_weierstrass_function_tanh,
    modified_weierstrass_function_relu,
    weierstrass_mandelbrot_function_xsinsquared,
    weierstrass_mandelbrot_function_xpsin,
    weierstrass_mandelbrot_function_relupsin,
    weierstrass_mandelbrot_function_tanhpsin,
    weierstrass_function_tf,
)


# ============================================================
# BASIC / CLASSICAL FRACTAL FUNCTIONS NOT YET IN COLLECTION
# ============================================================
def basic_blancmange_function_tf(x, num_terms=12):
    """
    Classical Blancmange / Takagi function:
        T(x) = sum_{n=0}^{N-1} 2^{-n} * phi(2^n x)
    where phi(u) = distance(u, nearest integer)
                 = abs(u - round(u))
    """
    x = tf.cast(x, tf.float64)
    y = tf.zeros_like(x, dtype=tf.float64)

    for n in range(num_terms):
        scale = 2.0 ** n
        u = scale * x
        phi = tf.abs(u - tf.round(u))
        y += phi / scale

    return tf.cast(y, tf.float32)


def basic_weierstrass_function_tf(x, a=0.5, b=7, num_terms=20):
    """
    Classical Weierstrass-type cosine series:
        W(x) = sum_{n=0}^{N-1} a^n cos(b^n pi x)
    """
    x = tf.cast(x, tf.float64)
    y = tf.zeros_like(x, dtype=tf.float64)

    for n in range(num_terms):
        y += (a ** n) * tf.cos((b ** n) * tnp.pi * x)

    return tf.cast(y, tf.float32)


# ============================================================
# REGULAR ACTIVATION FUNCTIONS
# ============================================================
def identity_tf(x):
    x = tf.cast(x, tf.float32)
    return x


def relu_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.nn.relu(x)


def leaky_relu_tf(x, alpha=0.2):
    x = tf.cast(x, tf.float32)
    return tf.nn.leaky_relu(x, alpha=alpha)


def sigmoid_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.nn.sigmoid(x)


def tanh_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.nn.tanh(x)


def elu_tf(x, alpha=1.0):
    x = tf.cast(x, tf.float32)
    return tf.nn.elu(x)


def selu_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.nn.selu(x)


def softplus_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.nn.softplus(x)


def swish_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.nn.swish(x)


def gelu_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.nn.gelu(x)


# ============================================================
# COLOR SETTINGS
# ============================================================
BG_COLOR = "#FFFFFF"      # background
LINE_COLOR = "#325066"    # curve color
ACCENT_COLOR = "#000000"  # text / frame / optional accent


# ============================================================
# PLOT SETTINGS
# ============================================================
ITERATIONS = 5                # maximum depth / number of partial approximations
LINE_WIDTH = 2.0
FIGSIZE = (12, 5)
DPI = 160
TRANSPARENT_BG = True          # True = transparent background where supported

# x-domain for plotting
X_MIN = -2.0
X_MAX = 2.0
NUM_X = 3000

# axis labels
X_AXIS_LABEL = "x"
Y_AXIS_LABEL = "f(x)"

# Output root folder
OUTPUT_ROOT = "fractal_activation_plots"
REGULAR_OUTPUT_ROOT = "regular_activation_plots"


# ============================================================
# HELPER: SAFE FOLDER / FILE NAME
# ============================================================
def slugify(text):
    keep = []
    for ch in text.lower():
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch in (" ", "/", "\\", ".", ":", ",", "(", ")"):
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ============================================================
# ZERO FUNCTION HELPER
# Used to create true depth-0 plots.
# ============================================================
def zero_function_tf(x):
    x = tf.cast(x, tf.float32)
    return tf.zeros_like(x, dtype=tf.float32)


# ============================================================
# FUNCTION WRAPPERS
# Each entry:
#   - name
#   - callable builder(depth) -> function(x_tensor)
#   - title
# ============================================================
FUNCTION_SPECS = [
    {
        "name": "basic_blancmange_function",
        "title": "Basic Blancmange Function",
        "builder": lambda depth: (
            lambda x: basic_blancmange_function_tf(x, num_terms=depth)
        ),
    },
    {
        "name": "modulated_blancmange_curve",
        "title": "Modulated Blancmange Curve",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: modulated_blancmange_curve(x, n_terms=depth, a=0.75))
        ),
    },
    {
        "name": "basic_weierstrass_function",
        "title": "Basic Weierstrass Function",
        "builder": lambda depth: (
            lambda x: basic_weierstrass_function_tf(x, a=0.5, b=7, num_terms=depth)
        ),
    },
    {
        "name": "weierstrass_function_tf",
        "title": "Weierstrass Function TF",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: weierstrass_function_tf(x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)))
        ),
    },
    {
        "name": "modified_weierstrass_function_tanh",
        "title": "Modified Weierstrass Function Tanh",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: modified_weierstrass_function_tanh(x, a=0.5, b=3, n_terms=depth))
        ),
    },
    {
        "name": "modified_weierstrass_function_relu",
        "title": "Modified Weierstrass Function ReLU",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: modified_weierstrass_function_relu(x, a=0.5, b=3, n_terms=depth))
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_xsinsquared",
        "title": "Weierstrass-Mandelbrot x*sin^2",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: weierstrass_mandelbrot_function_xsinsquared(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            ))
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_xpsin",
        "title": "Weierstrass-Mandelbrot x+sin",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: weierstrass_mandelbrot_function_xpsin(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            ))
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_relupsin",
        "title": "Weierstrass-Mandelbrot relu+sin",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: weierstrass_mandelbrot_function_relupsin(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            ))
        ),
    },
    {
        "name": "weierstrass_mandelbrot_function_tanhpsin",
        "title": "Weierstrass-Mandelbrot tanh+sin",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: weierstrass_mandelbrot_function_tanhpsin(
                x, gamma=0.5, lambda_val=2, num_terms=max(depth, 2)
            ))
        ),
    },
    {
        "name": "decaying_cosine_function_tf",
        "title": "Decaying Cosine Function TF",
        "builder": lambda depth: (
            (lambda x: zero_function_tf(x))
            if depth == 0 else
            (lambda x: decaying_cosine_function_tf(
                x, a=0.5, b=3, c=0.5, d=2, n_terms=max(depth, 2), zeta=0.2666
            ))
        ),
    },
]


# ============================================================
# REGULAR FUNCTION WRAPPERS
# ============================================================
REGULAR_FUNCTION_SPECS = [
    {
        "name": "identity",
        "title": "Identity",
        "fn": identity_tf,
    },
    {
        "name": "relu",
        "title": "ReLU",
        "fn": relu_tf,
    },
    {
        "name": "leaky_relu",
        "title": "Leaky ReLU",
        "fn": lambda x: leaky_relu_tf(x, alpha=0.2),
    },
    {
        "name": "sigmoid",
        "title": "Sigmoid",
        "fn": sigmoid_tf,
    },
    {
        "name": "tanh",
        "title": "Tanh",
        "fn": tanh_tf,
    },
    {
        "name": "elu",
        "title": "ELU",
        "fn": elu_tf,
    },
    {
        "name": "selu",
        "title": "SELU",
        "fn": selu_tf,
    },
    {
        "name": "softplus",
        "title": "Softplus",
        "fn": softplus_tf,
    },
    {
        "name": "swish",
        "title": "Swish",
        "fn": swish_tf,
    },
    {
        "name": "gelu",
        "title": "GELU",
        "fn": gelu_tf,
    },
]


# ============================================================
# SAMPLING
# ============================================================
def make_x_tensor(x_min=X_MIN, x_max=X_MAX, num_x=NUM_X):
    x_np = np.linspace(x_min, x_max, num_x, dtype=np.float32)
    x_tf = tf.constant(x_np, dtype=tf.float32)
    return x_np, x_tf


# ============================================================
# BUILD LEVELS
# Each level is a partial approximation with increasing depth.
# Includes depth 0.
# ============================================================
def build_levels(function_builder, iterations):
    x_np, x_tf = make_x_tensor()
    levels = []

    for depth in range(0, iterations + 1):
        fn = function_builder(depth)
        y_tf = fn(x_tf)
        y_np = np.asarray(y_tf.numpy(), dtype=np.float32)
        levels.append(
            {
                "depth": depth,
                "x": x_np.copy(),
                "y": y_np.copy(),
            }
        )

    return levels


# ============================================================
# BUILD REGULAR FUNCTION VALUES
# ============================================================
def build_regular_level(function_fn):
    x_np, x_tf = make_x_tensor()
    y_tf = function_fn(x_tf)
    y_np = np.asarray(y_tf.numpy(), dtype=np.float32)
    return {
        "x": x_np.copy(),
        "y": y_np.copy(),
    }


# ============================================================
# BOUNDS
# For fractal variants, axis bounds are taken from all depths
# so depth 0 is included in the global view.
# ============================================================
def get_bounds_from_levels(levels, pad_x=0.03, pad_y=0.08):
    xs = []
    ys = []

    for level in levels:
        xs.extend(level["x"].tolist())
        ys.extend(level["y"].tolist())

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    dx = x_max - x_min if x_max > x_min else 1.0
    dy = y_max - y_min if y_max > y_min else 1.0

    return (
        x_min - pad_x * dx,
        x_max + pad_x * dx,
        y_min - pad_y * dy,
        y_max + pad_y * dy,
    )


def get_bounds_from_xy(x, y, pad_x=0.03, pad_y=0.08):
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))

    dx = x_max - x_min if x_max > x_min else 1.0
    dy = y_max - y_min if y_max > y_min else 1.0

    return (
        x_min - pad_x * dx,
        x_max + pad_x * dx,
        y_min - pad_y * dy,
        y_max + pad_y * dy,
    )


# ============================================================
# AXIS STYLING
# ============================================================
def style_axes(ax, bounds):
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ACCENT_COLOR)
    ax.spines["bottom"].set_color(ACCENT_COLOR)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        colors=ACCENT_COLOR,
        labelsize=10,
        width=1.0,
    )

    ax.set_xlabel(X_AXIS_LABEL, fontsize=12, color=ACCENT_COLOR)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=12, color=ACCENT_COLOR)

    ax.grid(False)


# ============================================================
# SAVE STATIC LEVEL PLOTS
# Saves BOTH PNG and EPS for each full realization / depth
# ============================================================
def save_level_plot(x, y, depth_idx, total_depths, bounds, title, png_outpath, eps_outpath):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    ax.plot(
        x,
        y,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
    )

    #ax.text(
    #    0.02,
    #    0.94,
    #    f"{title}  |  Depth {depth_idx}/{total_depths}",
    #    transform=ax.transAxes,
    #    fontsize=16,
    #    color=ACCENT_COLOR,
    #    ha="left",
    #    va="top",
    #    fontweight="bold",
    #)

    style_axes(ax, bounds)

    plt.savefig(
        png_outpath,
        format="png",
        dpi=DPI,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=TRANSPARENT_BG,
    )

    plt.savefig(
        eps_outpath,
        format="eps",
        dpi=DPI,
        facecolor="white" if TRANSPARENT_BG else fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=False,
    )

    plt.close(fig)


# ============================================================
# SAVE STATIC REGULAR ACTIVATION PLOTS
# ============================================================
def save_regular_plot(x, y, bounds, title, png_outpath, eps_outpath):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    ax.plot(
        x,
        y,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
    )

    #ax.text(
    #    0.02,
    #    0.94,
    #    title,
    #    transform=ax.transAxes,
    #    fontsize=16,
    #    color=ACCENT_COLOR,
    #    ha="left",
    #    va="top",
    #    fontweight="bold",
    #)

    style_axes(ax, bounds)

    plt.savefig(
        png_outpath,
        format="png",
        dpi=DPI,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=TRANSPARENT_BG,
    )

    plt.savefig(
        eps_outpath,
        format="eps",
        dpi=DPI,
        facecolor="white" if TRANSPARENT_BG else fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=False,
    )

    plt.close(fig)


# ============================================================
# CREATE STATIC DEPTH PLOTS FOR ONE FRACTAL FUNCTION
# Saves:
#   plots/png/*.png
#   plots/eps/*.eps
# Includes depth 0.
# ============================================================
def create_function_plots(function_spec):
    function_name = function_spec["name"]
    title = function_spec["title"]
    builder = function_spec["builder"]

    function_slug = slugify(function_name)
    function_dir = os.path.join(OUTPUT_ROOT, function_slug)

    plots_dir = os.path.join(function_dir, "plots")
    plots_png_dir = os.path.join(plots_dir, "png")
    plots_eps_dir = os.path.join(plots_dir, "eps")

    os.makedirs(function_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plots_png_dir, exist_ok=True)
    os.makedirs(plots_eps_dir, exist_ok=True)

    levels = build_levels(builder, ITERATIONS)
    bounds = get_bounds_from_levels(levels)

    for level in levels:
        depth_idx = level["depth"]
        x_full = level["x"]
        y_full = level["y"]

        save_level_plot(
            x_full,
            y_full,
            depth_idx,
            ITERATIONS,
            bounds,
            title,
            os.path.join(plots_png_dir, f"{function_slug}_depth_{depth_idx:03d}.png"),
            os.path.join(plots_eps_dir, f"{function_slug}_depth_{depth_idx:03d}.eps"),
        )

    print(f"[DONE] {title}")
    print(f"  PNG plots: {os.path.abspath(plots_png_dir)}")
    print(f"  EPS plots: {os.path.abspath(plots_eps_dir)}")


# ============================================================
# CREATE STATIC PLOTS FOR REGULAR ACTIVATION FUNCTIONS
# ============================================================
def create_regular_activation_plots():
    os.makedirs(REGULAR_OUTPUT_ROOT, exist_ok=True)

    for function_spec in REGULAR_FUNCTION_SPECS:
        function_name = function_spec["name"]
        title = function_spec["title"]
        fn = function_spec["fn"]

        function_slug = slugify(function_name)
        function_dir = os.path.join(REGULAR_OUTPUT_ROOT, function_slug)
        plots_dir = os.path.join(function_dir, "plots")
        plots_png_dir = os.path.join(plots_dir, "png")
        plots_eps_dir = os.path.join(plots_dir, "eps")

        os.makedirs(function_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(plots_png_dir, exist_ok=True)
        os.makedirs(plots_eps_dir, exist_ok=True)

        level = build_regular_level(fn)
        bounds = get_bounds_from_xy(level["x"], level["y"])

        save_regular_plot(
            level["x"],
            level["y"],
            bounds,
            title,
            os.path.join(plots_png_dir, f"{function_slug}.png"),
            os.path.join(plots_eps_dir, f"{function_slug}.eps"),
        )

        print(f"[DONE] {title}")
        print(f"  PNG plot: {os.path.abspath(os.path.join(plots_png_dir, f'{function_slug}.png'))}")
        print(f"  EPS plot: {os.path.abspath(os.path.join(plots_eps_dir, f'{function_slug}.eps'))}")


# ============================================================
# MAIN
# ============================================================
def create_all_fractal_activation_plots():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for function_spec in FUNCTION_SPECS:
        create_function_plots(function_spec)


if __name__ == "__main__":
    create_all_fractal_activation_plots()
    create_regular_activation_plots()