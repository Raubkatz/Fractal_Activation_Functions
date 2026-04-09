import math
import os

import matplotlib.pyplot as plt


# ============================================================
# COLOR SETTINGS (edit these three hex colors as you want)
# ============================================================
BG_COLOR = "#FFFFFF"      # background
LINE_COLOR = "#325066"    # Koch curve color
ACCENT_COLOR = "#000000"  # text / frame / optional accent


# ============================================================
# PLOT SETTINGS
# ============================================================
ITERATIONS = 5                 # recursion depth; 4 or 5 is good for presentations
LINE_WIDTH = 2.5
FIGSIZE = (12, 4)
DPI = 160
TRANSPARENT_BG = True          # True = transparent background

OUTPUT_ROOT = "koch_curve_outputs"


# ============================================================
# HELPERS
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
# KOCH CURVE GEOMETRY
# ============================================================
def interpolate(p1, p2, t):
    """Linear interpolation between two 2D points."""
    return (p1[0] + (p2[0] - p1[0]) * t,
            p1[1] + (p2[1] - p1[1]) * t)


def koch_subdivide(p1, p2):
    """
    Replace one line segment by the 4 Koch segments.
    Returns 5 points: A, B, C, D, E
    where segment A->E becomes A->B->C->D->E.
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    # points at 1/3 and 2/3
    b = (x1 + dx / 3.0, y1 + dy / 3.0)
    d = (x1 + 2.0 * dx / 3.0, y1 + 2.0 * dy / 3.0)

    # rotate vector (d - b) by +60° around b to get peak c
    angle = math.radians(60)
    vx = d[0] - b[0]
    vy = d[1] - b[1]

    cx = b[0] + vx * math.cos(angle) - vy * math.sin(angle)
    cy = b[1] + vx * math.sin(angle) + vy * math.cos(angle)
    c = (cx, cy)

    return [p1, b, c, d, p2]


def refine_polyline(points):
    """Apply one Koch refinement step to the whole polyline."""
    new_points = []
    for i in range(len(points) - 1):
        sub = koch_subdivide(points[i], points[i + 1])
        if i == 0:
            new_points.extend(sub)
        else:
            new_points.extend(sub[1:])  # avoid duplicate point
    return new_points


def build_levels(iterations):
    """
    Create list of polyline levels.
    level 0 = straight line
    level n = Koch refinement of previous level
    """
    levels = []
    base = [(0.0, 0.0), (1.0, 0.0)]
    levels.append(base)

    current = base
    for _ in range(iterations):
        current = refine_polyline(current)
        levels.append(current)

    return levels


# ============================================================
# PLOTTING
# ============================================================
def get_bounds(levels, pad=0.06):
    xs = []
    ys = []
    for level in levels:
        for x, y in level:
            xs.append(x)
            ys.append(y)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    dx = x_max - x_min
    dy = y_max - y_min

    return (
        x_min - pad * dx,
        x_max + pad * dx,
        y_min - pad * max(dx, dy),
        y_max + pad * max(dx, dy),
    )


def save_iteration_plot(points, level_idx, total_levels, bounds, png_outpath, eps_outpath):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    ax.plot(
        xs,
        ys,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        solid_joinstyle="round",
        solid_capstyle="round",
    )

    #ax.text(
    #    0.02,
    #    0.93,
    #    f"Koch Curve Construction  |  Iteration {level_idx}/{total_levels}",
    #    transform=ax.transAxes,
    #    fontsize=16,
    #    color=ACCENT_COLOR,
    #    ha="left",
    #    va="top",
    #    fontweight="bold",
    #)

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.axis("off")

    for spine in ax.spines.values():
        spine.set_visible(False)

    # PNG with transparency
    plt.savefig(
        png_outpath,
        format="png",
        dpi=DPI,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=TRANSPARENT_BG
    )

    # EPS without true transparency
    plt.savefig(
        eps_outpath,
        format="eps",
        dpi=DPI,
        facecolor="white" if TRANSPARENT_BG else fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
        transparent=False
    )

    plt.close(fig)


# ============================================================
# MAIN ITERATION PLOT CREATION
# ============================================================
def create_koch_iteration_plots(
    output_root=OUTPUT_ROOT,
    iterations=ITERATIONS
):
    os.makedirs(output_root, exist_ok=True)

    plots_dir = os.path.join(output_root, "iterations")
    plots_png_dir = os.path.join(plots_dir, "png")
    plots_eps_dir = os.path.join(plots_dir, "eps")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(plots_png_dir, exist_ok=True)
    os.makedirs(plots_eps_dir, exist_ok=True)

    levels = build_levels(iterations)
    bounds = get_bounds(levels)

    for level_idx, level_points in enumerate(levels):
        save_iteration_plot(
            level_points,
            level_idx,
            iterations,
            bounds,
            os.path.join(plots_png_dir, f"koch_curve_iteration_{level_idx:03d}.png"),
            os.path.join(plots_eps_dir, f"koch_curve_iteration_{level_idx:03d}.eps"),
        )

    print(f"PNG iterations saved to: {os.path.abspath(plots_png_dir)}")
    print(f"EPS iterations saved to: {os.path.abspath(plots_eps_dir)}")


if __name__ == "__main__":
    create_koch_iteration_plots()