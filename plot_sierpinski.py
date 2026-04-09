import math
import os

import matplotlib.pyplot as plt


# ============================================================
# COLOR SETTINGS (edit these three hex colors as you want)
# ============================================================
BG_COLOR = "#FFFFFF"      # background
LINE_COLOR = "#325066"    # Sierpinski triangle color
ACCENT_COLOR = "#000000"  # text / frame / optional accent


# ============================================================
# PLOT SETTINGS
# ============================================================
ITERATIONS = 5                 # recursion depth; 4 or 5 is good for presentations
LINE_WIDTH = 1.5
FIGSIZE = (8, 8)
DPI = 160
TRANSPARENT_BG = True          # True = transparent background

OUTPUT_ROOT = "sierpinski_triangle_outputs"


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
# SIERPINSKI TRIANGLE GEOMETRY
# ============================================================
def interpolate(p1, p2, t):
    """Linear interpolation between two 2D points."""
    return (
        p1[0] + (p2[0] - p1[0]) * t,
        p1[1] + (p2[1] - p1[1]) * t
    )


def midpoint(p1, p2):
    """Midpoint between two 2D points."""
    return (
        0.5 * (p1[0] + p2[0]),
        0.5 * (p1[1] + p2[1])
    )


def sierpinski_subdivide(triangle):
    """
    Replace one triangle by the 3 corner triangles of the
    Sierpinski construction.
    """
    a, b, c = triangle
    ab = midpoint(a, b)
    bc = midpoint(b, c)
    ca = midpoint(c, a)

    t1 = (a, ab, ca)
    t2 = (ab, b, bc)
    t3 = (ca, bc, c)

    return [t1, t2, t3]


def refine_triangles(triangles):
    """Apply one Sierpinski refinement step to the whole set."""
    new_triangles = []
    for triangle in triangles:
        new_triangles.extend(sierpinski_subdivide(triangle))
    return new_triangles


def build_levels(iterations):
    """
    Create list of triangle sets.
    level 0 = one solid triangle
    level n = Sierpinski refinement of previous level
    """
    levels = []

    h = math.sqrt(3) / 2.0
    base_triangle = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.5, h)
    )

    current = [base_triangle]
    levels.append(current)

    for _ in range(iterations):
        current = refine_triangles(current)
        levels.append(current)

    return levels


# ============================================================
# PLOTTING
# ============================================================
def get_bounds(levels, pad=0.06):
    xs = []
    ys = []
    for level in levels:
        for triangle in level:
            for x, y in triangle:
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


def save_iteration_plot(triangles, level_idx, total_levels, bounds, png_outpath, eps_outpath):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0.0)
        ax.set_facecolor("none")
    else:
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

    for triangle in triangles:
        xs = [triangle[0][0], triangle[1][0], triangle[2][0], triangle[0][0]]
        ys = [triangle[0][1], triangle[1][1], triangle[2][1], triangle[0][1]]

        ax.fill(xs, ys, color=LINE_COLOR, linewidth=0)
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
    #    f"Sierpinski Triangle Construction  |  Iteration {level_idx}/{total_levels}",
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
def create_sierpinski_iteration_plots(
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

    for level_idx, level_triangles in enumerate(levels):
        save_iteration_plot(
            level_triangles,
            level_idx,
            iterations,
            bounds,
            os.path.join(plots_png_dir, f"sierpinski_triangle_iteration_{level_idx:03d}.png"),
            os.path.join(plots_eps_dir, f"sierpinski_triangle_iteration_{level_idx:03d}.eps"),
        )

    print(f"PNG iterations saved to: {os.path.abspath(plots_png_dir)}")
    print(f"EPS iterations saved to: {os.path.abspath(plots_eps_dir)}")


if __name__ == "__main__":
    create_sierpinski_iteration_plots()