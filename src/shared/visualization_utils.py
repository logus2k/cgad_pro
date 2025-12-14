"""
Visualization utilities for QUAD8 FEM

Only generates:
- potential_field_quad8_<implementation>.png
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib import colormaps


def plot_fem_field_detailed(
    x, y, quad8, u,
    title,
    output_path,
    cmap_name="viridis"
):
    """
    Detailed FEM potential field visualization for Quad-8 meshes.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    field_min, field_max = u.min(), u.max()
    norm = mcolors.Normalize(vmin=field_min, vmax=field_max)
    cmap = colormaps[cmap_name]

    for e in range(len(quad8)):
        n = quad8[e]

        # Quad-8 drawing order
        edofs = [n[0], n[4], n[1], n[5], n[2], n[6], n[3], n[7]]

        poly = Polygon(
            list(zip(x[edofs], y[edofs])),
            closed=True,
            facecolor=cmap(norm(u[edofs].mean())),
            edgecolor="k",
            linewidth=0.15
        )
        ax.add_patch(poly)

    # Axis setup
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    margin = 0.05

    ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
    ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Potential u")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_visualizations(
    x, y, quad8, u,
    output_dir,
    implementation_name="CPU"
):
    """
    Generate only the detailed Quad-8 potential field visualization.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"potential_field_quad8_{implementation_name}.png"

    plot_fem_field_detailed(
        x, y, quad8, u,
        f"FEM Potential Field ({implementation_name})",
        output_file,
        cmap_name="viridis"
    )

    return {"potential": output_file}
