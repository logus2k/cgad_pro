"""
Visualization utilities for QUAD8 FEM

Provides reusable plotting functions for FEM results visualization.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import colormaps


def plot_fem_scalar_2d(
    x, y, quad8, field,
    title,
    output_path,
    cmap_name="RdBu_r"
):
    """
    2D FEM scalar visualization for Quad-8 meshes using PatchCollection.
    
    Args:
        x: Node x-coordinates (1D array)
        y: Node y-coordinates (1D array)
        quad8: Element connectivity (Nels × 8 array)
        field: Scalar field to visualize (nodal or element-based)
        title: Plot title
        output_path: Path to save the figure
        cmap_name: Colormap name (default: "RdBu_r")
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    patches = []
    values  = []

    vmin = field.min()
    vmax = field.max()

    for e in range(len(quad8)):
        n = quad8[e]

        # Quad-8 drawing order (corners + mid-edges)
        edofs = [
            n[0], n[4],
            n[1], n[5],
            n[2], n[6],
            n[3], n[7]
        ]

        poly_xy = np.column_stack((x[edofs], y[edofs]))
        poly = Polygon(poly_xy, closed=True)

        patches.append(poly)

        # Handle both element-based and nodal fields
        if field.size == len(quad8):
            values.append(field[e])
        else:
            values.append(field[edofs].mean())

    collection = PatchCollection(
        patches,
        cmap=cmap_name,
        edgecolor="k",
        linewidth=0.2
    )

    collection.set_array(np.array(values))
    collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    cbar = fig.colorbar(collection, ax=ax)
    cbar.set_label("Value")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_fem_field_detailed(
    x, y, quad8, field,
    title,
    output_path,
    cmap_name="viridis"
):
    """
    Detailed FEM field visualization with individual polygon rendering.
    
    This version renders each element individually for better control
    over element appearance and compatibility with complex geometries.
    
    Args:
        x: Node x-coordinates (1D array)
        y: Node y-coordinates (1D array)
        quad8: Element connectivity (Nels × 8 array)
        field: Scalar field to visualize (nodal values)
        title: Plot title
        output_path: Path to save the figure
        cmap_name: Colormap name (default: "viridis")
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    field_min, field_max = field.min(), field.max()
    norm = mcolors.Normalize(vmin=field_min, vmax=field_max)
    cmap = colormaps[cmap_name]

    # Render each element individually
    for e in range(len(quad8)):
        n = quad8[e]
        
        # Quad-8 drawing order
        edofs = [n[0], n[4], n[1], n[5], n[2], n[6], n[3], n[7]]
        
        poly = Polygon(
            list(zip(x[edofs], y[edofs])),
            closed=True,
            facecolor=cmap(norm(field[edofs].mean())),
            edgecolor="k",
            linewidth=0.15
        )
        ax.add_patch(poly)

    # Set axis limits with margin
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    margin = 0.05

    ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
    ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    # Add colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Potential u")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_visualizations(
    x, y, quad8, u, abs_vel, pressure,
    output_dir,
    implementation_name="CPU"
):
    """
    Generate complete set of FEM visualizations.
    
    Args:
        x: Node x-coordinates
        y: Node y-coordinates
        quad8: Element connectivity
        u: Velocity potential (nodal values)
        abs_vel: Velocity magnitude (element values)
        pressure: Pressure field (element values)
        output_dir: Directory to save figures
        implementation_name: Name to add to titles (e.g., "CPU", "GPU", "Numba")
    
    Returns:
        Dictionary of output file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = {}
    
    # 1. Potential field (PatchCollection)
    output_files['potential'] = output_dir / f"potential_u_2D_{implementation_name}.png"
    plot_fem_scalar_2d(
        x, y, quad8, u,
        f"FEM Potential Field ({implementation_name})",
        output_files['potential'],
        cmap_name="RdBu_r"
    )
    
    # 2. Velocity magnitude
    output_files['velocity'] = output_dir / f"abs_velocity_2D_{implementation_name}.png"
    plot_fem_scalar_2d(
        x, y, quad8, abs_vel,
        f"Velocity Magnitude |V| ({implementation_name})",
        output_files['velocity'],
        cmap_name="RdBu_r"
    )
    
    # 3. Pressure field
    output_files['pressure'] = output_dir / f"pressure_2D_{implementation_name}.png"
    plot_fem_scalar_2d(
        x, y, quad8, pressure,
        f"Pressure Field ({implementation_name})",
        output_files['pressure'],
        cmap_name="RdBu_r"
    )
    
    # 4. Main detailed visualization (viridis)
    output_files['main'] = output_dir / f"potential_field_quad8_{implementation_name}.png"
    plot_fem_field_detailed(
        x, y, quad8, u,
        f"FEM Potential Field ({implementation_name})",
        output_files['main'],
        cmap_name="viridis"
    )
    
    return output_files
