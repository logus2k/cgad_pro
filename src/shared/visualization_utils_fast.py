"""
Fast Visualization utilities for QUAD8 FEM

Optimized for large meshes (100K+ elements) using matplotlib collections.
Performance: ~2 seconds for 48K elements vs 120+ seconds with patches.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib import colormaps


def plot_fem_field_fast(
    x, y, quad8, u,
    title,
    output_path,
    cmap_name="viridis",
    show_edges=True,
    dpi=300
):
    """
    Fast FEM potential field visualization using PolyCollection.
    
    Performance comparison for 48K elements:
    - Old method (Polygon loop): ~120 seconds
    - New method (PolyCollection): ~2 seconds (60x faster)
    
    Args:
        x, y: Node coordinates (1D arrays)
        quad8: Element connectivity (Nels x 8)
        u: Nodal potential values
        title: Plot title
        output_path: Save path
        cmap_name: Colormap name
        show_edges: Show element edges (disable for very large meshes)
        dpi: Image resolution
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Prepare vertex coordinates for all elements at once
    Nels = len(quad8)
    
    # Quad-8 drawing order: corners + midpoints
    # Order: 0 -> 4 -> 1 -> 5 -> 2 -> 6 -> 3 -> 7 -> back to 0
    drawing_order = [0, 4, 1, 5, 2, 6, 3, 7]
    
    # Build vertices array: (Nels, 8, 2) for all polygons
    vertices = np.zeros((Nels, 8, 2))
    for i, idx in enumerate(drawing_order):
        vertices[:, i, 0] = x[quad8[:, idx]]  # x coordinates
        vertices[:, i, 1] = y[quad8[:, idx]]  # y coordinates
    
    # Compute element-averaged potential for coloring
    u_elem = np.zeros(Nels)
    for i in range(8):
        u_elem += u[quad8[:, i]]
    u_elem /= 8.0
    
    # Create color normalization
    field_min, field_max = u.min(), u.max()
    norm = mcolors.Normalize(vmin=field_min, vmax=field_max)
    cmap = colormaps[cmap_name]
    
    # Create PolyCollection (single matplotlib object for all elements)
    # Convert to list for proper type compatibility
    vertices_list = [vertices[i] for i in range(Nels)]
    
    if show_edges:
        pc = PolyCollection(
            vertices_list,
            array=u_elem,
            cmap=cmap,
            norm=norm,
            edgecolors='black',
            linewidths=0.1
        )
    else:
        pc = PolyCollection(
            vertices_list,
            array=u_elem,
            cmap=cmap,
            norm=norm,
            edgecolors='none'
        )
    
    ax.add_collection(pc)
    
    # Axis setup
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    margin = 0.05
    
    ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
    ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(pc, ax=ax, label="Potential u")
    cbar.ax.tick_params(labelsize=10)
    
    # Save with tight layout
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return output_path


def plot_fem_field_nodal(
    x, y, quad8, u,
    title,
    output_path,
    cmap_name="viridis",
    dpi=300
):
    """
    Alternative visualization using tripcolor (interpolated nodal values).
    Even faster than PolyCollection but loses element structure.
    
    Best for: Very large meshes (>100K elements), smooth fields
    Not ideal for: Showing element boundaries, discontinuous fields
    
    Performance: <1 second for 48K elements
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert Quad8 to triangles (split each quad into 2 triangles)
    # Use corner nodes only: 0, 1, 2, 3
    triangles = []
    for elem in quad8:
        # Triangle 1: nodes 0, 1, 2
        triangles.append([elem[0], elem[1], elem[2]])
        # Triangle 2: nodes 0, 2, 3
        triangles.append([elem[0], elem[2], elem[3]])
    
    triangles = np.array(triangles)
    
    # Create triangular mesh plot
    tcf = ax.tripcolor(  # type: ignore[call-overload]
        x, y, triangles,
        u,
        cmap=cmap_name,
        shading='flat'
    )
    
    # Axis setup
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    margin = 0.05
    
    ax.set_xlim(x.min() - margin * x_range, x.max() + margin * x_range)
    ax.set_ylim(y.min() - margin * y_range, y.max() + margin * y_range)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(tcf, ax=ax, label="Potential u")
    cbar.ax.tick_params(labelsize=10)
    
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return output_path


def generate_all_visualizations(
    x, y, quad8, u,
    output_dir,
    implementation_name="CPU",
    method="fast",
    show_edges=True,
    dpi=300
):
    """
    Generate FEM potential field visualization.
    
    Args:
        x, y: Node coordinates
        quad8: Element connectivity
        u: Nodal potential values
        output_dir: Output directory
        implementation_name: Label for filename
        method: Visualization method
            - "fast": PolyCollection (recommended, ~2s for 48K elements)
            - "nodal": Tripcolor interpolation (fastest, <1s, smooth)
            - "detailed": Original patch-by-patch (slow, ~120s, most accurate)
        show_edges: Show element edges (only for "fast" method)
        dpi: Image resolution (default 300)
    
    Returns:
        dict: {"potential": output_file_path}
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"potential_field_quad8_{implementation_name}.png"
    
    if method == "fast":
        plot_fem_field_fast(
            x, y, quad8, u,
            f"FEM Potential Field ({implementation_name})",
            output_file,
            cmap_name="viridis",
            show_edges=show_edges,
            dpi=dpi
        )
    elif method == "nodal":
        plot_fem_field_nodal(
            x, y, quad8, u,
            f"FEM Potential Field ({implementation_name})",
            output_file,
            cmap_name="viridis",
            dpi=dpi
        )
    elif method == "detailed":
        # Fall back to original slow method
        plot_fem_field_detailed(
            x, y, quad8, u,
            f"FEM Potential Field ({implementation_name})",
            output_file,
            cmap_name="viridis"
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'fast', 'nodal', or 'detailed'")
    
    return {"potential": output_file}


def plot_fem_field_detailed(
    x, y, quad8, u,
    title,
    output_path,
    cmap_name="viridis"
):
    """
    Original detailed visualization (SLOW - kept for compatibility).
    Use plot_fem_field_fast() instead for large meshes.
    
    WARNING: Takes ~120 seconds for 48K elements.
    """
    from matplotlib.patches import Polygon
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    field_min, field_max = u.min(), u.max()
    norm = mcolors.Normalize(vmin=field_min, vmax=field_max)
    cmap = colormaps[cmap_name]
    
    for e in range(len(quad8)):
        n = quad8[e]
        edofs = [n[0], n[4], n[1], n[5], n[2], n[6], n[3], n[7]]
        
        poly = Polygon(
            list(zip(x[edofs], y[edofs])),
            closed=True,
            facecolor=cmap(norm(u[edofs].mean())),
            edgecolor="k",
            linewidth=0.15
        )
        ax.add_patch(poly)
    
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


# Convenience function for benchmarking
def benchmark_visualization_methods(x, y, quad8, u, output_dir):
    """
    Benchmark all visualization methods and compare performance.
    
    Returns:
        dict: Timing results for each method
    """
    import time
    
    results = {}
    
    print("Benchmarking visualization methods...")
    print(f"Mesh size: {len(quad8)} elements, {len(x)} nodes\n")
    
    # Method 1: Fast (PolyCollection)
    print("Testing 'fast' method (PolyCollection)...")
    t0 = time.perf_counter()
    generate_all_visualizations(
        x, y, quad8, u, output_dir,
        implementation_name="benchmark_fast",
        method="fast"
    )
    t1 = time.perf_counter()
    results['fast'] = t1 - t0
    print(f"  Time: {results['fast']:.2f}s\n")
    
    # Method 2: Nodal (Tripcolor)
    print("Testing 'nodal' method (Tripcolor)...")
    t0 = time.perf_counter()
    generate_all_visualizations(
        x, y, quad8, u, output_dir,
        implementation_name="benchmark_nodal",
        method="nodal"
    )
    t1 = time.perf_counter()
    results['nodal'] = t1 - t0
    print(f"  Time: {results['nodal']:.2f}s\n")
    
    # Method 3: Detailed (skip for large meshes)
    if len(quad8) < 10000:
        print("Testing 'detailed' method (Patch loop)...")
        t0 = time.perf_counter()
        generate_all_visualizations(
            x, y, quad8, u, output_dir,
            implementation_name="benchmark_detailed",
            method="detailed"
        )
        t1 = time.perf_counter()
        results['detailed'] = t1 - t0
        print(f"  Time: {results['detailed']:.2f}s\n")
    else:
        print("Skipping 'detailed' method (mesh too large)\n")
        results['detailed'] = None
    
    # Summary
    print("="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    for method, time_val in results.items():
        if time_val is not None:
            print(f"{method:12s}: {time_val:6.2f}s")
        else:
            print(f"{method:12s}: (skipped)")
    
    if results['detailed'] is not None:
        speedup = results['detailed'] / results['fast']
        print(f"\nSpeedup (fast vs detailed): {speedup:.1f}x")
    
    return results
