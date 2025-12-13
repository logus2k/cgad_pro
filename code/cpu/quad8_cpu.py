"""
QUAD8 FEM – CPU baseline (NumPy + SciPy)

Key properties:
- Dynamic mesh size (any Quad-8 Excel mesh)
- LIL assembly → CSR once
- Robin BCs assembled BEFORE Dirichlet
- Dirichlet enforced by elimination
- DIRECT SOLVER (for testing)
- Iteration progress printed
"""

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import colormaps

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from elem_quad8_cpu import Elem_Quad8
from robin_quadr_cpu import Robin_quadr
from genip2dq_cpu import Genip2DQ
from shape_n_der8_cpu import Shape_N_Der8

# -------------------------------------------------
# Physics parameters
# -------------------------------------------------
P0   = 101328.8281
RHO  = 0.6125
GAMA = 2.5

# -------------------------------------------------
# Input
# -------------------------------------------------
HERE = Path(__file__).resolve().parent.parent.parent
MESH_FILE = HERE / "data/input/converted_mesh_v5.xlsx"

coord = pd.read_excel(MESH_FILE, sheet_name="coord")
conec = pd.read_excel(MESH_FILE, sheet_name="conec")

x = coord["X"].to_numpy(dtype=float) / 1000.0
y = coord["Y"].to_numpy(dtype=float) / 1000.0
quad8 = conec.iloc[:, :8].to_numpy(dtype=int) - 1

Nnds = x.size
Nels = quad8.shape[0]

print(f"Loaded mesh: {Nnds} nodes, {Nels} Quad-8 elements")

# -------------------------------------------------
# Global system (use lil_matrix for dynamic assembly)
# -------------------------------------------------
Kg = lil_matrix((Nnds, Nnds), dtype=np.float64)
fg = np.zeros(Nnds, dtype=np.float64)

# -------------------------------------------------
# Element assembly (fast, using lil_matrix)
# -------------------------------------------------
for e in range(Nels):
    edofs = quad8[e]
    XN = np.column_stack((x[edofs], y[edofs]))
    Ke, fe = Elem_Quad8(XN, fL=0.0)

    for i in range(8):
        fg[edofs[i]] += fe[i]
        for j in range(8):
            Kg[edofs[i], edofs[j]] += Ke[i, j]

    if (e + 1) % 10000 == 0:
        print(f"  Assembled {e + 1}/{Nels} elements")

# -------------------------------------------------
# Apply Robin boundary conditions (still in lil_matrix form)
# -------------------------------------------------
tol = 1e-9
x_min = x.min()
boundary_nodes = set(np.where(np.abs(x - x_min) < tol)[0].tolist())

robin_edges = []
for e in range(Nels):
    n = quad8[e]
    edges = [
        (n[0], n[4], n[1]),
        (n[1], n[5], n[2]),
        (n[2], n[6], n[3]),
        (n[3], n[7], n[0]),
    ]
    for edge in edges:
        if all(k in boundary_nodes for k in edge):
            robin_edges.append(edge)

print(f"Robin edges detected: {len(robin_edges)}")

for (n1, n2, n3) in robin_edges:
    He, Pe = Robin_quadr(
        x[n1], y[n1],
        x[n2], y[n2],
        x[n3], y[n3],
        p=0.0,
        gama=GAMA
    )

    ed = [n1, n2, n3]
    for i in range(3):
        fg[ed[i]] += Pe[i]
        for j in range(3):
            Kg[ed[i], ed[j]] += He[i, j]

# -------------------------------------------------
# Apply Dirichlet BCs (maximum-x boundary) while still in lil_matrix
# -------------------------------------------------
x_max = x.max()
exit_nodes = np.where(np.abs(x - x_max) < tol)[0]
print(f"Dirichlet nodes detected: {exit_nodes.size}")

for n in exit_nodes:
    Kg[n, :] = 0.0
    Kg[:, n] = 0.0
    Kg[n, n] = 1.0
    fg[n] = 0.0




# Add this RIGHT AFTER the Dirichlet BC section and BEFORE CSR conversion:

print("\nMatrix diagnostics BEFORE CSR conversion:")
print(f"  Matrix shape: {Kg.shape}")
print(f"  Matrix nnz: {Kg.nnz}")

# Check which nodes actually have Dirichlet BCs applied
dirichlet_rows = []
for n in exit_nodes:
    row = Kg.getrow(n)  # Get sparse row
    row_nnz = row.nnz
    diag_val = Kg[n, n]
    if row_nnz == 1 and abs(diag_val - 1.0) < 1e-10:
        dirichlet_rows.append(n)

print(f"  Dirichlet BCs properly applied: {len(dirichlet_rows)}/{len(exit_nodes)}")
print(f"  Dirichlet node indices: {exit_nodes[:10]}...")

# Check if any rows are completely zero
zero_rows = []
for i in range(Nnds):
    row = Kg.getrow(i)
    if row.nnz == 0:
        zero_rows.append(i)

print(f"  Zero rows found: {len(zero_rows)}")
if len(zero_rows) > 0:
    print(f"    Zero row indices: {zero_rows[:10]}...")
    print(f"    Coordinates of zero rows:")
    for idx in zero_rows[:5]:
        print(f"      Node {idx}: ({x[idx]:.4f}, {y[idx]:.4f})")

# Check matrix diagonal
diag_array = np.array([Kg[i, i] for i in range(Nnds)])
zero_diag = np.where(np.abs(diag_array) < 1e-15)[0]
print(f"  Zero diagonal entries: {len(zero_diag)}")
if len(zero_diag) > 0:
    print(f"    Zero diag indices: {zero_diag[:10]}...")
    print(f"    Coordinates of zero diag nodes:")
    for idx in zero_diag[:5]:
        print(f"      Node {idx}: ({x[idx]:.4f}, {y[idx]:.4f})")



# Fix: Set unused nodes to Dirichlet BC (u=0) to make matrix non-singular
used_nodes_set = set(quad8.flatten())
for n in range(Nnds):
    if n not in used_nodes_set:
        Kg[n, :] = 0.0
        Kg[:, n] = 0.0
        Kg[n, n] = 1.0
        fg[n] = 0.0


print("\nChecking if any elements reference unused nodes...")
used_nodes_set = set(quad8.flatten())
unused_nodes_set = set(range(Nnds)) - used_nodes_set

elements_with_unused = []
for e in range(Nels):
    edofs = quad8[e]
    if any(node in unused_nodes_set for node in edofs):
        elements_with_unused.append(e)
        if len(elements_with_unused) <= 5:  # Show first 5
            print(f"  Element {e}: nodes {edofs}")
            print(f"    Unused nodes in this element: {[n for n in edofs if n in unused_nodes_set]}")

print(f"Total elements referencing unused nodes: {len(elements_with_unused)}")

# Check Element 154 specifically since it's in the hole region
if 154 in elements_with_unused or 898 in unused_nodes_set:
    print(f"\nElement 154 investigation:")
    print(f"  Node 898 in unused set: {898 in unused_nodes_set}")
    print(f"  Element 154 nodes: {quad8[154]}")


# -------------------------------------------------
# Convert to CSR ONCE all assembly and modifications are done
# -------------------------------------------------
Kg = csr_matrix(Kg)

# -------------------------------------------------
# DIRECT SOLVER (testing)
# -------------------------------------------------
print("Solving linear system (DIRECT solver - spsolve)...")
print("This may take a moment for large systems...")

try:
    u = spsolve(Kg, fg)
    if hasattr(u, 'toarray'):
        u = u.toarray().flatten()
    u = np.asarray(u, dtype=np.float64)    
    print("Direct solve completed successfully")
except Exception as e:
    print(f"Direct solver failed: {e}")
    raise

# -------------------------------------------------
# DIAGNOSTICS: Check for issues
# -------------------------------------------------
print("\n" + "="*60)
print("DIAGNOSTICS")
print("="*60)

# Check for NaN values in solution
nan_count = np.isnan(u).sum()
print(f"NaN nodes in u: {nan_count}")
if nan_count > 0:
    nan_indices = np.where(np.isnan(u))[0]
    print(f"NaN indices: {nan_indices[:10]}...")
    print(f"NaN coordinates:")
    for idx in nan_indices[:5]:
        print(f"  Node {idx}: ({x[idx]:.4f}, {y[idx]:.4f})")

# Check for inf values
inf_count = np.isinf(u).sum()
print(f"Inf nodes in u: {inf_count}")

# Check solution statistics
print(f"u min: {u.min():.6e}")
print(f"u max: {u.max():.6e}")
print(f"u mean: {u.mean():.6e}")
print(f"u std: {u.std():.6e}")

# Check for degenerate elements (zero or very small area)
print("\nChecking for degenerate elements...")
degenerate_elements = []
for e in range(Nels):
    edofs = quad8[e]
    poly_x = x[edofs]
    poly_y = y[edofs]
    area = 0.5 * np.abs(
        np.sum(poly_x[:-1] * poly_y[1:]) + poly_x[-1] * poly_y[0] -
        np.sum(poly_x[1:] * poly_y[:-1]) - poly_x[0] * poly_y[-1]
    )
    if area < 1e-12:
        degenerate_elements.append((e, area))

if degenerate_elements:
    print(f"Found {len(degenerate_elements)} degenerate elements:")
    for e, area in degenerate_elements[:5]:
        print(f"  Element {e}: area = {area:.6e}")
        print(f"    Nodes: {quad8[e]}")
else:
    print("No degenerate elements found")

# Check mesh coverage in the hole region
print("\nChecking mesh coverage in hole region (x∈[0.5,0.7], y∈[0.1,0.3])...")
mask = (x > 0.5) & (x < 0.7) & (y > 0.1) & (y < 0.3)
nodes_in_region = np.where(mask)[0]
print(f"Nodes in hole region: {len(nodes_in_region)}")
if len(nodes_in_region) > 0:
    print(f"  Node indices: {nodes_in_region[:10]}...")
    print(f"  Sample coordinates and u values:")
    for idx in nodes_in_region[:5]:
        print(f"    Node {idx}: ({x[idx]:.4f}, {y[idx]:.4f}), u={u[idx]:.6e}")

# Find elements in the hole region
elements_in_region = []
for e in range(Nels):
    edofs = quad8[e]
    elem_x_center = x[edofs].mean()
    elem_y_center = y[edofs].mean()
    if 0.5 < elem_x_center < 0.7 and 0.1 < elem_y_center < 0.3:
        elements_in_region.append(e)

print(f"Elements in hole region: {len(elements_in_region)}")
if len(elements_in_region) > 0:
    print(f"  Element indices: {elements_in_region}")
    
    # Detailed inspection
    print("\nDetailed inspection of elements in hole region:")
    for e in elements_in_region:
        edofs = quad8[e]
        print(f"\nElement {e}:")
        print(f"  Nodes: {edofs}")
        print(f"  u values: {[f'{u[n]:.4f}' for n in edofs]}")
        print(f"  u mean: {u[edofs].mean():.6f}")
        print(f"  Center: ({x[edofs].mean():.4f}, {y[edofs].mean():.4f})")


    # Add this diagnostic right after the "Detailed inspection of elements in hole region" section:
    print("\nGeometry validation for hole region elements:")
    for e in elements_in_region:
        edofs_ordered = [quad8[e][0], quad8[e][4], quad8[e][1], quad8[e][5], 
                        quad8[e][2], quad8[e][6], quad8[e][3], quad8[e][7]]
        
        poly_x = x[edofs_ordered]
        poly_y = y[edofs_ordered]
        
        # Check for self-intersection using signed area
        # Shoelace formula - if negative, polygon is wound clockwise
        signed_area = 0.5 * (
            np.sum(poly_x[:-1] * poly_y[1:]) + poly_x[-1] * poly_y[0] -
            np.sum(poly_x[1:] * poly_y[:-1]) - poly_x[0] * poly_y[-1]
        )
        
        # Check aspect ratio
        x_range = poly_x.max() - poly_x.min()
        y_range = poly_y.max() - poly_y.min()
        aspect = x_range / y_range if y_range > 0 else float('inf')
        
        print(f"\nElement {e}:")
        print(f"  Signed area: {signed_area:.6e}")
        print(f"  X range: [{poly_x.min():.4f}, {poly_x.max():.4f}] = {x_range:.4f}")
        print(f"  Y range: [{poly_y.min():.4f}, {poly_y.max():.4f}] = {y_range:.4f}")
        print(f"  Aspect ratio: {aspect:.2f}")
        print(f"  Vertices (x,y):")
        for i, (px, py) in enumerate(zip(poly_x, poly_y)):
            print(f"    {i}: ({px:.4f}, {py:.4f})")


        # Diagnostic to find the missing element
        print("\nSearching for missing element in hole region (x≈0.65, y≈0.2)...")

        # Get all elements near the hole
        hole_x, hole_y = 0.65, 0.2
        search_radius = 0.05

        nearby_elements = []
        for e in range(Nels):
            edofs = quad8[e]
            elem_x_center = x[edofs].mean()
            elem_y_center = y[edofs].mean()
            dist = np.sqrt((elem_x_center - hole_x)**2 + (elem_y_center - hole_y)**2)
            if dist < search_radius:
                nearby_elements.append((e, elem_x_center, elem_y_center, dist))

        nearby_elements.sort(key=lambda t: t[3])

        print(f"Elements within {search_radius} of hole center:")
        for e, ex, ey, dist in nearby_elements[:10]:
            print(f"  Element {e}: center=({ex:.4f}, {ey:.4f}), dist={dist:.4f}")

        # Check mesh topology - find which nodes are in the hole region
        hole_nodes = []
        for i in range(Nnds):
            if abs(x[i] - hole_x) < search_radius and abs(y[i] - hole_y) < search_radius:
                hole_nodes.append(i)

        print(f"\nNodes near hole center: {len(hole_nodes)}")
        for ni in hole_nodes[:15]:
            print(f"  Node {ni}: ({x[ni]:.4f}, {y[ni]:.4f}), u={u[ni]:.4f}")

        # Check if these nodes are all referenced by elements
        hole_nodes_in_elements = set()
        for e in range(Nels):
            edofs = quad8[e]
            for node in edofs:
                if node in hole_nodes:
                    hole_nodes_in_elements.add(node)

        unreferenced = set(hole_nodes) - hole_nodes_in_elements
        if unreferenced:
            print(f"\nUnreferenced nodes in hole region: {unreferenced}")
            for ni in unreferenced:
                print(f"  Node {ni}: ({x[ni]:.4f}, {y[ni]:.4f})")


# Check for duplicate nodes
print("\nChecking for duplicate node coordinates...")
coords = np.column_stack((x, y))
unique_coords, inverse, counts = np.unique(
    coords, axis=0, return_inverse=True, return_counts=True
)
duplicates = np.where(counts > 1)[0]
if len(duplicates) > 0:
    print(f"Found {len(duplicates)} duplicate coordinate locations")
    for dup_idx in duplicates[:5]:
        dup_nodes = np.where(inverse == dup_idx)[0]
        print(f"  Coordinate ({unique_coords[dup_idx, 0]:.4f}, {unique_coords[dup_idx, 1]:.4f})")
        print(f"    appears at nodes: {dup_nodes}")
else:
    print("No duplicate coordinates found")

print("="*60 + "\n")

# -------------------------------------------------
# Visualization helper function
# -------------------------------------------------
def plot_fem_scalar_2d(
    x, y, quad8, field,
    title,
    output_path,
    cmap_name="RdBu_r"
):
    """
    2D FEM scalar visualization for Quad-8 meshes.
    field can be u (nodal), abs_vel (element), pressure (element), etc.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    patches = []
    values  = []

    vmin = field.min()
    vmax = field.max()

    for e in range(len(quad8)):
        n = quad8[e]

        edofs = [
            n[0], n[4],
            n[1], n[5],
            n[2], n[6],
            n[3], n[7]
        ]

        poly_xy = np.column_stack((x[edofs], y[edofs]))
        poly = Polygon(poly_xy, closed=True)

        patches.append(poly)

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

# -------------------------------------------------
# Post-processing
# -------------------------------------------------
OUTPUT_DIR = HERE / "data/output/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plot_fem_scalar_2d(
    x=x,
    y=y,
    quad8=quad8,
    field=u,
    title="FEM Potential Field (Quad-8) - DIRECT SOLVER",
    output_path=OUTPUT_DIR / "potential_u_2D_direct.png"
)

abs_vel = np.zeros(Nels)
vel = np.zeros((Nels, 2))

for e in range(Nels):
    edofs = quad8[e]
    XN = np.column_stack((x[edofs], y[edofs]))

    xp, _ = Genip2DQ(4)
    v_ip = np.zeros(4)

    for ip in range(4):
        B, _, _ = Shape_N_Der8(XN, xp[ip, 0], xp[ip, 1])
        grad = B.T @ u[edofs]
        vel[e, 0] = grad[0]
        vel[e, 1] = grad[1]
        v_ip[ip] = np.linalg.norm(grad)

    abs_vel[e] = v_ip.mean()

plot_fem_scalar_2d(
    x=x,
    y=y,
    quad8=quad8,
    field=abs_vel,
    title="Velocity Magnitude |V| (Quad-8) - DIRECT SOLVER",
    output_path=OUTPUT_DIR / "abs_velocity_2D_direct.png"
)

pressure = P0 - RHO * abs_vel**2

plot_fem_scalar_2d(
    x=x,
    y=y,
    quad8=quad8,
    field=pressure,
    title="Pressure Field (Quad-8) - DIRECT SOLVER",
    output_path=OUTPUT_DIR / "pressure_2D_direct.png"
)

# -------------------------------------------------
# Additional visualization with viridis colormap
# -------------------------------------------------
print("\nGenerating alternative visualization (individual polygons)...")
fig, ax = plt.subplots(figsize=(8, 6))

u_min, u_max = u.min(), u.max()
norm = mcolors.Normalize(vmin=u_min, vmax=u_max)
cmap = colormaps["viridis"]

# Track which element might be failing
render_count = 0
failed_elements = []

for e in range(Nels):
    n = quad8[e]
    edofs = [n[0], n[4], n[1], n[5], n[2], n[6], n[3], n[7]]
    
    try:
        poly = Polygon(
            list(zip(x[edofs], y[edofs])),
            closed=True,
            facecolor=cmap(norm(u[edofs].mean())),
            edgecolor="k",
            linewidth=0.15
        )
        ax.add_patch(poly)
        render_count += 1
    except Exception as ex:
        failed_elements.append((e, str(ex)))
        if len(failed_elements) <= 3:
            print(f"  Failed to render element {e}: {ex}")

print(f"Successfully rendered {render_count}/{Nels} elements")
if failed_elements:
    print(f"Failed elements: {[e for e, _ in failed_elements]}")

ax.set_xlim(x.min() - 0.05, x.max() + 0.05)
ax.set_ylim(y.min() - 0.05, y.max() + 0.05)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("FEM Potential Field (individual polygons) - DIRECT SOLVER")

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Potential u")

outfile = OUTPUT_DIR / "potential_field_quad8_direct_v2.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved alternative plot → {outfile}")

# -------------------------------------------------
# Export results to Excel
# -------------------------------------------------
outfile = HERE / "data/output/Results_quad8_CPU_v2_direct.xlsx"

with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
    header = [
        "#NODE", "VELOCITY POTENTIAL", "",
        "#ELEMENT", "U m/s (x  vel)", "V m/s (y vel)",
        "|V| m/s", "Pressure (Pa)"
    ]
    pd.DataFrame([header]).to_excel(writer, index=False, header=False)

    pd.DataFrame({
        "#NODE": np.arange(1, Nnds + 1),
        "VELOCITY POTENTIAL": u
    }).to_excel(writer, startrow=1, startcol=0,
                index=False, header=False)

    pd.DataFrame(
        np.column_stack([
            np.arange(1, Nels + 1),
            vel[:, 0],
            vel[:, 1],
            abs_vel,
            pressure
        ])
    ).to_excel(writer, startrow=1, startcol=3,
                index=False, header=False)

print(f"Results written to: {outfile}")
