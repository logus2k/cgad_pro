from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

HERE = Path(__file__).resolve().parent.parent.parent
MESH_FILE = HERE / "data/input/converted_mesh_compact.xlsx"

# -----------------------------
# Load data (robust to headers)
# -----------------------------
coord = pd.read_excel(MESH_FILE, sheet_name="coord")
conec = pd.read_excel(MESH_FILE, sheet_name="conec")

# Coordinate columns
if {"X", "Y"}.issubset(coord.columns):
    x = coord["X"].to_numpy(dtype=float)
    y = coord["Y"].to_numpy(dtype=float)
else:
    x = coord.iloc[:, 0].to_numpy(dtype=float)
    y = coord.iloc[:, 1].to_numpy(dtype=float)

# Connectivity (first 8 columns only)
quad8 = conec.iloc[:, :8].to_numpy(dtype=int)


Nnds = len(x)
Nels = quad8.shape[0]

print(f"\nMesh loaded: {Nnds} nodes, {Nels} elements")

errors = []
warnings = []

# -----------------------------
# 1. Coordinate checks
# -----------------------------
if np.isnan(x).any() or np.isnan(y).any():
    errors.append("NaN values found in node coordinates")

if not np.isfinite(x).all() or not np.isfinite(y).all():
    errors.append("Non-finite values found in node coordinates")

coords = np.column_stack((x, y))
duplicates = [
    i for i, c in enumerate(Counter(map(tuple, coords)).values()) if c > 1
]
if duplicates:
    warnings.append(f"Duplicate node coordinates detected ({len(duplicates)} duplicates)")

# -----------------------------
# 2. Connectivity checks
# -----------------------------
if quad8.shape[1] != 8:
    errors.append("Connectivity does not have 8 nodes per element")

# Convert to zero-based
quad8_zb = quad8 - 1

# Out-of-range indices
bad_indices = np.where((quad8_zb < 0) | (quad8_zb >= Nnds))
if bad_indices[0].size > 0:
    errors.append(f"Invalid node indices in connectivity (out of bounds)")

# Repeated nodes inside elements
for e in range(Nels):
    if len(set(quad8_zb[e])) != 8:
        warnings.append(f"Element {e+1} has repeated node indices")

# -----------------------------
# 3. Degenerate elements (area check)
# -----------------------------
def quad_area(nodes):
    """Approximate area using corner nodes only"""
    pts = nodes[[0, 2, 4, 6]]
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(
        np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    )

for e in range(Nels):
    nodes = quad8_zb[e]
    pts = np.column_stack((x[nodes], y[nodes]))
    area = quad_area(pts)
    if area < 1e-12:
        warnings.append(f"Element {e+1} has near-zero area")

# -----------------------------
# 4. Orphan nodes
# -----------------------------
used_nodes = set(quad8_zb.flatten())
all_nodes = set(range(Nnds))
orphan_nodes = sorted(all_nodes - used_nodes)

if orphan_nodes:
    warnings.append(f"{len(orphan_nodes)} orphan nodes (not used by any element)")

# -----------------------------
# 5. Element numbering gaps
# -----------------------------
expected_elements = set(range(1, Nels + 1))
actual_elements = set(conec.index + 1)

missing_elements = expected_elements - actual_elements
if missing_elements:
    warnings.append(f"Missing element rows: {sorted(missing_elements)}")

# -----------------------------
# Report
# -----------------------------
print("\n========== MESH VALIDATION REPORT ==========")

if errors:
    print("\n❌ ERRORS:")
    for e in errors:
        print("  -", e)
else:
    print("\n✅ No fatal errors detected")

if warnings:
    print("\n⚠ WARNINGS:")
    for w in warnings:
        print("  -", w)
else:
    print("\n✅ No warnings detected")

print("\n============================================")

if errors:
    print("\nMesh is NOT safe to solve.")
else:
    print("\nMesh is safe to solve (review warnings if any).")
