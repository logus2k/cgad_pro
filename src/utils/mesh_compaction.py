"""
Quad-8 mesh compaction utility

- Removes orphan nodes (unused by any element)
- Renumbers nodes densely [0..Nused-1]
- Preserves element connectivity and geometry
- Writes a clean Excel mesh

Input format:
  coord sheet: columns [X, Y] (mm or m)
  conec sheet: 8 columns (1-based node indices)

Output:
  compacted_mesh_quad8.xlsx
"""

from pathlib import Path
import pandas as pd
import numpy as np

# -------------------------------------------------
# Paths
# -------------------------------------------------
HERE = Path(__file__).resolve().parent.parent.parent
INPUT_MESH  = HERE / "data/input/converted_mesh.xlsx"
OUTPUT_MESH = HERE / "data/input/converted_mesh_compact.xlsx"

# -------------------------------------------------
# Load mesh
# -------------------------------------------------
coord = pd.read_excel(INPUT_MESH, sheet_name="coord")
conec = pd.read_excel(INPUT_MESH, sheet_name="conec")

# Strip headers if present
coord = coord.iloc[:, :2]
conec = conec.iloc[:, :8]

coord.columns = ["X", "Y"]

# Convert to numpy
xy = coord.to_numpy(dtype=float)
conn = conec.to_numpy(dtype=int) - 1   # → zero-based

Nnds = xy.shape[0]
Nels = conn.shape[0]

print(f"Original mesh: {Nnds} nodes, {Nels} elements")

# -------------------------------------------------
# Detect used nodes
# -------------------------------------------------
used_nodes = np.unique(conn.flatten())
used_nodes.sort()

Nused = used_nodes.size
Norphan = Nnds - Nused

print(f"Used nodes   : {Nused}")
print(f"Orphan nodes : {Norphan}")

# -------------------------------------------------
# Build old → new index map
# -------------------------------------------------
old_to_new = {old: new for new, old in enumerate(used_nodes)}

# -------------------------------------------------
# Compact coordinates
# -------------------------------------------------
xy_compact = xy[used_nodes]

# -------------------------------------------------
# Remap connectivity
# -------------------------------------------------
conn_compact = np.zeros_like(conn)

for e in range(Nels):
    for i in range(8):
        conn_compact[e, i] = old_to_new[conn[e, i]]

# -------------------------------------------------
# Sanity checks
# -------------------------------------------------
assert conn_compact.min() == 0
assert conn_compact.max() == Nused - 1

# -------------------------------------------------
# Write compacted mesh
# -------------------------------------------------
with pd.ExcelWriter(OUTPUT_MESH, engine="openpyxl") as writer:
    pd.DataFrame(
        xy_compact,
        columns=["X", "Y"]
    ).to_excel(writer, sheet_name="coord", index=False)

    pd.DataFrame(
        conn_compact + 1   # back to 1-based
    ).to_excel(writer, sheet_name="conec", index=False, header=False)

print("--------------------------------------------------")
print(f"Compacted mesh written to:\n  {OUTPUT_MESH}")
print("Ready for CPU / GPU FEM runs")
print("--------------------------------------------------")
