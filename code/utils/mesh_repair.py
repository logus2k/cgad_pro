"""
Mesh Validation and Repair Tool for Quad-8 FEM

This script validates and optionally repairs Quad-8 meshes by:
1. Detecting unused/orphaned nodes
2. Identifying missing elements (topology gaps)
3. Removing unused nodes and renumbering connectivity
4. Generating a clean, repaired mesh file
"""

from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Set, List, Dict

def load_mesh(mesh_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh coordinates and connectivity from Excel file."""
    coord = pd.read_excel(mesh_file, sheet_name="coord")
    conec = pd.read_excel(mesh_file, sheet_name="conec")
    
    # Extract X and Y, handling potential missing header
    if 'X' in coord.columns and 'Y' in coord.columns:
        x = coord["X"].to_numpy(dtype=float)
        y = coord["Y"].to_numpy(dtype=float)
    else:
        # Assume first two columns are X, Y
        x = coord.iloc[:, 0].to_numpy(dtype=float)
        y = coord.iloc[:, 1].to_numpy(dtype=float)
    
    # Extract connectivity - take first 8 columns (excluding Element column if present)
    if conec.shape[1] > 8:
        # Has Element column, skip it
        quad8 = conec.iloc[:, 1:9].to_numpy(dtype=int) - 1  # Convert to 0-indexed
    else:
        quad8 = conec.iloc[:, :8].to_numpy(dtype=int) - 1  # Convert to 0-indexed
    
    return x, y, quad8

def find_unused_nodes(Nnds: int, quad8: np.ndarray) -> Set[int]:
    """Find nodes that are not referenced by any element."""
    used_nodes = set(quad8.flatten())
    all_nodes = set(range(Nnds))
    unused_nodes = all_nodes - used_nodes
    return unused_nodes

def find_topology_gaps(x: np.ndarray, y: np.ndarray, quad8: np.ndarray, 
                       unused_nodes: Set[int], search_radius: float = 50.0) -> List[Dict]:
    """
    Find potential missing elements by detecting unused nodes surrounded by elements.
    
    Returns list of suspected gaps with their surrounding topology.
    """
    gaps = []
    
    for node in unused_nodes:
        nx, ny = x[node], y[node]
        
        # Find nearby elements
        nearby_elements = []
        for e in range(len(quad8)):
            edofs = quad8[e]
            elem_x_center = x[edofs].mean()
            elem_y_center = y[edofs].mean()
            dist = np.sqrt((elem_x_center - nx)**2 + (elem_y_center - ny)**2)
            
            if dist < search_radius * 2:  # Wider search for context
                nearby_elements.append(e)
        
        # Find nearby used nodes
        nearby_used_nodes = []
        for i in range(len(x)):
            if i in unused_nodes:
                continue
            dist = np.sqrt((x[i] - nx)**2 + (y[i] - ny)**2)
            if dist < search_radius:
                nearby_used_nodes.append(i)
        
        if nearby_elements and nearby_used_nodes:
            gaps.append({
                'unused_node': node,
                'coords': (nx, ny),
                'nearby_elements': nearby_elements,
                'nearby_used_nodes': nearby_used_nodes
            })
    
    return gaps

def create_node_mapping(unused_nodes: Set[int], Nnds: int) -> Tuple[Dict[int, int], int]:
    """
    Create mapping from old node indices to new (compacted) indices.
    
    Returns:
        mapping: dict mapping old_index -> new_index
        new_Nnds: total number of nodes after removal
    """
    mapping = {}
    new_index = 0
    
    for old_index in range(Nnds):
        if old_index not in unused_nodes:
            mapping[old_index] = new_index
            new_index += 1
    
    return mapping, new_index

def repair_mesh(x: np.ndarray, y: np.ndarray, quad8: np.ndarray, 
                unused_nodes: Set[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove unused nodes and renumber connectivity.
    
    Returns:
        x_new: compacted x coordinates
        y_new: compacted y coordinates
        quad8_new: renumbered connectivity (still 0-indexed)
    """
    Nnds = len(x)
    node_mapping, new_Nnds = create_node_mapping(unused_nodes, Nnds)
    
    # Create new coordinate arrays
    x_new = np.zeros(new_Nnds)
    y_new = np.zeros(new_Nnds)
    
    for old_idx, new_idx in node_mapping.items():
        x_new[new_idx] = x[old_idx]
        y_new[new_idx] = y[old_idx]
    
    # Renumber connectivity
    quad8_new = np.zeros_like(quad8)
    for e in range(len(quad8)):
        for i in range(8):
            old_node = quad8[e, i]
            quad8_new[e, i] = node_mapping[old_node]
    
    return x_new, y_new, quad8_new

def save_mesh(output_file: Path, x: np.ndarray, y: np.ndarray, quad8: np.ndarray):
    """Save repaired mesh to Excel file."""
    # Create coordinate dataframe (1-indexed for output)
    coord_df = pd.DataFrame({
        'X': x,
        'Y': y
    })
    
    # Create connectivity dataframe (1-indexed for output)
    conec_df = pd.DataFrame(
        quad8 + 1,  # Convert back to 1-indexed
        columns=['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8']
    )
    
    # Write to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        coord_df.to_excel(writer, sheet_name='coord', index=False)
        conec_df.to_excel(writer, sheet_name='conec', index=False)

def print_validation_report(x: np.ndarray, y: np.ndarray, quad8: np.ndarray, 
                           unused_nodes: Set[int], gaps: List[Dict]):
    """Print detailed validation report."""
    Nnds = len(x)
    Nels = len(quad8)
    
    print("="*70)
    print("MESH VALIDATION REPORT")
    print("="*70)
    print(f"\nMesh Statistics:")
    print(f"  Total nodes:     {Nnds}")
    print(f"  Total elements:  {Nels}")
    print(f"  Unused nodes:    {len(unused_nodes)}")
    print(f"  Used nodes:      {Nnds - len(unused_nodes)}")
    print(f"  Utilization:     {100 * (1 - len(unused_nodes)/Nnds):.1f}%")
    
    if unused_nodes:
        print(f"\nUnused Nodes Details:")
        print(f"  Count: {len(unused_nodes)}")
        print(f"  Indices (first 20): {sorted(list(unused_nodes))[:20]}")
        
        if len(unused_nodes) <= 20:
            print(f"\n  Coordinates of unused nodes:")
            for node in sorted(unused_nodes):
                print(f"    Node {node+1}: ({x[node]:.2f}, {y[node]:.2f})")
    
    if gaps:
        print(f"\nTopology Gaps Detected:")
        print(f"  Suspected missing elements: {len(gaps)}")
        for i, gap in enumerate(gaps[:10], 1):  # Show first 10
            node = gap['unused_node']
            coords = gap['coords']
            print(f"\n  Gap {i}:")
            print(f"    Unused node {node+1} at ({coords[0]:.2f}, {coords[1]:.2f})")
            print(f"    Nearby elements: {[e+1 for e in gap['nearby_elements'][:5]]}")
            print(f"    Nearby used nodes: {[n+1 for n in gap['nearby_used_nodes'][:5]]}")
    
    # Check for degenerate elements
    print(f"\nElement Quality Check:")
    degenerate_count = 0
    min_area = float('inf')
    max_area = 0.0
    
    for e in range(Nels):
        edofs = quad8[e]
        poly_x = x[edofs]
        poly_y = y[edofs]
        area = 0.5 * np.abs(
            np.sum(poly_x[:-1] * poly_y[1:]) + poly_x[-1] * poly_y[0] -
            np.sum(poly_x[1:] * poly_y[:-1]) - poly_x[0] * poly_y[-1]
        )
        
        if area < 1e-12:
            degenerate_count += 1
        
        min_area = min(min_area, area)
        max_area = max(max_area, area)
    
    print(f"  Degenerate elements (area < 1e-12): {degenerate_count}")
    print(f"  Minimum element area: {min_area:.6e}")
    print(f"  Maximum element area: {max_area:.6e}")
    print(f"  Area ratio (max/min): {max_area/min_area:.2f}")
    
    print("\n" + "="*70)

def main():
    """Main validation and repair workflow."""
    # Configuration
    HERE = Path(__file__).resolve().parent.parent.parent
    INPUT_MESH = HERE / "data/input/converted_mesh.xlsx"
    OUTPUT_MESH = HERE / "data/input/converted_mesh_repaired.xlsx"
    
    print("Loading mesh...")
    x, y, quad8 = load_mesh(INPUT_MESH)
    Nnds = len(x)
    Nels = len(quad8)
    
    print(f"Loaded: {Nnds} nodes, {Nels} elements")
    print(f"Coordinate range: X=[{x.min():.1f}, {x.max():.1f}], Y=[{y.min():.1f}, {y.max():.1f}]")
    
    # Validation
    print("\nValidating mesh...")
    unused_nodes = find_unused_nodes(Nnds, quad8)
    gaps = find_topology_gaps(x, y, quad8, unused_nodes)
    
    # Print report
    print_validation_report(x, y, quad8, unused_nodes, gaps)
    
    # Repair option
    if unused_nodes:
        print("\n" + "="*70)
        print("IMPORTANT: Mesh repair will remove unused nodes and renumber connectivity.")
        print("This is ONLY safe if unused nodes are truly orphaned.")
        print("If nodes are part of intentional gaps (holes/windows), DO NOT repair.")
        response = input(f"\nRepair mesh by removing {len(unused_nodes)} unused nodes? (yes/no): ")
        
        if response.lower() in ['yes', 'y']:
            print("\nRepairing mesh...")
            x_new, y_new, quad8_new = repair_mesh(x, y, quad8, unused_nodes)
            
            print(f"Repaired mesh:")
            print(f"  New node count:    {len(x_new)} (removed {Nnds - len(x_new)})")
            print(f"  Element count:     {len(quad8_new)} (unchanged)")
            
            # Validate repaired mesh
            print("\nValidating repaired mesh...")
            unused_after = find_unused_nodes(len(x_new), quad8_new)
            print(f"  Unused nodes after repair: {len(unused_after)}")
            
            if unused_after:
                print("  WARNING: Still have unused nodes after repair!")
            
            # Save repaired mesh
            print(f"\nSaving repaired mesh to: {OUTPUT_MESH}")
            save_mesh(OUTPUT_MESH, x_new, y_new, quad8_new)
            print("✓ Repaired mesh saved successfully")
            
            # Generate comparison report
            print("\n" + "="*70)
            print("BEFORE vs AFTER COMPARISON")
            print("="*70)
            print(f"Nodes:    {Nnds:5d} → {len(x_new):5d} (Δ {Nnds - len(x_new)})")
            print(f"Elements: {Nels:5d} → {len(quad8_new):5d} (Δ 0)")
            print(f"Unused:   {len(unused_nodes):5d} → {len(unused_after):5d}")
            print("="*70)
        else:
            print("\nMesh repair cancelled. Original mesh unchanged.")
    else:
        print("\n✓ Mesh is valid - no unused nodes detected.")
        print("  No repair needed.")
    
    print("\nValidation complete.")

if __name__ == "__main__":
    main()
