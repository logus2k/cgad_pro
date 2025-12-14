"""
Export utilities for QUAD8 FEM results

Provides functions to export FEM results to Excel and other formats.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def export_results_to_excel(
    output_path,
    x, y, quad8,
    u, vel, abs_vel, pressure,
    implementation_name="CPU"
):
    """
    Export FEM results to Excel file.
    
    Args:
        output_path: Path to save Excel file (can be Path or str)
        x: Node x-coordinates
        y: Node y-coordinates
        quad8: Element connectivity
        u: Velocity potential (nodal values)
        vel: Velocity components (element values, Nels × 2)
        abs_vel: Velocity magnitude (element values)
        pressure: Pressure field (element values)
        implementation_name: Implementation identifier for filename
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Nnds = len(x)
    Nels = len(quad8)
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Write header row
        header = [
            "#NODE", "VELOCITY POTENTIAL", "",
            "#ELEMENT", "U m/s (x  vel)", "V m/s (y vel)",
            "|V| m/s", "Pressure (Pa)"
        ]
        pd.DataFrame([header]).to_excel(writer, index=False, header=False)
        
        # Write nodal results
        pd.DataFrame({
            "#NODE": np.arange(1, Nnds + 1),
            "VELOCITY POTENTIAL": u
        }).to_excel(
            writer,
            startrow=1,
            startcol=0,
            index=False,
            header=False
        )
        
        # Write element results
        pd.DataFrame(
            np.column_stack([
                np.arange(1, Nels + 1),
                vel[:, 0],
                vel[:, 1],
                abs_vel,
                pressure
            ])
        ).to_excel(
            writer,
            startrow=1,
            startcol=3,
            index=False,
            header=False
        )
    
    return output_path


def export_solution_vector(
    output_path,
    u,
    implementation_name="CPU"
):
    """
    Export solution vector to NumPy binary format for quick loading.
    
    Args:
        output_path: Path to save .npy file
        u: Solution vector
        implementation_name: Implementation identifier
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, u)
    return output_path


def export_summary_stats(
    output_path,
    u, abs_vel, pressure,
    solve_time=None,
    iterations=None,
    implementation_name="CPU"
):
    """
    Export summary statistics to text file.
    
    Args:
        output_path: Path to save summary file
        u: Velocity potential
        abs_vel: Velocity magnitude
        pressure: Pressure field
        solve_time: Total solve time in seconds (optional)
        iterations: Number of solver iterations (optional)
        implementation_name: Implementation identifier
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"FEM Solution Summary ({implementation_name})\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Velocity Potential (u):\n")
        f.write(f"  Min:    {u.min():.6e}\n")
        f.write(f"  Max:    {u.max():.6e}\n")
        f.write(f"  Mean:   {u.mean():.6e}\n")
        f.write(f"  Std:    {u.std():.6e}\n\n")
        
        f.write("Velocity Magnitude (|V|):\n")
        f.write(f"  Min:    {abs_vel.min():.6e}\n")
        f.write(f"  Max:    {abs_vel.max():.6e}\n")
        f.write(f"  Mean:   {abs_vel.mean():.6e}\n")
        f.write(f"  Std:    {abs_vel.std():.6e}\n\n")
        
        f.write("Pressure (Pa):\n")
        f.write(f"  Min:    {pressure.min():.6e}\n")
        f.write(f"  Max:    {pressure.max():.6e}\n")
        f.write(f"  Mean:   {pressure.mean():.6e}\n")
        f.write(f"  Std:    {pressure.std():.6e}\n\n")
        
        if solve_time is not None:
            f.write(f"Solve Time: {solve_time:.3f} seconds\n")
        
        if iterations is not None:
            f.write(f"Iterations: {iterations}\n")
    
    return output_path
