"""
Export utilities for QUAD8 FEM results

Provides functions to export FEM results to Excel and other formats.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def _ensure_numpy_array(arr):
    """
    Ensure the input array is a NumPy array, converting from CuPy if necessary.
    
    Args:
        arr: Input array (NumPy, CuPy, or other array-like)
    
    Returns:
        NumPy array
    """
    if hasattr(arr, '__cuda_array_interface__') and hasattr(arr, 'get'):
        # This is likely a CuPy array
        return arr.get()  # Transfer from GPU to CPU and convert to NumPy
    elif hasattr(arr, 'numpy'):  # PyTorch tensors also have this method
        return arr.numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        # Try to convert to NumPy array
        return np.asarray(arr)


def export_results_to_excel(
    output_path,
    x, y, quad8,
    u, vel, abs_vel, pressure,
    implementation_name="CPU"
):
    """
    Export FEM results to Excel file (legacy function for backward compatibility).
    DEPRECATED: Use export_results with format='excel' instead.
    
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
    # Call the new function with excel format for backward compatibility
    result = export_results(
        output_path,
        x, y, quad8,
        u, vel, abs_vel, pressure,
        implementation_name=implementation_name,
        formats=['excel']
    )
    # Return the excel path specifically
    if isinstance(result, dict) and 'excel' in result:
        return result['excel']
    else:
        return result


def export_results(
    output_path,
    x, y, quad8,
    u, vel, abs_vel, pressure,
    implementation_name="CPU",
    formats=None
):
    """
    Export FEM results to multiple formats (CSV, HDF5, NPZ, Excel).
    
    Args:
        output_path: Path to save file(s) (can be Path or str)
        x: Node x-coordinates
        y: Node y-coordinates
        quad8: Element connectivity
        u: Velocity potential (nodal values)
        vel: Velocity components (element values, Nels × 2)
        abs_vel: Velocity magnitude (element values)
        pressure: Pressure field (element values)
        implementation_name: Implementation identifier for filename
        formats: List of formats to export ['csv', 'hdf5', 'npz', 'excel']. 
                Defaults to ['hdf5'] if None.
    
    Returns:
        Dict of paths to saved files if multiple formats, or single path if single format
    """
    if formats is None:
        formats = ['hdf5']
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure all arrays are NumPy arrays (convert from CuPy if necessary)
    x = _ensure_numpy_array(x)
    y = _ensure_numpy_array(y)
    quad8 = _ensure_numpy_array(quad8)
    u = _ensure_numpy_array(u)
    vel = _ensure_numpy_array(vel)
    abs_vel = _ensure_numpy_array(abs_vel)
    pressure = _ensure_numpy_array(pressure)
    
    Nnds = len(x)
    Nels = len(quad8)
    
    results = {}
    
    for fmt in formats:
        if fmt == 'excel':
            # Legacy Excel export - wrapped in try-catch to prevent crashes
            excel_path = output_path.with_suffix('.xlsx')
            try:
                with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
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
                results['excel'] = excel_path
            except Exception as e:
                print(f"WARNING: Could not export to Excel (dataset too large?): {e}")
                # Continue processing other formats even if Excel fails
                continue
            
        elif fmt == 'csv':
            # Export to separate CSV files for nodes and elements
            csv_coord_path = output_path.with_name(output_path.stem + '_nodes.csv')
            csv_elem_path = output_path.with_name(output_path.stem + '_elements.csv')
            
            # Node data
            node_df = pd.DataFrame({
                "#NODE": np.arange(1, Nnds + 1),
                "X_COORD": x,
                "Y_COORD": y,
                "VELOCITY_POTENTIAL": u
            })
            node_df.to_csv(csv_coord_path, index=False)
            
            # Element data
            elem_df = pd.DataFrame(
                np.column_stack([
                    np.arange(1, Nels + 1),
                    quad8,  # connectivity
                    vel[:, 0],  # U velocity
                    vel[:, 1],  # V velocity
                    abs_vel,    # absolute velocity
                    pressure    # pressure
                ]),
                columns=["#ELEMENT"] + [f"N{i+1}" for i in range(quad8.shape[1])] + 
                         ["U_m_s", "V_m_s", "ABS_VEL_m_s", "PRESSURE_PA"]
            )
            elem_df.to_csv(csv_elem_path, index=False)
            
            results['csv_nodes'] = csv_coord_path
            results['csv_elements'] = csv_elem_path
            
        elif fmt == 'npz':
            # Export to NPZ format
            npz_path = output_path.with_suffix('.npz')
            np.savez_compressed(
                npz_path,
                x=x,
                y=y,
                quad8=quad8,
                u=u,
                vel=vel,
                abs_vel=abs_vel,
                pressure=pressure,
                implementation_name=implementation_name
            )
            results['npz'] = npz_path
            
        elif fmt == 'hdf5' or fmt == 'h5':
            # Export to HDF5 format
            try:
                import h5py
            except ImportError:
                print("ERROR: h5py not installed. Install with: pip install h5py")
                continue
                
            h5_path = output_path.with_suffix('.h5')
            with h5py.File(h5_path, 'w') as f:
                # Store mesh data with compression
                f.create_dataset('x', data=x, compression='gzip', compression_opts=9)
                f.create_dataset('y', data=y, compression='gzip', compression_opts=9)
                f.create_dataset('quad8', data=quad8, compression='gzip', compression_opts=9)
                
                # Store results with compression
                f.create_dataset('u', data=u, compression='gzip', compression_opts=9)
                f.create_dataset('vel', data=vel, compression='gzip', compression_opts=9)
                f.create_dataset('abs_vel', data=abs_vel, compression='gzip', compression_opts=9)
                f.create_dataset('pressure', data=pressure, compression='gzip', compression_opts=9)
                
                # Store metadata
                f.attrs['num_nodes'] = Nnds
                f.attrs['num_elements'] = Nels
                f.attrs['element_type'] = 'QUAD8'
                f.attrs['implementation_name'] = implementation_name
                f.attrs['data_description'] = 'FEM results: x,y coordinates; quad8 connectivity; u velocity potential; vel (U,V) components; abs_vel magnitude; pressure'
            
            results['hdf5'] = h5_path
    
    # Return single path if only one format was requested, otherwise return dict
    if len(results) == 1:
        return list(results.values())[0]
    elif len(results) == 0:
        # If no formats were successfully exported, warn the user
        print("No formats were successfully exported. All requested formats may have failed.")
        return None
    else:
        return results


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
    
    # Ensure the array is a NumPy array
    u = _ensure_numpy_array(u)
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
    
    # Ensure all arrays are NumPy arrays
    u = _ensure_numpy_array(u)
    abs_vel = _ensure_numpy_array(abs_vel)
    pressure = _ensure_numpy_array(pressure)
    
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
