import numpy as np
import pandas as pd
from pathlib import Path
import time
import argparse


def parse_mesh_to_formats(input_filename, output_prefix=None, formats=['csv', 'npz', 'hdf5'], compress_npz=True):
    """
    Parses a specific text mesh format and exports to multiple optimized formats.
    
    Args:
        input_filename (str): Path to the input text file.
        output_prefix (str): Prefix for output files (default: input filename without extension)
        formats (list): List of formats to export ('csv', 'npz', 'hdf5')
        compress_npz (bool): Whether to compress NPZ files
    """
    nodes = {}
    elements = []
    
    print(f"Parsing input file: {input_filename}")
    
    with open(input_filename, 'r') as f:
        lines = f.readlines()

    i = 0
    # Find the start of the NODE INFORMATION section
    while i < len(lines) and "NODE INFORMATION" not in lines[i]:
        i += 1
    
    if i >= len(lines):
        raise ValueError("Could not find 'NODE INFORMATION' section.")

    print("Parsing node coordinates...")
    # Parse nodes until we hit the ELEMENT INFORMATION section or run out of lines
    while i < len(lines):
        line = lines[i].strip()
        
        if "ELEMENT INFORMATION" in line:
            break # Stop parsing nodes when we reach elements
        
        if line.startswith("Label"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    node_id = int(parts[1])
                    
                    # Read the next line for coordinates
                    i += 1
                    coord_line = lines[i].strip()
                    if "Global coordinates" in coord_line:
                        coords = coord_line.replace("Global coordinates", "").replace(":", "").split()
                        if len(coords) >= 2: # Assuming X, Y (Z ignored if present for 2D)
                            x = float(coords[0])
                            y = float(coords[1])
                            nodes[node_id] = (x, y)
                        else:
                            print(f"Warning: Could not parse coordinates for node {node_id} on line: {coord_line}")
                    else:
                         print(f"Warning: Expected coordinate line after node {node_id}, found: {coord_line}")
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse node ID or coordinates from line: {line}")
                    if len(lines) > i+1: print(f"Next line was: {lines[i+1]}")
        
        i += 1

    # Find the start of the ELEMENT INFORMATION table body
    # Look for the header line containing 'Label', 'Mesh', 'Connected Nodes'
    while i < len(lines) and not ("Label" in lines[i] and "Mesh" in lines[i] and "Connected Nodes" in lines[i]):
        i += 1
    
    if i >= len(lines):
         raise ValueError("Could not find 'ELEMENT INFORMATION' table header.")

    print("Parsing element connectivity...")
    # Parse elements after the header
    i += 1 # Move past the header line
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop parsing elements if we hit another section header or end of file
        # You might need to adjust this condition based on your full file structure
        if not line or line.startswith("---") or "NODE INFORMATION" in line or "SECTION" in line.upper():
             break 

        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
            # Expected format: [Element_Label, Mesh_Name, "Connected_Node_List"]
            if len(parts) >= 3:
                try:
                    # Parse the connected nodes part (third column)
                    node_list_str = parts[2]
                    node_ids = [int(n) for n in node_list_str.split()]
                    # Take only the first 8 nodes for Q8 connectivity as per your requirement
                    if len(node_ids) >= 8:
                        q8_connectivity = node_ids[:8] 
                        elements.append(q8_connectivity)
                    else:
                        print(f"Warning: Element line has fewer than 8 nodes: {line}. Found: {len(node_ids)}")
                except ValueError:
                    print(f"Warning: Could not parse node IDs from element line: {line}")
            else:
                 print(f"Warning: Unexpected element line format: {line}")
        
        i += 1

    if not nodes:
        raise ValueError("No nodes were parsed from the file.")
    if not elements:
        raise ValueError("No elements were parsed from the file.")

    print(f"Found {len(nodes)} nodes and {len(elements)} elements.")

    # Prepare data for export
    sorted_node_ids = sorted(nodes.keys())
    coord_data = {'X': [], 'Y': []}
    id_map = {} # Optional: map original ID to index if needed later
    for idx, orig_id in enumerate(sorted_node_ids):
        x, y = nodes[orig_id]
        coord_data['X'].append(x)
        coord_data['Y'].append(y)
        id_map[orig_id] = idx # Map original ID to zero-based index

    # Convert to numpy arrays with unit scaling (divide by 1000)
    x_array = np.array(coord_data['X'], dtype=np.float64) / 1000.0
    y_array = np.array(coord_data['Y'], dtype=np.float64) / 1000.0
    
    # Conec array: Each row is an element's 8 node IDs.
    conec_data_mapped = []
    for elem_nodes_orig_ids in elements:
        mapped_row = [id_map[orig_id] for orig_id in elem_nodes_orig_ids]  # Convert to 0-based index
        conec_data_mapped.append(mapped_row)
    
    quad8_array = np.array(conec_data_mapped, dtype=np.int32)

    # Determine output prefix
    if output_prefix is None:
        input_path = Path(input_filename)
        output_prefix = str(input_path.with_suffix(''))

    # Export to requested formats
    results = {}
    
    if 'csv' in formats:
        csv_coord_path = f"{output_prefix}_coord.csv"
        csv_conec_path = f"{output_prefix}_conec.csv"
        
        print(f"Writing CSV files...")
        coord_df = pd.DataFrame({'X': x_array, 'Y': y_array})
        conec_df = pd.DataFrame(quad8_array, columns=['N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8'])
        
        coord_df.to_csv(csv_coord_path, index=False)
        conec_df.to_csv(csv_conec_path, index=False)
        
        results['csv_coord'] = csv_coord_path
        results['csv_conec'] = csv_conec_path
        print(f"  - Coordinates: {csv_coord_path}")
        print(f"  - Connectivity: {csv_conec_path}")

    if 'npz' in formats:
        npz_path = f"{output_prefix}.npz"
        print(f"Writing NPZ file...")
        
        if compress_npz:
            np.savez_compressed(npz_path, x=x_array, y=y_array, quad8=quad8_array)
        else:
            np.savez(npz_path, x=x_array, y=y_array, quad8=quad8_array)
        
        results['npz'] = npz_path
        print(f"  - NPZ: {npz_path}")

    if 'hdf5' in formats or 'h5' in formats:
        h5_path = f"{output_prefix}.h5"
        print(f"Writing HDF5 file...")
        
        try:
            import h5py
        except ImportError:
            print("ERROR: h5py not installed. Install with: pip install h5py")
            return results
        
        with h5py.File(h5_path, 'w') as f:
            # Store mesh data with compression
            f.create_dataset('x', data=x_array, compression='gzip', compression_opts=9)
            f.create_dataset('y', data=y_array, compression='gzip', compression_opts=9)
            f.create_dataset('quad8', data=quad8_array, compression='gzip', compression_opts=9)
            
            # Store metadata
            f.attrs['num_nodes'] = len(x_array)
            f.attrs['num_elements'] = len(quad8_array)
            f.attrs['element_type'] = 'QUAD8'
            f.attrs['source_file'] = str(input_filename)
            f.attrs['unit_scaling'] = 'divided_by_1000'
        
        results['hdf5'] = h5_path
        print(f"  - HDF5: {h5_path}")

    print(f"\n✓ Conversion complete! Output files:")
    for fmt, path in results.items():
        size_mb = Path(path).stat().st_size / (1024**2) if Path(path).exists() else 0
        print(f"  {fmt.upper()}: {path} ({size_mb:.2f} MB)")

    return results


def benchmark_loading(mesh_path: Path):
    """
    Benchmark loading times for different formats.
    
    Args:
        mesh_path: Path to mesh file (any supported format)
    """
    suffix = mesh_path.suffix.lower()
    
    print(f"\n{'='*70}")
    print(f"LOADING BENCHMARK: {mesh_path.name}")
    print(f"{'='*70}\n")
    
    # Benchmark loading
    times = []
    for i in range(3):
        t0 = time.perf_counter()
        
        if suffix in ['.csv', '.txt']:  # For CSV coordinate file
            coord_df = pd.read_csv(mesh_path)
            x = coord_df["X"].to_numpy(dtype=np.float64)
            y = coord_df["Y"].to_numpy(dtype=np.float64)
            
        elif suffix == '.npz':
            data = np.load(mesh_path)
            x = data['x']
            y = data['y']
            quad8 = data['quad8']
            
        elif suffix in ['.h5', '.hdf5']:
            import h5py
            with h5py.File(mesh_path, 'r') as f:
                x = f['x'][:]
                y = f['y'][:]
                quad8 = f['quad8'][:]
        else:
            print(f"Unsupported format: {suffix}")
            return
        
        t1 = time.perf_counter()
        times.append(t1 - t0)
        
        if i == 0:
            if 'quad8' in locals():
                print(f"Loaded {len(x)} nodes, {len(quad8)} elements")
            else:
                print(f"Loaded {len(x)} nodes")
    
    # Report statistics
    times = np.array(times)
    print(f"\nLoading times (3 runs):")
    print(f"  Min:  {times.min():.4f}s")
    print(f"  Mean: {times.mean():.4f}s")
    print(f"  Max:  {times.max():.4f}s")
    print(f"  Std:  {times.std():.4f}s")
    
    # File size
    size_mb = mesh_path.stat().st_size / 1024**2
    print(f"\nFile size: {size_mb:.2f} MB")
    print(f"Load speed: {size_mb / times.mean():.1f} MB/s")


def main():
    parser = argparse.ArgumentParser(
        description="Parse text mesh format and convert to optimized formats"
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input text mesh file'
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        choices=['csv', 'npz', 'hdf5', 'h5'],
        default=['csv', 'npz', 'hdf5'],
        help='Output formats (default: csv npz hdf5)'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        help='Output file prefix (default: input filename without extension)'
    )
    parser.add_argument(
        '--compress-npz',
        action='store_true',
        default=True,
        help='Enable compression for NPZ format (default: enabled)'
    )
    parser.add_argument(
        '--no-compress-npz',
        dest='compress_npz',
        action='store_false',
        help='Disable compression for NPZ format'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark loading time for output files'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}")
        return 1
    
    # Perform conversion
    results = parse_mesh_to_formats(
        input_filename=args.input_file,
        output_prefix=args.output_prefix,
        formats=args.formats,
        compress_npz=args.compress_npz
    )
    
    # Benchmark if requested
    if args.benchmark:
        for fmt, path in results.items():
            if fmt in ['npz', 'hdf5', 'h5']:
                benchmark_loading(Path(path))
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
