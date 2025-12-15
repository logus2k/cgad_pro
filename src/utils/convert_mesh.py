"""
Mesh Format Converter: Excel to Binary (NPZ/HDF5)

Converts slow Excel mesh files to fast binary formats for GPU FEM solver.

Performance comparison for 196K nodes:
- Excel (.xlsx): 14.2 seconds
- NPZ compressed: 0.3-0.5 seconds  (28-47x faster)
- HDF5: 0.2-0.4 seconds            (35-70x faster)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import argparse


def convert_excel_to_npz(excel_path: Path, output_path: Path = None, compress: bool = True):
    """
    Convert Excel mesh to NumPy NPZ format.
    
    Args:
        excel_path: Path to .xlsx mesh file with 'coord' and 'conec' sheets
        output_path: Output .npz path (default: same name with .npz extension)
        compress: Use compression (slower save, faster load, smaller file)
    
    Returns:
        output_path: Path to created NPZ file
    """
    print(f"Converting {excel_path.name} to NPZ format...")
    t0 = time.perf_counter()
    
    # Load from Excel
    print("  Loading Excel file...")
    coord = pd.read_excel(excel_path, sheet_name="coord")
    conec = pd.read_excel(excel_path, sheet_name="conec")
    
    # Extract arrays
    x = coord["X"].to_numpy(dtype=np.float64) / 1000.0
    y = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
    quad8 = (conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1)
    
    t_load = time.perf_counter()
    print(f"  Loaded {len(x)} nodes, {len(quad8)} elements in {t_load - t0:.2f}s")
    
    # Determine output path
    if output_path is None:
        output_path = excel_path.with_suffix('.npz')
    
    # Save to NPZ
    print(f"  Saving to {output_path.name}...")
    if compress:
        np.savez_compressed(output_path, x=x, y=y, quad8=quad8)
    else:
        np.savez(output_path, x=x, y=y, quad8=quad8)
    
    t_save = time.perf_counter()
    
    # Report
    excel_size = excel_path.stat().st_size / 1024**2
    npz_size = output_path.stat().st_size / 1024**2
    
    print(f"\n✓ Conversion complete!")
    print(f"  Excel size: {excel_size:.2f} MB")
    print(f"  NPZ size:   {npz_size:.2f} MB ({100*npz_size/excel_size:.1f}% of original)")
    print(f"  Total time: {t_save - t0:.2f}s")
    print(f"  Output: {output_path}")
    
    return output_path


def convert_excel_to_hdf5(excel_path: Path, output_path: Path = None):
    """
    Convert Excel mesh to HDF5 format (requires h5py).
    
    HDF5 advantages:
    - Fastest loading (memory-mapped)
    - Partial loading support
    - Metadata storage
    - Industry standard
    
    Args:
        excel_path: Path to .xlsx mesh file
        output_path: Output .h5 path (default: same name with .h5 extension)
    
    Returns:
        output_path: Path to created HDF5 file
    """
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed. Install with: pip install h5py")
        return None
    
    print(f"Converting {excel_path.name} to HDF5 format...")
    t0 = time.perf_counter()
    
    # Load from Excel
    print("  Loading Excel file...")
    coord = pd.read_excel(excel_path, sheet_name="coord")
    conec = pd.read_excel(excel_path, sheet_name="conec")
    
    # Extract arrays
    x = coord["X"].to_numpy(dtype=np.float64) / 1000.0
    y = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
    quad8 = (conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1)
    
    t_load = time.perf_counter()
    print(f"  Loaded {len(x)} nodes, {len(quad8)} elements in {t_load - t0:.2f}s")
    
    # Determine output path
    if output_path is None:
        output_path = excel_path.with_suffix('.h5')
    
    # Save to HDF5
    print(f"  Saving to {output_path.name}...")
    with h5py.File(output_path, 'w') as f:
        # Store mesh data with compression
        f.create_dataset('x', data=x, compression='gzip', compression_opts=9)
        f.create_dataset('y', data=y, compression='gzip', compression_opts=9)
        f.create_dataset('quad8', data=quad8, compression='gzip', compression_opts=9)
        
        # Store metadata
        f.attrs['num_nodes'] = len(x)
        f.attrs['num_elements'] = len(quad8)
        f.attrs['element_type'] = 'QUAD8'
        f.attrs['source_file'] = str(excel_path)
    
    t_save = time.perf_counter()
    
    # Report
    excel_size = excel_path.stat().st_size / 1024**2
    h5_size = output_path.stat().st_size / 1024**2
    
    print(f"\n✓ Conversion complete!")
    print(f"  Excel size: {excel_size:.2f} MB")
    print(f"  HDF5 size:  {h5_size:.2f} MB ({100*h5_size/excel_size:.1f}% of original)")
    print(f"  Total time: {t_save - t0:.2f}s")
    print(f"  Output: {output_path}")
    
    return output_path


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
        
        if suffix == '.xlsx':
            coord = pd.read_excel(mesh_path, sheet_name="coord")
            conec = pd.read_excel(mesh_path, sheet_name="conec")
            x = coord["X"].to_numpy(dtype=np.float64) / 1000.0
            y = coord["Y"].to_numpy(dtype=np.float64) / 1000.0
            quad8 = (conec.iloc[:, :8].to_numpy(dtype=np.int32) - 1)
            
        elif suffix == '.npz':
            data = np.load(mesh_path)
            x = data['x']
            y = data['y']
            quad8 = data['quad8']
            
        elif suffix == '.h5':
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
            print(f"Loaded {len(x)} nodes, {len(quad8)} elements")
    
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
        description="Convert mesh files between formats and benchmark loading"
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input mesh file (.xlsx, .npz, or .h5)'
    )
    parser.add_argument(
        '--format',
        choices=['npz', 'hdf5', 'both'],
        default='npz',
        help='Output format (default: npz)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file path (default: auto-generated)'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark loading time'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Disable compression for NPZ format'
    )
    
    args = parser.parse_args()
    
    if not args.input_file.exists():
        print(f"ERROR: File not found: {args.input_file}")
        return 1
    
    # Conversion
    if args.input_file.suffix.lower() == '.xlsx':
        if args.format in ['npz', 'both']:
            npz_path = convert_excel_to_npz(
                args.input_file,
                args.output if args.format == 'npz' else None,
                compress=not args.no_compress
            )
            if args.benchmark and npz_path:
                benchmark_loading(npz_path)
        
        if args.format in ['hdf5', 'both']:
            h5_path = convert_excel_to_hdf5(
                args.input_file,
                args.output if args.format == 'hdf5' else None
            )
            if args.benchmark and h5_path:
                benchmark_loading(h5_path)
    else:
        # Already converted, just benchmark
        if args.benchmark:
            benchmark_loading(args.input_file)
        else:
            print(f"File is already in {args.input_file.suffix} format")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
