"""
ProfilingHDF5Transformer - GPU-accelerated nsys HDF5 to client HDF5 transformation.

Transforms Nsight Systems native HDF5 export directly to client-optimized HDF5
using CuPy for GPU-accelerated processing. No intermediate Python dicts.

Performance target: <1s for 200K+ events (vs 16s+ with CPU parsing).

Usage:
    transformer = ProfilingHDF5Transformer()
    stats = transformer.transform(nsys_h5_path, output_h5_path, session_id)

Location: /src/app/server/profiling/profiling_hdf5_transformer.py
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time

# CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("[HDF5Transformer] Warning: CuPy not available, falling back to NumPy")


class ProfilingHDF5Transformer:
    """
    GPU-accelerated transformer from nsys HDF5 to client-optimized HDF5.
    
    Falls back to NumPy if CuPy is not available.
    """
    
    # Category mapping from nsys tables to client categories
    CATEGORIES = [
        'cuda_kernel',
        'cuda_memcpy_h2d',
        'cuda_memcpy_d2h',
        'cuda_memcpy_d2d',
        'cuda_sync',
        'nvtx_range'
    ]
    
    # NVTX event types for ranges (start/end)
    NVTX_RANGE_TYPES = (59, 60)
    
    # NVTX colors for known range names
    NVTX_COLORS = {
        "load_mesh": 0x3498db,
        "assemble_system": 0x2ecc71,
        "apply_bc": 0xf1c40f,
        "solve_system": 0xe74c3c,
        "compute_derived": 0x9b59b6,
        "export_results": 0x95a5a6,
    }
    DEFAULT_NVTX_COLOR = 0x9b59b6
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize transformer.
        
        Args:
            use_gpu: Use CuPy GPU acceleration if available (default: True)
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np  # Array module (CuPy or NumPy)
        
        if self.use_gpu:
            print("[HDF5Transformer] Using CuPy GPU acceleration")
        else:
            print("[HDF5Transformer] Using NumPy CPU processing")
    
    def transform(
        self,
        nsys_h5_path: Path,
        output_path: Path,
        session_id: str
    ) -> Dict[str, Any]:
        """
        Transform nsys HDF5 to client-optimized HDF5.
        
        Args:
            nsys_h5_path: Path to nsys native HDF5 export
            output_path: Path for output client HDF5
            session_id: Session identifier
            
        Returns:
            Dict with transformation statistics
        """
        nsys_h5_path = Path(nsys_h5_path)
        output_path = Path(output_path)
        
        total_start = time.perf_counter()
        stats = {
            'total_events': 0,
            'categories': {},
            'timings': {},
            'used_gpu': self.use_gpu
        }
        
        # Read nsys HDF5
        t0 = time.perf_counter()
        with h5py.File(nsys_h5_path, 'r') as src:
            # Build string lookup table
            string_ids = self._load_string_ids(src)
            stats['timings']['load_strings'] = time.perf_counter() - t0
            
            # Extract and transform each data type
            t0 = time.perf_counter()
            kernels = self._extract_kernels(src, string_ids)
            stats['timings']['extract_kernels'] = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            memcpy = self._extract_memcpy(src)
            stats['timings']['extract_memcpy'] = time.perf_counter() - t0
            
            t0 = time.perf_counter()
            nvtx = self._extract_nvtx(src, string_ids)
            stats['timings']['extract_nvtx'] = time.perf_counter() - t0
        
        # Compute global min_start for normalization
        t0 = time.perf_counter()
        all_starts = []
        all_ends = []
        
        for data in [kernels, memcpy['h2d'], memcpy['d2h'], memcpy['d2d'], nvtx]:
            if data is not None and len(data['start_ns']) > 0:
                all_starts.append(self.xp.min(data['start_ns']))
                all_ends.append(self.xp.max(data['end_ns']))
        
        if all_starts:
            min_start = int(self.xp.min(self.xp.array(all_starts)))
            max_end = int(self.xp.max(self.xp.array(all_ends)))
        else:
            min_start = 0
            max_end = 0
        
        total_duration_ns = max_end - min_start
        stats['timings']['compute_bounds'] = time.perf_counter() - t0
        
        # Normalize timestamps and write output
        t0 = time.perf_counter()
        with h5py.File(output_path, 'w') as dst:
            # Count total events
            total_events = 0
            categories_present = []
            
            # Process kernels
            if kernels is not None and len(kernels['start_ns']) > 0:
                self._write_category(dst, 'cuda_kernel', kernels, min_start)
                n = len(kernels['start_ns'])
                total_events += n
                categories_present.append('cuda_kernel')
                stats['categories']['cuda_kernel'] = {
                    'count': n,
                    'unique_names': len(kernels['names'])
                }
            
            # Process memcpy categories
            for kind, cat_name in [('h2d', 'cuda_memcpy_h2d'), 
                                    ('d2h', 'cuda_memcpy_d2h'), 
                                    ('d2d', 'cuda_memcpy_d2d')]:
                data = memcpy[kind]
                if data is not None and len(data['start_ns']) > 0:
                    self._write_category(dst, cat_name, data, min_start)
                    n = len(data['start_ns'])
                    total_events += n
                    categories_present.append(cat_name)
                    stats['categories'][cat_name] = {
                        'count': n,
                        'unique_names': len(data['names'])
                    }
            
            # Process NVTX
            if nvtx is not None and len(nvtx['start_ns']) > 0:
                self._write_category(dst, 'nvtx_range', nvtx, min_start)
                n = len(nvtx['start_ns'])
                total_events += n
                categories_present.append('nvtx_range')
                stats['categories']['nvtx_range'] = {
                    'count': n,
                    'unique_names': len(nvtx['names'])
                }
            
            # Write metadata
            meta = dst.create_group('meta')
            meta.attrs['session_id'] = session_id
            meta.attrs['total_duration_ns'] = np.uint64(total_duration_ns)
            meta.attrs['total_events'] = np.uint32(total_events)
            meta.create_dataset(
                'categories',
                data=np.array(categories_present, dtype=h5py.special_dtype(vlen=str))
            )
            
            stats['total_events'] = total_events
        
        stats['timings']['write_output'] = time.perf_counter() - t0
        stats['timings']['total'] = time.perf_counter() - total_start
        stats['file_size_bytes'] = output_path.stat().st_size
        stats['total_duration_ns'] = total_duration_ns
        
        return stats
    
    def _load_string_ids(self, f: h5py.File) -> Dict[int, str]:
        """Load string ID lookup table from nsys HDF5."""
        string_ids = {}
        
        if 'StringIds' not in f:
            return string_ids
        
        str_data = f['StringIds'][:]
        dtype_names = str_data.dtype.names
        
        for row in str_data:
            if dtype_names:
                str_id = int(row['id']) if 'id' in dtype_names else int(row[0])
                str_val = row['value'] if 'value' in dtype_names else row[1]
            else:
                str_id = int(row[0])
                str_val = row[1]
            
            if isinstance(str_val, bytes):
                str_val = str_val.decode('utf-8', errors='replace')
            string_ids[str_id] = str_val
        
        return string_ids
    
    def _extract_kernels(self, f: h5py.File, string_ids: Dict[int, str]) -> Optional[Dict[str, Any]]:
        """Extract CUDA kernel data from nsys HDF5."""
        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in f:
            return None
        
        kernels = f['CUPTI_ACTIVITY_KIND_KERNEL'][:]
        n = len(kernels)
        if n == 0:
            return None
        
        dtype_names = kernels.dtype.names or []
        
        def get_col(name, default=0):
            if name in dtype_names:
                return kernels[name]
            return np.full(n, default)
        
        # Load to GPU/CPU arrays
        start_ns = self.xp.asarray(get_col('start'), dtype=self.xp.uint64)
        end_ns = self.xp.asarray(get_col('end'), dtype=self.xp.uint64)
        duration_ns = end_ns - start_ns
        stream = self.xp.asarray(get_col('streamId'), dtype=self.xp.uint8)
        
        # Grid/block dimensions
        grid_x = self.xp.asarray(get_col('gridX', 1), dtype=self.xp.uint16)
        grid_y = self.xp.asarray(get_col('gridY', 1), dtype=self.xp.uint16)
        grid_z = self.xp.asarray(get_col('gridZ', 1), dtype=self.xp.uint16)
        block_x = self.xp.asarray(get_col('blockX', 1), dtype=self.xp.uint16)
        block_y = self.xp.asarray(get_col('blockY', 1), dtype=self.xp.uint16)
        block_z = self.xp.asarray(get_col('blockZ', 1), dtype=self.xp.uint16)
        
        # Other metadata
        registers = self.xp.asarray(get_col('registersPerThread'), dtype=self.xp.uint16)
        shared_static = self.xp.asarray(get_col('staticSharedMemory'), dtype=self.xp.uint32)
        shared_dynamic = self.xp.asarray(get_col('dynamicSharedMemory'), dtype=self.xp.uint32)
        
        # Build name index (CPU operation - string handling)
        name_ids = get_col('demangledName', 0)
        names, name_idx = self._build_name_index(name_ids, string_ids)
        name_idx = self.xp.asarray(name_idx, dtype=self.xp.uint16)
        
        # Sort by start time on GPU
        sort_idx = self.xp.argsort(start_ns)
        
        return {
            'start_ns': start_ns[sort_idx],
            'end_ns': end_ns[sort_idx],
            'duration_ns': duration_ns[sort_idx],
            'stream': stream[sort_idx],
            'name_idx': name_idx[sort_idx],
            'names': names,
            'grid_x': grid_x[sort_idx],
            'grid_y': grid_y[sort_idx],
            'grid_z': grid_z[sort_idx],
            'block_x': block_x[sort_idx],
            'block_y': block_y[sort_idx],
            'block_z': block_z[sort_idx],
            'registers': registers[sort_idx],
            'shared_static': shared_static[sort_idx],
            'shared_dynamic': shared_dynamic[sort_idx],
        }
    
    def _extract_memcpy(self, f: h5py.File) -> Dict[str, Optional[Dict[str, Any]]]:
        """Extract memory transfer data from nsys HDF5, split by copy kind."""
        result = {'h2d': None, 'd2h': None, 'd2d': None}
        
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' not in f:
            return result
        
        memcpy = f['CUPTI_ACTIVITY_KIND_MEMCPY'][:]
        n = len(memcpy)
        if n == 0:
            return result
        
        dtype_names = memcpy.dtype.names or []
        
        def get_col(name, default=0):
            if name in dtype_names:
                return memcpy[name]
            return np.full(n, default)
        
        # Load all data
        start_ns = get_col('start')
        end_ns = get_col('end')
        copy_kind = get_col('copyKind')
        stream_ids = get_col('streamId')
        bytes_arr = get_col('bytes')
        
        # Split by copy kind: 1=HtoD, 2=DtoH, others=DtoD
        for kind_val, kind_name, cat_name in [(1, 'h2d', 'cudaMemcpy HtoD'),
                                               (2, 'd2h', 'cudaMemcpy DtoH')]:
            mask = copy_kind == kind_val
            if not np.any(mask):
                continue
            
            start = self.xp.asarray(start_ns[mask], dtype=self.xp.uint64)
            end = self.xp.asarray(end_ns[mask], dtype=self.xp.uint64)
            
            # Sort by start time
            sort_idx = self.xp.argsort(start)
            
            result[kind_name] = {
                'start_ns': start[sort_idx],
                'end_ns': end[sort_idx],
                'duration_ns': (end - start)[sort_idx],
                'stream': self.xp.asarray(stream_ids[mask], dtype=self.xp.uint8)[sort_idx],
                'name_idx': self.xp.zeros(int(mask.sum()), dtype=self.xp.uint16),
                'names': [cat_name],
                'bytes': self.xp.asarray(bytes_arr[mask], dtype=self.xp.uint32)[sort_idx],
            }
        
        # D2D (everything else)
        mask = (copy_kind != 1) & (copy_kind != 2)
        if np.any(mask):
            start = self.xp.asarray(start_ns[mask], dtype=self.xp.uint64)
            end = self.xp.asarray(end_ns[mask], dtype=self.xp.uint64)
            sort_idx = self.xp.argsort(start)
            
            result['d2d'] = {
                'start_ns': start[sort_idx],
                'end_ns': end[sort_idx],
                'duration_ns': (end - start)[sort_idx],
                'stream': self.xp.asarray(stream_ids[mask], dtype=self.xp.uint8)[sort_idx],
                'name_idx': self.xp.zeros(int(mask.sum()), dtype=self.xp.uint16),
                'names': ['cudaMemcpy D2D'],
                'bytes': self.xp.asarray(bytes_arr[mask], dtype=self.xp.uint32)[sort_idx],
            }
        
        return result
    
    def _extract_nvtx(self, f: h5py.File, string_ids: Dict[int, str]) -> Optional[Dict[str, Any]]:
        """Extract NVTX range data from nsys HDF5."""
        if 'NVTX_EVENTS' not in f:
            return None
        
        nvtx = f['NVTX_EVENTS'][:]
        n = len(nvtx)
        if n == 0:
            return None
        
        dtype_names = nvtx.dtype.names or []
        
        def get_col(name, default=0):
            if name in dtype_names:
                return nvtx[name]
            return np.full(n, default)
        
        # Filter to range events only (eventType 59 or 60)
        event_types = get_col('eventType')
        mask = np.isin(event_types, self.NVTX_RANGE_TYPES)
        
        if not np.any(mask):
            return None
        
        # Also filter to valid ranges (end > start)
        start_all = get_col('start')
        end_all = get_col('end')
        valid_mask = mask & (end_all > start_all) & (start_all > 0) & (end_all > 0)
        
        if not np.any(valid_mask):
            return None
        
        start_ns = self.xp.asarray(start_all[valid_mask], dtype=self.xp.uint64)
        end_ns = self.xp.asarray(end_all[valid_mask], dtype=self.xp.uint64)
        duration_ns = end_ns - start_ns
        
        # Build name index and colors
        text_ids = get_col('textId')[valid_mask]
        names, name_idx = self._build_name_index(text_ids, string_ids, default_name='nvtx_range')
        name_idx = self.xp.asarray(name_idx, dtype=self.xp.uint16)
        
        # Assign colors based on names
        # Need to get indices as numpy for iteration
        name_idx_cpu = name_idx.get() if self.use_gpu else name_idx
        colors = np.array([
            self.NVTX_COLORS.get(names[idx], self.DEFAULT_NVTX_COLOR)
            for idx in name_idx_cpu
        ], dtype=np.uint32)
        colors = self.xp.asarray(colors)
        
        # Sort by start time
        sort_idx = self.xp.argsort(start_ns)
        
        return {
            'start_ns': start_ns[sort_idx],
            'end_ns': end_ns[sort_idx],
            'duration_ns': duration_ns[sort_idx],
            'stream': self.xp.zeros(len(start_ns), dtype=self.xp.uint8),
            'name_idx': name_idx[sort_idx],
            'names': names,
            'color': colors[sort_idx],
        }
    
    def _build_name_index(
        self, 
        name_ids: np.ndarray, 
        string_ids: Dict[int, str],
        default_name: str = 'unknown'
    ) -> Tuple[list, np.ndarray]:
        """Build deduplicated name list and index array."""
        unique_names = []
        name_to_idx = {}
        indices = np.zeros(len(name_ids), dtype=np.uint16)
        
        for i, name_id in enumerate(name_ids):
            name = string_ids.get(int(name_id), default_name)
            if name not in name_to_idx:
                name_to_idx[name] = len(unique_names)
                unique_names.append(name)
            indices[i] = name_to_idx[name]
        
        return unique_names, indices
    
    def _write_category(
        self, 
        f: h5py.File, 
        category: str, 
        data: Dict[str, Any],
        min_start: int
    ):
        """Write a category group to output HDF5."""
        cat_group = f.create_group(category)
        
        # Normalize timestamps
        start_ns = data['start_ns'] - min_start
        duration_ns = data['duration_ns']
        
        # Transfer from GPU to CPU if needed
        def to_numpy(arr):
            if self.use_gpu and hasattr(arr, 'get'):
                return arr.get()
            return np.asarray(arr)
        
        # Core fields
        cat_group.create_dataset('start_ns', data=to_numpy(start_ns).astype(np.uint64),
                                  compression='gzip', compression_opts=1)
        cat_group.create_dataset('duration_ns', data=to_numpy(duration_ns).astype(np.uint64),
                                  compression='gzip', compression_opts=1)
        cat_group.create_dataset('stream', data=to_numpy(data['stream']).astype(np.uint8),
                                  compression='gzip', compression_opts=1)
        cat_group.create_dataset('name_idx', data=to_numpy(data['name_idx']).astype(np.uint16),
                                  compression='gzip', compression_opts=1)
        
        # Names group
        if 'names' not in f:
            f.create_group('names')
        f['names'].create_dataset(
            category,
            data=np.array(data['names'], dtype=h5py.special_dtype(vlen=str)),
            compression='gzip', compression_opts=1
        )
        
        # Category-specific fields
        if category == 'cuda_kernel':
            for field in ['grid_x', 'grid_y', 'grid_z', 'block_x', 'block_y', 'block_z']:
                cat_group.create_dataset(field, data=to_numpy(data[field]).astype(np.uint16),
                                          compression='gzip', compression_opts=1)
            cat_group.create_dataset('registers', data=to_numpy(data['registers']).astype(np.uint16),
                                      compression='gzip', compression_opts=1)
            cat_group.create_dataset('shared_static', data=to_numpy(data['shared_static']).astype(np.uint32),
                                      compression='gzip', compression_opts=1)
            cat_group.create_dataset('shared_dynamic', data=to_numpy(data['shared_dynamic']).astype(np.uint32),
                                      compression='gzip', compression_opts=1)
        
        elif category.startswith('cuda_memcpy'):
            cat_group.create_dataset('bytes', data=to_numpy(data['bytes']).astype(np.uint32),
                                      compression='gzip', compression_opts=1)
        
        elif category == 'nvtx_range':
            cat_group.create_dataset('color', data=to_numpy(data['color']).astype(np.uint32),
                                      compression='gzip', compression_opts=1)


def transform_nsys_to_client_hdf5(
    nsys_h5_path: Path,
    output_path: Path,
    session_id: str,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for transforming nsys HDF5 to client format.
    
    Args:
        nsys_h5_path: Path to nsys native HDF5 export
        output_path: Path for output client HDF5
        session_id: Session identifier
        use_gpu: Use GPU acceleration if available
        
    Returns:
        Transformation statistics
    """
    transformer = ProfilingHDF5Transformer(use_gpu=use_gpu)
    return transformer.transform(nsys_h5_path, output_path, session_id)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python profiling_hdf5_transformer.py <input.nsys.h5> <output.h5> [session_id]")
        sys.exit(1)
    
    nsys_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    session_id = sys.argv[3] if len(sys.argv) > 3 else 'test'
    
    stats = transform_nsys_to_client_hdf5(nsys_path, output_path, session_id)
    
    print(f"\nTransformed {stats['total_events']} events")
    print(f"Output size: {stats['file_size_bytes'] / 1024:.1f} KB")
    print(f"Total time: {stats['timings']['total']:.3f}s (GPU: {stats['used_gpu']})")
    print(f"\nTimings:")
    for k, v in stats['timings'].items():
        if k != 'total':
            print(f"  {k}: {v:.3f}s")
    print(f"\nCategories:")
    for cat, cat_stats in stats['categories'].items():
        print(f"  {cat}: {cat_stats['count']} events, {cat_stats['unique_names']} unique names")
