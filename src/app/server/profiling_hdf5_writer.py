"""
ProfilingHDF5Writer - Writes profiling timeline data to HDF5 format.

Optimized for fast client-side loading with h5wasm:
- Pre-grouped by category (no client-side grouping needed)
- Pre-sorted by start_ns (no client-side sorting needed)
- Typed arrays ready for InstancedMesh
- Deduplicated names (kernel names stored once, referenced by index)

Usage:
    writer = ProfilingHDF5Writer()
    writer.write(events, output_path, session_id, total_duration_ns)

HDF5 Schema:
    /meta
        session_id: string (attribute)
        total_duration_ns: uint64 (attribute)
        total_events: uint32 (attribute)
        categories: string[] (dataset) - list of category names present
    
    /<category>/  (e.g., /cuda_kernel/, /cuda_memcpy_h2d/, etc.)
        start_ns: uint64[N]
        duration_ns: uint32[N]  
        stream: uint8[N]
        name_idx: uint16[N]  # index into /names/<category>
        # Category-specific fields:
        # cuda_kernel: grid_x, grid_y, grid_z, block_x, block_y, block_z (uint16)
        # cuda_memcpy_*: bytes (uint32)
    
    /names/
        <category>: string[]  # deduplicated names for each category

Location: /src/app/server/profiling/profiling_hdf5_writer.py
"""

import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


class ProfilingHDF5Writer:
    """Writes profiling events to HDF5 format optimized for browser rendering."""
    
    # Category order (matches client-side)
    CATEGORIES = [
        'cuda_kernel',
        'cuda_memcpy_h2d',
        'cuda_memcpy_d2h',
        'cuda_memcpy_d2d',
        'cuda_sync',
        'nvtx_range'
    ]
    
    def write(
        self,
        events: List[Dict[str, Any]],
        output_path: Path,
        session_id: str,
        total_duration_ns: int
    ) -> Dict[str, Any]:
        """
        Write events to HDF5 file.
        
        Args:
            events: List of event dicts with keys:
                - id: unique identifier
                - category: one of CATEGORIES
                - name: event name (e.g., kernel name)
                - start_ns: start time in nanoseconds
                - end_ns: end time in nanoseconds
                - duration_ns: duration in nanoseconds
                - stream: CUDA stream ID
                - metadata: dict with category-specific fields
            output_path: Path to write HDF5 file
            session_id: Session identifier
            total_duration_ns: Total profiling duration
            
        Returns:
            Dict with write statistics
        """
        output_path = Path(output_path)
        
        # Group events by category
        events_by_category = defaultdict(list)
        for event in events:
            cat = event.get('category', 'unknown')
            if cat in self.CATEGORIES:
                events_by_category[cat].append(event)
        
        stats = {
            'total_events': len(events),
            'categories': {},
            'file_size_bytes': 0
        }
        
        with h5py.File(output_path, 'w') as f:
            # Meta group
            meta = f.create_group('meta')
            meta.attrs['session_id'] = session_id
            meta.attrs['total_duration_ns'] = np.uint64(total_duration_ns)
            meta.attrs['total_events'] = np.uint32(len(events))
            
            # Store category list
            categories_present = [c for c in self.CATEGORIES if c in events_by_category]
            meta.create_dataset(
                'categories',
                data=np.array(categories_present, dtype=h5py.special_dtype(vlen=str))
            )
            
            # Names group (deduplicated strings)
            names_group = f.create_group('names')
            
            # Write each category
            for category in self.CATEGORIES:
                cat_events = events_by_category.get(category, [])
                if not cat_events:
                    continue
                
                # Sort by start time
                cat_events.sort(key=lambda e: e['start_ns'])
                n = len(cat_events)
                
                # Build name index (deduplicate names)
                unique_names = []
                name_to_idx = {}
                name_indices = np.zeros(n, dtype=np.uint16)
                
                for i, event in enumerate(cat_events):
                    name = event.get('name', '')
                    if name not in name_to_idx:
                        name_to_idx[name] = len(unique_names)
                        unique_names.append(name)
                    name_indices[i] = name_to_idx[name]
                
                # Store names for this category
                if unique_names:
                    names_group.create_dataset(
                        category,
                        data=np.array(unique_names, dtype=h5py.special_dtype(vlen=str)),
                        compression='gzip',
                        compression_opts=1
                    )
                
                # Create category group
                cat_group = f.create_group(category)
                
                # Core fields (all categories)
                start_ns = np.array([e['start_ns'] for e in cat_events], dtype=np.uint64)
                duration_ns = np.array([e['duration_ns'] for e in cat_events], dtype=np.uint32)
                stream = np.array([e.get('stream', 0) for e in cat_events], dtype=np.uint8)
                
                cat_group.create_dataset('start_ns', data=start_ns, compression='gzip', compression_opts=1)
                cat_group.create_dataset('duration_ns', data=duration_ns, compression='gzip', compression_opts=1)
                cat_group.create_dataset('stream', data=stream, compression='gzip', compression_opts=1)
                cat_group.create_dataset('name_idx', data=name_indices, compression='gzip', compression_opts=1)
                
                # Category-specific fields
                if category == 'cuda_kernel':
                    self._write_kernel_metadata(cat_group, cat_events)
                elif category.startswith('cuda_memcpy'):
                    self._write_memcpy_metadata(cat_group, cat_events)
                elif category == 'nvtx_range':
                    self._write_nvtx_metadata(cat_group, cat_events)
                
                stats['categories'][category] = {
                    'count': n,
                    'unique_names': len(unique_names)
                }
        
        stats['file_size_bytes'] = output_path.stat().st_size
        return stats
    
    def _write_kernel_metadata(self, group: h5py.Group, events: List[Dict]):
        """Write CUDA kernel-specific metadata."""
        n = len(events)
        
        # Grid dimensions
        grid_x = np.zeros(n, dtype=np.uint16)
        grid_y = np.zeros(n, dtype=np.uint16)
        grid_z = np.zeros(n, dtype=np.uint16)
        
        # Block dimensions
        block_x = np.zeros(n, dtype=np.uint16)
        block_y = np.zeros(n, dtype=np.uint16)
        block_z = np.zeros(n, dtype=np.uint16)
        
        # Other kernel info
        registers = np.zeros(n, dtype=np.uint16)
        shared_static = np.zeros(n, dtype=np.uint32)
        shared_dynamic = np.zeros(n, dtype=np.uint32)
        
        for i, event in enumerate(events):
            meta = event.get('metadata', {})
            
            grid = meta.get('grid', [1, 1, 1])
            if isinstance(grid, (list, tuple)) and len(grid) >= 3:
                grid_x[i] = min(grid[0], 65535)
                grid_y[i] = min(grid[1], 65535)
                grid_z[i] = min(grid[2], 65535)
            
            block = meta.get('block', [1, 1, 1])
            if isinstance(block, (list, tuple)) and len(block) >= 3:
                block_x[i] = min(block[0], 65535)
                block_y[i] = min(block[1], 65535)
                block_z[i] = min(block[2], 65535)
            
            registers[i] = min(meta.get('registers_per_thread', 0), 65535)
            shared_static[i] = meta.get('shared_memory_static', 0)
            shared_dynamic[i] = meta.get('shared_memory_dynamic', 0)
        
        group.create_dataset('grid_x', data=grid_x, compression='gzip', compression_opts=1)
        group.create_dataset('grid_y', data=grid_y, compression='gzip', compression_opts=1)
        group.create_dataset('grid_z', data=grid_z, compression='gzip', compression_opts=1)
        group.create_dataset('block_x', data=block_x, compression='gzip', compression_opts=1)
        group.create_dataset('block_y', data=block_y, compression='gzip', compression_opts=1)
        group.create_dataset('block_z', data=block_z, compression='gzip', compression_opts=1)
        group.create_dataset('registers', data=registers, compression='gzip', compression_opts=1)
        group.create_dataset('shared_static', data=shared_static, compression='gzip', compression_opts=1)
        group.create_dataset('shared_dynamic', data=shared_dynamic, compression='gzip', compression_opts=1)
    
    def _write_memcpy_metadata(self, group: h5py.Group, events: List[Dict]):
        """Write memcpy-specific metadata."""
        n = len(events)
        bytes_transferred = np.zeros(n, dtype=np.uint32)
        
        for i, event in enumerate(events):
            meta = event.get('metadata', {})
            bytes_transferred[i] = meta.get('bytes', 0)
        
        group.create_dataset('bytes', data=bytes_transferred, compression='gzip', compression_opts=1)
    
    def _write_nvtx_metadata(self, group: h5py.Group, events: List[Dict]):
        """Write NVTX range-specific metadata."""
        n = len(events)
        
        # NVTX ranges can have color
        colors = np.zeros(n, dtype=np.uint32)
        
        for i, event in enumerate(events):
            meta = event.get('metadata', {})
            colors[i] = meta.get('color', 0)
        
        group.create_dataset('color', data=colors, compression='gzip', compression_opts=1)


def convert_json_to_hdf5(json_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Convenience function to convert existing JSON timeline to HDF5.
    
    Args:
        json_path: Path to JSON file with timeline data
        output_path: Output HDF5 path (default: same name with .h5 extension)
        
    Returns:
        Path to created HDF5 file
    """
    import json
    
    json_path = Path(json_path)
    if output_path is None:
        output_path = json_path.with_suffix('.h5')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    writer = ProfilingHDF5Writer()
    stats = writer.write(
        events=data.get('events', []),
        output_path=output_path,
        session_id=data.get('session_id', 'unknown'),
        total_duration_ns=data.get('total_duration_ns', 0)
    )
    
    print(f"Converted {stats['total_events']} events to {output_path}")
    print(f"File size: {stats['file_size_bytes'] / 1024:.1f} KB")
    for cat, cat_stats in stats['categories'].items():
        print(f"  {cat}: {cat_stats['count']} events, {cat_stats['unique_names']} unique names")
    
    return output_path


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python profiling_hdf5_writer.py <input.json> [output.h5]")
        sys.exit(1)
    
    json_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    convert_json_to_hdf5(json_path, output_path)
