"""
NVTX Helper - NVIDIA Tools Extension for profiling instrumentation.

Provides graceful fallback when nvtx is not installed, allowing
solver code to run unchanged whether profiling is enabled or not.

Location: /src/app/server/shared/nvtx_helper.py

Usage:
    from nvtx_helper import nvtx_range, nvtx_available, NVTX_COLORS
    
    # Context manager style
    with nvtx_range("assemble_system"):
        # ... code to profile ...
    
    # Or check availability
    if nvtx_available:
        # do something nvtx-specific

Installation:
    pip install nvtx
"""

from contextlib import contextmanager
from typing import Optional

# Attempt to import nvtx
try:
    import nvtx
    nvtx_available = True
except ImportError:
    nvtx = None
    nvtx_available = False


# =============================================================================
# Color Definitions (matching profiling_service.py)
# =============================================================================

NVTX_COLORS = {
    # Pipeline stages - consistent with profiling_service.py
    "load_mesh": 0x3498db,        # Blue
    "assemble_system": 0x2ecc71,  # Green
    "apply_bc": 0xf1c40f,         # Yellow
    "solve_system": 0xe74c3c,     # Red
    "compute_derived": 0x9b59b6,  # Purple
    "export_results": 0x95a5a6,   # Gray
    "visualize": 0x1abc9c,        # Teal
    
    # Sub-operations
    "kernel_launch": 0xe67e22,    # Orange
    "memory_transfer": 0x3498db,  # Blue
    "sparse_convert": 0x9b59b6,   # Purple
    "cg_iteration": 0xe74c3c,     # Red
}


# =============================================================================
# NVTX Range Context Manager
# =============================================================================

@contextmanager
def nvtx_range(name: str, color: Optional[int] = None, domain: str = "FEMulator"):
    """
    Context manager for NVTX range annotation.
    
    Wraps code in an NVTX range for visualization in Nsight Systems.
    Falls back to no-op if nvtx is not installed.
    
    Args:
        name: Range name (appears in timeline)
        color: Optional color as hex int (e.g., 0xFF0000 for red)
        domain: NVTX domain name for grouping
        
    Usage:
        with nvtx_range("assemble_system"):
            assemble_system()
    """
    if nvtx_available and nvtx is not None:
        # Get color from predefined map or use provided
        if color is None:
            color = NVTX_COLORS.get(name, 0x7f8c8d)  # Default gray
        
        # Push range with color
        nvtx.push_range(name, color=color)
        try:
            yield
        finally:
            nvtx.pop_range()
    else:
        # No-op fallback
        yield


def nvtx_mark(message: str, color: Optional[int] = None):
    """
    Place an instantaneous marker in the timeline.
    
    Args:
        message: Marker message
        color: Optional color as hex int
    """
    if nvtx_available and nvtx is not None:
        if color is None:
            color = 0x7f8c8d
        nvtx.mark(message, color=color)


# =============================================================================
# Decorator for Function-Level Annotation
# =============================================================================

def nvtx_annotate(name: Optional[str] = None, color: Optional[int] = None):
    """
    Decorator to wrap a function in an NVTX range.
    
    Args:
        name: Range name (defaults to function name)
        color: Optional color as hex int
        
    Usage:
        @nvtx_annotate("my_function")
        def my_function():
            ...
    """
    def decorator(func):
        range_name = name or func.__name__
        range_color = color or NVTX_COLORS.get(range_name, 0x7f8c8d)
        
        if nvtx_available and nvtx is not None:
            # Use nvtx's built-in annotate decorator
            return nvtx.annotate(range_name, color=range_color)(func)
        else:
            # Return function unchanged
            return func
    
    return decorator


# =============================================================================
# Sync Marker for GPU Operations
# =============================================================================

def nvtx_sync_mark(message: str = "cuda_sync"):
    """
    Mark a CUDA synchronization point.
    
    Call this after cp.cuda.Stream.null.synchronize() or similar
    to mark sync points in the timeline.
    """
    nvtx_mark(message, color=0x95a5a6)


# =============================================================================
# Module Info
# =============================================================================

def get_nvtx_status() -> dict:
    """Get NVTX availability status."""
    return {
        "available": nvtx_available,
        "version": getattr(nvtx, "__version__", None) if nvtx_available else None,
        "colors_defined": len(NVTX_COLORS)
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print(f"NVTX Helper Status: {get_nvtx_status()}")
    
    # Test context manager
    with nvtx_range("test_range"):
        print("Inside NVTX range")
    
    # Test marker
    nvtx_mark("test_marker")
    
    # Test decorator
    @nvtx_annotate("decorated_function")
    def test_func():
        print("Inside decorated function")
    
    test_func()
    
    print("All tests passed (no-op if nvtx not installed)")
