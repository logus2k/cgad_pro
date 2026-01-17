# NVTX Integration - Surgical Changes for quad8_gpu_v3.py

This document describes the minimal changes needed to add NVTX profiling
instrumentation to the GPU solver without modifying its core functionality.

## Change 1: Add Import (after line 36)

**Location:** After the existing imports, around line 36

**Add these lines:**

```python
# NVTX profiling support (optional - graceful fallback if not installed)
try:
    from nvtx_helper import nvtx_range, nvtx_available
except ImportError:
    from contextlib import contextmanager
    nvtx_available = False
    @contextmanager
    def nvtx_range(name, color=None, domain=None):
        yield
```

---

## Change 2: Modify _time_step Method (around line 436)

**Current code:**

```python
def _time_step(self, name: str, fn: Callable) -> Any:
    t0 = time.perf_counter()
    result = fn()
    self.timing_metrics[name] = time.perf_counter() - t0
    if self.verbose:
        print(f"  {name}: {self.timing_metrics[name]:.3f}s")
    return result
```

**Replace with:**

```python
def _time_step(self, name: str, fn: Callable) -> Any:
    """Execute a pipeline stage with timing and NVTX annotation."""
    with nvtx_range(name):
        t0 = time.perf_counter()
        result = fn()
        # Ensure GPU work is complete before timing
        cp.cuda.Stream.null.synchronize()
        self.timing_metrics[name] = time.perf_counter() - t0
    if self.verbose:
        print(f"  {name}: {self.timing_metrics[name]:.3f}s")
    return result
```

---

## Change 3: Add NVTX Range Around CG Solve Loop (around line 832)

**Location:** Inside the `solve()` method, around the CG iteration loop

**Current code (approximately):**

```python
t0_solve = time.perf_counter()

# ... CG solve call ...
self.u_cpu, info = cg(
    K_cpu,
    rhs_cpu,
    x0=x0,
    M=M_linop,
    tol=self.rtol,
    maxiter=self.maxiter,
    callback=gpu_monitor
)

t1_solve = time.perf_counter()
```

**Replace with:**

```python
t0_solve = time.perf_counter()

with nvtx_range("cg_iteration"):
    # ... CG solve call ...
    self.u_cpu, info = cg(
        K_cpu,
        rhs_cpu,
        x0=x0,
        M=M_linop,
        tol=self.rtol,
        maxiter=self.maxiter,
        callback=gpu_monitor
    )

t1_solve = time.perf_counter()
```

---

## Change 4: Optional - Add Kernel-Level NVTX (in assemble_system)

For finer granularity in the assembly stage, add NVTX around kernel launches.

**Location:** Inside `assemble_system()` method, around line 560

**Current code:**

```python
# Launch kernel
self.quad8_kernel(
    grid_size,
    block_size,
    (self.x, self.y, self.quad8_gpu, ...)
)
cp.cuda.Stream.null.synchronize()
```

**Replace with:**

```python
# Launch kernel
with nvtx_range("kernel_launch"):
    self.quad8_kernel(
        grid_size,
        block_size,
        (self.x, self.y, self.quad8_gpu, ...)
    )
    cp.cuda.Stream.null.synchronize()
```

---

## Summary of Changes

| Location | Change Type | Lines Affected |
|----------|-------------|----------------|
| After imports (~line 36) | Add import | +7 lines |
| `_time_step` method (~line 436) | Modify | ~3 lines changed |
| `solve()` CG loop (~line 832) | Wrap with nvtx_range | +2 lines |
| `assemble_system()` kernel (~line 560) | Optional wrap | +2 lines |

**Total impact:** ~12-14 lines added/modified in a ~1100 line file

---

## Verification

After applying changes, verify with:

```bash
# Without profiling (should work unchanged)
python quad8_gpu_v3.py --mesh test_mesh.h5

# With profiling
nsys profile -o test_profile python quad8_gpu_v3.py --mesh test_mesh.h5

# View NVTX ranges
nsys-ui test_profile.nsys-rep
```

The NVTX ranges should appear as colored bars in the Nsight Systems timeline,
grouped by pipeline stage (load_mesh, assemble_system, apply_bc, etc.).
