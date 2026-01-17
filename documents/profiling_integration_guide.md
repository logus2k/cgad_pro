# Profiling Service Integration - fem_api_server.py

This document describes the changes needed to integrate the profiling service
into the FEM API server.

---

## Change 1: Add Import (near line 15)

**Location:** After existing service imports

**Add:**

```python
from profiling_service import init_profiling_service, create_profiling_router
```

---

## Change 2: Add Directory Constant (near line 30)

**Location:** After REPORT_DIR definition

**Add:**

```python
PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
BENCHMARK_SCRIPT = PROJECT_ROOT / "app" / "server" / "run_benchmark.py"  # Adjust path as needed
```

---

## Change 3: Initialize Service (near line 31)

**Location:** After report_service initialization

**Add:**

```python
profiling_service = init_profiling_service(PROFILES_DIR, BENCHMARK_SCRIPT)
```

---

## Change 4: Register Router (near line 88)

**Location:** After report_router registration

**Add:**

```python
profiling_router = create_profiling_router(profiling_service)
app.include_router(profiling_router)
```

---

## Complete Integration Block

Here's the complete block to add (can be inserted after report service setup):

```python
# =============================================================================
# Profiling Service (Nsight Systems / Compute Integration)
# =============================================================================

from profiling_service import init_profiling_service, create_profiling_router

PROFILES_DIR = PROJECT_ROOT / "data" / "profiles"
BENCHMARK_SCRIPT = PROJECT_ROOT / "app" / "server" / "run_benchmark.py"

profiling_service = init_profiling_service(PROFILES_DIR, BENCHMARK_SCRIPT)
profiling_router = create_profiling_router(profiling_service)
app.include_router(profiling_router)

print(f"[Server] Profiling service initialized - profiles dir: {PROFILES_DIR}")
```

---

## Optional: Add Profiling Flag to Benchmark Endpoint

If you want to trigger profiling from the existing benchmark endpoint,
modify the benchmark run endpoint to accept a profiling flag:

**In benchmark_service.py or the benchmark router:**

```python
class BenchmarkRunRequest(BaseModel):
    solver: str
    mesh_file: str
    # ... existing fields ...
    profiling_enabled: bool = False
    profiling_mode: str = "timeline"  # timeline, kernels, full


@router.post("/run")
async def run_benchmark(request: BenchmarkRunRequest):
    if request.profiling_enabled:
        # Delegate to profiling service
        return profiling_service.start_profiled_run(
            solver=request.solver,
            mesh_file=request.mesh_file,
            mode=request.profiling_mode
        )
    else:
        # Normal benchmark run
        return benchmark_service.run(...)
```

---

## API Endpoints After Integration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profiling/modes` | GET | Available profiling modes |
| `/api/profiling/run` | POST | Start profiled run |
| `/api/profiling/sessions` | GET | List all sessions |
| `/api/profiling/session/{id}` | GET | Session metadata |
| `/api/profiling/timeline/{id}` | GET | Timeline data (Gantt-ready) |
| `/api/profiling/kernels/{id}` | GET | Kernel metrics |
| `/api/profiling/session/{id}` | DELETE | Delete session |
| `/api/profiling/cleanup` | POST | Remove old sessions |

---

## Directory Structure After Integration

```
project_root/
├── src/app/server/
│   ├── fem_api_server.py      # Modified - router registration
│   ├── profiling_service.py   # New
│   ├── benchmark_service.py   # Existing
│   ├── report_service.py      # Existing
│   └── shared/
│       └── nvtx_helper.py     # New
│
├── data/
│   ├── profiles/              # New - auto-created
│   │   ├── sessions.json
│   │   ├── nsys/
│   │   │   ├── {session_id}.nsys-rep
│   │   │   └── {session_id}.sqlite
│   │   └── ncu/
│   │       ├── {session_id}.ncu-rep
│   │       └── {session_id}.csv
│   ├── input/
│   └── output/
```

---

## Verification

After integration, test with:

```bash
# Start server
python fem_api_server.py

# Check profiling modes
curl http://localhost:8000/api/profiling/modes

# Expected response:
{
  "modes": [
    {"id": "timeline", "name": "Timeline (nsys)", "available": true, ...},
    {"id": "kernels", "name": "Kernel Analysis (ncu)", "available": true, ...},
    {"id": "full", "name": "Full Analysis", "available": true, ...}
  ],
  "nsys_available": true,
  "ncu_available": true
}
```
