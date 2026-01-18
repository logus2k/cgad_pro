# FEMulator Pro - GPU Profiling Service

## Technical Implementation

This document describes the complete architecture and evolution of the GPU Profiling Service in FEMulator Pro, from initial prototype to production-ready implementation with GPU-accelerated processing and WebGL rendering.

## Overview

The Profiling Service provides detailed GPU performance analysis for FEM solver operations by integrating with NVIDIA Nsight Systems and Nsight Compute. It captures, processes, and visualizes:

- **CUDA Kernel executions** - timing, grid/block dimensions, register usage, shared memory
- **Memory transfers** - Host-to-Device, Device-to-Host, Device-to-Device operations
- **NVTX ranges** - custom application-defined profiling regions (load_mesh, assemble_system, solve_system, etc.)

The system handles datasets of **200,000+ events** with sub-second processing times and smooth 60fps visualization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SERVER (Python/FastAPI)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────────┐   │
│  │    nsys      │───▶│  .nsys-rep   │───▶│     nsys export --hdf       │   │
│  │  profiler    │    │   (binary)   │    │                             │   │
│  └──────────────┘    └──────────────┘    └──────────────┬──────────────┘   │
│                                                         │                   │
│                                                         ▼                   │
│                                          ┌──────────────────────────────┐   │
│                                          │       .nsys.h5               │   │
│                                          │   (Nsight native HDF5)       │   │
│                                          └──────────────┬───────────────┘   │
│                                                         │                   │
│                                                         ▼                   │
│                                          ┌──────────────────────────────┐   │
│                                          │  ProfilingHDF5Transformer    │   │
│                                          │    (CuPy GPU-accelerated)    │   │
│                                          └──────────────┬───────────────┘   │
│                                                         │                   │
│                                                         ▼                   │
│                                          ┌──────────────────────────────┐   │
│                                          │          .h5                 │   │
│                                          │  (Client-optimized HDF5)     │   │
│                                          └──────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      ProfilingService                                 │   │
│  │  • Session management          • Socket.IO event emission            │   │
│  │  • Background profiling        • FastAPI REST endpoints              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       │ HTTP/WebSocket
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLIENT (Browser)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌────────────────────┐    │
│  │   ProfilingAPI   │───▶│ ProfilingHDF5    │───▶│   h5wasm (WASM)    │    │
│  │   (fetch + progress)  │    Loader        │    │                    │    │
│  └──────────────────┘    └──────────────────┘    └────────────────────┘    │
│           │                       │                                         │
│           │                       ▼                                         │
│           │              ┌──────────────────┐                               │
│           │              │   Web Worker     │                               │
│           │              │ (data processing)│                               │
│           │              └────────┬─────────┘                               │
│           │                       │                                         │
│           ▼                       ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       ProfilingView                                   │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │   │
│  │  │ Session List   │  │ Control Panel  │  │   TimelineRenderer     │  │   │
│  │  │                │  │                │  │   (Three.js/WebGL)     │  │   │
│  │  │                │  │                │  │   • InstancedMesh      │  │   │
│  │  │                │  │                │  │   • GPU instancing     │  │   │
│  │  │                │  │                │  │   • Virtual scrolling  │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Evolution & Optimizations

### Phase 1: Initial Prototype (Baseline)

**Server-Side:**
- NVIDIA Nsight exports to **SQLite** format
- Python parses SQLite with row-by-row iteration
- Creates Python dictionaries for each event
- Serves data as **JSON** via REST endpoint

**Client-Side:**
- Fetches JSON (large payload, no progress indication)
- Uses **vis-timeline** library (DOM-based)
- Creates individual **DOM elements** for each event
- Synchronous rendering blocks UI thread

**Problems:**
- 16+ seconds to process 236K events on server
- JSON payload of 50+ MB for large sessions
- Browser freezes during DOM element creation
- vis-timeline cannot handle >10K items efficiently
- No progress feedback during long operations

### Phase 2: HDF5 + Web Worker (Intermediate)

**Server-Side:**
- Added HDF5 export using `ProfilingHDF5Writer`
- Still parsed SQLite first, then converted to HDF5
- HDF5 schema optimized for client consumption:
  - Pre-grouped by category
  - Pre-sorted by start time
  - Deduplicated kernel names with index references
  - Typed arrays ready for GPU upload

**Client-Side:**
- **h5wasm** (WebAssembly HDF5 reader) replaces JSON parsing
- **Web Worker** (`profiling.worker.js`) offloads data processing
- Download progress via `Content-Length` header
- Still used vis-timeline for rendering

**Improvements:**
- HDF5 files ~10-20x smaller than JSON
- Accurate download progress indication
- UI no longer blocks during data processing
- But server-side processing still slow (16s)

### Phase 3: Three.js WebGL Renderer (Client Optimization)

**Replaced vis-timeline with custom Three.js renderer:**

```javascript
// TimelineRenderer using InstancedMesh
class TimelineRenderer {
    constructor(container) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(...);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        
        // One InstancedMesh per category - renders 100K+ items in single draw call
        this.meshes = new Map(); // category -> InstancedMesh
    }
    
    rebuildMeshes(data) {
        for (const [category, catData] of Object.entries(data.byCategory)) {
            const geometry = new THREE.PlaneGeometry(1, 1);
            const material = new THREE.MeshBasicMaterial({ color: categoryColors[category] });
            
            // Single mesh with N instances - GPU handles positioning
            const mesh = new THREE.InstancedMesh(geometry, material, catData.count);
            
            // Upload instance transforms directly from typed arrays
            for (let i = 0; i < catData.count; i++) {
                matrix.makeTranslation(catData.startMs[i], streamY, 0);
                matrix.scale(catData.durationMs[i], barHeight, 1);
                mesh.setMatrixAt(i, matrix);
            }
            
            mesh.instanceMatrix.needsUpdate = true;
            this.meshes.set(category, mesh);
        }
    }
}
```

**Key optimizations:**
- **InstancedMesh**: Renders 200K+ rectangles in ~6 draw calls (one per category)
- **Typed arrays**: Float32Array data uploads directly to GPU
- **Virtual viewport**: Only renders visible time range
- **RequestAnimationFrame**: Smooth 60fps pan/zoom

### Phase 4: CuPy GPU-Accelerated Transform (Final)

**Server-Side Revolution:**

Eliminated Python dict creation entirely with direct HDF5-to-HDF5 GPU transform:

```python
class ProfilingHDF5Transformer:
    def __init__(self, use_gpu=True):
        self.xp = cp if use_gpu else np  # CuPy or NumPy
    
    def transform(self, nsys_h5_path, output_path, session_id):
        with h5py.File(nsys_h5_path, 'r') as src:
            # Load directly to GPU memory
            kernels = src['CUPTI_ACTIVITY_KIND_KERNEL'][:]
            start_ns = self.xp.asarray(kernels['start'], dtype=self.xp.uint64)
            end_ns = self.xp.asarray(kernels['end'], dtype=self.xp.uint64)
            
            # GPU-accelerated sorting
            sort_idx = self.xp.argsort(start_ns)
            start_ns = start_ns[sort_idx]
            
            # Transfer back and write
            with h5py.File(output_path, 'w') as dst:
                dst.create_dataset('start_ns', data=start_ns.get())
```

**Data flow comparison:**

```
BEFORE (Phase 2):
  nsys-rep → SQLite → Python dicts → HDF5 → Client
            (slow)    (16s, 236K iterations)

AFTER (Phase 4):
  nsys-rep → HDF5 → CuPy GPU arrays → HDF5 → Client
            (0.4s)  (0.2s, parallel)
```

## Server-Side Implementation

### Core Components

#### `profiling_service.py`

Main service orchestrating profiling operations:

```python
class ProfilingService:
    def __init__(self, profiles_dir: Path):
        self.nsys_dir = profiles_dir / "nsys"
        self.ncu_dir = profiles_dir / "ncu"
        self._event_callback = None  # Socket.IO integration
    
    def start_profiled_run(self, solver, mesh_file, mode):
        """Launch benchmark wrapped with nsys/ncu profiling."""
        session_id = str(uuid.uuid4())[:8]
        
        # Background thread for non-blocking execution
        thread = threading.Thread(
            target=self._run_profiling,
            args=(session_id, solver, mesh_file, mode),
            daemon=True
        )
        thread.start()
        
        return {"session_id": session_id, "status": "pending"}
    
    def generate_timeline_hdf5(self, session_id):
        """Generate client-optimized HDF5 using GPU acceleration."""
        # Step 1: Export nsys-rep to native HDF5
        self._export_nsys_hdf5(nsys_rep, nsys_h5_path)
        
        # Step 2: GPU-accelerated transform
        transformer = ProfilingHDF5Transformer(use_gpu=True)
        stats = transformer.transform(nsys_h5_path, output_path, session_id)
        
        return stats
```

#### `profiling_hdf5_transformer.py`

GPU-accelerated HDF5 transformation:

```python
class ProfilingHDF5Transformer:
    def _extract_kernels(self, f, string_ids):
        """Extract CUDA kernels with GPU acceleration."""
        kernels = f['CUPTI_ACTIVITY_KIND_KERNEL'][:]
        
        # Load to GPU
        start_ns = self.xp.asarray(kernels['start'], dtype=self.xp.uint64)
        end_ns = self.xp.asarray(kernels['end'], dtype=self.xp.uint64)
        duration_ns = end_ns - start_ns
        
        # GPU-parallel operations
        sort_idx = self.xp.argsort(start_ns)
        
        return {
            'start_ns': start_ns[sort_idx],
            'duration_ns': duration_ns[sort_idx],
            # ... other fields
        }
```

### HDF5 Schema (Client-Optimized)

```
/meta
    session_id: string (attribute)
    total_duration_ns: uint64 (attribute)
    total_events: uint32 (attribute)
    categories: string[] (dataset)

/<category>/  (cuda_kernel, cuda_memcpy_h2d, etc.)
    start_ns: uint64[N]      # Normalized timestamps
    duration_ns: uint64[N]   # Event durations
    stream: uint8[N]         # CUDA stream IDs
    name_idx: uint16[N]      # Index into /names/<category>
    
    # Category-specific:
    # cuda_kernel: grid_x/y/z, block_x/y/z, registers, shared_static/dynamic
    # cuda_memcpy_*: bytes
    # nvtx_range: color

/names/
    <category>: string[]     # Deduplicated name lookup table
```

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/profiling/modes` | GET | Available profiling modes |
| `/api/profiling/run` | POST | Start profiled benchmark |
| `/api/profiling/sessions` | GET | List profiling sessions |
| `/api/profiling/session/{id}` | GET | Session metadata |
| `/api/profiling/timeline/{id}.h5` | GET | Download HDF5 timeline |
| `/api/profiling/timeline/{id}/status` | GET | Generation status |
| `/api/profiling/kernels/{id}` | GET | Nsight Compute metrics |
| `/api/profiling/session/{id}` | DELETE | Delete session |

### Socket.IO Events

Real-time progress updates via WebSocket:

```python
def _emit_event(self, event_type, session_id, **data):
    """Emit profiling events to connected clients."""
    if self._event_callback:
        self._event_callback(event_type, session_id, data)

# Events emitted:
# - profiling_started: {session_id, solver, mesh, mode}
# - profiling_progress: {session_id, status, stage, message}
# - profiling_complete: {session_id, status}
# - profiling_error: {session_id, error}
```

## Client-Side Implementation

### Core Components

#### `profiling.api.js`

REST client with HDF5 preference:

```javascript
class ProfilingAPI {
    async getTimeline(sessionId, options = {}) {
        if (this.#preferHDF5) {
            try {
                return await this.#getTimelineHDF5(sessionId, onProgress);
            } catch (err) {
                console.warn('HDF5 failed, falling back to JSON:', err);
            }
        }
        return await this.#getTimelineJSON(sessionId, onProgress);
    }
    
    async #getTimelineHDF5(sessionId, onProgress) {
        const url = `${this.#baseUrl}/api/profiling/timeline/${sessionId}.h5`;
        
        // Load with progress tracking
        const data = await loadProfilingHDF5(url, (loaded, total, phase) => {
            onProgress(phase, loaded, total);
        });
        
        return {
            session_id: data.sessionId,
            events: data.toEvents(),  // Or data.getRendererData() for typed arrays
            total_duration_ns: data.totalDurationNs
        };
    }
}
```

#### `profiling_hdf5_loader.js`

WebAssembly HDF5 parser:

```javascript
export async function loadProfilingHDF5(url, onProgress) {
    await initH5Wasm();  // Load h5wasm library once
    
    // Download with progress
    const buffer = await fetchWithProgress(url, (loaded, total) => {
        onProgress(loaded, total, 'download');
    });
    
    // Parse HDF5 in virtual filesystem
    const { FS } = await h5wasm.ready;
    FS.writeFile(tempFilename, new Uint8Array(buffer));
    
    const file = new h5wasm.File(tempFilename, 'r');
    return parseHDF5FromFile(file);
}

function parseHDF5FromFile(file) {
    const meta = file.get('meta');
    const categories = file.get('meta/categories').value;
    
    const categoryData = {};
    for (const category of categories) {
        categoryData[category] = {
            start_ns: file.get(`${category}/start_ns`).value,    // BigUint64Array
            duration_ns: file.get(`${category}/duration_ns`).value,
            stream: file.get(`${category}/stream`).value,
            name_idx: file.get(`${category}/name_idx`).value,
            names: file.get(`names/${category}`).value
        };
    }
    
    return {
        sessionId: meta.attrs['session_id'].value,
        totalDurationNs: Number(meta.attrs['total_duration_ns'].value),
        categoryData,
        
        toEvents() { return convertToEvents(this); },
        getRendererData() { return prepareRendererData(this); }
    };
}
```

#### `profiling.worker.js`

Web Worker for data processing:

```javascript
self.onmessage = function(e) {
    const { type, payload, requestId } = e.data;
    
    switch (type) {
        case 'transform':
            result = transformEvents(payload);  // Convert to vis.js format
            break;
        case 'filter':
            result = filterEvents(payload);     // Category filtering
            break;
        case 'summary':
            result = computeSummary(payload);   // Statistics
            break;
        case 'buildGroups':
            result = buildGroups(payload);      // Timeline groups
            break;
    }
    
    self.postMessage({ requestId, type: 'result', payload: result });
};
```

#### `timeline_renderer.js`

Three.js WebGL renderer:

```javascript
class TimelineRenderer {
    constructor(container) {
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(...);
        
        // Category colors
        this.colors = {
            cuda_kernel: 0xe74c3c,
            cuda_memcpy_h2d: 0x3498db,
            cuda_memcpy_d2h: 0x2ecc71,
            cuda_memcpy_d2d: 0xf39c12,
            nvtx_range: 0x9b59b6
        };
    }
    
    setData(rendererData) {
        this.data = rendererData;
        this.rebuildMeshes();
    }
    
    rebuildMeshes() {
        // Clear existing meshes
        this.meshes.forEach(mesh => this.scene.remove(mesh));
        this.meshes.clear();
        
        for (const [category, catData] of Object.entries(this.data.byCategory)) {
            const geometry = new THREE.PlaneGeometry(1, 1);
            const material = new THREE.MeshBasicMaterial({
                color: this.colors[category]
            });
            
            // InstancedMesh: single draw call for all events in category
            const mesh = new THREE.InstancedMesh(
                geometry, material, catData.count
            );
            
            const matrix = new THREE.Matrix4();
            
            for (let i = 0; i < catData.count; i++) {
                const x = catData.startMs[i];
                const width = Math.max(catData.durationMs[i], 0.001);
                const y = this.getStreamY(catData.stream[i]);
                
                matrix.makeTranslation(x + width/2, y, 0);
                matrix.scale(new THREE.Vector3(width, this.barHeight, 1));
                mesh.setMatrixAt(i, matrix);
            }
            
            mesh.instanceMatrix.needsUpdate = true;
            this.scene.add(mesh);
            this.meshes.set(category, mesh);
        }
    }
    
    render() {
        this.renderer.render(this.scene, this.camera);
    }
}
```

## Data Flow

### Complete Request Flow

```
1. User clicks "Load Timeline" for session ABC123

2. CLIENT: ProfilingAPI.getTimeline('ABC123')
   ├── HEAD /api/profiling/timeline/ABC123.h5
   │   └── Check if HDF5 exists, get Content-Length
   └── GET /api/profiling/timeline/ABC123.h5
       └── Stream download with progress callback

3. SERVER: get_timeline_hdf5 endpoint
   ├── Check if ABC123.h5 exists
   ├── If not: generate_timeline_hdf5()
   │   ├── _export_nsys_hdf5() → ABC123.nsys.h5
   │   └── ProfilingHDF5Transformer.transform()
   │       ├── Load nsys HDF5 to CuPy GPU arrays
   │       ├── Sort, filter, normalize on GPU
   │       └── Write client-optimized HDF5
   └── Return FileResponse(ABC123.h5)

4. CLIENT: loadProfilingHDF5()
   ├── fetchWithProgress() → ArrayBuffer
   ├── h5wasm.File.parse() → Typed arrays
   └── Return ProfilingData object

5. CLIENT: TimelineRenderer.setData()
   ├── getRendererData() → Float32Arrays
   ├── rebuildMeshes() → InstancedMesh per category
   └── render() → WebGL draw calls

6. User sees interactive timeline at 60fps
```

### Timing Breakdown (236K events)

| Stage | Time | Location |
|-------|------|----------|
| nsys export to HDF5 | 0.43s | Server (nsys CLI) |
| GPU transform | 0.19s | Server (CuPy) |
| Network transfer | ~0.2s | 1.3MB HDF5 |
| h5wasm parse | ~0.05s | Browser (WASM) |
| Mesh creation | ~0.006s | Browser (Three.js) |
| **Total** | **~0.9s** | End-to-end |

## Performance Results

### Server-Side Comparison

| Metric | SQLite + Python | HDF5 + CuPy | Improvement |
|--------|-----------------|-------------|-------------|
| Processing time | 16.5s | 0.62s | **26x faster** |
| Peak memory | ~2GB (dicts) | ~200MB (arrays) | **10x less** |
| Output size | 50MB (JSON) | 1.3MB (HDF5) | **38x smaller** |

### Client-Side Comparison

| Metric | vis-timeline (DOM) | Three.js (WebGL) | Improvement |
|--------|-------------------|------------------|-------------|
| Render time | 30+ seconds | 6ms | **5000x faster** |
| Max events | ~10,000 | 500,000+ | **50x more** |
| Frame rate | <1 fps | 60 fps | Smooth |
| Memory | Crashes | ~100MB | Stable |

### Network Comparison

| Format | Size (236K events) | Parse time |
|--------|-------------------|------------|
| JSON | ~50MB | 2-3s |
| HDF5 | 1.3MB | 50ms |

## File Reference

### Server Files

| File | Purpose |
|------|---------|
| `profiling_service.py` | Main service, session management, REST endpoints |
| `profiling_hdf5_transformer.py` | CuPy GPU-accelerated HDF5 transform |
| `profiling_hdf5_writer.py` | Legacy HDF5 writer (deprecated) |

### Client Files

| File | Purpose |
|------|---------|
| `profiling.api.js` | REST client, HDF5/JSON loading |
| `profiling_hdf5_loader.js` | h5wasm integration, typed array extraction |
| `profiling.worker.js` | Web Worker for data processing |
| `profiling.view.js` | Main UI component, session list, controls |
| `timeline_renderer.js` | Three.js WebGL timeline visualization |

### Dependencies

**Server:**
- Python 3.10+
- FastAPI + Uvicorn
- h5py
- CuPy (CUDA toolkit required)
- NVIDIA Nsight Systems (`nsys`)
- NVIDIA Nsight Compute (`ncu`)

**Client:**
- h5wasm (WebAssembly HDF5 reader)
- Three.js (WebGL rendering)
- Socket.IO (real-time updates)

## Future Optimizations

Potential areas for further improvement:

1. **Streaming HDF5**: Progressive loading for very large sessions
2. **GPU-based LOD**: Level-of-detail rendering for zoom levels
3. **Shared ArrayBuffer**: Zero-copy transfer between Worker and main thread
4. **IndexedDB caching**: Persist parsed data for instant reload
5. **WebGPU**: Next-gen GPU API when widely available

## Conclusion

The Profiling Service evolution demonstrates how targeted optimizations at each layer of the stack compound into dramatic performance improvements:

- **Server**: Python loops → CuPy GPU parallelism (87x faster transform)
- **Format**: JSON → HDF5 (38x smaller, typed arrays)
- **Rendering**: DOM elements → WebGL instancing (5000x faster)
- **Threading**: Synchronous → Web Workers (non-blocking UI)

The result is a system capable of handling 200K+ GPU events with sub-second load times and smooth 60fps interaction, compared to the original prototype that would freeze for 30+ seconds on the same dataset.

