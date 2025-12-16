# Statement of Work (SoW)

**Real-Time CPU vs GPU QUAD8 FEM Solver Visualization Platform**

---

## 1. Purpose & Scope

### 1.1 Purpose

Design and implement a **real-time visualization frontend** for a QUAD8 FEM solver that:

* Compares **CPU vs GPU implementations**
* Streams **metrics and results in real time**
* Renders **large meshes (100k–millions of nodes) efficiently**
* Uses **Three.js** for GPU-accelerated visualization
* Provides **clear, meaningful UX for technical users**

This platform is **not** a CFD time-marching simulator; it visualizes **static FEM solutions** with **animated visual cues**.

---

## 2. Architectural Principles (Locked)

1. **Solver-agnostic frontend**

   * Frontend consumes events, not solver logic
2. **One physical Socket.IO connection**
3. **Two logical channels (“rooms”)**

   * Metrics
   * Rendering
4. **Binary data where it matters**
5. **All heavy preprocessing happens server-side**
6. **Frontend is a GPU buffer consumer, not a numerical engine**
7. **Scalability to millions of nodes is a hard requirement**

---

## 3. Solver Assumptions (Explicit)

### 3.1 Solver behavior

* Primary GPU solver: **CuPy CG**
* GMRES:

  * Exists as **fallback / reference**
  * May run on CPU
* System is assumed **SPD after BC enforcement**
* Solver provides:

  * Nodal scalar fields
  * Element or nodal velocity fields
  * Timing metrics per stage

### 3.2 Solver stages (canonical)

1. Process start
2. Mesh load
3. Assembly
4. Boundary conditions
5. Solve
6. Post-processing
7. Process end

---

## 4. Communication Model

### 4.1 Transport

* **Socket.IO**
* Single TCP/WebSocket connection
* Two logical rooms:

  * `metrics`
  * `render`

---

## 5. Data Format Decisions (Locked)

### 5.1 Metrics channel

* **JSON-first**
* Binary only if strictly needed in the future
* Payloads are small and frequent

### 5.2 Render channel

* **Binary-only for numeric data**
* TypedArrays (`Float32Array`, `Uint32Array`)
* Small JSON envelopes for metadata

---

## 6. Event Specification (Checklist)

### 6.1 Process-level events (metrics channel)

| Event           | Required | Description             |
| --------------- | -------- | ----------------------- |
| `process:start` | ✅        | Global start timestamp  |
| `process:end`   | ✅        | Total wall time, status |
| `session:error` | ✅        | Fatal solver error      |

**Notes**

* Total process time is a **first-class metric**
* Must be explicitly emitted

---

### 6.2 Stage events (metrics channel)

| Event         | Required |
| ------------- | -------- |
| `stage:start` | ✅        |
| `stage:end`   | ✅        |

Payload includes:

* stage name
* duration (on end)

---

### 6.3 Solver iteration events (metrics channel)

| Event              | Required | Notes         |
| ------------------ | -------- | ------------- |
| `solver:iteration` | ✅        | Throttled     |
| `solver:converged` | ✅        | Once          |
| `solver:fallback`  | ⚠️       | If CG → GMRES |

Payload includes:

* iteration index
* residual norm
* ETA (if available)

---

### 6.4 Mesh events (render channel)

#### Static geometry (one-time)

| Event                    | Required |
| ------------------------ | -------- |
| `render:geometry:static` | ✅        |

Contains:

* Pre-triangulated indexed geometry
* 2D positions (z=0)
* Indices
* Bounding box

---

### 6.5 Field events (render channel)

| Field              | Required |
| ------------------ | -------- |
| Potential          | ✅        |
| Velocity magnitude | ✅        |
| Pressure           | ✅        |

Each field:

* Sent **only once per computation**
* Sent as `Float32Array`
* Matches preprocessed vertex layout

---

## 7. Server-Side Pre-Processing (Critical)

### 7.1 Geometry preprocessing (once)

**Triggered:** immediately after mesh load

Server responsibilities:

* Quad8 → triangle triangulation
* Vertex list generation
* Index buffer generation
* Bounding box computation
* Cache geometry for reuse

Frontend must **not**:

* Know about Quad8
* Triangulate
* Duplicate vertices

---

### 7.2 Field preprocessing

**Triggered:** immediately after field computation

Server responsibilities:

* Map solver output → render vertices
* Compute extrusion:

  ```
  z = scale * scalar_field
  ```
* Prepare `Float32Array` buffers

---

## 8. Three.js Rendering Requirements

### 8.1 Geometry

* `THREE.BufferGeometry`
* Indexed
* Uploaded once

### 8.2 Scalar visualization

* Color-mapped scalar fields
* Toggleable (potential, pressure, velocity magnitude)
* No geometry rebuild on field switch

### 8.3 Extrusion (“fake 3D”)

* z-coordinate encodes scalar
* Scaling adjustable in UI
* Geometry remains static in XY

---

## 9. Velocity Visualization (Critical Clarification)

### 9.1 What is explicitly **not** required

* ❌ One arrow per node
* ❌ One arrow per element
* ❌ Particle advection
* ❌ Time-dependent solver data

---

### 9.2 Approved velocity visualization model

**Sparse, sampled vector field**

* Independent of mesh resolution
* Density controlled by UI
* Typical count: 1k–10k arrows

---

### 9.3 Sampling strategy (locked)

* Regular 2D grid over domain
* Sampling performed **server-side**
* Server sends:

  * Arrow centers
  * Velocity vectors
  * Optional magnitude

---

### 9.4 Animation model

* Purely visual
* Time-based phase offset along velocity direction
* No solver involvement after data is sent

---

## 10. Performance & Scalability Constraints

| Constraint         | Requirement             |
| ------------------ | ----------------------- |
| Mesh size          | Up to millions of nodes |
| Geometry uploads   | Once                    |
| JS per-frame loops | ❌                       |
| JS triangulation   | ❌                       |
| GPU draw calls     | Minimal                 |
| Memory duplication | Avoided                 |

---

## 11. UX / UI Requirements

### 11.1 Always visible

* Total process time
* Solver backend (CPU / GPU)
* Solver type (CG / GMRES)
* Convergence status

### 11.2 Panels

* Metrics panel (progress, residual plot)
* Main 3D viewport
* Field selector
* Velocity overlay toggle

---

## 12. Explicit Non-Goals

* No CFD time marching
* No transient simulation
* No solver parameter tuning UI
* No FEM logic in browser
* No browser-side preprocessing

---

## 13. Deliverables (Checklist)

* [ ] Socket.IO event schema
* [ ] Server preprocessing pipeline
* [ ] Geometry cache
* [ ] Render channel binary emitters
* [ ] Metrics channel emitters
* [ ] Three.js static geometry renderer
* [ ] Scalar field updates
* [ ] Sparse velocity arrow overlay
* [ ] Residual plot UI
* [ ] Process timing UI

---

## 14. Acceptance Criteria

The system is considered complete when:

* Mesh renders **before** solver finishes
* Scalar field fills appear immediately on availability
* Metrics update continuously without UI stalls
* Velocity arrows are readable at large mesh sizes
* CPU vs GPU runs are visually and metrically comparable
* No browser freezes on million-node meshes

---

## 15. Final Notes

This SoW:

* Captures **all agreed decisions**
* Explicitly documents assumptions
* Prevents scope creep
* Is suitable as a **development checklist**

---
