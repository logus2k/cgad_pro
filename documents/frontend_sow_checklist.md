# Implementation Checklist

## 1. Core Architecture

* [ ] Single physical Socket.IO connection
* [ ] Two logical rooms:

  * [ ] `metrics`
  * [ ] `render`
* [ ] Binary payload support enabled
* [ ] Solver-agnostic event emission layer

---

## 2. Process Lifecycle Events (Metrics)

* [ ] Emit `process:start` (global start timestamp)
* [ ] Emit `process:end` (total wall time, status)
* [ ] Emit `session:error` on fatal failure
* [ ] Store total process time explicitly (not derived)

---

## 3. Stage-Level Metrics

* [ ] Emit `stage:start` for each solver stage
* [ ] Emit `stage:end` with duration
* [ ] Stages covered:

  * [ ] load_mesh
  * [ ] assemble_system
  * [ ] apply_bc
  * [ ] solve
  * [ ] postprocess

---

## 4. Solver Iteration Telemetry

* [ ] Emit `solver:iteration` (throttled)
* [ ] Emit `solver:converged`
* [ ] Emit `solver:fallback` (CG → GMRES, if triggered)
* [ ] Report solver type and backend (CPU/GPU)

---

## 5. Mesh Geometry Preprocessing (Server)

* [ ] Triggered immediately after mesh load
* [ ] Quad8 → triangle triangulation
* [ ] Build indexed geometry buffers
* [ ] Compute bounding box
* [ ] Cache static geometry
* [ ] Emit `render:geometry:static`

---

## 6. Field Preprocessing (Server)

### Scalar fields

* [ ] Potential field mapped to vertices
* [ ] Pressure field mapped to vertices or elements
* [ ] Velocity magnitude prepared (scalar)

### Extrusion

* [ ] Compute `z = scale * scalar`
* [ ] No extrusion logic in browser

---

## 7. Render Channel (Binary)

* [ ] Geometry sent as TypedArrays
* [ ] Fields sent as TypedArrays
* [ ] No JSON-encoded numeric arrays
* [ ] No FEM logic in frontend

---

## 8. Three.js Rendering (Client)

* [ ] Indexed `BufferGeometry`
* [ ] Geometry uploaded once
* [ ] Scalar field updates without geometry rebuild
* [ ] Field selector (potential / pressure / velocity)
* [ ] Adjustable extrusion scale

---

## 9. Velocity Visualization (Sparse & Scalable)

* [ ] No arrows per node
* [ ] No arrows per element
* [ ] Regular sampling grid defined
* [ ] Velocity sampling done server-side
* [ ] Send sampled centers + velocity vectors
* [ ] Render with instanced arrows
* [ ] Animate visually (time-based phase shift)
* [ ] Density controlled by UI

---

## 10. UI / UX

* [ ] Total process time always visible
* [ ] Solver backend & type visible
* [ ] Convergence status visible
* [ ] Residual plot (log scale)
* [ ] Mesh visible before solve completion
* [ ] Field fills appear immediately when available
* [ ] Velocity overlay toggle

---

## 11. Performance & Scalability Guards

* [ ] No JS triangulation
* [ ] No per-frame geometry rebuild
* [ ] No per-node arrow rendering
* [ ] No solver logic in browser
* [ ] Handles 100k–1M+ nodes without UI freeze

---

## 12. Acceptance Checks

* [ ] Early mesh rendering works
* [ ] Metrics stream continuously
* [ ] Binary rendering does not block metrics
* [ ] CPU vs GPU runs comparable
* [ ] Velocity visualization remains readable at scale

---
