/**
 * Geometry Worker - Offloads heavy mesh computation to a separate thread
 * 
 * Handles:
 * - Y-segment cache building (for SDF) - OPTIMIZED with spatial index
 * - Marching Cubes via isosurface library
 * - Element grid building (spatial index)
 * - Vertex-element mapping (for solution interpolation)
 * 
 * Returns raw typed arrays that main thread uses to create Three.js geometry.
 */

import * as isosurface from '../library/isosurface.mjs';

// ============================================================================
// Message Handler
// ============================================================================
self.onmessage = function(e) {
    const { type, payload } = e.data;
    
    switch (type) {
        case 'createGeometry':
            createGeometry(payload);
            break;
        default:
            console.warn('Worker: Unknown message type:', type);
    }
};

// ============================================================================
// Main Geometry Creation Pipeline
// ============================================================================
function createGeometry(payload) {
    try {
        const { meshData, config } = payload;
        const zFactor = config.zFactor || 1.0;
        
        // Step 1: Calculate bounds
        postProgress('bounds', 'Calculating bounds...');
        const bounds = calculateBounds(meshData, zFactor);
        const originalBounds = {
            xMin: bounds.xMin,
            xMax: bounds.xMax,
            yMin: bounds.yMin,
            yMax: bounds.yMax
        };
        
        // Step 2: Build element X-index (for fast Y-segment cache building)
        postProgress('xIndex', 'Building element X-index...');
        const elementXIndex = buildElementXIndex(meshData, originalBounds, 500);
        
        // Step 3: Build Y-segment cache using X-index (OPTIMIZED)
        postProgress('yCache', 'Building Y-segment cache...');
        const ySegmentCache = buildYRangeCacheOptimized(meshData, originalBounds, 500, elementXIndex);
        
        // Step 4: Build element grid (spatial index for vertex mapping)
        postProgress('elementGrid', 'Building element grid...');
        const elementGrid = buildElementGrid(meshData, originalBounds, 50);
        
        // Step 5: Create 2D geometry arrays
        postProgress('geometry2D', 'Creating 2D geometry...');
        const geometry2D = createQuad8Geometry2D(meshData);
        
        // Step 6: Create 3D geometry via Marching Cubes
        postProgress('geometry3D', 'Running Marching Cubes...');
        const geometry3D = create3DGeometry(bounds, originalBounds, ySegmentCache);
        
        // Step 7: Build vertex-element mapping
        postProgress('vertexMapping', 'Building vertex mapping...');
        const vertexMapping = buildVertexElementMapping(
            geometry3D.positions,
            meshData,
            originalBounds,
            elementGrid
        );
        
        // Send results back
        postProgress('complete', 'Geometry complete');
        
        self.postMessage({
            type: 'complete',
            result: {
                bounds,
                originalBounds,
                ySegmentCache,
                elementGrid: serializeElementGrid(elementGrid),
                geometry2D,
                geometry3D,
                vertexMapping
            }
        }, [
            // Transfer ownership of typed arrays for zero-copy
            geometry2D.positions.buffer,
            geometry2D.indices.buffer,
            geometry3D.positions.buffer,
            geometry3D.indices.buffer
        ]);
        
    } catch (error) {
        self.postMessage({ 
            type: 'error', 
            error: error.message,
            stack: error.stack
        });
    }
}

function postProgress(stage, message) {
    self.postMessage({ type: 'progress', stage, message });
}

// ============================================================================
// Bounds Calculation
// ============================================================================
function calculateBounds(meshData, zFactor) {
    const coords = meshData.coordinates;
    let xMin = Infinity, xMax = -Infinity;
    let yMin = Infinity, yMax = -Infinity;

    for (let i = 0; i < coords.x.length; i++) {
        if (coords.x[i] < xMin) xMin = coords.x[i];
        if (coords.x[i] > xMax) xMax = coords.x[i];
        if (coords.y[i] < yMin) yMin = coords.y[i];
        if (coords.y[i] > yMax) yMax = coords.y[i];
    }
    
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const maxDim = Math.max(xRange, yRange);
    const zExtent = maxDim * zFactor;
    
    return { 
        xMin, xMax, yMin, yMax, 
        zMin: -zExtent / 2, 
        zMax: zExtent / 2,
        maxDim
    };
}

// ============================================================================
// Element X-Index (spatial index for fast Y-segment cache building)
// ============================================================================
function buildElementXIndex(meshData, bounds, resolution) {
    const coords = meshData.coordinates;
    const conn = meshData.connectivity;
    const { xMin, xMax } = bounds;
    const xRange = xMax - xMin;
    const cellWidth = xRange / resolution;
    
    // Create buckets for each X slice
    const buckets = new Array(resolution);
    for (let i = 0; i < resolution; i++) {
        buckets[i] = [];
    }
    
    // For each element, add it to all X buckets it overlaps
    for (let elemIdx = 0; elemIdx < conn.length; elemIdx++) {
        const elem = conn[elemIdx];
        
        // Find element X bounds
        let elemXMin = Infinity, elemXMax = -Infinity;
        let elemYMin = Infinity, elemYMax = -Infinity;
        
        for (let i = 0; i < 8; i++) {
            const nodeId = elem[i];
            const nx = coords.x[nodeId];
            const ny = coords.y[nodeId];
            
            if (nx < elemXMin) elemXMin = nx;
            if (nx > elemXMax) elemXMax = nx;
            if (ny < elemYMin) elemYMin = ny;
            if (ny > elemYMax) elemYMax = ny;
        }
        
        // Find which X buckets this element overlaps
        const iMin = Math.max(0, Math.floor((elemXMin - xMin) / cellWidth));
        const iMax = Math.min(resolution - 1, Math.floor((elemXMax - xMin) / cellWidth));
        
        // Store element with its Y bounds (pre-computed)
        const elemData = { elemIdx, yMin: elemYMin, yMax: elemYMax };
        
        for (let i = iMin; i <= iMax; i++) {
            buckets[i].push(elemData);
        }
    }
    
    return { buckets, xMin, xMax, cellWidth, resolution };
}

// ============================================================================
// Y-Segment Cache - OPTIMIZED (uses X-index instead of scanning all elements)
// ============================================================================
function buildYRangeCacheOptimized(meshData, bounds, cacheResolution, xIndex) {
    const { xMin, xMax } = bounds;
    const xRange = xMax - xMin;
    const gapThreshold = 0.05;
    
    const cache = new Array(cacheResolution);
    
    for (let i = 0; i < cacheResolution; i++) {
        const x = xMin + (i / (cacheResolution - 1)) * xRange;
        
        // Get elements that overlap this X from the index
        const bucketIdx = Math.min(i, xIndex.resolution - 1);
        const elements = xIndex.buckets[bucketIdx];
        
        // Collect Y ranges from overlapping elements
        const yValues = [];
        for (const elem of elements) {
            yValues.push({ min: elem.yMin, max: elem.yMax });
        }
        
        if (yValues.length === 0) {
            cache[i] = null;
            continue;
        }
        
        // Sort and merge overlapping Y ranges
        yValues.sort((a, b) => a.min - b.min);
        
        const segments = [];
        let current = { ...yValues[0] };
        
        for (let j = 1; j < yValues.length; j++) {
            const next = yValues[j];
            if (next.min <= current.max + gapThreshold) {
                current.max = Math.max(current.max, next.max);
            } else {
                // Add segment with centerY and radius for cylindrical particle mode
                segments.push({
                    yMin: current.min,
                    yMax: current.max,
                    centerY: (current.min + current.max) / 2,
                    radius: (current.max - current.min) / 2
                });
                current = { ...next };
            }
        }
        // Add final segment with all properties
        segments.push({
            yMin: current.min,
            yMax: current.max,
            centerY: (current.min + current.max) / 2,
            radius: (current.max - current.min) / 2
        });
        
        cache[i] = segments;
    }
    
    return cache;
}

// ============================================================================
// Element Grid (spatial index for fast element lookup)
// ============================================================================
function buildElementGrid(meshData, bounds, gridSize) {
    const coords = meshData.coordinates;
    const conn = meshData.connectivity;
    const { xMin, xMax, yMin, yMax } = bounds;
    
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const cellWidth = xRange / gridSize;
    const cellHeight = yRange / gridSize;
    
    const cells = new Array(gridSize * gridSize);
    for (let i = 0; i < cells.length; i++) {
        cells[i] = [];
    }
    
    for (let elemIdx = 0; elemIdx < conn.length; elemIdx++) {
        const elem = conn[elemIdx];
        
        let elemXMin = Infinity, elemXMax = -Infinity;
        let elemYMin = Infinity, elemYMax = -Infinity;
        
        for (let i = 0; i < 8; i++) {
            const nodeId = elem[i];
            const nx = coords.x[nodeId];
            const ny = coords.y[nodeId];
            
            if (nx < elemXMin) elemXMin = nx;
            if (nx > elemXMax) elemXMax = nx;
            if (ny < elemYMin) elemYMin = ny;
            if (ny > elemYMax) elemYMax = ny;
        }
        
        const iMin = Math.max(0, Math.floor((elemXMin - xMin) / cellWidth));
        const iMax = Math.min(gridSize - 1, Math.floor((elemXMax - xMin) / cellWidth));
        const jMin = Math.max(0, Math.floor((elemYMin - yMin) / cellHeight));
        const jMax = Math.min(gridSize - 1, Math.floor((elemYMax - yMin) / cellHeight));
        
        for (let i = iMin; i <= iMax; i++) {
            for (let j = jMin; j <= jMax; j++) {
                cells[j * gridSize + i].push(elemIdx);
            }
        }
    }
    
    return {
        resX: gridSize,
        resY: gridSize,
        xMin, xMax, yMin, yMax,
        cellWidth, cellHeight,
        cells
    };
}

function serializeElementGrid(grid) {
    return {
        resX: grid.resX,
        resY: grid.resY,
        xMin: grid.xMin,
        xMax: grid.xMax,
        yMin: grid.yMin,
        yMax: grid.yMax,
        cellWidth: grid.cellWidth,
        cellHeight: grid.cellHeight,
        cells: grid.cells
    };
}

// ============================================================================
// 2D Geometry (Quad-8 elements)
// ============================================================================
function createQuad8Geometry2D(meshData) {
    const coords = meshData.coordinates;
    const conn = meshData.connectivity;
    
    const vertexCount = conn.length * 8;
    const positions = new Float32Array(vertexCount * 3);
    const indicesArr = [];
    
    let vertexIndex = 0;
    
    for (let elemIdx = 0; elemIdx < conn.length; elemIdx++) {
        const elem = conn[elemIdx];
        const baseVertex = vertexIndex;
        
        for (let i = 0; i < 8; i++) {
            const nodeId = elem[i];
            positions[vertexIndex * 3 + 0] = coords.x[nodeId];
            positions[vertexIndex * 3 + 1] = coords.y[nodeId];
            positions[vertexIndex * 3 + 2] = 0;
            vertexIndex++;
        }
        
        // 6-triangle subdivision for Quad-8
        indicesArr.push(
            baseVertex + 0, baseVertex + 1, baseVertex + 7,
            baseVertex + 1, baseVertex + 2, baseVertex + 3,
            baseVertex + 3, baseVertex + 4, baseVertex + 5,
            baseVertex + 5, baseVertex + 6, baseVertex + 7,
            baseVertex + 1, baseVertex + 3, baseVertex + 7,
            baseVertex + 3, baseVertex + 5, baseVertex + 7
        );
    }
    
    return {
        positions,
        indices: new Uint32Array(indicesArr),
        vertexCount
    };
}

// ============================================================================
// 3D Geometry via Marching Cubes
// ============================================================================
function create3DGeometry(bounds, originalBounds, ySegmentCache) {
    const { xMin, xMax, yMin, yMax, zMin, zMax } = bounds;
    
    // Add margin for marching cubes
    const margin = 0.05;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const zRange = zMax - zMin;
    
    const mcBounds = {
        xMin: xMin - xRange * margin,
        xMax: xMax + xRange * margin,
        yMin: yMin - yRange * margin,
        yMax: yMax + yRange * margin,
        zMin: zMin - zRange * margin,
        zMax: zMax + zRange * margin
    };
    
    const mcXRange = mcBounds.xMax - mcBounds.xMin;
    const mcYRange = mcBounds.yMax - mcBounds.yMin;
    const mcZRange = mcBounds.zMax - mcBounds.zMin;
    
    // Resolution for marching cubes
    const resX = 120;
    const resY = 80;
    const resZ = 80;
    
    const cacheResolution = ySegmentCache.length;
    
    const sdfFunc = (gx, gy, gz) => {
        const x = mcBounds.xMin + (gx / resX) * mcXRange;
        const y = mcBounds.yMin + (gy / resY) * mcYRange;
        const z = mcBounds.zMin + (gz / resZ) * mcZRange;
        return rectangularSDF(x, y, z, originalBounds, bounds, ySegmentCache, cacheResolution);
    };
    
    const result = isosurface.surfaceNets(
        [resX + 1, resY + 1, resZ + 1],
        sdfFunc,
        [[0, 0, 0], [resX, resY, resZ]]
    );
    
    if (!result.positions || result.positions.length === 0) {
        return { 
            positions: new Float32Array(0), 
            indices: new Uint32Array(0),
            vertexCount: 0
        };
    }
    
    // Convert positions from grid coords to world coords
    const positions = new Float32Array(result.positions.length * 3);
    for (let i = 0; i < result.positions.length; i++) {
        const [gx, gy, gz] = result.positions[i];
        positions[i * 3 + 0] = mcBounds.xMin + (gx / resX) * mcXRange;
        positions[i * 3 + 1] = mcBounds.yMin + (gy / resY) * mcYRange;
        positions[i * 3 + 2] = mcBounds.zMin + (gz / resZ) * mcZRange;
    }
    
    // Convert indices
    let indices = new Uint32Array(result.cells.length * 3);
    for (let i = 0; i < result.cells.length; i++) {
        indices[i * 3 + 0] = result.cells[i][0];
        indices[i * 3 + 1] = result.cells[i][1];
        indices[i * 3 + 2] = result.cells[i][2];
    }
    
    // Correct MC geometry bounds
    let geoXMin = Infinity, geoXMax = -Infinity;
    let geoYMin = Infinity, geoYMax = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
        if (positions[i] < geoXMin) geoXMin = positions[i];
        if (positions[i] > geoXMax) geoXMax = positions[i];
        if (positions[i + 1] < geoYMin) geoYMin = positions[i + 1];
        if (positions[i + 1] > geoYMax) geoYMax = positions[i + 1];
    }
    
    const geoWidth = geoXMax - geoXMin;
    const geoHeight = geoYMax - geoYMin;
    const intendedWidth = originalBounds.xMax - originalBounds.xMin;
    const intendedHeight = originalBounds.yMax - originalBounds.yMin;
    
    if (geoWidth > 0 && geoHeight > 0) {
        const scaleCorrectX = intendedWidth / geoWidth;
        const scaleCorrectY = intendedHeight / geoHeight;
        
        for (let i = 0; i < positions.length; i += 3) {
            positions[i] = originalBounds.xMin + (positions[i] - geoXMin) * scaleCorrectX;
            positions[i + 1] = originalBounds.yMin + (positions[i + 1] - geoYMin) * scaleCorrectY;
        }
    }
    
    // Remove end caps
    indices = removeEndCaps(positions, indices, originalBounds);
    
    return {
        positions,
        indices,
        vertexCount: result.positions.length
    };
}

function rectangularSDF(x, y, z, originalBounds, bounds, ySegmentCache, cacheResolution) {
    const { xMin, xMax } = originalBounds;
    const { zMin, zMax } = bounds;
    
    if (x < xMin || x > xMax) {
        return 1.0;
    }
    
    // Get segments from cache
    const t = (x - xMin) / (xMax - xMin);
    const idx = Math.floor(t * (cacheResolution - 1));
    const segments = ySegmentCache[Math.min(idx, cacheResolution - 1)];
    
    if (!segments || segments.length === 0) {
        return 1.0;
    }
    
    let minDist = Infinity;
    
    for (const seg of segments) {
        const { yMin, yMax } = seg;
        
        const dTop = y - yMax;
        const dBottom = yMin - y;
        const dY = Math.max(dTop, dBottom);
        
        const dFront = z - zMax;
        const dBack = zMin - z;
        const dZ = Math.max(dFront, dBack);
        
        const dist = Math.max(dY, dZ);
        
        if (dist < minDist) {
            minDist = dist;
        }
    }
    
    return minDist;
}

function removeEndCaps(positions, indices, originalBounds) {
    const { xMin, xMax } = originalBounds;
    const endDistance = 0.02;
    const flatThreshold = 0.01;
    
    const newIndices = [];
    
    for (let i = 0; i < indices.length; i += 3) {
        const i0 = indices[i + 0];
        const i1 = indices[i + 1];
        const i2 = indices[i + 2];
        
        const x0 = positions[i0 * 3 + 0];
        const x1 = positions[i1 * 3 + 0];
        const x2 = positions[i2 * 3 + 0];
        
        const xSpread = Math.max(x0, x1, x2) - Math.min(x0, x1, x2);
        const isFlat = xSpread < flatThreshold;
        
        const avgX = (x0 + x1 + x2) / 3;
        const atInlet = avgX < xMin + endDistance && isFlat;
        const atOutlet = avgX > xMax - endDistance && isFlat;
        
        if (!atInlet && !atOutlet) {
            newIndices.push(i0, i1, i2);
        }
    }
    
    return new Uint32Array(newIndices);
}

// ============================================================================
// Vertex-Element Mapping
// ============================================================================
function buildVertexElementMapping(positions, meshData, bounds, elementGrid) {
    const vertexCount = positions.length / 3;
    const mapping = new Array(vertexCount);
    
    let mapped = 0, unmapped = 0;
    
    for (let i = 0; i < vertexCount; i++) {
        const x = positions[i * 3 + 0];
        const y = positions[i * 3 + 1];
        
        const result = findNearestElement(x, y, meshData, bounds, elementGrid);
        
        if (result) {
            mapping[i] = result;
            mapped++;
        } else {
            mapping[i] = null;
            unmapped++;
        }
    }
    
    return { mapping, mapped, unmapped };
}

function findNearestElement(x, y, meshData, bounds, elementGrid) {
    const { resX, resY, xMin, yMin, cellWidth, cellHeight, cells } = elementGrid;
    const coords = meshData.coordinates;
    const conn = meshData.connectivity;
    
    // Calculate grid cell for this point
    const i = Math.floor((x - xMin) / cellWidth);
    const j = Math.floor((y - yMin) / cellHeight);
    
    // Search in expanding rings around the target cell
    let bestResult = null;
    let bestDistSq = Infinity;
    
    // Start with radius 0 (just the target cell), expand if needed
    const maxRadius = Math.max(resX, resY);
    
    for (let radius = 0; radius <= maxRadius && bestResult === null; radius++) {
        // Check all cells at this radius
        for (let di = -radius; di <= radius; di++) {
            for (let dj = -radius; dj <= radius; dj++) {
                // Only check cells on the perimeter of current radius (optimization)
                if (radius > 0 && Math.abs(di) !== radius && Math.abs(dj) !== radius) continue;
                
                const ci = i + di;
                const cj = j + dj;
                
                if (ci < 0 || ci >= resX || cj < 0 || cj >= resY) continue;
                
                const cellElements = cells[cj * resX + ci];
                
                for (const elemIdx of cellElements) {
                    const elem = conn[elemIdx];
                    
                    // Get corner nodes (0, 2, 4, 6 are corners in Quad-8)
                    const x0 = coords.x[elem[0]], y0 = coords.y[elem[0]];
                    const x2 = coords.x[elem[2]], y2 = coords.y[elem[2]];
                    const x4 = coords.x[elem[4]], y4 = coords.y[elem[4]];
                    const x6 = coords.x[elem[6]], y6 = coords.y[elem[6]];
                    
                    // Element bounding box
                    const bxMin = Math.min(x0, x2, x4, x6);
                    const bxMax = Math.max(x0, x2, x4, x6);
                    const byMin = Math.min(y0, y2, y4, y6);
                    const byMax = Math.max(y0, y2, y4, y6);
                    
                    // Element center
                    const cx = (bxMin + bxMax) / 2;
                    const cy = (byMin + byMax) / 2;
                    
                    // Distance squared from point to element center
                    const dx = x - cx;
                    const dy = y - cy;
                    const distSq = dx * dx + dy * dy;
                    
                    if (distSq < bestDistSq) {
                        // Calculate parametric coordinates
                        const xi = 2 * (x - cx) / (bxMax - bxMin);
                        const eta = 2 * (y - cy) / (byMax - byMin);
                        
                        bestDistSq = distSq;
                        bestResult = {
                            elemIdx,
                            xi: Math.max(-1, Math.min(1, xi)),
                            eta: Math.max(-1, Math.min(1, eta))
                        };
                    }
                }
            }
        }
        
        // If we found something in this radius, we can stop
        // (elements in further cells will be farther away)
        if (bestResult !== null) break;
    }
    
    return bestResult;
}
