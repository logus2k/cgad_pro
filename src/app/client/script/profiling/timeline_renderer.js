/**
 * TimelineRenderer - Three.js renderer for profiling timeline events.
 * 
 * Optimized version that works directly with typed arrays from HDF5.
 * Eliminates intermediate object allocation for maximum performance.
 * 
 * Uses InstancedMesh for efficient rendering of 100k+ events.
 * One InstancedMesh per category = ~6 draw calls total.
 * 
 * Location: /src/app/client/script/profiling/timeline_renderer.js
 */

import * as THREE from '../../library/three.module.min.js';

export class TimelineRenderer {
    
    #canvas;
    #renderer;
    #scene;
    #camera;
    #categoryColors;
    
    // InstancedMesh per category
    #instancedMeshes = new Map();
    #geometry;
    #materials = new Map();
    
    // Rendering state
    #needsRebuild = false;
    
    // Data - supports both legacy events array and typed array format
    #dataMode = 'none'; // 'none' | 'legacy' | 'typed'
    #events = [];       // Legacy: array of event objects
    #typedData = null;  // Typed: { byCategory: { category: { startMs, durationMs, stream, ... } } }
    
    // Filtered indices per category (for typed array mode)
    #visibleIndices = new Map(); // category -> Uint32Array of visible indices
    
    // For hit testing
    #visibleEvents = [];  // Legacy mode
    #visibleRanges = new Map(); // Typed mode: category -> { startMs, endMs, originalIdx }[]
    
    #groupMapping = new Map();
    #visibleGroups = [];
    #visibleCategories = new Set();

    // Grid lines
    #gridLines = null;
    #gridColor = 0xcccccc;
    #gridMinorColor = 0xe8e8e8;

    // NVTX range colors (matching backend NVTX_COLORS)
    #nvtxColors = new Map([
        ['load_mesh', new THREE.Color(0x3498db)],        // Blue
        ['assemble_system', new THREE.Color(0x2ecc71)],  // Green  
        ['apply_bc', new THREE.Color(0xf1c40f)],         // Yellow
        ['solve_system', new THREE.Color(0xe74c3c)],     // Red
        ['compute_derived', new THREE.Color(0x9b59b6)],  // Purple
        ['export_results', new THREE.Color(0x95a5a6)],   // Gray
    ]);
    #nvtxDefaultColor = new THREE.Color(0x9b59b6); 

    // Known phase names for NVTX splitting
    #nvtxPhaseNames = new Set([
        'load_mesh',
        'assemble_system',
        'apply_bc',
        'solve_system',
        'compute_derived',
        'export_results'
    ]);
    #nvtxCudaColor = new THREE.Color(0x888888);    
    
    // View state
    #timeRange = { start: 0, end: 1000 };
    #resolution = { width: 800, height: 600 };
    
    // Reusable objects to avoid GC
    #tempMatrix = new THREE.Matrix4();
    #tempPosition = new THREE.Vector3();
    #tempQuaternion = new THREE.Quaternion();
    #tempScale = new THREE.Vector3();
    
    constructor(canvas) {
        this.#canvas = canvas;
        this.#categoryColors = new Map();
        
        this.#initThree();
        this.#extractCSSColors();
        this.#createMaterials();
    }
    
    #initThree() {
        this.#renderer = new THREE.WebGLRenderer({
            canvas: this.#canvas,
            antialias: false,
            alpha: true,
            preserveDrawingBuffer: true
        });
        this.#renderer.setClearColor(0x000000, 0);
        this.#renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        
        this.#scene = new THREE.Scene();
        
        // Orthographic camera
        this.#camera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
        
        // Shared geometry - a 1x1 plane
        this.#geometry = new THREE.PlaneGeometry(1, 1);
    }
    
    #extractCSSColors() {
        const categories = [
            'cuda_kernel',
            'cuda_memcpy_h2d',
            'cuda_memcpy_d2h',
            'cuda_memcpy_d2d',
            'cuda_sync',
            'nvtx_range'
        ];
        
        const defaults = {
            cuda_kernel: 0xe74c3c,
            cuda_memcpy_h2d: 0x3498db,
            cuda_memcpy_d2h: 0x2ecc71,
            cuda_memcpy_d2d: 0xf39c12,
            cuda_sync: 0x95a5a6,
            nvtx_range: 0x9b59b6
        };
        
        for (const cat of categories) {
            let color = defaults[cat];
            
            try {
                const el = document.createElement('div');
                el.className = `profiling-item-${cat}`;
                el.style.display = 'none';
                document.body.appendChild(el);
                
                const style = getComputedStyle(el);
                const bgColor = style.backgroundColor || style.background;
                document.body.removeChild(el);
                
                if (bgColor) {
                    const match = bgColor.match(/rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/);
                    if (match) {
                        const r = parseInt(match[1]);
                        const g = parseInt(match[2]);
                        const b = parseInt(match[3]);
                        if (r > 0 || g > 0 || b > 0) {
                            color = (r << 16) | (g << 8) | b;
                        }
                    }
                }
            } catch (e) {
                // Use default color
            }
            
            this.#categoryColors.set(cat, color);
        }
    }
    
    #createMaterials() {
        for (const [cat, color] of this.#categoryColors) {
            this.#materials.set(cat, new THREE.MeshBasicMaterial({
                color: color,
                transparent: true,
                opacity: 1.0,
                side: THREE.FrontSide
            }));
        }
        this.#materials.set('default', new THREE.MeshBasicMaterial({
            color: 0x666666,
            transparent: true,
            opacity: 1.0,
            side: THREE.FrontSide
        }));

        // Special material for NVTX with per-instance colors
        this.#materials.set('nvtx_range_colored', new THREE.MeshBasicMaterial({
            color: 0xffffff,  // Base white color, will be multiplied by instance color
            transparent: true,
            opacity: 1.0,
            side: THREE.FrontSide
        }));     
    }
    
    /**
     * Set data from legacy event objects array.
     * @param {Array} events - Array of event objects
     * @param {Array} groups - Array of group definitions
     */
    setData(events, groups) {
        console.time('[TimelineRenderer] setData (legacy)');
        this.#dataMode = 'legacy';
        this.#events = events;
        this.#typedData = null;
        this.#visibleGroups = groups;
        
        this.#groupMapping.clear();
        groups.forEach((g, i) => this.#groupMapping.set(g.id, i));
        
        this.#visibleCategories = new Set(groups.map(g => g.id));
        
        this.#needsRebuild = true;
        console.timeEnd('[TimelineRenderer] setData (legacy)');
    }
    
    /**
     * Set data from typed arrays (HDF5 format).
     * This is the optimized path - no intermediate object allocation.
     * 
     * @param {Object} rendererData - Output from ProfilingHDF5Loader.getRendererData()
     * @param {Array} groups - Array of group definitions
     */
    setTypedData(rendererData, groups) {
        console.time('[TimelineRenderer] setTypedData');
        this.#dataMode = 'typed';
        this.#events = [];
        this.#typedData = rendererData;
        this.#visibleGroups = groups;
        
        this.#groupMapping.clear();
        groups.forEach((g, i) => this.#groupMapping.set(g.id, i));
        
        this.#visibleCategories = new Set(groups.map(g => g.id));
        
        this.#needsRebuild = true;
        console.timeEnd('[TimelineRenderer] setTypedData');
    }
    
    setTimeRange(start, end) {
        this.#timeRange = { start, end };
        this.#needsRebuild = true;
    }

    /**
     * Clear all rendered content.
     */
    clear() {
        // Clear all meshes
        for (const mesh of this.#instancedMeshes.values()) {
            this.#scene.remove(mesh);
            mesh.geometry.dispose();
            mesh.material.dispose();
        }
        this.#instancedMeshes.clear();
        this.#visibleRanges.clear();
        
        // Clear grid
        if (this.#gridLines) {
            this.#scene.remove(this.#gridLines);
            this.#gridLines.geometry.dispose();
            this.#gridLines.material.dispose();
            this.#gridLines = null;
        }
        
        // Reset time range to prevent grid rendering
        this.#timeRange = { start: 0, end: 0 };
        
        // Render empty scene
        this.#renderer.render(this.#scene, this.#camera);
    }    
    
    resize(width, height) {
        if (width <= 0 || height <= 0) return;
        
        if (this.#resolution.width === width && this.#resolution.height === height) {
            return;
        }
        
        this.#resolution = { width, height };
        this.#renderer.setSize(width, height, false);
        
        this.#camera.left = 0;
        this.#camera.right = width;
        this.#camera.top = height;
        this.#camera.bottom = 0;
        this.#camera.near = -1;
        this.#camera.far = 1;
        this.#camera.updateProjectionMatrix();
        
        this.#needsRebuild = true;
    }
    
    #rebuildMeshes() {
        if (!this.#needsRebuild) return;
        
        console.time('[TimelineRenderer] rebuildMeshes');
        
        // Remove existing instanced meshes
        for (const mesh of this.#instancedMeshes.values()) {
            this.#scene.remove(mesh);
            mesh.dispose();
        }
        this.#instancedMeshes.clear();
        this.#visibleRanges.clear();
        
        const hasData = this.#dataMode === 'typed' 
            ? (this.#typedData && this.#typedData.categories?.length > 0)
            : (this.#events.length > 0);
            
        if (!hasData || this.#visibleGroups.length === 0) {
            this.#needsRebuild = false;
            console.timeEnd('[TimelineRenderer] rebuildMeshes');
            return;
        }
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        if (timeSpan <= 0) {
            this.#needsRebuild = false;
            console.timeEnd('[TimelineRenderer] rebuildMeshes');
            return;
        }
        
        if (this.#dataMode === 'typed') {
            this.#rebuildMeshesTyped(timeSpan);
        } else {
            this.#rebuildMeshesLegacy(timeSpan);
        }
        
        // Update grid lines
        this.#updateGrid();
        
        this.#needsRebuild = false;
        console.timeEnd('[TimelineRenderer] rebuildMeshes');
    }
    
    /**
     * Optimized mesh building using typed arrays directly.
     * No intermediate object allocation.
     */
    #rebuildMeshesTyped(timeSpan) {
        const pxPerMs = this.#resolution.width / timeSpan;
        const rowHeight = this.#resolution.height / this.#visibleGroups.length;
        const margin = timeSpan * 0.1;
        const minTime = this.#timeRange.start - margin;
        const maxTime = this.#timeRange.end + margin;
        
        let totalVisible = 0;
        
        for (const category of this.#typedData.categories) {
            const catData = this.#typedData.byCategory[category];
            if (!catData || catData.count === 0) continue;
            
            // Special handling for nvtx_range - split into phases and cuda rows
            if (category === 'nvtx_range') {
                totalVisible += this.#renderNvtxSplit(catData, pxPerMs, rowHeight, minTime, maxTime);
                continue;
            }
            
            if (!this.#visibleCategories.has(category)) continue;
            
            const rowIndex = this.#groupMapping.get(category);
            if (rowIndex === undefined) continue;
            
            const startMs = catData.startMs;
            const durationMs = catData.durationMs;
            const count = catData.count;
            
            // Filter to visible time range
            const visibleIdx = [];
            for (let i = 0; i < count; i++) {
                const start = startMs[i];
                const end = start + durationMs[i];
                if (end >= minTime && start <= maxTime) {
                    visibleIdx.push(i);
                }
            }
            
            if (visibleIdx.length === 0) continue;
            totalVisible += visibleIdx.length;
            
            // Create mesh
            const material = this.#materials.get(category) || this.#materials.get('default');
            const instancedMesh = new THREE.InstancedMesh(this.#geometry, material, visibleIdx.length);
            instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            
            const hitTestData = new Array(visibleIdx.length);
            const stack = [];
            
            for (let j = 0; j < visibleIdx.length; j++) {
                const i = visibleIdx[j];
                const start = startMs[i];
                const duration = durationMs[i];
                const end = start + duration;
                
                // Stacking
                let stackLevel = 0;
                let foundSlot = false;
                for (let s = 0; s < stack.length; s++) {
                    if (stack[s] <= start) {
                        stackLevel = s;
                        stack[s] = end;
                        foundSlot = true;
                        break;
                    }
                }
                if (!foundSlot) {
                    stackLevel = stack.length;
                    stack.push(end);
                }
                
                const x = (start - this.#timeRange.start) * pxPerMs;
                const barWidth = Math.max(duration * pxPerMs, 1);
                const maxStackLevels = Math.max(stack.length, 1);
                const barHeight = Math.max((rowHeight * 0.85) / maxStackLevels, 1);
                const y = rowIndex * rowHeight + rowHeight * 0.075 + stackLevel * barHeight;
                
                const screenY = y + barHeight / 2;
                const threeY = this.#resolution.height - screenY;
                
                this.#tempPosition.set(x + barWidth / 2, threeY, 0);
                this.#tempScale.set(barWidth, barHeight, 1);
                this.#tempMatrix.compose(this.#tempPosition, this.#tempQuaternion, this.#tempScale);
                instancedMesh.setMatrixAt(j, this.#tempMatrix);
                
                hitTestData[j] = { startMs: start, endMs: end, originalIdx: i };
            }
            
            instancedMesh.instanceMatrix.needsUpdate = true;
            this.#scene.add(instancedMesh);
            this.#instancedMeshes.set(category, instancedMesh);
            this.#visibleRanges.set(category, hitTestData);
        }
        
        // console.log(`[TimelineRenderer] Rendered ${totalVisible} events in ${this.#instancedMeshes.size} draw calls (typed)`);
    }

    #renderNvtxSplit(catData, pxPerMs, rowHeight, minTime, maxTime) {
        const startMs = catData.startMs;
        const durationMs = catData.durationMs;
        const count = catData.count;
        
        const phasesIdx = [];
        const cudaIdx = [];
        
        for (let i = 0; i < count; i++) {
            const start = startMs[i];
            const end = start + durationMs[i];
            if (end < minTime || start > maxTime) continue;
            
            const name = catData.names[catData.nameIdx[i]] || '';
            if (this.#nvtxPhaseNames.has(name)) {
                phasesIdx.push(i);
            } else {
                cudaIdx.push(i);
            }
        }
        
        let total = 0;
        
        const phasesRowIndex = this.#groupMapping.get('nvtx_phases');
        if (phasesRowIndex !== undefined && phasesIdx.length > 0) {
            this.#renderNvtxRow(catData, phasesIdx, phasesRowIndex, rowHeight, pxPerMs, 'nvtx_phases', true);
            total += phasesIdx.length;
        }
        
        const cudaRowIndex = this.#groupMapping.get('nvtx_cuda');
        if (cudaRowIndex !== undefined && cudaIdx.length > 0) {
            this.#renderNvtxRow(catData, cudaIdx, cudaRowIndex, rowHeight, pxPerMs, 'nvtx_cuda', false);
            total += cudaIdx.length;
        }
        
        return total;
    }

    #renderNvtxRow(catData, visibleIdx, rowIndex, rowHeight, pxPerMs, rowId, usePhaseColors) {
        const startMs = catData.startMs;
        const durationMs = catData.durationMs;
        
        const material = this.#materials.get('nvtx_range_colored');
        const instancedMesh = new THREE.InstancedMesh(this.#geometry, material, visibleIdx.length);
        instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        
        // Initialize colors
        const tempColor = new THREE.Color();
        for (let j = 0; j < visibleIdx.length; j++) {
            instancedMesh.setColorAt(j, tempColor);
        }
        
        const hitTestData = new Array(visibleIdx.length);
        const stack = [];
        
        for (let j = 0; j < visibleIdx.length; j++) {
            const i = visibleIdx[j];
            const start = startMs[i];
            const duration = durationMs[i];
            const end = start + duration;
            
            // Stacking
            let stackLevel = 0;
            let foundSlot = false;
            for (let s = 0; s < stack.length; s++) {
                if (stack[s] <= start) { stackLevel = s; stack[s] = end; foundSlot = true; break; }
            }
            if (!foundSlot) { stackLevel = stack.length; stack.push(end); }
            
            const x = (start - this.#timeRange.start) * pxPerMs;
            const barWidth = Math.max(duration * pxPerMs, 1);
            const maxStackLevels = Math.max(stack.length, 1);
            const barHeight = Math.max((rowHeight * 0.85) / maxStackLevels, 1);
            const y = rowIndex * rowHeight + rowHeight * 0.075 + stackLevel * barHeight;
            
            const screenY = y + barHeight / 2;
            const threeY = this.#resolution.height - screenY;
            
            this.#tempPosition.set(x + barWidth / 2, threeY, 0);
            this.#tempScale.set(barWidth, barHeight, 1);
            this.#tempMatrix.compose(this.#tempPosition, this.#tempQuaternion, this.#tempScale);
            instancedMesh.setMatrixAt(j, this.#tempMatrix);
            
            // Set color
            const name = catData.names[catData.nameIdx[i]] || '';
            const color = usePhaseColors ? (this.#nvtxColors.get(name) || this.#nvtxDefaultColor) : this.#nvtxCudaColor;
            instancedMesh.setColorAt(j, color);
            
            hitTestData[j] = { startMs: start, endMs: end, originalIdx: i };
        }
        
        instancedMesh.instanceMatrix.needsUpdate = true;
        if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;
        this.#scene.add(instancedMesh);
        this.#instancedMeshes.set(rowId, instancedMesh);
        this.#visibleRanges.set(rowId, hitTestData);
    }
    
    /**
     * Legacy mesh building using event objects.
     */
    #rebuildMeshesLegacy(timeSpan) {
        const pxPerMs = this.#resolution.width / timeSpan;
        const rowHeight = this.#resolution.height / this.#visibleGroups.length;
        
        // Filter to visible time range
        console.time('[TimelineRenderer] filter events');
        const margin = timeSpan * 0.1;
        this.#visibleEvents = this.#events.filter(e => {
            const startMs = e.start_ns / 1e6;
            const endMs = e.end_ns / 1e6;
            return endMs >= this.#timeRange.start - margin &&
                   startMs <= this.#timeRange.end + margin &&
                   this.#groupMapping.has(this.#getEventGroup(e));
        });
        console.timeEnd('[TimelineRenderer] filter events');
        
        // Group events by category
        console.time('[TimelineRenderer] group by category');
        const eventsByCategory = new Map();
        for (const event of this.#visibleEvents) {
            const cat = event.category;
            if (!eventsByCategory.has(cat)) {
                eventsByCategory.set(cat, []);
            }
            eventsByCategory.get(cat).push(event);
        }
        console.timeEnd('[TimelineRenderer] group by category');
        
        // Sort each category by start time for stacking
        console.time('[TimelineRenderer] sort events');
        for (const events of eventsByCategory.values()) {
            events.sort((a, b) => a.start_ns - b.start_ns);
        }
        console.timeEnd('[TimelineRenderer] sort events');
        
        // Create InstancedMesh for each category
        console.time('[TimelineRenderer] create instances');
        for (const [category, events] of eventsByCategory) {
            if (events.length === 0) continue;
            
            const material = this.#materials.get(category) || this.#materials.get('default');
            const instancedMesh = new THREE.InstancedMesh(this.#geometry, material, events.length);
            instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
            
            // Compute stacking for this category
            const groupStacks = new Map();
            
            for (let i = 0; i < events.length; i++) {
                const event = events[i];
                const groupId = this.#getEventGroup(event);
                const rowIndex = this.#groupMapping.get(groupId);
                if (rowIndex === undefined) continue;
                
                const startMs = event.start_ns / 1e6;
                const durationMs = event.duration_ns / 1e6;
                const endMs = startMs + durationMs;
                
                // Stacking logic
                if (!groupStacks.has(groupId)) {
                    groupStacks.set(groupId, []);
                }
                const stack = groupStacks.get(groupId);
                
                let stackLevel = 0;
                let foundSlot = false;
                for (let s = 0; s < stack.length; s++) {
                    if (stack[s] <= startMs) {
                        stackLevel = s;
                        stack[s] = endMs;
                        foundSlot = true;
                        break;
                    }
                }
                if (!foundSlot) {
                    stackLevel = stack.length;
                    stack.push(endMs);
                }
                
                // Calculate position in pixels
                const x = (startMs - this.#timeRange.start) * pxPerMs;
                const barWidth = Math.max(durationMs * pxPerMs, 1);
                
                const maxStackLevels = Math.max(stack.length, 1);
                const barHeight = Math.max((rowHeight * 0.85) / maxStackLevels, 1);
                const y = rowIndex * rowHeight + rowHeight * 0.075 + stackLevel * barHeight;
                
                // Convert to Three.js coordinates
                const screenY = y + barHeight / 2;
                const threeY = this.#resolution.height - screenY;
                
                // Set instance matrix
                this.#tempPosition.set(x + barWidth / 2, threeY, 0);
                this.#tempScale.set(barWidth, barHeight, 1);
                this.#tempMatrix.compose(this.#tempPosition, this.#tempQuaternion, this.#tempScale);
                instancedMesh.setMatrixAt(i, this.#tempMatrix);
            }
            
            instancedMesh.instanceMatrix.needsUpdate = true;
            instancedMesh.renderOrder = 0;
            this.#scene.add(instancedMesh);
            this.#instancedMeshes.set(category, instancedMesh);
        }
        console.timeEnd('[TimelineRenderer] create instances');
        
        console.log(`[TimelineRenderer] Rendered ${this.#visibleEvents.length} events in ${this.#instancedMeshes.size} draw calls (legacy)`);
    }
    
    #getEventGroup(event) {
        if (this.#visibleGroups.some(g => g.id === event.category)) {
            return event.category;
        }
        return event.category === 'nvtx_range' ? 'nvtx' : `stream_${event.stream}`;
    }

    #updateGrid() {
        // Remove existing grid
        if (this.#gridLines) {
            this.#scene.remove(this.#gridLines);
            this.#gridLines.geometry.dispose();
            this.#gridLines.material.dispose();
            this.#gridLines = null;
        }
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        if (timeSpan <= 0) return;
        
        const width = this.#resolution.width;
        const height = this.#resolution.height;
        
        // Compute tick interval
        const { interval, minorInterval } = this.#computeTickInterval(timeSpan, width);
        
        const positions = [];
        const colors = [];
        
        const majorColor = new THREE.Color(this.#gridColor);
        const minorColor = new THREE.Color(this.#gridMinorColor);
        
        // Minor ticks
        const firstMinorTick = Math.ceil(this.#timeRange.start / minorInterval) * minorInterval;
        for (let t = firstMinorTick; t <= this.#timeRange.end; t += minorInterval) {
            // Skip major tick positions
            if (Math.abs(t % interval) < minorInterval * 0.1) continue;
            
            const x = ((t - this.#timeRange.start) / timeSpan) * width;
            
            positions.push(x, 0, 0);
            positions.push(x, height, 0);
            colors.push(minorColor.r, minorColor.g, minorColor.b);
            colors.push(minorColor.r, minorColor.g, minorColor.b);
        }
        
        // Major ticks
        const firstMajorTick = Math.ceil(this.#timeRange.start / interval) * interval;
        for (let t = firstMajorTick; t <= this.#timeRange.end; t += interval) {
            const x = ((t - this.#timeRange.start) / timeSpan) * width;
            
            positions.push(x, 0, 0);
            positions.push(x, height, 0);
            colors.push(majorColor.r, majorColor.g, majorColor.b);
            colors.push(majorColor.r, majorColor.g, majorColor.b);
        }
        
        if (positions.length === 0) return;
        
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        
        const material = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            depthTest: false
        });
        
        this.#gridLines = new THREE.LineSegments(geometry, material);
        this.#gridLines.renderOrder = -1;  // Render before event bars
        this.#scene.add(this.#gridLines);
    }

    #computeTickInterval(timeSpanMs, widthPx) {
        const targetIntervalMs = (timeSpanMs / widthPx) * 100;
        
        const niceIntervals = [
            0.000001, 0.000002, 0.000005,
            0.00001, 0.00002, 0.00005,
            0.0001, 0.0002, 0.0005,
            0.001, 0.002, 0.005,
            0.01, 0.02, 0.05,
            0.1, 0.2, 0.5,
            1, 2, 5,
            10, 20, 50,
            100, 200, 500,
            1000, 2000, 5000,
            10000, 20000, 50000,
            100000, 200000, 500000
        ];
        
        let interval = niceIntervals[0];
        for (const ni of niceIntervals) {
            if (ni >= targetIntervalMs) {
                interval = ni;
                break;
            }
            interval = ni;
        }
        
        return {
            interval,
            minorInterval: interval / 5
        };
    }    
    
    render() {
        this.#rebuildMeshes();
        this.#renderer.render(this.#scene, this.#camera);
    }
    
    invalidate() {
        this.#needsRebuild = true;
    }

    /**
     * Get NVTX range label data for overlay rendering.
     * Returns array of { name, startMs, durationMs, rowIndex, rowHeight, stackLevel, maxStackLevels }
     */
    getNvtxLabels() {
        if (this.#visibleGroups.length === 0) return [];
        
        const rowHeight = this.#resolution.height / this.#visibleGroups.length;
        
        // Only show labels for phases row
        const rowIndex = this.#groupMapping.get('nvtx_phases');
        if (rowIndex === undefined) return [];
        
        const ranges = this.#visibleRanges.get('nvtx_phases');
        if (!ranges || ranges.length === 0) return [];
        
        const catData = this.#typedData?.byCategory['nvtx_range'];
        if (!catData) return [];
        
        const labels = [];
        const stack = [];
        
        for (const range of ranges) {
            const idx = range.originalIdx;
            const name = catData.names[catData.nameIdx[idx]] || '';
            
            // Stacking
            let stackLevel = 0;
            let foundSlot = false;
            for (let s = 0; s < stack.length; s++) {
                if (stack[s] <= range.startMs) { stackLevel = s; stack[s] = range.endMs; foundSlot = true; break; }
            }
            if (!foundSlot) { stackLevel = stack.length; stack.push(range.endMs); }
            
            labels.push({
                name,
                startMs: range.startMs,
                durationMs: range.endMs - range.startMs,
                rowIndex,
                rowHeight,
                stackLevel,
                maxStackLevels: Math.max(stack.length, 1)
            });
        }
        
        return labels;
    }   
    
    /**
     * Hit test at canvas coordinates.
     * Returns event data for tooltips.
     */
    hitTest(x, y) {
        if (this.#visibleGroups.length === 0) return null;
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        const timeAtX = this.#timeRange.start + (x / this.#resolution.width) * timeSpan;
        
        // Minimum hit tolerance of 3 pixels converted to time units
        const hitToleranceMs = (3 / this.#resolution.width) * timeSpan;
        
        const rowHeight = this.#resolution.height / this.#visibleGroups.length;
        const rowIndex = Math.floor(y / rowHeight);
        
        if (rowIndex < 0 || rowIndex >= this.#visibleGroups.length) return null;
        
        const group = this.#visibleGroups[rowIndex];
        const category = group.id;
        
        if (this.#dataMode === 'typed') {
            return this.#hitTestTyped(category, timeAtX, hitToleranceMs);
        } else {
            return this.#hitTestLegacy(group.id, timeAtX * 1e6, hitToleranceMs * 1e6);
        }
    }
    
    #hitTestTyped(category, timeAtXMs, toleranceMs = 0) {
        const ranges = this.#visibleRanges.get(category);
        if (!ranges) return null;
        
        // Map nvtx_phases/nvtx_cuda back to nvtx_range for data lookup
        const dataCategory = (category === 'nvtx_phases' || category === 'nvtx_cuda') ? 'nvtx_range' : category;
        const catData = this.#typedData.byCategory[dataCategory];
        if (!catData) return null;
        
        // Binary search would be faster for large datasets, but linear is fine for visible subset
        for (const range of ranges) {
            // Expand hit area by tolerance for thin bars
            const hitStart = range.startMs - toleranceMs;
            const hitEnd = range.endMs + toleranceMs;
            if (hitStart <= timeAtXMs && hitEnd >= timeAtXMs) {
                const idx = range.originalIdx;
                
                // Build event object for tooltip
                return {
                    id: `${category}_${idx}`,
                    category,
                    name: catData.names[catData.nameIdx[idx]] || '',
                    start_ns: range.startMs * 1e6,
                    end_ns: range.endMs * 1e6,
                    duration_ns: (range.endMs - range.startMs) * 1e6,
                    stream: catData.stream[idx],
                    metadata: this.#buildMetadata(category, catData, idx)
                };
            }
        }
        
        return null;
    }
    
    #buildMetadata(category, catData, idx) {
        const meta = catData.metadata || {};
        const result = {};
        
        if (category === 'cuda_kernel') {
            if (meta.grid_x) {
                result.grid = [meta.grid_x[idx], meta.grid_y[idx], meta.grid_z[idx]];
            }
            if (meta.block_x) {
                result.block = [meta.block_x[idx], meta.block_y[idx], meta.block_z[idx]];
            }
            if (meta.registers) {
                result.registers_per_thread = meta.registers[idx];
            }
            if (meta.shared_static) {
                result.shared_memory_static = meta.shared_static[idx];
            }
            if (meta.shared_dynamic) {
                result.shared_memory_dynamic = meta.shared_dynamic[idx];
            }
        } else if (category.startsWith('cuda_memcpy')) {
            if (meta.bytes) {
                result.bytes = meta.bytes[idx];
            }
        } else if (category === 'nvtx_range') {
            if (meta.color) {
                result.color = meta.color[idx];
            }
        }
        
        return result;
    }
    
    #hitTestLegacy(groupId, timeAtXNs, toleranceNs = 0) {
        for (const event of this.#visibleEvents) {
            if (this.#getEventGroup(event) !== groupId) continue;
            const hitStart = event.start_ns - toleranceNs;
            const hitEnd = event.end_ns + toleranceNs;
            if (hitStart <= timeAtXNs && hitEnd >= timeAtXNs) {
                return event;
            }
        }
        return null;
    }
    
    /**
     * Get total event count.
     */
    getEventCount() {
        if (this.#dataMode === 'typed' && this.#typedData) {
            return this.#typedData.totalEvents;
        }
        return this.#events.length;
    }
    
    /**
     * Get visible event count.
     */
    getVisibleEventCount() {
        if (this.#dataMode === 'typed') {
            let count = 0;
            for (const ranges of this.#visibleRanges.values()) {
                count += ranges.length;
            }
            return count;
        }
        return this.#visibleEvents.length;
    }
    
    destroy() {
        // Clean up grid
        if (this.#gridLines) {
            this.#scene.remove(this.#gridLines);
            this.#gridLines.geometry.dispose();
            this.#gridLines.material.dispose();
            this.#gridLines = null;
        }
        
        for (const mesh of this.#instancedMeshes.values()) {
            this.#scene.remove(mesh);
            mesh.dispose();
        }
        this.#instancedMeshes.clear();
        
        if (this.#geometry) {
            this.#geometry.dispose();
        }
        for (const mat of this.#materials.values()) {
            mat.dispose();
        }
        if (this.#renderer) {
            this.#renderer.dispose();
        }
    }
}
