/**
 * TimelineRenderer - Three.js renderer for profiling timeline events.
 * 
 * Uses InstancedMesh for efficient rendering of thousands of event bars.
 * 
 * Location: /src/app/client/script/profiling/timeline_renderer.js
 */

import * as THREE from 'three';

export class TimelineRenderer {
    
    #canvas;
    #renderer;
    #scene;
    #camera;
    #instancedMesh;
    #categoryColors;
    
    // Rendering state
    #instanceCount = 0;
    #maxInstances = 100000;  // Pre-allocate for performance
    #needsUpload = false;
    #eventBounds = [];  // {event, x, y, width, height} for hit testing
    
    // Data
    #events = [];
    #visibleEvents = [];
    #groupMapping = new Map();  // groupId -> row index
    #visibleGroups = [];
    
    // View state
    #timeRange = { start: 0, end: 1000 };
    #resolution = { width: 800, height: 600 };
    
    // Reusable objects (avoid GC)
    #tempMatrix = new THREE.Matrix4();
    #tempColor = new THREE.Color();
    #tempPosition = new THREE.Vector3();
    #tempScale = new THREE.Vector3();
    #tempQuaternion = new THREE.Quaternion();
    
    constructor(canvas) {
        this.#canvas = canvas;
        this.#categoryColors = new Map();
        
        this.#initThree();
        this.#initMesh();
        this.#extractCSSColors();
    }
    
    #initThree() {
        // Renderer
        this.#renderer = new THREE.WebGLRenderer({
            canvas: this.#canvas,
            antialias: false,
            alpha: false,
            powerPreference: 'high-performance'
        });
        this.#renderer.setClearColor(0x1e1e1e, 1);
        
        // Scene
        this.#scene = new THREE.Scene();
        
        // Orthographic camera (2D view)
        this.#camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
        this.#camera.position.z = 1;
    }
    
    #initMesh() {
        // Simple plane geometry for each event bar
        const geometry = new THREE.PlaneGeometry(1, 1);
        
        // Basic material with vertex colors
        const material = new THREE.MeshBasicMaterial({
            vertexColors: false,
            transparent: true,
            opacity: 0.9
        });
        
        // Create instanced mesh with pre-allocated capacity
        this.#instancedMesh = new THREE.InstancedMesh(geometry, material, this.#maxInstances);
        this.#instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        
        // Add instance colors
        const colors = new Float32Array(this.#maxInstances * 3);
        this.#instancedMesh.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
        this.#instancedMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
        
        // Initially hide all instances
        this.#instancedMesh.count = 0;
        
        this.#scene.add(this.#instancedMesh);
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
            cuda_kernel: '#e74c3c',
            cuda_memcpy_h2d: '#3498db',
            cuda_memcpy_d2h: '#2ecc71',
            cuda_memcpy_d2d: '#f39c12',
            cuda_sync: '#95a5a6',
            nvtx_range: '#9b59b6'
        };
        
        for (const cat of categories) {
            const el = document.createElement('div');
            el.className = `profiling-item-${cat}`;
            el.style.display = 'none';
            document.body.appendChild(el);
            
            const style = getComputedStyle(el);
            const bgColor = style.backgroundColor;
            document.body.removeChild(el);
            
            // Parse color
            const match = bgColor.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
            if (match) {
                this.#categoryColors.set(cat, new THREE.Color(
                    parseInt(match[1]) / 255,
                    parseInt(match[2]) / 255,
                    parseInt(match[3]) / 255
                ));
            } else {
                this.#categoryColors.set(cat, new THREE.Color(defaults[cat]));
            }
        }
    }
    
    /**
     * Set events data and prepare for rendering.
     */
    setData(events, groups) {
        this.#events = events;
        this.#visibleGroups = groups;
        
        this.#groupMapping.clear();
        groups.forEach((g, i) => this.#groupMapping.set(g.id, i));
        
        this.#needsUpload = true;
    }
    
    /**
     * Update visible time range.
     */
    setTimeRange(start, end) {
        this.#timeRange = { start, end };
        this.#needsUpload = true;
    }
    
    /**
     * Resize renderer.
     */
    resize(width, height) {
        this.#canvas.width = width;
        this.#canvas.height = height;
        this.#resolution = { width, height };
        
        this.#renderer.setSize(width, height, false);
        
        // Update orthographic camera to match pixel coordinates
        // Camera views [0, width] x [0, height], Y down
        this.#camera.left = 0;
        this.#camera.right = width;
        this.#camera.top = 0;
        this.#camera.bottom = height;
        this.#camera.updateProjectionMatrix();
        
        this.#needsUpload = true;
    }
    
    #uploadData() {
        if (!this.#needsUpload || this.#events.length === 0 || this.#visibleGroups.length === 0) {
            if (this.#events.length === 0 || this.#visibleGroups.length === 0) {
                this.#instancedMesh.count = 0;
                this.#eventBounds = [];
            }
            this.#needsUpload = false;
            return;
        }
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        const pxPerMs = this.#resolution.width / timeSpan;
        const rowHeight = this.#resolution.height / this.#visibleGroups.length;
        
        // Filter to visible time range
        const margin = timeSpan * 0.1;
        this.#visibleEvents = this.#events.filter(e => {
            const startMs = e.start_ns / 1e6;
            const endMs = e.end_ns / 1e6;
            return endMs >= this.#timeRange.start - margin &&
                startMs <= this.#timeRange.end + margin &&
                this.#groupMapping.has(this.#getEventGroup(e));
        });
        
        // Sort by start time for stacking
        this.#visibleEvents.sort((a, b) => a.start_ns - b.start_ns);
        
        // Compute stacking within each group
        const groupStacks = new Map();
        
        // Reset event bounds
        this.#eventBounds = [];
        
        let instanceIdx = 0;
        const maxVisible = Math.min(this.#visibleEvents.length, this.#maxInstances);
        
        for (let i = 0; i < maxVisible; i++) {
            const event = this.#visibleEvents[i];
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
            for (let s = 0; s < stack.length; s++) {
                if (stack[s] <= startMs) {
                    stackLevel = s;
                    stack[s] = endMs;
                    break;
                }
                stackLevel = s + 1;
            }
            if (stackLevel >= stack.length) {
                stack.push(endMs);
            }
            
            // Calculate pixel positions
            const x = (startMs - this.#timeRange.start) * pxPerMs;
            const width = Math.max(durationMs * pxPerMs, 1);  // Min 1px width
            
            const maxStackLevels = Math.max(stack.length, 1);
            const eventHeight = (rowHeight * 0.9) / maxStackLevels;
            const y = rowIndex * rowHeight + stackLevel * eventHeight + rowHeight * 0.05;
            
            // Store bounds for hit testing
            this.#eventBounds.push({
                event,
                x,
                y,
                width,
                height: eventHeight
            });
            
            // Set transform matrix
            // PlaneGeometry is centered, so offset by half width/height
            this.#tempPosition.set(x + width / 2, y + eventHeight / 2, 0);
            this.#tempScale.set(width, eventHeight, 1);
            this.#tempMatrix.compose(this.#tempPosition, this.#tempQuaternion, this.#tempScale);
            this.#instancedMesh.setMatrixAt(instanceIdx, this.#tempMatrix);
            
            // Set color
            const color = this.#categoryColors.get(event.category) || new THREE.Color(0x666666);
            this.#instancedMesh.setColorAt(instanceIdx, color);
            
            instanceIdx++;
        }
        
        this.#instanceCount = instanceIdx;
        this.#instancedMesh.count = instanceIdx;
        this.#instancedMesh.instanceMatrix.needsUpdate = true;
        if (this.#instancedMesh.instanceColor) {
            this.#instancedMesh.instanceColor.needsUpdate = true;
        }
        
        this.#needsUpload = false;
    }
    
    #getEventGroup(event) {
        if (this.#visibleGroups.some(g => g.id === event.category)) {
            return event.category;
        }
        return event.category === 'nvtx_range' ? 'nvtx' : `stream_${event.stream}`;
    }
    
    /**
     * Render frame.
     */
    render() {
        this.#uploadData();
        this.#renderer.render(this.#scene, this.#camera);
    }
    
    /**
     * Mark data as needing re-upload.
     */
    invalidate() {
        this.#needsUpload = true;
    }
    
    /**
     * Hit test at canvas coordinates.
     * Uses pre-computed bounds from #uploadData for accurate stacking support.
     */
    hitTest(x, y) {
        const minHitWidth = 6;  // Minimum 6px hit area
        
        // Iterate in reverse to find topmost (last rendered) event first
        for (let i = this.#eventBounds.length - 1; i >= 0; i--) {
            const b = this.#eventBounds[i];
            const hitWidth = Math.max(b.width, minHitWidth);
            const hitX = b.x - (hitWidth - b.width) / 2;  // Center the expanded hit area
            
            if (x >= hitX && x <= hitX + hitWidth &&
                y >= b.y && y <= b.y + b.height) {
                return b.event;
            }
        }
        return null;
    }
    
    /**
     * Cleanup resources.
     */
    destroy() {
        if (this.#instancedMesh) {
            this.#instancedMesh.geometry.dispose();
            this.#instancedMesh.material.dispose();
            this.#scene.remove(this.#instancedMesh);
        }
        if (this.#renderer) {
            this.#renderer.dispose();
        }
    }
}
