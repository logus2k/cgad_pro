/**
 * TimelineRenderer - Three.js renderer for profiling timeline events.
 * 
 * Location: /src/app/client/script/profiling/timeline_renderer.js
 */

import * as THREE from '../../library/three.module.min.js';

export class TimelineRenderer {
    
    #canvas;
    #renderer;
    #scene;
    #camera;
    #meshes = [];
    #categoryColors;
    
    // Rendering state
    #needsRebuild = false;
    
    // Data
    #events = [];
    #visibleEvents = [];
    #groupMapping = new Map();
    #visibleGroups = [];
    
    // View state
    #timeRange = { start: 0, end: 1000 };
    #resolution = { width: 800, height: 600 };
    
    // Shared geometry and materials
    #geometry;
    #materials = new Map();
    
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
            preserveDrawingBuffer: true  // Prevents flicker on resize
        });
        this.#renderer.setClearColor(0x000000, 0);
        
        this.#scene = new THREE.Scene();
        
        // Orthographic camera - configured properly in resize()
        // Using a coordinate system where (0,0) is top-left
        this.#camera = new THREE.OrthographicCamera(0, 1, 1, 0, -1, 1);
        
        // Shared geometry - a 1x1 plane centered at origin
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
                transparent: false,
                side: THREE.FrontSide
            }));
        }
        this.#materials.set('default', new THREE.MeshBasicMaterial({
            color: 0x666666,
            transparent: false,
            side: THREE.FrontSide
        }));
    }
    
    setData(events, groups) {
        this.#events = events;
        this.#visibleGroups = groups;
        
        this.#groupMapping.clear();
        groups.forEach((g, i) => this.#groupMapping.set(g.id, i));
        
        this.#needsRebuild = true;
    }
    
    setTimeRange(start, end) {
        this.#timeRange = { start, end };
        this.#needsRebuild = true;
    }
    
    resize(width, height) {
        if (width <= 0 || height <= 0) return;
        
        // Only resize if dimensions actually changed
        if (this.#resolution.width === width && this.#resolution.height === height) {
            return;
        }
        
        this.#resolution = { width, height };
        this.#renderer.setSize(width, height, false);
        
        // Standard orthographic: Y increases upward in Three.js
        // We want (0,0) top-left, so:
        // left=0, right=width, top=height, bottom=0
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
        
        // Clear existing meshes
        for (const mesh of this.#meshes) {
            this.#scene.remove(mesh);
        }
        this.#meshes = [];
        
        if (this.#events.length === 0 || this.#visibleGroups.length === 0) {
            this.#needsRebuild = false;
            return;
        }
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        if (timeSpan <= 0) {
            this.#needsRebuild = false;
            return;
        }
        
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
        
        // Limit for performance
        const maxMeshes = 10000;
        const eventsToRender = this.#visibleEvents.slice(0, maxMeshes);
        
        // Sort by start time for stacking
        eventsToRender.sort((a, b) => a.start_ns - b.start_ns);
        
        // Compute stacking
        const groupStacks = new Map();
        
        for (const event of eventsToRender) {
            const groupId = this.#getEventGroup(event);
            const rowIndex = this.#groupMapping.get(groupId);
            if (rowIndex === undefined) continue;
            
            const startMs = event.start_ns / 1e6;
            const durationMs = event.duration_ns / 1e6;
            const endMs = startMs + durationMs;
            
            // Stacking
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
            const barWidth = Math.max(durationMs * pxPerMs, 2);
            
            const maxStackLevels = Math.max(stack.length, 1);
            const barHeight = Math.max((rowHeight * 0.85) / maxStackLevels, 2);
            const y = rowIndex * rowHeight + rowHeight * 0.075 + stackLevel * barHeight;
            
            // Create mesh
            const material = this.#materials.get(event.category) || this.#materials.get('default');
            const mesh = new THREE.Mesh(this.#geometry, material);
            
            // Scale to bar size
            mesh.scale.set(barWidth, barHeight, 1);
            
            // Position at bar center
            // Y: convert from screen coords (0 at top) to Three.js (0 at bottom)
            // screenY -> threeY = height - screenY
            const screenY = y + barHeight / 2;
            mesh.position.set(
                x + barWidth / 2,
                this.#resolution.height - screenY,
                0
            );
            
            this.#scene.add(mesh);
            this.#meshes.push(mesh);
        }
        
        this.#needsRebuild = false;
    }
    
    #getEventGroup(event) {
        if (this.#visibleGroups.some(g => g.id === event.category)) {
            return event.category;
        }
        return event.category === 'nvtx_range' ? 'nvtx' : `stream_${event.stream}`;
    }
    
    render() {
        this.#rebuildMeshes();
        this.#renderer.render(this.#scene, this.#camera);
    }
    
    invalidate() {
        this.#needsRebuild = true;
    }
    
    hitTest(x, y) {
        if (this.#visibleGroups.length === 0) return null;
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        const timeAtX = this.#timeRange.start + (x / this.#resolution.width) * timeSpan;
        const timeAtXNs = timeAtX * 1e6;
        
        const rowHeight = this.#resolution.height / this.#visibleGroups.length;
        const rowIndex = Math.floor(y / rowHeight);
        
        if (rowIndex < 0 || rowIndex >= this.#visibleGroups.length) return null;
        
        const group = this.#visibleGroups[rowIndex];
        
        for (const event of this.#visibleEvents) {
            if (this.#getEventGroup(event) !== group.id) continue;
            if (event.start_ns <= timeAtXNs && event.end_ns >= timeAtXNs) {
                return event;
            }
        }
        
        return null;
    }
    
    destroy() {
        for (const mesh of this.#meshes) {
            this.#scene.remove(mesh);
        }
        this.#meshes = [];
        
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
