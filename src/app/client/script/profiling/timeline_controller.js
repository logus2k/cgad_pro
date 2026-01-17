/**
 * TimelineController - Main controller for WebGL profiling timeline.
 * 
 * Coordinates renderer, axis, groups, and handles user interaction.
 * Drop-in replacement for vis.js ProfilingTimeline.
 * 
 * Location: /src/app/client/script/profiling/timeline_controller.js
 */

import { TimelineRenderer } from './timeline_renderer.js';
import { TimelineAxis } from './timeline_axis.js';
import { TimelineGroups } from './timeline_groups.js';

export class TimelineController {
    
    // Category definitions (same as original)
    static CATEGORIES = {
        cuda_kernel: { label: 'CUDA Kernels', order: 1 },
        cuda_memcpy_h2d: { label: 'MemCpy H-D', order: 2 },
        cuda_memcpy_d2h: { label: 'MemCpy D-H', order: 3 },
        cuda_memcpy_d2d: { label: 'MemCpy D-D', order: 4 },
        cuda_sync: { label: 'CUDA Sync', order: 5 },
        nvtx_range: { label: 'NVTX Ranges', order: 6 }
    };
    
    static GROUP_BY = {
        CATEGORY: 'category',
        STREAM: 'stream'
    };
    
    #container;
    #renderer;
    #axis;
    #groupsSidebar;
    
    // DOM elements
    #mainEl;
    #axisCanvas;
    #glCanvas;
    #groupsEl;
    #tooltipEl;
    
    // Data
    #rawEvents = [];
    #visibleCategories = new Set(Object.keys(TimelineController.CATEGORIES));
    #currentGroupBy = TimelineController.GROUP_BY.CATEGORY;
    #groups = [];
    
    // View state
    #timeRange = { start: 0, end: 1000 };
    #totalTimeRange = { start: 0, end: 1000 };
    
    // Interaction state
    #isDragging = false;
    #dragStartX = 0;
    #dragStartTime = 0;
    #hoverTimeout = null;
    #lastMousePos = { x: 0, y: 0 };
    
    // Event handlers
    #eventHandlers = {};
    
    // Animation
    #animationFrame = null;
    #needsRender = true;
    
    // Config
    #groupsWidth = 120;
    #axisHeight = 30;
    #tooltipDelay = 1000;
    
    constructor(containerId) {
        this.#container = document.getElementById(containerId);
        if (!this.#container) {
            throw new Error(`Container not found: ${containerId}`);
        }
        
        this.#buildDOM();
        this.#initComponents();
        this.#bindEvents();
        this.#startRenderLoop();
    }
    
    #buildDOM() {
        this.#container.innerHTML = '';
        this.#container.style.position = 'relative';
        this.#container.style.overflow = 'hidden';
        this.#container.style.backgroundColor = '#ffffff';
        
        // Main layout: [groups sidebar] [timeline area]
        // Timeline area: [axis] on top, [canvas] below
        
        const wrapper = document.createElement('div');
        wrapper.style.cssText = 'display: flex; width: 100%; height: 100%;';
        
        // Groups sidebar
        this.#groupsEl = document.createElement('div');
        this.#groupsEl.style.cssText = `
            width: ${this.#groupsWidth}px;
            flex-shrink: 0;
            display: flex;
            flex-direction: column;
        `;
        
        // Spacer for axis alignment
        const groupsSpacer = document.createElement('div');
        groupsSpacer.style.cssText = `
            height: ${this.#axisHeight}px;
            background: #ffffff;
            border-bottom: 1px solid #e0e0e0;
            border-right: 1px solid #e0e0e0;
            box-sizing: border-box;
        `;
        this.#groupsEl.appendChild(groupsSpacer);
        
        // Groups list container
        const groupsList = document.createElement('div');
        groupsList.style.cssText = 'flex: 1; overflow: hidden;';
        this.#groupsEl.appendChild(groupsList);
        
        // Timeline area
        const timelineArea = document.createElement('div');
        timelineArea.style.cssText = 'flex: 1; display: flex; flex-direction: column; min-width: 0;';
        
        // Axis canvas
        this.#axisCanvas = document.createElement('canvas');
        this.#axisCanvas.style.cssText = `
            width: 100%;
            height: ${this.#axisHeight}px;
            display: block;
        `;
        timelineArea.appendChild(this.#axisCanvas);
        
        // WebGL canvas
        this.#glCanvas = document.createElement('canvas');
        this.#glCanvas.style.cssText = 'flex: 1; display: block; cursor: grab;';
        timelineArea.appendChild(this.#glCanvas);
        
        wrapper.appendChild(this.#groupsEl);
        wrapper.appendChild(timelineArea);
        this.#container.appendChild(wrapper);
        
        // Tooltip (hidden by default)
        this.#tooltipEl = document.createElement('div');
        this.#tooltipEl.style.cssText = `
            position: absolute;
            display: none;
            background: rgba(30, 30, 30, 0.95);
            border: 1px solid #555;
            border-radius: 4px;
            padding: 8px 12px;
            font-size: 12px;
            color: #e0e0e0;
            pointer-events: none;
            z-index: 1000;
            max-width: 350px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            line-height: 1.5;
        `;
        this.#container.appendChild(this.#tooltipEl);
        
        // Store groups list reference
        this.#groupsEl._listContainer = groupsList;
    }
    
    #initComponents() {
        this.#renderer = new TimelineRenderer(this.#glCanvas);
        this.#axis = new TimelineAxis(this.#axisCanvas, this.#axisHeight);
        this.#groupsSidebar = new TimelineGroups(this.#groupsEl._listContainer);
        
        // Handle group toggle
        this.#groupsSidebar.onToggle((groupId, enabled) => {
            if (enabled) {
                this.#visibleCategories.add(groupId);
            } else {
                this.#visibleCategories.delete(groupId);
            }
            this.#updateVisibleData();
        });
        
        // Initial resize
        this.#handleResize();
        
        // Watch for container resize
        this.#resizeObserver = new ResizeObserver(() => this.#handleResize());
        this.#resizeObserver.observe(this.#container);
    }
    
    #resizeObserver = null;
    #resizeTimeout = null;
    #lastResizeTime = 0;
    
    #handleResize() {
        // Debounce resize events - wait for resize to settle
        const now = Date.now();
        
        // Clear pending timeout
        if (this.#resizeTimeout) {
            clearTimeout(this.#resizeTimeout);
        }
        
        // If resizing rapidly, defer the update
        if (now - this.#lastResizeTime < 32) {
            this.#resizeTimeout = setTimeout(() => this.#doResize(), 50);
            return;
        }
        
        this.#lastResizeTime = now;
        this.#doResize();
    }
    
    #doResize() {
        const rect = this.#container.getBoundingClientRect();
        const timelineWidth = rect.width - this.#groupsWidth;
        const timelineHeight = rect.height - this.#axisHeight;
        
        if (timelineWidth <= 0 || timelineHeight <= 0) return;
        
        // Resize components
        this.#renderer.resize(timelineWidth, timelineHeight);
        this.#axis.resize(timelineWidth);
        
        // Update group row heights
        if (this.#groups.length > 0) {
            const rowHeight = timelineHeight / this.#groups.length;
            this.#groupsSidebar.updateRowHeight(rowHeight);
        }
        
        // Render immediately to prevent flicker
        this.#renderer.render();
        this.#axis.render();
        this.#needsRender = false;
    }
    
    #bindEvents() {
        const canvas = this.#glCanvas;
        
        // Mouse drag for panning
        canvas.addEventListener('mousedown', (e) => this.#onMouseDown(e));
        window.addEventListener('mousemove', (e) => this.#onMouseMove(e));
        window.addEventListener('mouseup', (e) => this.#onMouseUp(e));
        
        // Wheel for zooming
        canvas.addEventListener('wheel', (e) => this.#onWheel(e), { passive: false });
        
        // Touch support
        canvas.addEventListener('touchstart', (e) => this.#onTouchStart(e), { passive: false });
        canvas.addEventListener('touchmove', (e) => this.#onTouchMove(e), { passive: false });
        canvas.addEventListener('touchend', (e) => this.#onTouchEnd(e));
        
        // Click for selection
        canvas.addEventListener('click', (e) => this.#onClick(e));
    }
    
    #onMouseDown(e) {
        if (e.button !== 0) return;
        
        this.#isDragging = true;
        this.#dragStartX = e.clientX;
        this.#dragStartTime = this.#timeRange.start;
        this.#glCanvas.style.cursor = 'grabbing';
        
        // Cancel tooltip
        this.#hideTooltip();
    }
    
    #onMouseMove(e) {
        const rect = this.#glCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.#lastMousePos = { x, y };
        
        if (this.#isDragging) {
            const dx = e.clientX - this.#dragStartX;
            const timeSpan = this.#timeRange.end - this.#timeRange.start;
            const pxPerMs = this.#glCanvas.width / timeSpan;
            const timeDelta = -dx / pxPerMs;
            
            const newStart = this.#dragStartTime + timeDelta;
            this.#setTimeRange(newStart, newStart + timeSpan);
        } else {
            // Hover - start tooltip timer
            this.#startTooltipTimer(x, y);
        }
    }
    
    #onMouseUp(e) {
        if (this.#isDragging) {
            this.#isDragging = false;
            this.#glCanvas.style.cursor = 'grab';
        }
    }
    
    #onWheel(e) {
        e.preventDefault();
        
        const rect = this.#glCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const zoomFactor = e.deltaY > 0 ? 1.15 : 0.87;
        
        this.#zoomAtPoint(x, zoomFactor);
    }
    
    #zoomAtPoint(x, factor) {
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        const timeAtCursor = this.#timeRange.start + (x / this.#glCanvas.width) * timeSpan;
        
        const newSpan = timeSpan * factor;
        
        // Clamp zoom
        const minSpan = 0.0001;  // 100 nanoseconds
        const maxSpan = this.#totalTimeRange.end - this.#totalTimeRange.start;
        const clampedSpan = Math.max(minSpan, Math.min(maxSpan * 2, newSpan));
        
        // Keep cursor position fixed
        const ratio = x / this.#glCanvas.width;
        const newStart = timeAtCursor - ratio * clampedSpan;
        
        this.#setTimeRange(newStart, newStart + clampedSpan);
        
        // Invalidate renderer for re-binning
        this.#renderer.invalidate();
    }
    
    #setTimeRange(start, end) {
        this.#timeRange = { start, end };
        this.#renderer.setTimeRange(start, end);
        this.#axis.setTimeRange(start, end);
        this.#needsRender = true;
    }
    
    // Touch handling
    #touchStartDist = 0;
    #touchStartSpan = 0;
    
    #onTouchStart(e) {
        if (e.touches.length === 1) {
            this.#isDragging = true;
            this.#dragStartX = e.touches[0].clientX;
            this.#dragStartTime = this.#timeRange.start;
        } else if (e.touches.length === 2) {
            // Pinch zoom start
            this.#touchStartDist = Math.hypot(
                e.touches[1].clientX - e.touches[0].clientX,
                e.touches[1].clientY - e.touches[0].clientY
            );
            this.#touchStartSpan = this.#timeRange.end - this.#timeRange.start;
        }
        e.preventDefault();
    }
    
    #onTouchMove(e) {
        if (e.touches.length === 1 && this.#isDragging) {
            const dx = e.touches[0].clientX - this.#dragStartX;
            const timeSpan = this.#timeRange.end - this.#timeRange.start;
            const pxPerMs = this.#glCanvas.width / timeSpan;
            const timeDelta = -dx / pxPerMs;
            
            const newStart = this.#dragStartTime + timeDelta;
            this.#setTimeRange(newStart, newStart + timeSpan);
        } else if (e.touches.length === 2) {
            // Pinch zoom
            const dist = Math.hypot(
                e.touches[1].clientX - e.touches[0].clientX,
                e.touches[1].clientY - e.touches[0].clientY
            );
            const scale = this.#touchStartDist / dist;
            const newSpan = this.#touchStartSpan * scale;
            
            const centerX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
            const rect = this.#glCanvas.getBoundingClientRect();
            const x = centerX - rect.left;
            const ratio = x / this.#glCanvas.width;
            
            const centerTime = this.#timeRange.start + ratio * (this.#timeRange.end - this.#timeRange.start);
            const newStart = centerTime - ratio * newSpan;
            
            this.#setTimeRange(newStart, newStart + newSpan);
            this.#renderer.invalidate();
        }
        e.preventDefault();
    }
    
    #onTouchEnd(e) {
        this.#isDragging = false;
    }
    
    #onClick(e) {
        const rect = this.#glCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const event = this.#renderer.hitTest(x, y);
        if (event && this.#eventHandlers.select) {
            this.#eventHandlers.select(event);
        }
    }
    
    // Tooltip handling
    #startTooltipTimer(x, y) {
        this.#cancelTooltipTimer();
        
        this.#hoverTimeout = setTimeout(() => {
            this.#showTooltip(x, y);
        }, this.#tooltipDelay);
    }
    
    #cancelTooltipTimer() {
        if (this.#hoverTimeout) {
            clearTimeout(this.#hoverTimeout);
            this.#hoverTimeout = null;
        }
    }
    
    #showTooltip(x, y) {
        const event = this.#renderer.hitTest(x, y);
        if (!event) {
            this.#hideTooltip();
            return;
        }
        
        // Build tooltip content
        const content = this.#buildTooltipContent(event);
        this.#tooltipEl.innerHTML = content;
        this.#tooltipEl.style.display = 'block';
        
        // Position tooltip
        const rect = this.#glCanvas.getBoundingClientRect();
        const containerRect = this.#container.getBoundingClientRect();
        
        let tooltipX = rect.left - containerRect.left + x + 15;
        let tooltipY = rect.top - containerRect.top + y + 15;
        
        // Keep within container
        const tooltipRect = this.#tooltipEl.getBoundingClientRect();
        if (tooltipX + tooltipRect.width > containerRect.width) {
            tooltipX = x - tooltipRect.width - 15;
        }
        if (tooltipY + tooltipRect.height > containerRect.height) {
            tooltipY = y - tooltipRect.height - 15;
        }
        
        this.#tooltipEl.style.left = `${tooltipX}px`;
        this.#tooltipEl.style.top = `${tooltipY}px`;
        
        // Emit hover event
        if (this.#eventHandlers.hover) {
            this.#eventHandlers.hover(event);
        }
    }
    
    #hideTooltip() {
        this.#cancelTooltipTimer();
        this.#tooltipEl.style.display = 'none';
        
        if (this.#eventHandlers.hover) {
            this.#eventHandlers.hover(null);
        }
    }
    
    #buildTooltipContent(event) {
        const durationMs = event.duration_ns / 1e6;
        const lines = [
            `<div style="font-weight: 600; margin-bottom: 4px; color: #fff;">${this.#escapeHtml(event.name)}</div>`,
            `<div><span style="color: #888;">Duration:</span> ${this.#formatDuration(durationMs)}</div>`,
            `<div><span style="color: #888;">Category:</span> ${TimelineController.CATEGORIES[event.category]?.label || event.category}</div>`
        ];
        
        if (event.stream !== undefined) {
            lines.push(`<div><span style="color: #888;">Stream:</span> ${event.stream}</div>`);
        }
        
        if (event.metadata) {
            if (event.metadata.grid) {
                lines.push(`<div><span style="color: #888;">Grid:</span> [${event.metadata.grid.join(', ')}]</div>`);
            }
            if (event.metadata.block) {
                lines.push(`<div><span style="color: #888;">Block:</span> [${event.metadata.block.join(', ')}]</div>`);
            }
            if (event.metadata.registers_per_thread) {
                lines.push(`<div><span style="color: #888;">Registers/Thread:</span> ${event.metadata.registers_per_thread}</div>`);
            }
            if (event.metadata.bytes) {
                lines.push(`<div><span style="color: #888;">Bytes:</span> ${this.#formatBytes(event.metadata.bytes)}</div>`);
            }
            if (event.metadata.throughput_gbps) {
                lines.push(`<div><span style="color: #888;">Throughput:</span> ${event.metadata.throughput_gbps.toFixed(2)} GB/s</div>`);
            }
        }
        
        return lines.join('');
    }
    
    #escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }
    
    // Render loop
    #startRenderLoop() {
        const render = () => {
            if (this.#needsRender) {
                this.#renderer.render();
                this.#axis.render();
                this.#needsRender = false;
            }
            this.#animationFrame = requestAnimationFrame(render);
        };
        render();
    }
    
    // Public API (matching original ProfilingTimeline)
    
    /**
     * Load profiling data.
     * @param {Object} data - { events, total_duration_ns }
     * @param {Function} onProgress - Progress callback (0-100)
     */
    async load(data, onProgress = null) {
        if (!data || !data.events) {
            console.warn('[TimelineController] No events to load');
            return;
        }
        
        this.#rawEvents = data.events;
        console.log(`[TimelineController] Loading ${this.#rawEvents.length} events`);
        
        if (onProgress) onProgress(10);
        
        // Compute time range
        let minTime = Infinity, maxTime = -Infinity;
        for (const event of this.#rawEvents) {
            const startMs = event.start_ns / 1e6;
            const endMs = event.end_ns / 1e6;
            if (startMs < minTime) minTime = startMs;
            if (endMs > maxTime) maxTime = endMs;
        }
        
        if (minTime === Infinity) {
            minTime = 0;
            maxTime = 1000;
        }
        
        this.#totalTimeRange = { start: minTime, end: maxTime };
        
        if (onProgress) onProgress(30);
        
        // Build groups and update view
        this.#updateVisibleData();
        
        if (onProgress) onProgress(70);
        
        // Fit to content
        this.fit();
        
        if (onProgress) onProgress(100);
    }
    
    #updateVisibleData() {
        // Build groups based on current grouping mode
        this.#groups = this.#buildGroups();
        
        // Update sidebar
        const rect = this.#container.getBoundingClientRect();
        const timelineHeight = rect.height - this.#axisHeight;
        const rowHeight = this.#groups.length > 0 ? timelineHeight / this.#groups.length : 60;
        this.#groupsSidebar.setGroups(this.#groups, rowHeight);
        
        // Update renderer
        const visibleEvents = this.#rawEvents.filter(e => this.#visibleCategories.has(e.category));
        this.#renderer.setData(visibleEvents, this.#groups);
        this.#renderer.invalidate();
        
        this.#needsRender = true;
    }
    
    #buildGroups() {
        const visibleEvents = this.#rawEvents.filter(e => this.#visibleCategories.has(e.category));
        const groups = [];
        
        if (this.#currentGroupBy === TimelineController.GROUP_BY.CATEGORY) {
            const seenCategories = new Set(visibleEvents.map(e => e.category));
            
            for (const [catId, catInfo] of Object.entries(TimelineController.CATEGORIES)) {
                if (seenCategories.has(catId) && this.#visibleCategories.has(catId)) {
                    groups.push({
                        id: catId,
                        content: catInfo.label,
                        order: catInfo.order,
                        className: `profiling-item-${catId}`
                    });
                }
            }
            
            groups.sort((a, b) => a.order - b.order);
        } else {
            const streams = [...new Set(visibleEvents.map(e => e.stream))].sort((a, b) => a - b);
            
            for (const stream of streams) {
                groups.push({
                    id: `stream_${stream}`,
                    content: `Stream ${stream}`,
                    order: stream,
                    className: 'profiling-group-stream'
                });
            }
            
            const hasNvtx = visibleEvents.some(e => e.category === 'nvtx_range');
            if (hasNvtx) {
                groups.push({
                    id: 'nvtx',
                    content: 'NVTX Ranges',
                    order: 1000,
                    className: 'profiling-item-nvtx_range'
                });
            }
        }
        
        return groups;
    }
    
    /**
     * Clear all data.
     */
    clear() {
        this.#rawEvents = [];
        this.#groups = [];
        this.#renderer.setData([], []);
        this.#groupsSidebar.setGroups([], 60);
        this.#needsRender = true;
    }
    
    /**
     * Set grouping mode.
     * @param {string} groupBy - 'category' or 'stream'
     */
    async setGroupBy(groupBy) {
        if (!Object.values(TimelineController.GROUP_BY).includes(groupBy)) {
            console.warn(`[TimelineController] Invalid groupBy: ${groupBy}`);
            return;
        }
        this.#currentGroupBy = groupBy;
        this.#updateVisibleData();
    }
    
    /**
     * Set visible categories.
     * @param {string[]} categories 
     */
    async setVisibleCategories(categories) {
        this.#visibleCategories = new Set(categories);
        this.#updateVisibleData();
    }
    
    /**
     * Toggle category visibility.
     */
    async toggleCategory(category, visible) {
        if (visible) {
            this.#visibleCategories.add(category);
        } else {
            this.#visibleCategories.delete(category);
        }
        this.#updateVisibleData();
    }
    
    /**
     * Register event handler.
     */
    on(event, handler) {
        this.#eventHandlers[event] = handler;
    }
    
    /**
     * Fit timeline to show all content.
     */
    fit() {
        const padding = (this.#totalTimeRange.end - this.#totalTimeRange.start) * 0.02;
        this.#setTimeRange(
            this.#totalTimeRange.start - padding,
            this.#totalTimeRange.end + padding
        );
        this.#renderer.invalidate();
    }
    
    /**
     * Focus on a specific event.
     */
    focus(itemId) {
        const event = this.#rawEvents.find(e => e.id === itemId);
        if (!event) return;
        
        const startMs = event.start_ns / 1e6;
        const endMs = event.end_ns / 1e6;
        const duration = endMs - startMs;
        const padding = Math.max(duration * 5, 1);
        
        this.#setTimeRange(startMs - padding, endMs + padding);
        this.#renderer.invalidate();
    }
    
    /**
     * Get summary statistics.
     */
    getSummary() {
        const summary = {
            totalEvents: this.#rawEvents.length,
            totalDurationMs: 0,
            categories: {}
        };
        
        for (const event of this.#rawEvents) {
            const durationMs = event.duration_ns / 1e6;
            summary.totalDurationMs += durationMs;
            
            if (!summary.categories[event.category]) {
                summary.categories[event.category] = { count: 0, durationMs: 0 };
            }
            summary.categories[event.category].count++;
            summary.categories[event.category].durationMs += durationMs;
        }
        
        return summary;
    }
    
    get isLoading() {
        return false;  // WebGL version loads synchronously
    }
    
    // Utility methods
    #formatDuration(ms) {
        if (ms < 0.001) {
            return `${(ms * 1e6).toFixed(1)} ns`;
        } else if (ms < 1) {
            return `${(ms * 1000).toFixed(1)} us`;
        } else if (ms < 1000) {
            return `${ms.toFixed(2)} ms`;
        } else {
            return `${(ms / 1000).toFixed(2)} s`;
        }
    }
    
    #formatBytes(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    }
    
    /**
     * Destroy and cleanup.
     */
    destroy() {
        if (this.#animationFrame) {
            cancelAnimationFrame(this.#animationFrame);
        }
        if (this.#resizeObserver) {
            this.#resizeObserver.disconnect();
        }
        this.#cancelTooltipTimer();
        this.#renderer.destroy();
        this.#axis.destroy();
        this.#groupsSidebar.destroy();
        this.#container.innerHTML = '';
    }
}
