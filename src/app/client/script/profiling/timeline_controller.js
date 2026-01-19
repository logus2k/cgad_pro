/**
 * TimelineController - Main controller for WebGL profiling timeline.
 * 
 * Coordinates renderer, axis, groups, and handles user interaction.
 * Supports both legacy event objects and optimized typed arrays from HDF5.
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
    
    // Data - supports both modes
    #dataMode = 'none'; // 'none' | 'legacy' | 'typed'
    #rawEvents = [];     // Legacy mode
    #typedData = null;   // Typed mode (from HDF5)
    
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
    #tooltipDelay = 0;
    
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
        
        // Tooltip
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
        
        this.#groupsEl._listContainer = groupsList;
    }
    
    #initComponents() {
        this.#renderer = new TimelineRenderer(this.#glCanvas);
        this.#axis = new TimelineAxis(this.#axisCanvas, this.#axisHeight);
        this.#groupsSidebar = new TimelineGroups(this.#groupsEl._listContainer);
        
        this.#groupsSidebar.onToggle((groupId, enabled) => {
            if (enabled) {
                this.#visibleCategories.add(groupId);
            } else {
                this.#visibleCategories.delete(groupId);
            }
            this.#updateVisibleData();
        });
        
        this.#handleResize();
        
        this.#resizeObserver = new ResizeObserver(() => this.#handleResize());
        this.#resizeObserver.observe(this.#container);
    }
    
    #resizeObserver = null;
    #resizeTimeout = null;
    #lastResizeTime = 0;
    
    #handleResize() {
        const now = Date.now();
        
        if (this.#resizeTimeout) {
            clearTimeout(this.#resizeTimeout);
        }
        
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
        
        this.#renderer.resize(timelineWidth, timelineHeight);
        this.#axis.resize(timelineWidth);
        
        if (this.#groups.length > 0) {
            const rowHeight = timelineHeight / this.#groups.length;
            this.#groupsSidebar.updateRowHeight(rowHeight);
        }
        
        this.#renderer.render();
        this.#axis.render();
        this.#needsRender = false;
    }
    
    #bindEvents() {
        const canvas = this.#glCanvas;
        
        canvas.addEventListener('mousedown', (e) => this.#onMouseDown(e));
        window.addEventListener('mousemove', (e) => this.#onMouseMove(e));
        window.addEventListener('mouseup', (e) => this.#onMouseUp(e));
        canvas.addEventListener('wheel', (e) => this.#onWheel(e), { passive: false });
        
        canvas.addEventListener('touchstart', (e) => this.#onTouchStart(e), { passive: false });
        canvas.addEventListener('touchmove', (e) => this.#onTouchMove(e), { passive: false });
        canvas.addEventListener('touchend', (e) => this.#onTouchEnd(e));
    }
    
    #startRenderLoop() {
        const loop = () => {
            if (this.#needsRender) {
                this.#renderer.setTimeRange(this.#timeRange.start, this.#timeRange.end);
                this.#axis.setTimeRange(this.#timeRange.start, this.#timeRange.end);
                this.#renderer.render();
                this.#axis.render();
                this.#needsRender = false;
            }
            this.#animationFrame = requestAnimationFrame(loop);
        };
        this.#animationFrame = requestAnimationFrame(loop);
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Mouse / Touch Handlers
    // ─────────────────────────────────────────────────────────────────────────
    
    #onMouseDown(e) {
        this.#isDragging = true;
        this.#dragStartX = e.clientX;
        this.#dragStartTime = this.#timeRange.start;
        this.#glCanvas.style.cursor = 'grabbing';
        this.#cancelTooltipTimer();
    }
    
    #onMouseMove(e) {
        const rect = this.#glCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        this.#lastMousePos = { x, y };
        
        if (this.#isDragging) {
            const dx = e.clientX - this.#dragStartX;
            const timeSpan = this.#timeRange.end - this.#timeRange.start;
            const timeDelta = -(dx / rect.width) * timeSpan;
            
            const newStart = this.#dragStartTime + timeDelta;
            const newEnd = newStart + timeSpan;
            
            this.#setTimeRange(newStart, newEnd);
        } else if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
            this.#scheduleTooltip(x, y);
        } else {
            this.#hideTooltip();
        }
    }
    
    #onMouseUp(e) {
        this.#isDragging = false;
        this.#glCanvas.style.cursor = 'grab';
    }
    
    #onWheel(e) {
        e.preventDefault();
        
        const rect = this.#glCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseRatio = mouseX / rect.width;
        
        const zoomFactor = e.deltaY > 0 ? 1.15 : 0.87;
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        const newSpan = timeSpan * zoomFactor;
        
        const mouseTime = this.#timeRange.start + mouseRatio * timeSpan;
        const newStart = mouseTime - mouseRatio * newSpan;
        const newEnd = mouseTime + (1 - mouseRatio) * newSpan;
        
        this.#setTimeRange(newStart, newEnd);
        this.#cancelTooltipTimer();
    }
    
    #onTouchStart(e) {
        if (e.touches.length === 1) {
            e.preventDefault();
            this.#isDragging = true;
            this.#dragStartX = e.touches[0].clientX;
            this.#dragStartTime = this.#timeRange.start;
        }
    }
    
    #onTouchMove(e) {
        if (e.touches.length === 1 && this.#isDragging) {
            e.preventDefault();
            const rect = this.#glCanvas.getBoundingClientRect();
            const dx = e.touches[0].clientX - this.#dragStartX;
            const timeSpan = this.#timeRange.end - this.#timeRange.start;
            const timeDelta = -(dx / rect.width) * timeSpan;
            
            const newStart = this.#dragStartTime + timeDelta;
            const newEnd = newStart + timeSpan;
            
            this.#setTimeRange(newStart, newEnd);
        }
    }
    
    #onTouchEnd(e) {
        this.#isDragging = false;
    }
    
    #setTimeRange(start, end) {
        this.#timeRange = { start, end };
        this.#needsRender = true;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Tooltip
    // ─────────────────────────────────────────────────────────────────────────
    
    #scheduleTooltip(x, y) {
        this.#cancelTooltipTimer();
        if (this.#tooltipDelay > 0) {
            this.#hoverTimeout = setTimeout(() => {
                this.#showTooltipAt(x, y);
            }, this.#tooltipDelay);
        } else {
            this.#showTooltipAt(x, y);
        }
    }
    
    #cancelTooltipTimer() {
        if (this.#hoverTimeout) {
            clearTimeout(this.#hoverTimeout);
            this.#hoverTimeout = null;
        }
    }
    
    #showTooltipAt(x, y) {
        const event = this.#renderer.hitTest(x, y);
        
        if (event) {
            const html = this.#buildTooltipHtml(event);
            this.#tooltipEl.innerHTML = html;
            
            const rect = this.#glCanvas.getBoundingClientRect();
            const containerRect = this.#container.getBoundingClientRect();
            
            let tooltipX = rect.left - containerRect.left + x + 15;
            let tooltipY = rect.top - containerRect.top + y + 15;
            
            this.#tooltipEl.style.display = 'block';
            
            const tooltipRect = this.#tooltipEl.getBoundingClientRect();
            if (tooltipX + tooltipRect.width > containerRect.width) {
                tooltipX = x - tooltipRect.width - 15;
            }
            if (tooltipY + tooltipRect.height > containerRect.height) {
                tooltipY = y - tooltipRect.height - 15;
            }
            
            this.#tooltipEl.style.left = `${Math.max(0, tooltipX)}px`;
            this.#tooltipEl.style.top = `${Math.max(0, tooltipY)}px`;
            
            if (this.#eventHandlers.hover) {
                this.#eventHandlers.hover(event);
            }
        } else {
            this.#hideTooltip();
        }
    }
    
    #buildTooltipHtml(event) {
        const duration = this.#formatDuration(event.duration_ns / 1e6);
        const category = TimelineController.CATEGORIES[event.category]?.label || event.category;
        
        let html = `
            <div style="font-weight: 600; margin-bottom: 4px; color: #fff;">${this.#escapeHtml(event.name || 'Unknown')}</div>
            <div style="color: #aaa; font-size: 11px;">
                <span>${category}</span> &bull; <span>${duration}</span> &bull; <span>Stream ${event.stream}</span>
            </div>
        `;
        
        if (event.metadata) {
            const meta = event.metadata;
            const details = [];
            
            if (meta.grid) {
                details.push(`Grid: [${meta.grid.join(', ')}]`);
            }
            if (meta.block) {
                details.push(`Block: [${meta.block.join(', ')}]`);
            }
            if (meta.registers_per_thread) {
                details.push(`Registers: ${meta.registers_per_thread}`);
            }
            if (meta.bytes) {
                details.push(`Size: ${this.#formatBytes(meta.bytes)}`);
            }
            
            if (details.length > 0) {
                html += `<div style="margin-top: 6px; font-size: 11px; color: #888;">${details.join(' &bull; ')}</div>`;
            }
        }
        
        return html;
    }
    
    #hideTooltip() {
        this.#tooltipEl.style.display = 'none';
    }
    
    #escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // ─────────────────────────────────────────────────────────────────────────
    // Data Loading
    // ─────────────────────────────────────────────────────────────────────────
    
    /**
     * Load timeline data from legacy event objects.
     * @param {Object} data - { events: Array, total_duration_ns: number }
     * @param {Function} onProgress - Progress callback (0-100)
     */
    async load(data, onProgress = null) {
        if (!data || !data.events) {
            console.warn('[TimelineController] No events to load');
            return;
        }
        
        this.#dataMode = 'legacy';
        this.#rawEvents = data.events;
        this.#typedData = null;
        
        console.log(`[TimelineController] Loading ${this.#rawEvents.length} events (legacy)`);
        
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
        
        this.#updateVisibleData();
        
        if (onProgress) onProgress(70);
        
        this.fit();
        
        if (onProgress) onProgress(100);
    }
    
    /**
     * Load timeline data from typed arrays (HDF5 format).
     * This is the optimized path - skips event object conversion.
     * 
     * @param {Object} rendererData - Output from ProfilingHDF5Loader.getRendererData()
     * @param {Function} onProgress - Progress callback (0-100)
     */
    async loadTyped(rendererData, onProgress = null) {
        if (!rendererData || !rendererData.categories) {
            console.warn('[TimelineController] No typed data to load');
            return;
        }
        
        this.#dataMode = 'typed';
        this.#rawEvents = [];
        this.#typedData = rendererData;
        
        console.log(`[TimelineController] Loading ${rendererData.totalEvents} events (typed)`);
        
        if (onProgress) onProgress(10);
        
        // Time range is already computed
        this.#totalTimeRange = { start: 0, end: rendererData.totalDurationMs };
        
        // Compute actual time range from data if needed
        let minTime = Infinity, maxTime = -Infinity;
        for (const category of rendererData.categories) {
            const catData = rendererData.byCategory[category];
            if (!catData || catData.count === 0) continue;
            
            const startMs = catData.startMs;
            const durationMs = catData.durationMs;
            
            for (let i = 0; i < catData.count; i++) {
                const start = startMs[i];
                const end = start + durationMs[i];
                if (start < minTime) minTime = start;
                if (end > maxTime) maxTime = end;
            }
        }
        
        if (minTime !== Infinity) {
            this.#totalTimeRange = { start: minTime, end: maxTime };
        }
        
        if (onProgress) onProgress(30);
        
        this.#updateVisibleDataTyped();
        
        if (onProgress) onProgress(70);
        
        this.fit();
        
        if (onProgress) onProgress(100);
    }
    
    #updateVisibleData() {
        this.#groups = this.#buildGroups();
        
        const rect = this.#container.getBoundingClientRect();
        const timelineHeight = rect.height - this.#axisHeight;
        const rowHeight = this.#groups.length > 0 ? timelineHeight / this.#groups.length : 60;
        this.#groupsSidebar.setGroups(this.#groups, rowHeight);
        
        const visibleEvents = this.#rawEvents.filter(e => this.#visibleCategories.has(e.category));
        this.#renderer.setData(visibleEvents, this.#groups);
        this.#renderer.invalidate();
        
        this.#needsRender = true;
    }
    
    #updateVisibleDataTyped() {
        this.#groups = this.#buildGroupsTyped();
        
        const rect = this.#container.getBoundingClientRect();
        const timelineHeight = rect.height - this.#axisHeight;
        const rowHeight = this.#groups.length > 0 ? timelineHeight / this.#groups.length : 60;
        this.#groupsSidebar.setGroups(this.#groups, rowHeight);
        
        // Filter typed data to visible categories
        const filteredData = {
            ...this.#typedData,
            categories: this.#typedData.categories.filter(c => this.#visibleCategories.has(c))
        };
        
        this.#renderer.setTypedData(filteredData, this.#groups);
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
    
    #buildGroupsTyped() {
        const groups = [];
        
        if (this.#currentGroupBy === TimelineController.GROUP_BY.CATEGORY) {
            for (const [catId, catInfo] of Object.entries(TimelineController.CATEGORIES)) {
                const catData = this.#typedData.byCategory[catId];
                if (catData && catData.count > 0 && this.#visibleCategories.has(catId)) {
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
            // Stream grouping - need to collect unique streams from typed data
            const streams = new Set();
            let hasNvtx = false;
            
            for (const category of this.#typedData.categories) {
                if (!this.#visibleCategories.has(category)) continue;
                
                const catData = this.#typedData.byCategory[category];
                if (!catData || catData.count === 0) continue;
                
                if (category === 'nvtx_range') {
                    hasNvtx = true;
                } else {
                    for (let i = 0; i < catData.count; i++) {
                        streams.add(catData.stream[i]);
                    }
                }
            }
            
            const sortedStreams = [...streams].sort((a, b) => a - b);
            for (const stream of sortedStreams) {
                groups.push({
                    id: `stream_${stream}`,
                    content: `Stream ${stream}`,
                    order: stream,
                    className: 'profiling-group-stream'
                });
            }
            
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
    
    // ─────────────────────────────────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────────────────────────────────
    
    clear() {
        this.#dataMode = 'none';
        this.#rawEvents = [];
        this.#typedData = null;
        this.#groups = [];
        this.#renderer.setData([], []);
        this.#groupsSidebar.setGroups([], 60);
        this.#needsRender = true;
    }
    
    async setGroupBy(groupBy) {
        if (!Object.values(TimelineController.GROUP_BY).includes(groupBy)) {
            console.warn(`[TimelineController] Invalid groupBy: ${groupBy}`);
            return;
        }
        this.#currentGroupBy = groupBy;
        
        if (this.#dataMode === 'typed') {
            this.#updateVisibleDataTyped();
        } else {
            this.#updateVisibleData();
        }
    }
    
    async setVisibleCategories(categories) {
        this.#visibleCategories = new Set(categories);
        
        if (this.#dataMode === 'typed') {
            this.#updateVisibleDataTyped();
        } else {
            this.#updateVisibleData();
        }
    }
    
    async toggleCategory(category, visible) {
        if (visible) {
            this.#visibleCategories.add(category);
        } else {
            this.#visibleCategories.delete(category);
        }
        
        if (this.#dataMode === 'typed') {
            this.#updateVisibleDataTyped();
        } else {
            this.#updateVisibleData();
        }
    }
    
    on(event, handler) {
        this.#eventHandlers[event] = handler;
    }
    
    fit() {
        const padding = (this.#totalTimeRange.end - this.#totalTimeRange.start) * 0.02;
        this.#setTimeRange(
            this.#totalTimeRange.start - padding,
            this.#totalTimeRange.end + padding
        );
        this.#renderer.invalidate();
    }
    
    focus(itemId) {
        // Legacy mode only
        const event = this.#rawEvents.find(e => e.id === itemId);
        if (!event) return;
        
        const startMs = event.start_ns / 1e6;
        const endMs = event.end_ns / 1e6;
        const duration = endMs - startMs;
        const padding = Math.max(duration * 5, 1);
        
        this.#setTimeRange(startMs - padding, endMs + padding);
        this.#renderer.invalidate();
    }
    
    getSummary() {
        if (this.#dataMode === 'typed') {
            return this.#getSummaryTyped();
        }
        
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
    
    #getSummaryTyped() {
        const summary = {
            totalEvents: this.#typedData.totalEvents,
            totalDurationMs: this.#typedData.totalDurationMs,
            categories: {}
        };
        
        for (const category of this.#typedData.categories) {
            const catData = this.#typedData.byCategory[category];
            if (!catData) continue;
            
            let totalDuration = 0;
            for (let i = 0; i < catData.count; i++) {
                totalDuration += catData.durationMs[i];
            }
            
            summary.categories[category] = {
                count: catData.count,
                durationMs: totalDuration
            };
        }
        
        return summary;
    }
    
    get isLoading() {
        return false;
    }
    
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
