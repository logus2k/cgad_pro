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
import { TimelineLabels } from './timeline_labels.js';


export class TimelineController {
    
    // Category definitions (same as original)
    static CATEGORIES = {
        cuda_kernel: { label: 'CUDA Kernels', order: 1 },
        cuda_memcpy_h2d: { label: 'MemCpy H-D', order: 2 },
        cuda_memcpy_d2h: { label: 'MemCpy D-H', order: 3 },
        cuda_memcpy_d2d: { label: 'MemCpy D-D', order: 4 },
        cuda_sync: { label: 'CUDA Sync', order: 5 },
        nvtx_phases: { label: 'NVTX Phases', order: 6 },
        nvtx_cuda: { label: 'NVTX CUDA', order: 7 }
    };

    static NVTX_PHASE_NAMES = new Set([
        'load_mesh',
        'assemble_system',
        'apply_bc',
        'solve_system',
        'compute_derived',
        'export_results'
    ]);
    
    static GROUP_BY = {
        CATEGORY: 'category',
        STREAM: 'stream'
    };
    
    #container;
    #renderer;
    #axis;
    #groupsSidebar;

    #labels;
    #labelsCanvas;
    #ncuDataByKernel = new Map();
    
    // DOM elements
    #mainEl;
    #axisCanvas;
    #glCanvas;
    #groupsEl;
    #tooltipEl;
    #cursorLineEl;
    #cursorBadgeEl;
    
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
        this.#container.style.backgroundColor = 'rgba(255, 255, 255, 0.6)';
        
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
        
        // WebGL canvas container (for overlay positioning)
        const canvasContainer = document.createElement('div');
        canvasContainer.style.cssText = 'flex: 1; position: relative; min-height: 0;';

        // WebGL canvas
        this.#glCanvas = document.createElement('canvas');
        this.#glCanvas.style.cssText = 'width: 100%; height: 100%; display: block; cursor: default;';
        canvasContainer.appendChild(this.#glCanvas);

        // Labels canvas (overlay)
        this.#labelsCanvas = document.createElement('canvas');
        this.#labelsCanvas.style.cssText = 'position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;';
        canvasContainer.appendChild(this.#labelsCanvas);

        timelineArea.appendChild(canvasContainer);
        
        wrapper.appendChild(this.#groupsEl);
        wrapper.appendChild(timelineArea);
        this.#container.appendChild(wrapper);
        
        // Tooltip
        this.#tooltipEl = document.createElement('div');
        this.#tooltipEl.style.cssText = `
            position: absolute;
            display: none;
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 0;
            font-family: 'Roboto', system-ui, sans-serif;
            font-size: 11px;
            color: #212529;
            pointer-events: none;
            z-index: 1000;
            max-width: 320px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            overflow: hidden;
        `;
        const overlay = document.getElementById('overlay');
        (overlay || document.body).appendChild(this.#tooltipEl);

        // Cursor time indicator (vertical line + time badge)
        this.#cursorLineEl = document.createElement('div');
        this.#cursorLineEl.className = 'timeline-cursor-line';
        this.#container.appendChild(this.#cursorLineEl);

        this.#cursorBadgeEl = document.createElement('div');
        this.#cursorBadgeEl.className = 'timeline-cursor-badge';
        this.#container.appendChild(this.#cursorBadgeEl); 

        this.#groupsEl._listContainer = groupsList;
    }
    
    #initComponents() {
        this.#renderer = new TimelineRenderer(this.#glCanvas);
        this.#axis = new TimelineAxis(this.#axisCanvas, this.#axisHeight);
        this.#labels = new TimelineLabels(this.#labelsCanvas);
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
        this.#labels.resize(timelineWidth, timelineHeight);
        
        if (this.#groups.length > 0) {
            const rowHeight = timelineHeight / this.#groups.length;
            this.#groupsSidebar.updateRowHeight(rowHeight);
        }
        
        this.#renderer.render();
        this.#axis.render();
        this.#updateLabels();
        this.#labels.render();
        this.#needsRender = false;
    }

    #bindEvents() {
        const canvas = this.#glCanvas;
        
        canvas.addEventListener('mousedown', (e) => this.#onMouseDown(e));
        window.addEventListener('mousemove', (e) => this.#onMouseMove(e));
        window.addEventListener('mouseup', (e) => this.#onMouseUp(e));
        canvas.addEventListener('wheel', (e) => this.#onWheel(e), { passive: false });

        canvas.addEventListener('mouseenter', () => {
            if (!this.#isDragging) {
                this.#showCursorLine(true);
            }
        });
        canvas.addEventListener('mouseleave', () => {
            if (!this.#isDragging) {
                this.#showCursorLine(false);
            }
        });  
        
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
                this.#updateLabels();
                this.#labels.render();
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
            const timeDelta = -(dx / rect.width) * timeSpan;
            
            const newStart = this.#dragStartTime + timeDelta;
            const newEnd = newStart + timeSpan;
            
            this.#setTimeRange(newStart, newEnd);
            } else if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {

                // Show and update cursor line position
                this.#showCursorLine(true);
                this.#updateCursorLine(x);

                // Check if hovering over an event
                const event = this.#renderer.hitTest(x, y);
                this.#glCanvas.style.cursor = event ? 'pointer' : 'grab';
                
                this.#scheduleTooltip(x, y);
            } else {
            this.#glCanvas.style.cursor = 'default';
            this.#hideTooltip();
            this.#showCursorLine(false);
        }
    }
    
    #onMouseUp(e) {
        if (this.#isDragging) {
            this.#isDragging = false;
            // Restore cursor based on current position
            const rect = this.#glCanvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
                const event = this.#renderer.hitTest(x, y);
                this.#glCanvas.style.cursor = event ? 'pointer' : 'grab';
                this.#showCursorLine(true);
                this.#updateCursorLine(x);                
            } else {
                this.#glCanvas.style.cursor = 'default';
            }
        }
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
            
            this.#tooltipEl.style.display = 'block';

            // Set z-index just above the profiling panel
            const profilingPanel = document.getElementById('hud-profiling');
            if (profilingPanel) {
                this.#tooltipEl.style.zIndex = (parseInt(profilingPanel.style.zIndex) || 20) + 1;
            }            
            
            const tooltipRect = this.#tooltipEl.getBoundingClientRect();
            
            // Position relative to page, below and to the right of cursor
            const cursorOffset = 20;
            let tooltipX = rect.left + x + cursorOffset;
            let tooltipY = rect.top + y + cursorOffset;
            
            // If tooltip goes beyond right edge, flip to left of cursor
            if (tooltipX + tooltipRect.width > window.innerWidth) {
                tooltipX = rect.left + x - tooltipRect.width - cursorOffset;
            }
            
            this.#tooltipEl.style.left = `${Math.max(0, tooltipX)}px`;
            this.#tooltipEl.style.top = `${tooltipY}px`;
            
            if (this.#eventHandlers.hover) {
                this.#eventHandlers.hover(event);
            }
        } else {
            this.#hideTooltip();
        }
    }
    
    #buildTooltipHtml(event) {
        const durationMs = event.duration_ns / 1e6;
        const startMs = event.start_ns / 1e6;
        const sessionDuration = this.#totalTimeRange.end - this.#totalTimeRange.start;
        const percent = sessionDuration > 0 ? ((durationMs / sessionDuration) * 100) : 0;
        
        const category = event.category;
        const categoryLabel = TimelineController.CATEGORIES[category]?.label || category;
        const accentColor = this.#getCategoryColor(category);
        
        // Header
        let html = `
            <div style="display: flex; border-bottom: 1px solid #e9ecef;">
                <div style="width: 4px; background: ${accentColor};"></div>
                <div style="flex: 1; padding: 8px 10px;">
                    <div style="font-weight: 600; font-size: 12px; color: #212529; margin-bottom: 2px; word-break: break-word;">
                        ${this.#escapeHtml(event.name || 'Unknown')}
                    </div>
                    <div style="font-size: 10px; color: #6c757d;">
                        <span style="background: ${accentColor}22; color: ${accentColor}; padding: 1px 5px; border-radius: 3px; font-weight: 500;">${categoryLabel}</span>
                        <span style="margin-left: 6px;">Stream ${event.stream}</span>
                    </div>
                </div>
            </div>
        `;
        
        // Nsight Systems section
        html += `<div style="padding: 6px 10px 2px 10px; font-size: 10px; font-weight: 600; color: #6c757d; border-bottom: 1px solid #f0f0f0;">Nsight Systems</div>`;
        html += `<div style="padding: 6px 10px 8px 10px; display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; font-size: 11px;">`;
        
        // Timing
        html += this.#tooltipRow('Start', this.#formatDuration(startMs));
        html += this.#tooltipRow('Duration', `${this.#formatDuration(durationMs)} <span style="color: #6c757d;">(${percent.toFixed(2)}%)</span>`);
        
        // Category-specific details from nsys
        if (event.metadata) {
            const meta = event.metadata;
            
            if (category === 'cuda_kernel') {
                if (meta.grid) {
                    html += this.#tooltipRow('Grid', `[${meta.grid.join(', ')}]`);
                }
                if (meta.block) {
                    html += this.#tooltipRow('Block', `[${meta.block.join(', ')}]`);
                }
                if (meta.grid && meta.block) {
                    const totalThreads = 
                        (meta.grid[0] * meta.grid[1] * meta.grid[2]) *
                        (meta.block[0] * meta.block[1] * meta.block[2]);
                    html += this.#tooltipRow('Threads', totalThreads.toLocaleString());
                }
                if (meta.registers_per_thread) {
                    html += this.#tooltipRow('Registers', `${meta.registers_per_thread}/thread`);
                }
                const sharedTotal = (meta.shared_memory_static || 0) + (meta.shared_memory_dynamic || 0);
                if (sharedTotal > 0) {
                    html += this.#tooltipRow('Shared Mem', this.#formatBytes(sharedTotal));
                }
            } else if (category.startsWith('cuda_memcpy')) {
                if (meta.bytes) {
                    html += this.#tooltipRow('Size', this.#formatBytes(meta.bytes));
                    const throughputGBs = (meta.bytes / (durationMs / 1000)) / (1024 * 1024 * 1024);
                    if (isFinite(throughputGBs) && throughputGBs > 0) {
                        html += this.#tooltipRow('Throughput', `${throughputGBs.toFixed(2)} GB/s`);
                    }
                }
            }
        }
        
        html += `</div>`;
        
        // Nsight Compute section (only for CUDA kernels with NCU data)
        if (category === 'cuda_kernel' && event.name) {
            const ncuData = this.#ncuDataByKernel.get(event.name);
            if (ncuData) {
                html += `<div style="padding: 6px 10px 2px 10px; font-size: 10px; font-weight: 600; color: #6c757d; border-bottom: 1px solid #f0f0f0; border-top: 1px solid #e9ecef;">Nsight Compute</div>`;
                html += `<div style="padding: 6px 10px 8px 10px; display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; font-size: 11px;">`;
                
                // Occupancy
                if (ncuData.occupancy_achieved !== undefined) {
                    const occAchieved = (ncuData.occupancy_achieved * 100).toFixed(1);
                    const occTheoretical = (ncuData.occupancy_theoretical * 100).toFixed(1);
                    html += this.#tooltipRow('Occupancy', `${occAchieved}% <span style="color: #6c757d;">(theo: ${occTheoretical}%)</span>`);
                }
                
                // Throughput
                if (ncuData.sm_throughput_pct !== undefined) {
                    html += this.#tooltipRow('SM Throughput', `${ncuData.sm_throughput_pct.toFixed(1)}%`);
                }
                if (ncuData.dram_throughput_pct !== undefined) {
                    html += this.#tooltipRow('DRAM Throughput', `${ncuData.dram_throughput_pct.toFixed(1)}%`);
                }
                
                // Cache hit rates
                if (ncuData.l1_hit_rate !== undefined && ncuData.l1_hit_rate > 0) {
                    html += this.#tooltipRow('L1 Hit Rate', `${(ncuData.l1_hit_rate * 100).toFixed(1)}%`);
                }
                if (ncuData.l2_hit_rate !== undefined && ncuData.l2_hit_rate > 0) {
                    html += this.#tooltipRow('L2 Hit Rate', `${(ncuData.l2_hit_rate * 100).toFixed(1)}%`);
                }
                
                // Warp efficiency
                if (ncuData.warp_execution_efficiency !== undefined && ncuData.warp_execution_efficiency > 0) {
                    html += this.#tooltipRow('Warp Efficiency', `${(ncuData.warp_execution_efficiency * 100).toFixed(1)}%`);
                }
                
                html += `</div>`;
            }
        }
        
        return html;
    }

    #tooltipRow(label, value) {
        return `
            <div style="color: #6c757d; white-space: nowrap;">${label}</div>
            <div style="color: #212529;">${value}</div>
        `;
    }

    #getCategoryColor(category) {
        const colors = {
            cuda_kernel: '#e74c3c',
            cuda_memcpy_h2d: '#3498db',
            cuda_memcpy_d2h: '#2ecc71',
            cuda_memcpy_d2d: '#f39c12',
            cuda_sync: '#95a5a6',
            nvtx_range: '#9b59b6',
            nvtx_phases: '#9b59b6',
            nvtx_cuda: '#888888'
        };
        return colors[category] || '#6c757d';
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

        // Force resize to ensure sidebar row heights match
        requestAnimationFrame(() => this.#doResize());

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
        
        // Include nvtx_range data if either nvtx_phases or nvtx_cuda is visible
        const visibleDataCategories = new Set();
        for (const cat of this.#typedData.categories) {
            if (cat === 'nvtx_range') {
                if (this.#visibleCategories.has('nvtx_phases') || this.#visibleCategories.has('nvtx_cuda')) {
                    visibleDataCategories.add(cat);
                }
            } else if (this.#visibleCategories.has(cat)) {
                visibleDataCategories.add(cat);
            }
        }
        
        const filteredData = {
            ...this.#typedData,
            categories: this.#typedData.categories.filter(c => visibleDataCategories.has(c))
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
            // Check if nvtx_range data exists (for the split categories)
            const hasNvtx = this.#typedData.byCategory['nvtx_range']?.count > 0;
            
            for (const [catId, catInfo] of Object.entries(TimelineController.CATEGORIES)) {
                // Handle nvtx_phases and nvtx_cuda specially
                if (catId === 'nvtx_phases' || catId === 'nvtx_cuda') {
                    if (hasNvtx && this.#visibleCategories.has(catId)) {
                        groups.push({
                            id: catId,
                            content: catInfo.label,
                            order: catInfo.order,
                            className: `profiling-item-${catId}`
                        });
                    }
                } else {
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
            }
            
            groups.sort((a, b) => a.order - b.order);
        } else {
            // Stream grouping
            const streams = new Set();
            let hasNvtx = false;
            
            for (const category of this.#typedData.categories) {
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
                if (this.#visibleCategories.has('nvtx_phases')) {
                    groups.push({ id: 'nvtx_phases', content: 'NVTX Phases', order: 1000, className: 'profiling-item-nvtx_phases' });
                }
                if (this.#visibleCategories.has('nvtx_cuda')) {
                    groups.push({ id: 'nvtx_cuda', content: 'NVTX CUDA', order: 1001, className: 'profiling-item-nvtx_cuda' });
                }
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
        this.#totalTimeRange = { start: 0, end: 0 };
        this.#timeRange = { start: 0, end: 0 };
        this.#ncuDataByKernel.clear();
        this.#renderer.clear();
        this.#groupsSidebar.setGroups([], 60);
        this.#labels.setLabels([]);
        this.#labels.render();
        this.#axis.setTimeRange(0, 0);
        this.#axis.render();
        this.#needsRender = false;
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

    /**
     * Set NCU kernel metrics for tooltip enrichment.
     * @param {Array} kernels - Array of kernel metrics from NCU
     */
    setNcuData(kernels) {
        this.#ncuDataByKernel.clear();
        for (const kernel of kernels) {
            // Index by kernel name for quick lookup
            const name = kernel.kernel_name;
            if (!this.#ncuDataByKernel.has(name)) {
                this.#ncuDataByKernel.set(name, kernel);
            }
        }
        console.log(`[TimelineController] Loaded NCU data for ${this.#ncuDataByKernel.size} unique kernels`);
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

    #showCursorLine(show) {
        if (this.#cursorLineEl) {
            this.#cursorLineEl.style.display = show ? 'block' : 'none';
        }
        if (this.#cursorBadgeEl) {
            this.#cursorBadgeEl.style.display = show ? 'block' : 'none';
        }
    }

    #updateCursorLine(x) {
        if (!this.#cursorLineEl || !this.#cursorBadgeEl) return;
        
        const rect = this.#glCanvas.getBoundingClientRect();
        const containerRect = this.#container.getBoundingClientRect();
        
        // Position relative to container
        const lineX = Math.round(rect.left - containerRect.left + x);
        
        // Update line position and height
        this.#cursorLineEl.style.left = `${lineX}px`;
        this.#cursorLineEl.style.top = `${rect.top - containerRect.top}px`;
        this.#cursorLineEl.style.height = `${rect.height}px`;
        
        // Calculate time at cursor position
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        const timeAtCursor = this.#timeRange.start + (x / rect.width) * timeSpan;
        
        // Update badge
        this.#cursorBadgeEl.textContent = this.#formatDuration(timeAtCursor);
        this.#cursorBadgeEl.style.left = `${lineX}px`;
        this.#cursorBadgeEl.style.top = `${rect.top - containerRect.top - 20}px`;
    }   
    
    #updateLabels() {
        // Get NVTX range data from renderer for label display
        const nvtxLabels = this.#renderer.getNvtxLabels();
        this.#labels.setTimeRange(this.#timeRange.start, this.#timeRange.end);
        this.#labels.setLabels(nvtxLabels);
    }    
    
    destroy() {

        if (this.#animationFrame) {
            cancelAnimationFrame(this.#animationFrame);
        }
        if (this.#resizeObserver) {
            this.#resizeObserver.disconnect();
        }
        if (this.#tooltipEl && this.#tooltipEl.parentNode) {
            this.#tooltipEl.parentNode.removeChild(this.#tooltipEl);
        }

        this.#cancelTooltipTimer();
        this.#renderer.destroy();
        this.#axis.destroy();
        this.#groupsSidebar.destroy();
        this.#labels.destroy();
        this.#container.innerHTML = '';
    }
}
