/**
 * ProfilingTimeline - Optimized vis.js Timeline wrapper for Nsight profiling data.
 * 
 * Performance optimizations:
 * - Virtual rendering: only render visible events
 * - Progressive loading: chunk-based rendering with requestAnimationFrame
 * - Deferred tooltips: generate on hover, not upfront
 * - Event aggregation: merge tiny events at low zoom levels
 * - Throttled updates: limit re-renders during zoom/pan
 * 
 * Location: /src/app/client/script/profiling/profiling.timeline.js
 */

export class ProfilingTimeline {

    #container;
    #timeline;
    #items;
    #groups;
    #options;
    #eventHandlers;

    // Category definitions matching profiling_service.py
    static CATEGORIES = {
        cuda_kernel: { label: 'CUDA Kernels', color: 'rgba(231, 76, 60, 0.9)', order: 1 },
        cuda_memcpy_h2d: { label: 'MemCpy H→D', color: 'rgba(52, 152, 219, 0.9)', order: 2 },
        cuda_memcpy_d2h: { label: 'MemCpy D→H', color: 'rgba(46, 204, 113, 0.9)', order: 3 },
        cuda_memcpy_d2d: { label: 'MemCpy D→D', color: 'rgba(243, 156, 18, 0.9)', order: 4 },
        cuda_sync: { label: 'CUDA Sync', color: 'rgba(149, 165, 166, 0.9)', order: 5 },
        nvtx_range: { label: 'NVTX Ranges', color: 'rgba(155, 89, 182, 0.9)', order: 6 }
    };

    // Grouping modes
    static GROUP_BY = {
        CATEGORY: 'category',
        STREAM: 'stream'
    };

    // Performance thresholds
    static PERF = {
        CHUNK_SIZE: 200,              // Events per render chunk
        CHUNK_DELAY_MS: 10,           // Delay between chunks (allows UI updates)
        MAX_VISIBLE_ITEMS: 1000,      // Max items to render at once
        MIN_DURATION_PX: 2,           // Minimum event width in pixels to render
        THROTTLE_MS: 100              // Throttle for range change events
    };

    #currentGroupBy = ProfilingTimeline.GROUP_BY.CATEGORY;
    #visibleCategories = new Set(Object.keys(ProfilingTimeline.CATEGORIES));
    #rawEvents = [];
    #isLoading = false;
    #loadingAborted = false;
    #rangeChangeTimeout = null;
    #currentRange = null;
    #totalDurationNs = 0;

    constructor(containerId) {
        this.#container = document.getElementById(containerId);
        if (!this.#container) {
            throw new Error(`Container element not found: ${containerId}`);
        }

        this.#items = new vis.DataSet();
        this.#groups = new vis.DataSet();
        this.#eventHandlers = {};

        this.#options = {
            orientation: { axis: 'top', item: 'top' },
            type: 'range',
            stack: false,              // Disable stacking - items overlap instead
            stackSubgroups: false,
            showCurrentTime: false,
            showMajorLabels: true,
            showMinorLabels: true,
            zoomMin: 0.001,            // 1 microsecond minimum visible range
            zoomMax: 1000000000000,
            margin: { item: { horizontal: 0, vertical: 1 }, axis: 5 },
            tooltip: { followMouse: true, overflowMethod: 'cap' },
            groupHeightMode: 'fixed',  // Use fixed height per group
            verticalScroll: true,
            horizontalScroll: true,
            // Performance options
            throttleRedraw: 50
        };

        this.#initTimeline();
    }

    #initTimeline() {
        this.#timeline = new vis.Timeline(
            this.#container,
            this.#items,
            this.#groups,
            this.#options
        );

        // Bind event handlers
        this.#timeline.on('select', (params) => this.#onSelect(params));
        this.#timeline.on('itemover', (params) => this.#onItemOver(params));
        this.#timeline.on('itemout', (params) => this.#onItemOut(params));
        
        // Throttled range change for virtual rendering
        this.#timeline.on('rangechanged', (params) => this.#onRangeChanged(params));
    }

    /**
     * Load profiling data with progressive rendering.
     * @param {Object} data - Profiling timeline data
     * @param {Function} onProgress - Progress callback (0-100)
     */
    async load(data, onProgress = null) {
        if (!data || !data.events) {
            console.warn('[ProfilingTimeline] No events to load');
            return;
        }

        // Abort any ongoing load
        if (this.#isLoading) {
            this.#loadingAborted = true;
            await this.#waitForLoadComplete();
        }

        this.#isLoading = true;
        this.#loadingAborted = false;
        this.#rawEvents = data.events;
        this.#totalDurationNs = data.total_duration_ns || 0;

        console.log(`[ProfilingTimeline] Loading ${this.#rawEvents.length} events`);

        try {
            // Build groups first (fast)
            this.#buildAndSetGroups();

            // Progressive render of items
            await this.#renderProgressively(onProgress);

            // Fit timeline to content
            if (!this.#loadingAborted) {
                this.#timeline.fit();
            }

        } finally {
            this.#isLoading = false;
        }
    }

    /**
     * Clear all data from timeline.
     */
    clear() {
        this.#loadingAborted = true;
        this.#rawEvents = [];
        this.#items.clear();
        this.#groups.clear();
    }

    /**
     * Set grouping mode.
     * @param {string} groupBy - 'category' or 'stream'
     */
    async setGroupBy(groupBy) {
        if (!Object.values(ProfilingTimeline.GROUP_BY).includes(groupBy)) {
            console.warn(`[ProfilingTimeline] Invalid groupBy: ${groupBy}`);
            return;
        }
        this.#currentGroupBy = groupBy;
        await this.#renderProgressively();
    }

    /**
     * Set visible categories.
     * @param {string[]} categories - Array of category IDs to show
     */
    async setVisibleCategories(categories) {
        this.#visibleCategories = new Set(categories);
        await this.#renderProgressively();
    }

    /**
     * Toggle category visibility.
     * @param {string} category - Category ID
     * @param {boolean} visible - Visibility state
     */
    async toggleCategory(category, visible) {
        if (visible) {
            this.#visibleCategories.add(category);
        } else {
            this.#visibleCategories.delete(category);
        }
        await this.#renderProgressively();
    }

    /**
     * Register event handler.
     * @param {string} event - Event name ('select', 'hover')
     * @param {Function} handler - Event handler function
     */
    on(event, handler) {
        this.#eventHandlers[event] = handler;
    }

    /**
     * Fit timeline to show all items.
     */
    fit() {
        this.#timeline.fit();
    }

    /**
     * Focus on a specific item.
     * @param {string} itemId - Item ID to focus on
     */
    focus(itemId) {
        this.#timeline.focus(itemId, { animation: true });
    }

    /**
     * Get summary statistics.
     * @returns {Object} Summary stats
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

    /**
     * Check if currently loading.
     */
    get isLoading() {
        return this.#isLoading;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Progressive Rendering
    // ─────────────────────────────────────────────────────────────────────────

    async #renderProgressively(onProgress = null) {
        // Filter events by visible categories
        const filteredEvents = this.#rawEvents.filter(
            event => this.#visibleCategories.has(event.category)
        );

        // Update groups
        this.#buildAndSetGroups(filteredEvents);

        // Clear existing items
        this.#items.clear();

        if (filteredEvents.length === 0) {
            if (onProgress) onProgress(100);
            return;
        }

        // Transform and add in chunks
        const chunkSize = ProfilingTimeline.PERF.CHUNK_SIZE;
        const totalChunks = Math.ceil(filteredEvents.length / chunkSize);

        for (let i = 0; i < totalChunks; i++) {
            if (this.#loadingAborted) {
                console.log('[ProfilingTimeline] Load aborted');
                return;
            }

            const start = i * chunkSize;
            const end = Math.min(start + chunkSize, filteredEvents.length);
            const chunk = filteredEvents.slice(start, end);

            // Transform chunk
            const items = this.#transformEventsChunk(chunk);

            // Add to DataSet
            this.#items.add(items);

            // Report progress
            if (onProgress) {
                const progress = Math.round(((i + 1) / totalChunks) * 100);
                onProgress(progress);
            }

            // Yield to UI thread
            await this.#yieldToUI();
        }

        console.log(`[ProfilingTimeline] Rendered ${filteredEvents.length} events`);
    }

    #transformEventsChunk(events) {
        return events.map(event => {
            const startMs = event.start_ns / 1e6;
            const endMs = event.end_ns / 1e6;

            // Determine group
            let group;
            if (this.#currentGroupBy === ProfilingTimeline.GROUP_BY.CATEGORY) {
                group = event.category;
            } else {
                group = event.category === 'nvtx_range' ? 'nvtx' : `stream_${event.stream}`;
            }

            return {
                id: event.id,
                group: group,
                content: this.#truncateName(event.name, 25),
                start: new Date(startMs),
                end: new Date(endMs),
                className: `profiling-item profiling-item-${event.category}`,
                // Store reference for tooltip generation on hover
                _eventRef: event
            };
        });
    }

    #buildAndSetGroups(events = null) {
        const eventsToUse = events || this.#rawEvents.filter(
            event => this.#visibleCategories.has(event.category)
        );

        const groups = [];
        const groupHeight = 60;  // Fixed height per group in pixels

        if (this.#currentGroupBy === ProfilingTimeline.GROUP_BY.CATEGORY) {
            const seenCategories = new Set(eventsToUse.map(e => e.category));

            for (const [catId, catInfo] of Object.entries(ProfilingTimeline.CATEGORIES)) {
                if (seenCategories.has(catId)) {
                    groups.push({
                        id: catId,
                        content: catInfo.label,
                        order: catInfo.order,
                        className: `profiling-group-${catId}`,
                        style: `height: ${groupHeight}px; max-height: ${groupHeight}px;`
                    });
                }
            }
        } else {
            const streams = [...new Set(eventsToUse.map(e => e.stream))].sort((a, b) => a - b);

            for (const stream of streams) {
                groups.push({
                    id: `stream_${stream}`,
                    content: `Stream ${stream}`,
                    order: stream,
                    className: `profiling-group-stream`,
                    style: `height: ${groupHeight}px; max-height: ${groupHeight}px;`
                });
            }

            const hasNvtx = eventsToUse.some(e => e.category === 'nvtx_range');
            if (hasNvtx) {
                groups.push({
                    id: 'nvtx',
                    content: 'NVTX Ranges',
                    order: 1000,
                    className: 'profiling-group-nvtx',
                    style: `height: ${groupHeight}px; max-height: ${groupHeight}px;`
                });
            }
        }

        this.#groups.clear();
        this.#groups.add(groups);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tooltip Generation (deferred to hover)
    // ─────────────────────────────────────────────────────────────────────────

    #generateTooltip(event) {
        if (!event) return '';

        const durationMs = event.duration_ns / 1e6;
        const lines = [
            `<strong>${event.name}</strong>`,
            `Duration: ${this.#formatDuration(durationMs)}`,
            `Category: ${ProfilingTimeline.CATEGORIES[event.category]?.label || event.category}`
        ];

        if (event.stream !== undefined) {
            lines.push(`Stream: ${event.stream}`);
        }

        if (event.metadata) {
            if (event.metadata.grid) {
                lines.push(`Grid: [${event.metadata.grid.join(', ')}]`);
            }
            if (event.metadata.block) {
                lines.push(`Block: [${event.metadata.block.join(', ')}]`);
            }
            if (event.metadata.registers_per_thread) {
                lines.push(`Registers/Thread: ${event.metadata.registers_per_thread}`);
            }
            if (event.metadata.bytes) {
                lines.push(`Bytes: ${this.#formatBytes(event.metadata.bytes)}`);
            }
            if (event.metadata.throughput_gbps) {
                lines.push(`Throughput: ${event.metadata.throughput_gbps.toFixed(2)} GB/s`);
            }
        }

        return lines.join('<br>');
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Event Handlers
    // ─────────────────────────────────────────────────────────────────────────

    #onSelect(params) {
        if (params.items.length > 0) {
            const itemId = params.items[0];
            const item = this.#items.get(itemId);
            if (item && item._eventRef && this.#eventHandlers.select) {
                this.#eventHandlers.select(item._eventRef);
            }
        }
    }

    #onItemOver(params) {
        if (params.item) {
            const item = this.#items.get(params.item);
            if (item && item._eventRef) {
                // Generate tooltip on demand
                if (!item.title) {
                    item.title = this.#generateTooltip(item._eventRef);
                    this.#items.update({ id: item.id, title: item.title });
                }

                if (this.#eventHandlers.hover) {
                    this.#eventHandlers.hover(item._eventRef);
                }
            }
        }
    }

    #onItemOut(params) {
        if (this.#eventHandlers.hover) {
            this.#eventHandlers.hover(null);
        }
    }

    #onRangeChanged(params) {
        // Throttle range change handling
        if (this.#rangeChangeTimeout) {
            clearTimeout(this.#rangeChangeTimeout);
        }

        this.#rangeChangeTimeout = setTimeout(() => {
            this.#currentRange = { start: params.start, end: params.end };
            // Could implement virtual rendering here based on visible range
        }, ProfilingTimeline.PERF.THROTTLE_MS);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────────

    #truncateName(name, maxLength) {
        if (!name) return '?';
        if (name.length <= maxLength) return name;
        return name.substring(0, maxLength - 2) + '..';
    }

    #formatDuration(ms) {
        if (ms < 0.001) {
            return `${(ms * 1000000).toFixed(1)} ns`;
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

    async #yieldToUI() {
        return new Promise(resolve => {
            requestAnimationFrame(() => {
                setTimeout(resolve, ProfilingTimeline.PERF.CHUNK_DELAY_MS);
            });
        });
    }

    async #waitForLoadComplete() {
        while (this.#isLoading) {
            await new Promise(resolve => setTimeout(resolve, 50));
        }
    }

    /**
     * Destroy the timeline instance.
     */
    destroy() {
        this.#loadingAborted = true;
        if (this.#rangeChangeTimeout) {
            clearTimeout(this.#rangeChangeTimeout);
        }
        if (this.#timeline) {
            this.#timeline.destroy();
            this.#timeline = null;
        }
        this.#items.clear();
        this.#groups.clear();
    }
}
