/**
 * ProfilingWorker - Web Worker for timeline data processing.
 * 
 * Offloads heavy computation from main thread:
 * - Event filtering
 * - Data transformation for vis.js
 * - Group building
 * - Summary statistics
 * 
 * Location: /src/app/client/script/profiling/profiling.worker.js
 */

// Category definitions (duplicated here since workers can't share imports easily)
const CATEGORIES = {
    cuda_kernel: { label: 'CUDA Kernels', order: 1 },
    cuda_memcpy_h2d: { label: 'MemCpy H→D', order: 2 },
    cuda_memcpy_d2h: { label: 'MemCpy D→H', order: 3 },
    cuda_memcpy_d2d: { label: 'MemCpy D→D', order: 4 },
    cuda_sync: { label: 'CUDA Sync', order: 5 },
    nvtx_range: { label: 'NVTX Ranges', order: 6 }
};

const GROUP_BY = {
    CATEGORY: 'category',
    STREAM: 'stream'
};

/**
 * Message handler
 */
self.onmessage = function(e) {
    const { type, payload, requestId } = e.data;

    try {
        let result;

        switch (type) {
            case 'transform':
                result = transformEvents(payload);
                break;

            case 'filter':
                result = filterEvents(payload);
                break;

            case 'summary':
                result = computeSummary(payload);
                break;

            case 'buildGroups':
                result = buildGroups(payload);
                break;

            default:
                throw new Error(`Unknown message type: ${type}`);
        }

        self.postMessage({ requestId, type: 'result', payload: result });

    } catch (error) {
        self.postMessage({ requestId, type: 'error', error: error.message });
    }
};

/**
 * Transform raw events to vis.js format.
 * This is the heavy operation.
 */
function transformEvents({ events, groupBy, visibleCategories, chunkIndex, chunkSize }) {
    const startIdx = chunkIndex * chunkSize;
    const endIdx = Math.min(startIdx + chunkSize, events.length);
    const chunk = events.slice(startIdx, endIdx);

    const items = [];

    for (let i = 0; i < chunk.length; i++) {
        const event = chunk[i];

        // Skip if category not visible
        if (!visibleCategories.includes(event.category)) {
            continue;
        }

        const startMs = event.start_ns / 1e6;
        const endMs = event.end_ns / 1e6;

        // Determine group
        let group;
        if (groupBy === GROUP_BY.CATEGORY) {
            group = event.category;
        } else {
            group = event.category === 'nvtx_range' ? 'nvtx' : `stream_${event.stream}`;
        }

        items.push({
            id: event.id,
            group: group,
            content: truncateName(event.name, 25),
            start: startMs,  // Send as number, convert to Date on main thread
            end: endMs,
            className: `profiling-item profiling-item-${event.category}`,
            // Store minimal event data for tooltips (generated on hover)
            _eventData: {
                name: event.name,
                category: event.category,
                duration_ns: event.duration_ns,
                stream: event.stream,
                metadata: event.metadata
            }
        });
    }

    return {
        items,
        processed: endIdx,
        total: events.length,
        done: endIdx >= events.length
    };
}

/**
 * Filter events by visible categories.
 */
function filterEvents({ events, visibleCategories }) {
    const filtered = events.filter(e => visibleCategories.includes(e.category));
    return { events: filtered, count: filtered.length };
}

/**
 * Compute summary statistics.
 */
function computeSummary({ events }) {
    const summary = {
        totalEvents: events.length,
        totalDurationMs: 0,
        categories: {}
    };

    for (const event of events) {
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
 * Build groups based on grouping mode.
 */
function buildGroups({ events, groupBy, visibleCategories }) {
    const filteredEvents = events.filter(e => visibleCategories.includes(e.category));
    const groups = [];

    if (groupBy === GROUP_BY.CATEGORY) {
        const seenCategories = new Set(filteredEvents.map(e => e.category));

        for (const [catId, catInfo] of Object.entries(CATEGORIES)) {
            if (seenCategories.has(catId)) {
                groups.push({
                    id: catId,
                    content: catInfo.label,
                    order: catInfo.order,
                    className: `profiling-group-${catId}`
                });
            }
        }
    } else {
        const streams = [...new Set(filteredEvents.map(e => e.stream))].sort((a, b) => a - b);

        for (const stream of streams) {
            groups.push({
                id: `stream_${stream}`,
                content: `Stream ${stream}`,
                order: stream,
                className: 'profiling-group-stream'
            });
        }

        const hasNvtx = filteredEvents.some(e => e.category === 'nvtx_range');
        if (hasNvtx) {
            groups.push({
                id: 'nvtx',
                content: 'NVTX Ranges',
                order: 1000,
                className: 'profiling-group-nvtx'
            });
        }
    }

    return { groups };
}

/**
 * Truncate name for display.
 */
function truncateName(name, maxLength) {
    if (!name) return '?';
    if (name.length <= maxLength) return name;
    return name.substring(0, maxLength - 2) + '..';
}
