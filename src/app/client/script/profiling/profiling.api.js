/**
 * ProfilingAPI - REST client for profiling service endpoints.
 * 
 * Supports both JSON (legacy) and HDF5 (optimized) timeline formats.
 * HDF5 provides ~10-20x smaller file sizes and accurate download progress.
 * 
 * Location: /src/app/client/script/profiling/profiling.api.js
 */

import { loadProfilingHDF5, initH5Wasm } from './profiling_hdf5_loader.js';

export class ProfilingAPI {

    #baseUrl;
    #preferHDF5;

    /**
     * @param {string} baseUrl - API base URL (auto-detected if null)
     * @param {Object} options
     * @param {boolean} options.preferHDF5 - Use HDF5 format when available (default: true)
     */
    constructor(baseUrl = null, options = {}) {
        // Dynamic URL resolution (same logic as FEMClient)
        if (baseUrl === null) {
            const hostname = window.location.hostname;
            const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';
            baseUrl = isLocal ? '' : '/fem';
        }
        
        this.#baseUrl = baseUrl;
        this.#preferHDF5 = options.preferHDF5 !== false;
        
        // Pre-initialize h5wasm if HDF5 preferred
        if (this.#preferHDF5) {
            initH5Wasm().catch(err => {
                console.warn('[ProfilingAPI] h5wasm init failed, will fall back to JSON:', err);
            });
        }
    }

    /**
     * Get the base URL for API requests.
     * @returns {string}
     */
    getBaseUrl() {
        return this.#baseUrl;
    }    

    /**
     * Get available profiling modes.
     * @returns {Promise<{modes: Array, nsys_available: boolean, ncu_available: boolean}>}
     */
    async getModes() {
        const response = await fetch(`${this.#baseUrl}/api/profiling/modes`);
        if (!response.ok) {
            throw new Error(`Failed to fetch modes: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Get list of profiling sessions.
     * @param {number} limit - Maximum number of sessions to retrieve
     * @returns {Promise<{sessions: Array}>}
     */
    async getSessions(limit = 50) {
        const response = await fetch(`${this.#baseUrl}/api/profiling/sessions?limit=${limit}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch sessions: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Get session details and summary.
     * @param {string} sessionId 
     * @returns {Promise<Object>}
     */
    async getSession(sessionId) {
        const response = await fetch(`${this.#baseUrl}/api/profiling/session/${sessionId}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch session: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Get timeline data for a session.
     * 
     * Automatically uses HDF5 format if available and preferHDF5 is true.
     * Falls back to JSON if HDF5 unavailable.
     * 
     * @param {string} sessionId 
     * @param {Object} options
     * @param {Function} options.onProgress - Progress callback: (phase, loaded, total) => void
     *   - phase: 'check' | 'download' | 'parse' | 'convert'
     *   - For 'download': loaded/total in bytes
     *   - For others: loaded/total as percentage (0-100)
     * @returns {Promise<{session_id: string, events: Array, total_duration_ns: number}>}
     */
    async getTimeline(sessionId, options = {}) {
        const { onProgress } = options;
        
        if (this.#preferHDF5) {
            try {
                return await this.#getTimelineHDF5(sessionId, onProgress);
            } catch (err) {
                console.warn('[ProfilingAPI] HDF5 load failed, falling back to JSON:', err);
                // Fall through to JSON
            }
        }
        
        return await this.#getTimelineJSON(sessionId, onProgress);
    }
    
    /**
     * Load timeline from HDF5 endpoint.
     */
    async #getTimelineHDF5(sessionId, onProgress) {
        if (onProgress) onProgress('check', 0, 100);
        
        const url = `${this.#baseUrl}/api/profiling/timeline/${sessionId}.h5`;
        
        // Check if HDF5 available
        const headResponse = await fetch(url, { method: 'HEAD' });
        if (!headResponse.ok) {
            throw new Error('HDF5 timeline not available');
        }
        
        const contentLength = parseInt(headResponse.headers.get('Content-Length') || '0', 10);
        console.log(`[ProfilingAPI] HDF5 file size: ${(contentLength / 1024).toFixed(1)} KB`);
        
        // Load with progress
        const data = await loadProfilingHDF5(url, (loaded, total, phase) => {
            if (onProgress) {
                if (phase === 'download') {
                    onProgress('download', loaded, total);
                } else {
                    onProgress('parse', loaded, total);
                }
            }
        });
        
        if (onProgress) onProgress('convert', 50, 100);
        
        // Convert to legacy format for compatibility with existing code
        const events = data.toEvents();
        
        if (onProgress) onProgress('convert', 100, 100);
        
        console.log(`[ProfilingAPI] Loaded ${events.length} events from HDF5`);
        
        return {
            session_id: data.sessionId,
            events,
            total_duration_ns: data.totalDurationNs
        };
    }
    
    /**
     * Load timeline from JSON endpoint (legacy).
     */
    async #getTimelineJSON(sessionId, onProgress) {
        if (onProgress) onProgress('download', 0, 100);
        
        console.time('[ProfilingAPI] fetch JSON');
        const response = await fetch(`${this.#baseUrl}/api/profiling/timeline/${sessionId}`);
        console.timeEnd('[ProfilingAPI] fetch JSON');
        
        if (!response.ok) {
            throw new Error(`Failed to fetch timeline: ${response.statusText}`);
        }
        
        if (onProgress) onProgress('parse', 0, 100);
        
        console.time('[ProfilingAPI] json parse');
        const data = await response.json();
        console.timeEnd('[ProfilingAPI] json parse');
        
        if (onProgress) onProgress('parse', 100, 100);
        
        console.log(`[ProfilingAPI] Received ${data.events?.length || 0} events (JSON)`);
        return data;
    }
    
    /**
     * Get timeline data optimized for renderer (HDF5 only).
     * 
     * Returns typed arrays ready for InstancedMesh, skipping conversion to event objects.
     * Use this for best performance when you don't need the legacy event format.
     * 
     * @param {string} sessionId
     * @param {Object} options
     * @param {Function} options.onProgress - Progress callback
     * @returns {Promise<RendererData>}
     */
    async getTimelineForRenderer(sessionId, options = {}) {
        const { onProgress } = options;
        
        const url = `${this.#baseUrl}/api/profiling/timeline/${sessionId}.h5`;
        
        const data = await loadProfilingHDF5(url, (loaded, total, phase) => {
            if (onProgress) {
                onProgress(phase, loaded, total);
            }
        });
        
        // Return renderer-optimized format (typed arrays)
        return data.getRendererData();
    }

    /**
     * Get kernel metrics for a session.
     * @param {string} sessionId 
     * @returns {Promise<{session_id: string, kernels: Array}>}
     */
    async getKernels(sessionId) {
        const response = await fetch(`${this.#baseUrl}/api/profiling/kernels/${sessionId}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch kernels: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Delete a single profiling session.
     * @param {string} sessionId 
     * @returns {Promise<{deleted: boolean, session_id: string}>}
     */
    async deleteSession(sessionId) {
        const response = await fetch(`${this.#baseUrl}/api/profiling/session/${sessionId}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            throw new Error(`Failed to delete session: ${response.statusText}`);
        }
        return response.json();
    }

    /**
     * Delete multiple profiling sessions.
     * @param {string[]} sessionIds 
     * @returns {Promise<{deleted: string[], failed: string[]}>}
     */
    async deleteSessions(sessionIds) {
        const results = { deleted: [], failed: [] };
        
        for (const sessionId of sessionIds) {
            try {
                await this.deleteSession(sessionId);
                results.deleted.push(sessionId);
            } catch (error) {
                console.error(`Failed to delete session ${sessionId}:`, error);
                results.failed.push(sessionId);
            }
        }
        
        return results;
    }

    /**
     * Cleanup old sessions.
     * @param {number} maxAgeDays 
     * @returns {Promise<{deleted_count: number, deleted_sessions: string[]}>}
     */
    async cleanup(maxAgeDays = 30) {
        const response = await fetch(`${this.#baseUrl}/api/profiling/cleanup?max_age_days=${maxAgeDays}`, {
            method: 'POST'
        });
        if (!response.ok) {
            throw new Error(`Failed to cleanup: ${response.statusText}`);
        }
        return response.json();
    }
}

/**
 * @typedef {Object} RendererData
 * @property {string[]} categories - Categories present in data
 * @property {number} totalDurationMs - Total duration in milliseconds
 * @property {number} totalEvents - Total event count
 * @property {Object.<string, CategoryRendererData>} byCategory - Data per category
 */

/**
 * @typedef {Object} CategoryRendererData
 * @property {number} count - Number of events
 * @property {Float32Array} startMs - Start times in ms
 * @property {Float32Array} durationMs - Durations in ms
 * @property {Uint8Array} stream - Stream IDs
 * @property {Uint16Array} nameIdx - Name indices
 * @property {string[]} names - Name lookup table
 * @property {Object} metadata - Category-specific metadata arrays
 */
