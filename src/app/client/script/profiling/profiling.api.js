/**
 * ProfilingAPI - REST client for profiling service endpoints.
 * 
 * Handles session list retrieval and deletion.
 * Timeline data comes via Socket.IO (see profiling.view.js).
 * 
 * Location: /src/app/client/script/profiling/profiling.api.js
 */

export class ProfilingAPI {

    #baseUrl;

    constructor(baseUrl = '') {
        this.#baseUrl = baseUrl;
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
     * Note: For real-time updates, use Socket.IO events instead.
     * @param {string} sessionId 
     * @returns {Promise<{session_id: string, events: Array, total_duration_ns: number}>}
     */
    async getTimeline(sessionId) {
        const response = await fetch(`${this.#baseUrl}/api/profiling/timeline/${sessionId}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch timeline: ${response.statusText}`);
        }
        return response.json();
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
