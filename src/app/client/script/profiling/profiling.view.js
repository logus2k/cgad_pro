/**
 * ProfilingView - Main controller for the profiling panel.
 * 
 * Features:
 * - Loading progress indicator during timeline render
 * - Non-blocking session loading
 * - Debounced filter changes
 * - Real-time Socket.IO updates for profiling progress
 * - Integration with Gallery "Run Profiling" button
 * 
 * Location: /src/app/client/script/profiling/profiling.view.js
 */

import { ProfilingAPI } from './profiling.api.js';
import { TimelineController as ProfilingTimeline } from './timeline_controller.js';

export class ProfilingView {

    #container;
    #api;
    #timeline;
    #socket;
    #currentSessionId;
    #sessions;

    // UI element references
    #elements = {
        sessionList: null,
        sessionSelect: null,
        timelineContainer: null,
        summaryCards: null,
        statusIndicator: null,
        groupBySelect: null,
        categoryFilters: null,
        deleteSelectedBtn: null,
        loadingOverlay: null,
        progressBar: null,
        activeRunIndicator: null
    };

    // Pending profiling sessions (running in background)
    #pendingSessions = new Map();
    
    // Currently active profiling run (triggered from gallery)
    #activeProfilingSession = null;
    
    // Debounce timer for filter changes
    #filterDebounceTimer = null;

    constructor(containerId, socket) {
        this.#container = document.getElementById(containerId);
        if (!this.#container) {
            throw new Error(`Container element not found: ${containerId}`);
        }

        this.#socket = socket;
        this.#api = new ProfilingAPI();
        this.#sessions = [];
        this.#currentSessionId = null;

        this.#buildUI();
        this.#bindSocketEvents();
        this.#bindUIEvents();
        this.#bindGalleryEvents();

        // Initial load (non-blocking)
        this.loadSessions();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public Methods
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Load sessions from server.
     */
    async loadSessions() {
        try {
            const data = await this.#api.getSessions(100);
            this.#sessions = data.sessions || [];
            this.#renderSessionList();
        } catch (error) {
            console.error('[ProfilingView] Failed to load sessions:', error);
            this.#showStatus('Failed to load sessions', 'error');
        }
    }

    /**
     * Load and display a specific session's timeline.
     * @param {string} sessionId 
     */
    async loadSession(sessionId) {
        if (!sessionId) return;

        this.#currentSessionId = sessionId;
        this.#showLoading(true, 'Loading timeline data...');

        try {
            // Fetch timeline data
            const timelineData = await this.#api.getTimeline(sessionId);
            
            const eventCount = timelineData.events?.length || 0;
            this.#showLoading(true, `Rendering ${eventCount.toLocaleString()} events...`);

            // Load with progress callback
            if (this.#timeline) {
                await this.#timeline.load(timelineData, (progress) => {
                    this.#updateProgress(progress);
                });
            }

            this.#updateSummaryCards(timelineData);
            this.#showLoading(false);
            this.#showStatus('', 'idle');
            this.#elements.exportBtn.disabled = false;  // Enable export after successful load
            
        } catch (error) {
            console.error('[ProfilingView] Failed to load timeline:', error);
            this.#showLoading(false);
            this.#showStatus('Failed to load timeline', 'error');
            this.#elements.exportBtn.disabled = true;   // Keep disabled on error
        }
    }

    /**
     * Delete selected sessions.
     */
    async deleteSelected() {
        const checkboxes = this.#elements.sessionList.querySelectorAll('input[type="checkbox"]:checked');
        const sessionIds = Array.from(checkboxes).map(cb => cb.dataset.sessionId).filter(Boolean);

        if (sessionIds.length === 0) {
            return;
        }

        const confirmed = confirm(`Delete ${sessionIds.length} selected session(s)?`);
        if (!confirmed) return;

        this.#showStatus('Deleting sessions...', 'loading');

        try {
            const results = await this.#api.deleteSessions(sessionIds);
            
            this.#sessions = this.#sessions.filter(s => !results.deleted.includes(s.id));
            this.#renderSessionList();

            if (results.deleted.includes(this.#currentSessionId)) {
                this.#currentSessionId = null;
                if (this.#timeline) {
                    this.#timeline.clear();
                }
                this.#clearSummaryCards();
                this.#elements.exportBtn.disabled = true;
            }

            this.#showStatus(`Deleted ${results.deleted.length} session(s)`, 'success');
            
        } catch (error) {
            console.error('[ProfilingView] Failed to delete sessions:', error);
            this.#showStatus('Failed to delete sessions', 'error');
        }
    }
    
    /**
     * Start tracking a profiling session (called when gallery triggers profiling)
     * @param {string} sessionId 
     * @param {object} metadata - solver, mesh, etc.
     */
    startTracking(sessionId, metadata = {}) {
        console.log('[ProfilingView] Start tracking session:', sessionId);
        
        this.#activeProfilingSession = {
            sessionId,
            ...metadata,
            startTime: Date.now()
        };
        
        // Join Socket.IO room for this session
        if (this.#socket) {
            this.#socket.emit('join_profiling', { session_id: sessionId });
        }
        
        // Show active run indicator
        this.#showActiveRunIndicator(true, metadata);
        this.#showStatus('Profiling started...', 'loading');
    }

    // ─────────────────────────────────────────────────────────────────────────
    // UI Building
    // ─────────────────────────────────────────────────────────────────────────

    #buildUI() {
        this.#container.innerHTML = `
            <div class="profiling-layout">
                <!-- Loading Overlay -->
                <div class="profiling-loading-overlay" style="display: none;">
                    <div class="profiling-loading-content">
                        <div class="profiling-loading-spinner"></div>
                        <div class="profiling-loading-text">Loading...</div>
                        <div class="profiling-loading-progress">
                            <div class="profiling-loading-progress-bar"></div>
                        </div>
                    </div>
                </div>

                <!-- Active Run Indicator (shown when profiling is in progress) -->
                <div class="profiling-active-run" style="display: none;">
                    <div class="profiling-active-run-spinner"></div>
                    <div class="profiling-active-run-info">
                        <div class="profiling-active-run-title">Profiling in progress...</div>
                        <div class="profiling-active-run-details"></div>
                    </div>
                    <div class="profiling-active-run-stage"></div>
                </div>

                <!-- Toolbar -->
                <div class="profiling-toolbar">
                    <div class="profiling-toolbar-left">

                    </div>
                    <div class="profiling-toolbar-center">
                        <span class="profiling-status"></span>
                    </div>
                    <div class="profiling-toolbar-right">

                    </div>
                </div>

                <!-- Filters -->
                <div class="profiling-filters">
                    <div class="profiling-filter-group">
                        <label>Profiling Session:</label>
                        <select class="profiling-session-select" title="Select Session">
                            <option value="">-- Select Session --</option>
                        </select>
                        <button class="profiling-btn profiling-btn-refresh" title="Refresh Sessions">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
                            </svg>
                        </button>
                    </div>
                    <div class="profiling-filter-group">
                        <label>Group by:</label>
                        <select class="profiling-groupby-select">
                            <option value="category" selected>Category</option>
                            <option value="stream">Stream</option>
                        </select>
                    </div>
                    <div class="profiling-filter-group profiling-category-filters">
                        <label>Show:</label>
                        <label class="profiling-filter-checkbox">
                            <input type="checkbox" data-category="cuda_kernel" checked>
                            <span class="profiling-filter-color" style="background: rgba(231, 76, 60, 0.9);"></span>
                            Kernels
                        </label>
                        <label class="profiling-filter-checkbox">
                            <input type="checkbox" data-category="cuda_memcpy_h2d" checked>
                            <span class="profiling-filter-color" style="background: rgba(52, 152, 219, 0.9);"></span>
                            H2D
                        </label>
                        <label class="profiling-filter-checkbox">
                            <input type="checkbox" data-category="cuda_memcpy_d2h" checked>
                            <span class="profiling-filter-color" style="background: rgba(46, 204, 113, 0.9);"></span>
                            D2H
                        </label>
                        <label class="profiling-filter-checkbox">
                            <input type="checkbox" data-category="nvtx_range" checked>
                            <span class="profiling-filter-color" style="background: rgba(155, 89, 182, 0.9);"></span>
                            NVTX
                        </label>
                    </div>
                </div>

                <!-- Summary Cards -->
                <div class="profiling-summary">
                    <div class="profiling-card">
                        <div class="profiling-card-value" id="prof-total-duration">--</div>
                        <div class="profiling-card-label">Total Duration</div>
                    </div>
                    <div class="profiling-card">
                        <div class="profiling-card-value" id="prof-kernel-count">--</div>
                        <div class="profiling-card-label">Kernels</div>
                    </div>
                    <div class="profiling-card">
                        <div class="profiling-card-value" id="prof-memcpy-count">--</div>
                        <div class="profiling-card-label">MemCpy</div>
                    </div>
                    <div class="profiling-card">
                        <div class="profiling-card-value" id="prof-nvtx-count">--</div>
                        <div class="profiling-card-label">NVTX Ranges</div>
                    </div>
                </div>

                <!-- Timeline Container -->
                <div class="profiling-timeline-wrapper">
                    <div class="profiling-timeline" id="profiling-timeline-container"></div>
                </div>

                <!-- Session List (collapsible) -->
                <details class="profiling-sessions-details" open>
                    <summary>Session History</summary>
                    <div class="profiling-session-list"></div>
                </details>

                <!-- Footer Actions -->
                <div class="profiling-footer">
                    <div class="profiling-footer-left">
                        <button class="profiling-btn profiling-btn-delete" title="Delete Selected" disabled>
                            Delete Selected
                        </button>
                        <button class="profiling-btn profiling-btn-export" title="Export timeline data" disabled>
                            Export
                        </button>
                    </div>
                    <div class="profiling-footer-right">
                        <button class="profiling-btn profiling-btn-close">
                            Close
                        </button>
                    </div>
                </div>                

            </div>
        `;

        // Cache element references
        this.#elements.sessionSelect = this.#container.querySelector('.profiling-session-select');
        this.#elements.sessionList = this.#container.querySelector('.profiling-session-list');
        this.#elements.timelineContainer = this.#container.querySelector('#profiling-timeline-container');
        this.#elements.summaryCards = this.#container.querySelector('.profiling-summary');
        this.#elements.detailsPanel = this.#container.querySelector('.profiling-details-panel');
        this.#elements.statusIndicator = this.#container.querySelector('.profiling-status');
        this.#elements.groupBySelect = this.#container.querySelector('.profiling-groupby-select');
        this.#elements.categoryFilters = this.#container.querySelector('.profiling-category-filters');
        this.#elements.deleteSelectedBtn = this.#container.querySelector('.profiling-btn-delete');
        this.#elements.loadingOverlay = this.#container.querySelector('.profiling-loading-overlay');
        this.#elements.progressBar = this.#container.querySelector('.profiling-loading-progress-bar');
        this.#elements.activeRunIndicator = this.#container.querySelector('.profiling-active-run');
        this.#elements.exportBtn = this.#container.querySelector('.profiling-btn-export');
        this.#elements.closeBtn = this.#container.querySelector('.profiling-btn-close');

        // Initialize timeline
        this.#timeline = new ProfilingTimeline('profiling-timeline-container');
    }

    #bindUIEvents() {

        // Session select dropdown
        this.#elements.sessionSelect.addEventListener('change', (e) => {
            this.loadSession(e.target.value);
        });

        // Refresh button
        this.#container.querySelector('.profiling-btn-refresh').addEventListener('click', () => {
            this.loadSessions();
        });

        // Delete button
        this.#elements.deleteSelectedBtn.addEventListener('click', () => {
            this.deleteSelected();
        });

        // Export button
        this.#elements.exportBtn.addEventListener('click', () => {
            this.#handleExport();
        });

        // Close button
        this.#elements.closeBtn.addEventListener('click', () => {
            this.#handleClose();
        });

        // Group by select (debounced)
        this.#elements.groupBySelect.addEventListener('change', (e) => {
            this.#debounceFilterChange(() => {
                if (this.#timeline) {
                    this.#showLoading(true, 'Regrouping events...');
                    this.#timeline.setGroupBy(e.target.value).then(() => {
                        this.#showLoading(false);
                    });
                }
            });
        });

        // Category filter checkboxes (debounced)
        this.#elements.categoryFilters.querySelectorAll('input[type="checkbox"]').forEach(cb => {
            cb.addEventListener('change', (e) => {
                this.#debounceFilterChange(() => {
                    if (this.#timeline) {
                        this.#showLoading(true, 'Filtering events...');
                        this.#timeline.toggleCategory(e.target.dataset.category, e.target.checked).then(() => {
                            this.#showLoading(false);
                        });
                    }
                });
            });
        });

        // Session list checkbox changes
        this.#elements.sessionList.addEventListener('change', (e) => {
            if (e.target.type === 'checkbox') {
                this.#updateDeleteButton();
            }
        });

        // Session list row click
        this.#elements.sessionList.addEventListener('click', (e) => {
            const row = e.target.closest('.profiling-session-row');
            if (row && !e.target.matches('input[type="checkbox"]')) {
                const sessionId = row.dataset.sessionId;
                if (sessionId) {
                    this.#elements.sessionSelect.value = sessionId;
                    this.loadSession(sessionId);
                }
            }
        });
    }
    
    /**
     * Listen for events from Gallery "Run Profiling" button
     */
    #bindGalleryEvents() {
        // Listen for profilingStarted event from gallery
        document.addEventListener('profilingStarted', (e) => {
            const { sessionId, model, mesh, solver } = e.detail;
            console.log('[ProfilingView] Received profilingStarted event:', e.detail);
            
            this.startTracking(sessionId, {
                model: model?.name,
                mesh: mesh?.label,
                solver: solver
            });
        });
        
        // Listen for profilingError event from gallery
        document.addEventListener('profilingError', (e) => {
            const { error } = e.detail;
            console.error('[ProfilingView] Received profilingError event:', error);
            this.#showStatus(`Failed to start profiling: ${error}`, 'error');
            this.#showActiveRunIndicator(false);
        });
    }

    #debounceFilterChange(callback, delay = 150) {
        if (this.#filterDebounceTimer) {
            clearTimeout(this.#filterDebounceTimer);
        }
        this.#filterDebounceTimer = setTimeout(callback, delay);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Loading & Progress Indicators
    // ─────────────────────────────────────────────────────────────────────────

    #showLoading(show, message = 'Loading...') {
        if (!this.#elements.loadingOverlay) return;

        if (show) {
            this.#elements.loadingOverlay.style.display = 'flex';
            this.#elements.loadingOverlay.querySelector('.profiling-loading-text').textContent = message;
            this.#updateProgress(0);
        } else {
            this.#elements.loadingOverlay.style.display = 'none';
        }
    }

    #updateProgress(percent) {
        if (this.#elements.progressBar) {
            this.#elements.progressBar.style.width = `${percent}%`;
        }
    }
    
    /**
     * Show/hide the active profiling run indicator
     */
    #showActiveRunIndicator(show, metadata = {}) {
        if (!this.#elements.activeRunIndicator) return;
        
        if (show) {
            this.#elements.activeRunIndicator.style.display = 'flex';
            
            const details = this.#elements.activeRunIndicator.querySelector('.profiling-active-run-details');
            if (details) {
                const parts = [];
                if (metadata.solver) parts.push(`Solver: ${metadata.solver}`);
                if (metadata.mesh) parts.push(`Mesh: ${metadata.mesh}`);
                details.textContent = parts.join(' | ') || '';
            }
            
            this.#updateActiveRunStage('starting', 'Starting profiler...');
        } else {
            this.#elements.activeRunIndicator.style.display = 'none';
            this.#activeProfilingSession = null;
        }
    }
    
    /**
     * Update the stage display in the active run indicator
     */
    #updateActiveRunStage(stage, message) {
        const stageEl = this.#elements.activeRunIndicator?.querySelector('.profiling-active-run-stage');
        if (stageEl) {
            stageEl.textContent = message || stage;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Socket.IO Events
    // ─────────────────────────────────────────────────────────────────────────

    #bindSocketEvents() {
        if (!this.#socket) {
            console.warn('[ProfilingView] No socket provided, real-time updates disabled');
            return;
        }

        // Join general profiling room on connect
        this.#socket.on('connect', () => {
            this.#socket.emit('join_profiling', {});
        });
        
        // If already connected, join now
        if (this.#socket.connected) {
            this.#socket.emit('join_profiling', {});
        }

        // Session started
        this.#socket.on('profiling_started', (data) => {
            console.log('[ProfilingView] Profiling started:', data);
            this.#pendingSessions.set(data.session_id, { ...data, status: 'pending' });
            
            // If this is our active session, update UI
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#updateActiveRunStage('pending', 'Profiling session created...');
            }
        });
        
        // Legacy event names (for backward compatibility)
        this.#socket.on('profiling_queued', (data) => {
            console.log('[ProfilingView] Profiling queued:', data);
            this.#pendingSessions.set(data.session_id, { ...data, status: 'queued' });
            this.#showStatus(`Profiling queued: ${data.mesh}`, 'loading');
            
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#updateActiveRunStage('queued', 'Queued...');
            }
        });

        this.#socket.on('profiling_running', (data) => {
            console.log('[ProfilingView] Profiling running:', data);
            const pending = this.#pendingSessions.get(data.session_id);
            if (pending) {
                pending.status = 'running';
                pending.message = data.message;
            }
            this.#showStatus(data.message || 'Profiling in progress...', 'loading');
            
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#updateActiveRunStage('running', data.message || 'Running solver...');
            }
        });

        this.#socket.on('profiling_extracting', (data) => {
            console.log('[ProfilingView] Profiling extracting:', data);
            const pending = this.#pendingSessions.get(data.session_id);
            if (pending) {
                pending.status = 'extracting';
                pending.message = data.message;
            }
            this.#showStatus(data.message || 'Parsing profiling data...', 'loading');
            
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#updateActiveRunStage('extracting', data.message || 'Extracting timeline...');
            }
        });

        // Progress updates
        this.#socket.on('profiling_progress', (data) => {
            console.log('[ProfilingView] Profiling progress:', data);
            
            const pending = this.#pendingSessions.get(data.session_id);
            if (pending) {
                pending.status = data.status;
                pending.stage = data.stage;
                pending.message = data.message;
            }
            
            // Update status bar
            this.#showStatus(data.message || `Profiling: ${data.stage}`, 'loading');
            
            // If this is our active session, update the indicator
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#updateActiveRunStage(data.stage, data.message);
            }
        });

        // Profiling complete
        this.#socket.on('profiling_complete', (data) => {
            console.log('[ProfilingView] Profiling complete:', data);
            this.#pendingSessions.delete(data.session_id);
            
            // Refresh session list
            this.loadSessions();
            
            // If this was our active session, auto-load it
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#showActiveRunIndicator(false);
                this.loadSession(data.session_id);
                this.#showStatus('Profiling complete', 'success');
            } else if (!this.#currentSessionId) {
                // If no session is loaded, load the completed one
                this.loadSession(data.session_id);
                this.#showStatus('Profiling complete', 'success');
            } else {
                this.#showStatus('Profiling complete', 'success');
            }
        });

        // Profiling failed/error
        this.#socket.on('profiling_failed', (data) => {
            console.error('[ProfilingView] Profiling failed:', data);
            this.#pendingSessions.delete(data.session_id);
            
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#showActiveRunIndicator(false);
            }
            
            this.#showStatus(`Profiling failed: ${data.error || 'Unknown error'}`, 'error');
        });
        
        this.#socket.on('profiling_error', (data) => {
            console.error('[ProfilingView] Profiling error:', data);
            this.#pendingSessions.delete(data.session_id);
            
            if (this.#activeProfilingSession?.sessionId === data.session_id) {
                this.#showActiveRunIndicator(false);
            }
            
            this.#showStatus(`Profiling failed: ${data.error || 'Unknown error'}`, 'error');
        });

        // Session deleted
        this.#socket.on('profiling_deleted', (data) => {
            console.log('[ProfilingView] Sessions deleted:', data);
            if (data.session_ids) {
                this.#sessions = this.#sessions.filter(s => !data.session_ids.includes(s.id));
                this.#renderSessionList();
            }
        });
        
        // Joined profiling room confirmation
        this.#socket.on('joined_profiling', (data) => {
            console.log('[ProfilingView] Joined profiling room:', data);
        });
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Rendering
    // ─────────────────────────────────────────────────────────────────────────

    #renderSessionList() {
        const currentValue = this.#elements.sessionSelect.value;
        this.#elements.sessionSelect.innerHTML = '<option value="">-- Select Session --</option>';
        
        for (const session of this.#sessions) {
            const option = document.createElement('option');
            option.value = session.id;
            option.textContent = `${session.mesh} (${session.solver}) - ${this.#formatDate(session.created_at)}`;
            this.#elements.sessionSelect.appendChild(option);
        }
        
        if (currentValue && this.#sessions.some(s => s.id === currentValue)) {
            this.#elements.sessionSelect.value = currentValue;
        }

        if (this.#sessions.length === 0) {
            this.#elements.sessionList.innerHTML = `
                <div class="profiling-session-empty">No profiling sessions available</div>
            `;
            return;
        }

        let html = `
            <div class="profiling-session-header">
                <div class="profiling-session-col-checkbox">
                    <input type="checkbox" class="profiling-select-all" title="Select All">
                </div>
                <div class="profiling-session-col-date">Date</div>
                <div class="profiling-session-col-solver">Solver</div>
                <div class="profiling-session-col-mesh">Mesh</div>
                <div class="profiling-session-col-status">Status</div>
                <div class="profiling-session-col-duration">Duration</div>
            </div>
        `;

        for (const session of this.#sessions) {
            const statusClass = session.status === 'completed' ? 'success' : 
                               session.status === 'failed' ? 'error' : 'pending';
            
            const duration = session.timeline_summary?.total_duration_ms 
                ? this.#formatDuration(session.timeline_summary.total_duration_ms)
                : '--';

            html += `
                <div class="profiling-session-row ${this.#currentSessionId === session.id ? 'selected' : ''}" 
                     data-session-id="${session.id}">
                    <div class="profiling-session-col-checkbox">
                        <input type="checkbox" data-session-id="${session.id}">
                    </div>
                    <div class="profiling-session-col-date">${this.#formatDate(session.created_at)}</div>
                    <div class="profiling-session-col-solver">${session.solver}</div>
                    <div class="profiling-session-col-mesh">${session.mesh}</div>
                    <div class="profiling-session-col-status">
                        <span class="profiling-status-badge ${statusClass}">${session.status}</span>
                    </div>
                    <div class="profiling-session-col-duration">${duration}</div>
                </div>
            `;
        }

        this.#elements.sessionList.innerHTML = html;

        const selectAll = this.#elements.sessionList.querySelector('.profiling-select-all');
        if (selectAll) {
            selectAll.addEventListener('change', (e) => {
                const checkboxes = this.#elements.sessionList.querySelectorAll('.profiling-session-row input[type="checkbox"]');
                checkboxes.forEach(cb => cb.checked = e.target.checked);
                this.#updateDeleteButton();
            });
        }
    }

    #updateSummaryCards(timelineData) {
        if (!timelineData) return;

        const summary = this.#timeline.getSummary();

        document.getElementById('prof-total-duration').textContent = 
            this.#formatDuration(summary.totalDurationMs);
        
        document.getElementById('prof-kernel-count').textContent = 
            summary.categories.cuda_kernel?.count || 0;
        
        const memcpyCount = (summary.categories.cuda_memcpy_h2d?.count || 0) +
                           (summary.categories.cuda_memcpy_d2h?.count || 0) +
                           (summary.categories.cuda_memcpy_d2d?.count || 0);
        document.getElementById('prof-memcpy-count').textContent = memcpyCount;
        
        document.getElementById('prof-nvtx-count').textContent = 
            summary.categories.nvtx_range?.count || 0;
    }

    #clearSummaryCards() {
        document.getElementById('prof-total-duration').textContent = '--';
        document.getElementById('prof-kernel-count').textContent = '--';
        document.getElementById('prof-memcpy-count').textContent = '--';
        document.getElementById('prof-nvtx-count').textContent = '--';
    }

    #showStatus(message, type = 'info') {
        const el = this.#elements.statusIndicator;
        el.textContent = message;
        el.className = `profiling-status profiling-status-${type}`;
        
        if (type === 'success' || type === 'error') {
            setTimeout(() => {
                if (el.textContent === message) {
                    el.textContent = '';
                    el.className = 'profiling-status';
                }
            }, 3000);
        }
    }

    #updateDeleteButton() {
        const checkboxes = this.#elements.sessionList.querySelectorAll('.profiling-session-row input[type="checkbox"]:checked');
        this.#elements.deleteSelectedBtn.disabled = checkboxes.length === 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utilities
    // ─────────────────────────────────────────────────────────────────────────

    #formatDate(isoString) {
        if (!isoString) return '--';
        const date = new Date(isoString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    #formatDuration(ms) {
        if (ms == null || isNaN(ms)) return '--';
        if (ms < 1) return `${(ms * 1000).toFixed(1)} us`;
        if (ms < 1000) return `${ms.toFixed(2)} ms`;
        return `${(ms / 1000).toFixed(2)} s`;
    }

    #formatBytes(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
        return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
    }

    #handleExport() {
        if (!this.#currentSessionId) {
            this.#showStatus('No session loaded', 'error');
            return;
        }
        
        // Trigger download of the Nsight report file
        const url = `${this.#api.getBaseUrl()}/api/profiling/report/${this.#currentSessionId}`;
        const a = document.createElement('a');
        a.href = url;
        a.download = '';  // Let server set filename
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        this.#showStatus('Download started', 'success');
    }

    #handleClose() {
        // Dispatch event for parent to handle (e.g., hide HUD panel)
        this.#container.dispatchEvent(new CustomEvent('profiling-close', { bubbles: true }));
    }


    // ─────────────────────────────────────────────────────────────────────────
    // Cleanup
    // ─────────────────────────────────────────────────────────────────────────
    destroy() {
        if (this.#filterDebounceTimer) {
            clearTimeout(this.#filterDebounceTimer);
        }
        if (this.#timeline) {
            this.#timeline.destroy();
            this.#timeline = null;
        }
        
        if (this.#socket) {
            this.#socket.emit('leave_profiling', {});
            this.#socket.off('profiling_started');
            this.#socket.off('profiling_queued');
            this.#socket.off('profiling_running');
            this.#socket.off('profiling_extracting');
            this.#socket.off('profiling_progress');
            this.#socket.off('profiling_complete');
            this.#socket.off('profiling_failed');
            this.#socket.off('profiling_error');
            this.#socket.off('profiling_deleted');
            this.#socket.off('joined_profiling');
        }
    }
}
