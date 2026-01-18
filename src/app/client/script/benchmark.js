/**
 * Benchmark Panel UI Component
 * 
 * Displays benchmark results in a sortable table with filtering and deletion capabilities.
 * Records are grouped by server configuration (collapsible groups).
 * Fetches data from /api/benchmark endpoint.
 * 
 * Location: /src/app/client/script/benchmark.js
 */

export class BenchmarkPanel {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId) || document.querySelector(containerId);
        if (!this.container) {
            console.error(`[Benchmark] Container not found: ${containerId}`);
            return;
        }
        
        this.options = {
            apiBase: options.apiBase || '',
            pollInterval: options.pollInterval || 5000, // 5 seconds
            autoRefresh: options.autoRefresh !== false,
            ...options
        };
        
        this.records = [];
        this.selectedRecords = new Set();
        this.sortColumn = 'timestamp';
        this.sortOrder = 'desc';
        this.filterSolver = '';
        this.filterModel = '';
        this.filterServer = '';
        this.filterTesting = '';  // '', 'manual', 'automated'
        this.serverConfig = null;
        this.serverHash = null;
        
        // Track expanded/collapsed state per server_hash
        this.expandedGroups = new Set();
        this.firstLoad = true;
        
        this.pollTimer = null;
        
        this.init();
    }
    
    async init() {
        this.render();
        await this.fetchData();
        
        if (this.options.autoRefresh) {
            this.startPolling();
        }
    }
    
    render() {
        this.container.innerHTML = `
            <div class="benchmark-controls">
                <div class="benchmark-controls-left">
                    <div class="benchmark-filter-group">
                        <label class="benchmark-filter-label">Solvers</label>
                        <select class="benchmark-filter" id="benchmark-filter-solver">
                            <option value="">All Solvers</option>
                        </select>
                    </div>
                    <div class="benchmark-filter-group">
                        <label class="benchmark-filter-label">Meshes</label>
                        <select class="benchmark-filter" id="benchmark-filter-model">
                            <option value="">All Meshes</option>
                        </select>
                    </div>
                    <div class="benchmark-filter-group">
                        <label class="benchmark-filter-label">Servers</label>
                        <select class="benchmark-filter" id="benchmark-filter-server">
                            <option value="">All Servers</option>
                        </select>
                    </div>
                    <div class="benchmark-filter-group">
                        <label class="benchmark-filter-label">Testing</label>
                        <select class="benchmark-filter" id="benchmark-filter-testing">
                            <option value="">All</option>
                            <option value="manual">Manual</option>
                            <option value="automated">Automated</option>
                        </select>
                    </div>
                </div>
                <div class="benchmark-controls-right">
                    <div class="benchmark-filter-group">
                        <label class="benchmark-filter-label">Reports</label>
                        <div class="benchmark-report-controls">
                            <select class="benchmark-filter" id="benchmark-report-section">
                            </select>
                            <button class="benchmark-btn benchmark-btn-secondary" id="benchmark-view-report-btn">View</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="benchmark-table-container" id="benchmark-table-container">
                <div class="benchmark-loading">
                    <div class="benchmark-loading-spinner"></div>
                    <div>Loading benchmark data...</div>
                </div>
            </div>
            
            <div class="benchmark-footer">
                <div class="benchmark-footer-left">
                    <button class="benchmark-btn benchmark-btn-danger" id="benchmark-delete-btn" disabled>
                        Delete Selected
                    </button>
                    <button class="benchmark-btn benchmark-btn-secondary" id="benchmark-refresh-btn">
                        Refresh
                    </button>
                </div>
                <div class="benchmark-footer-right">
                    <button class="benchmark-btn benchmark-btn-secondary" id="benchmark-copy-table-btn">
                        Copy
                    </button>
                    <button class="benchmark-btn benchmark-btn-secondary" id="benchmark-export-table-btn">
                        Export
                    </button>
                    <button class="benchmark-btn benchmark-btn-close" id="benchmark-close-btn">Close</button>
                </div>
            </div>
        `;
        
        this.bindEvents();
    }
    
    bindEvents() {
        // Refresh button
        const refreshBtn = this.container.querySelector('#benchmark-refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.fetchData());
        }
        
        // Delete button
        const deleteBtn = this.container.querySelector('#benchmark-delete-btn');
        if (deleteBtn) {
            deleteBtn.addEventListener('click', () => this.deleteSelected());
        }
        
        // Close button
        const closeBtn = this.container.querySelector('#benchmark-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closePanel());
        }

        // Copy table button
        const copyTableBtn = this.container.querySelector('#benchmark-copy-table-btn');
        if (copyTableBtn) {
            copyTableBtn.addEventListener('click', () => this.copyTableAsMarkdown());
        }

        // Export table button
        const exportTableBtn = this.container.querySelector('#benchmark-export-table-btn');
        if (exportTableBtn) {
            exportTableBtn.addEventListener('click', () => this.exportTableAsMarkdown());
        }        

        // Report section dropdown
        const reportSection = this.container.querySelector('#benchmark-report-section');
        const viewReportBtn = this.container.querySelector('#benchmark-view-report-btn');

        if (reportSection) {
            // Load available sections
            this.loadReportSections();
            
            reportSection.addEventListener('change', (e) => {
                if (viewReportBtn) {
                    viewReportBtn.disabled = !e.target.value;
                }
            });
        }

        if (viewReportBtn) {
            viewReportBtn.addEventListener('click', () => {
                const sectionId = reportSection?.value;
                if (sectionId) {
                    this.openReportViewer(sectionId);
                }
            });
        }        
        
        // Solver filter
        const solverFilter = this.container.querySelector('#benchmark-filter-solver');
        if (solverFilter) {
            solverFilter.addEventListener('change', (e) => {
                this.filterSolver = e.target.value;
                this.renderTable();
            });
        }
        
        // Model filter
        const modelFilter = this.container.querySelector('#benchmark-filter-model');
        if (modelFilter) {
            modelFilter.addEventListener('change', (e) => {
                this.filterModel = e.target.value;
                this.renderTable();
            });
        }
        
        // Server filter
        const serverFilter = this.container.querySelector('#benchmark-filter-server');
        if (serverFilter) {
            serverFilter.addEventListener('change', (e) => {
                this.filterServer = e.target.value;
                this.renderTable();
            });
        }
        
        // Testing filter (manual/automated)
        const testingFilter = this.container.querySelector('#benchmark-filter-testing');
        if (testingFilter) {
            testingFilter.addEventListener('change', (e) => {
                this.filterTesting = e.target.value;
                // Reset server filter when testing filter changes
                this.filterServer = '';
                const serverSelect = this.container.querySelector('#benchmark-filter-server');
                if (serverSelect) serverSelect.value = '';
                this.updateFilters({ solver_types: [], models: [] });
                this.renderTable();
            });
        }
    }
    
    closePanel() {
        if (window.menuManager) {
            window.menuManager.hidePanel('benchmark');
        }
    }
    
    async fetchData() {
        try {
            // First, trigger backend to reload files from disk
            await fetch(`${this.options.apiBase}/api/benchmark/refresh`, { method: 'POST' });
            
            // Then fetch the updated records
            const response = await fetch(`${this.options.apiBase}/api/benchmark?sort_by=${this.sortColumn}&sort_order=${this.sortOrder}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const data = await response.json();
            this.records = data.records || [];
            
            // Fetch summary
            const summaryResponse = await fetch(`${this.options.apiBase}/api/benchmark/summary`);
            if (summaryResponse.ok) {
                const summary = await summaryResponse.json();
                this.serverConfig = summary.server_config;
                this.serverHash = summary.server_hash;
                this.updateFilters(summary);
            }
            
            this.renderTable();
            
        } catch (error) {
            console.error('[Benchmark] Failed to fetch data:', error);
            this.renderError(error.message);
        }
    }
    
    updateFilters(summary) {
        const solverFilter = this.container.querySelector('#benchmark-filter-solver');
        const modelFilter = this.container.querySelector('#benchmark-filter-model');
        
        if (solverFilter && summary.solver_types) {
            const currentValue = solverFilter.value;
            solverFilter.innerHTML = '<option value="">All Solvers</option>';
            summary.solver_types.sort().forEach(solver => {
                const option = document.createElement('option');
                option.value = solver;
                option.textContent = this.formatSolverName(solver);
                solverFilter.appendChild(option);
            });
            solverFilter.value = currentValue;
        }
        
        if (modelFilter && summary.models) {
            const currentValue = modelFilter.value;
            modelFilter.innerHTML = '<option value="">All Meshes</option>';
            summary.models.sort().forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                modelFilter.appendChild(option);
            });
            modelFilter.value = currentValue;
        }
        
        // Populate server filter from records
        // Distinguishes between manual and automated records from the same server
        // Respects the Testing filter to show only relevant servers
        const serverFilter = this.container.querySelector('#benchmark-filter-server');
        if (serverFilter && this.records.length > 0) {
            const currentValue = serverFilter.value;
            serverFilter.innerHTML = '<option value="">All Servers</option>';
            
            // Extract unique servers from records
            // Use composite key: server_hash + automated flag
            const servers = new Map();
            this.records.forEach(record => {
                const hash = record.server_hash;
                const isAutomated = record.client_hash === 'automated';
                
                // Apply Testing filter to server list
                if (this.filterTesting === 'manual' && isAutomated) return;
                if (this.filterTesting === 'automated' && !isAutomated) return;
                
                const compositeKey = `${hash}${isAutomated ? '_automated' : ''}`;
                
                if (hash && !servers.has(compositeKey)) {
                    const config = record.server_config || {};
                    const hostname = (config.hostname || 'Unknown').toUpperCase();
                    const cpu = this.shortenCpuName(config.cpu_model);
                    const gpu = this.shortenGpuName(config.gpu_model);
                    
                    // Add "(Automated)" suffix for automated records
                    const displayName = isAutomated ? `${hostname} (Automated)` : hostname;
                    const label = `${displayName} - ${cpu}, ${gpu}`;
                    
                    servers.set(compositeKey, {
                        label: label,
                        hash: hash,
                        isAutomated: isAutomated,
                        displayName: displayName
                    });
                }
            });
            
            // Sort by label and add options
            [...servers.entries()]
                .sort((a, b) => a[1].displayName.localeCompare(b[1].displayName))
                .forEach(([compositeKey, info]) => {
                    const option = document.createElement('option');
                    option.value = compositeKey;
                    option.textContent = info.label;
                    serverFilter.appendChild(option);
                });
            
            serverFilter.value = currentValue;
        }
    }
    
    /**
     * Group records by server_hash and automated flag
     * @param {Array} records - Array of benchmark records
     * @returns {Array} Array of {hash, compositeKey, isAutomated, config, records, filteredRecords, mostRecent}
     */
    groupRecordsByServer(records) {
        const groups = new Map();
        
        records.forEach(record => {
            const hash = record.server_hash || 'unknown';
            const isAutomated = record.client_hash === 'automated';
            const compositeKey = `${hash}${isAutomated ? '_automated' : ''}`;
            
            if (!groups.has(compositeKey)) {
                groups.set(compositeKey, {
                    hash: hash,
                    compositeKey: compositeKey,
                    isAutomated: isAutomated,
                    config: record.server_config || {},
                    records: [],
                    filteredRecords: [],
                    mostRecent: null
                });
            }
            
            const group = groups.get(compositeKey);
            group.records.push(record);
            
            // Track most recent timestamp for sorting groups
            const timestamp = new Date(record.timestamp);
            if (!group.mostRecent || timestamp > group.mostRecent) {
                group.mostRecent = timestamp;
            }
        });
        
        // Convert to array and sort by most recent record (descending)
        return Array.from(groups.values()).sort((a, b) => b.mostRecent - a.mostRecent);
    }
    
    /**
     * Apply filters to records within each group
     */
    applyFiltersToGroups(groups) {
        groups.forEach(group => {
            let filtered = group.records;
            
            if (this.filterSolver) {
                filtered = filtered.filter(r => r.solver_type === this.filterSolver);
            }
            if (this.filterModel) {
                filtered = filtered.filter(r => r.model_name === this.filterModel);
            }
            
            // Sort filtered records by current sort column
            filtered = this.sortRecords(filtered);
            
            group.filteredRecords = filtered;
        });
        
        return groups;
    }
    
    /**
     * Sort records by current sort column/order
     */
    sortRecords(records) {
        const sorted = [...records];
        const column = this.sortColumn;
        const order = this.sortOrder;
        
        sorted.sort((a, b) => {
            let aVal, bVal;
            
            if (column === 'total_time') {
                aVal = a.timings?.total_program_time || 0;
                bVal = b.timings?.total_program_time || 0;
            } else if (column === 'peak_ram') {
                aVal = a.memory?.peak_ram_mb || 0;
                bVal = b.memory?.peak_ram_mb || 0;
            } else if (column === 'peak_vram') {
                aVal = a.memory?.peak_vram_mb || 0;
                bVal = b.memory?.peak_vram_mb || 0;
            } else if (column.startsWith('timings.')) {
                const key = column.split('.')[1];
                aVal = a.timings?.[key] || 0;
                bVal = b.timings?.[key] || 0;
            } else {
                aVal = a[column] || '';
                bVal = b[column] || '';
            }
            
            if (typeof aVal === 'string') {
                aVal = aVal.toLowerCase();
                bVal = bVal.toLowerCase();
            }
            
            if (aVal < bVal) return order === 'asc' ? -1 : 1;
            if (aVal > bVal) return order === 'asc' ? 1 : -1;
            return 0;
        });
        
        return sorted;
    }
    
    renderTable() {
        const container = this.container.querySelector('#benchmark-table-container');
        if (!container) return;
        
        if (this.records.length === 0) {
            container.innerHTML = `
                <div class="benchmark-empty">
                    <div class="benchmark-empty-icon">📊</div>
                    <div class="benchmark-empty-text">
                        No benchmark records yet. Run a solver to record results.
                    </div>
                </div>
            `;
            return;
        }
        
        // Apply Testing filter first (manual/automated)
        let filteredRecords = this.records;
        if (this.filterTesting === 'manual') {
            filteredRecords = this.records.filter(r => r.client_hash !== 'automated');
        } else if (this.filterTesting === 'automated') {
            filteredRecords = this.records.filter(r => r.client_hash === 'automated');
        }
        
        if (filteredRecords.length === 0) {
            container.innerHTML = `
                <div class="benchmark-empty">
                    <div class="benchmark-empty-icon">📊</div>
                    <div class="benchmark-empty-text">
                        No ${this.filterTesting || ''} benchmark records found.
                    </div>
                </div>
            `;
            return;
        }
        
        // Group records by server
        let groups = this.groupRecordsByServer(filteredRecords);
        
        // Filter groups by server if selected (using composite key)
        if (this.filterServer) {
            groups = groups.filter(g => g.compositeKey === this.filterServer);
        }
        
        groups = this.applyFiltersToGroups(groups);
        
        // On first load, expand only the first group
        if (this.firstLoad && groups.length > 0) {
            this.expandedGroups.add(groups[0].compositeKey);
            this.firstLoad = false;
        }
        
        // Build table HTML
        let tableHtml = `
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th data-column="select" style="width: 30px;"></th>
                        <th data-column="model_name" class="${this.getSortClass('model_name')}">Model</th>
                        <th data-column="solver_type" class="${this.getSortClass('solver_type')}">Solver</th>
                        <th data-column="model_nodes" class="${this.getSortClass('model_nodes')}">Nodes</th>
                        <th data-column="model_elements" class="${this.getSortClass('model_elements')}">Elements</th>
                        <th data-column="total_time" class="${this.getSortClass('total_time')}">Total Time</th>
                        <th data-column="timings.assemble_system" class="${this.getSortClass('timings.assemble_system')}">Assembly</th>
                        <th data-column="timings.solve_system" class="${this.getSortClass('timings.solve_system')}">Solve</th>
                        <th data-column="iterations" class="${this.getSortClass('iterations')}">Iterations</th>
                        <th data-column="peak_ram" class="${this.getSortClass('peak_ram')}">Peak RAM</th>
                        <th data-column="peak_vram" class="${this.getSortClass('peak_vram')}">Peak VRAM</th>
                        <th data-column="converged" class="${this.getSortClass('converged')}">Status</th>
                        <th data-column="timestamp" class="${this.getSortClass('timestamp')}">Date</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Render each group
        groups.forEach(group => {
            const isExpanded = this.expandedGroups.has(group.compositeKey);
            const totalCount = group.records.length;
            const filteredCount = group.filteredRecords.length;
            
            // Render group header
            tableHtml += this.renderGroupHeader(group, isExpanded, totalCount, filteredCount);
            
            // Render records if expanded
            if (isExpanded) {
                group.filteredRecords.forEach(record => {
                    tableHtml += this.renderRow(record, group.compositeKey);
                });
            }
        });
        
        tableHtml += `
                </tbody>
            </table>
        `;
        
        container.innerHTML = tableHtml;
        
        // Bind table events
        this.bindTableEvents();
    }
    
    renderGroupHeader(group, isExpanded, totalCount, filteredCount) {
        const config = group.config;
        const toggleIcon = isExpanded ? '▼' : '►';
        
        const cpuShort = this.shortenCpuName(config.cpu_model || 'Unknown CPU');
        const gpuShort = this.shortenGpuName(config.gpu_model);
        const ram = config.ram_gb ? `${config.ram_gb} GB` : '-';
        const vram = config.gpu_memory_gb ? `${config.gpu_memory_gb} GB` : '-';
        
        // Add "(Automated)" suffix for automated records
        let hostname = (config.hostname || 'Unknown').toUpperCase();
        if (group.isAutomated) {
            hostname += ' (Automated)';
        }
        
        // Build count display
        let countDisplay;
        if (filteredCount === totalCount) {
            countDisplay = `${totalCount} record${totalCount !== 1 ? 's' : ''}`;
        } else {
            countDisplay = `${filteredCount} of ${totalCount} records`;
        }
        
        // Group checkbox - only show if this server owns the records
        const canDelete = group.hash === this.serverHash;
        const recordIds = group.filteredRecords.map(r => r.id);
        const allSelected = recordIds.length > 0 && recordIds.every(id => this.selectedRecords.has(id));
        const someSelected = recordIds.some(id => this.selectedRecords.has(id));
        
        const checkboxHtml = canDelete 
            ? `<input type="checkbox" 
                   class="benchmark-checkbox benchmark-group-checkbox" 
                   data-group-key="${group.compositeKey}"
                   data-record-ids="${recordIds.join(',')}"
                   ${allSelected ? 'checked' : ''}
                   ${someSelected && !allSelected ? 'indeterminate' : ''}
                   title="Select all ${filteredCount} records in this group">`
            : '';
        
        return `
            <tr class="benchmark-group-header ${isExpanded ? 'expanded' : 'collapsed'}" 
                data-group-hash="${group.compositeKey}">
                <td colspan="13">
                    <div class="group-header-content">
                        <span class="group-checkbox-container">${checkboxHtml}</span>
                        <span class="group-toggle">${toggleIcon}</span>
                        <span class="group-hostname">${hostname}</span>
                        <span class="group-cpu" title="${config.cpu_model || ''}">${cpuShort}</span>
                        <span class="group-separator">|</span>
                        <span class="group-ram">${ram} RAM</span>
                        <span class="group-separator">|</span>
                        <span class="group-gpu" title="${config.gpu_model || ''}">${gpuShort}</span>
                        <span class="group-separator">|</span>
                        <span class="group-vram">${vram} VRAM</span>
                        <span class="group-count">${countDisplay}</span>
                    </div>
                </td>
            </tr>
        `;
    }
    
    renderRow(record, groupHash) {
        const totalTime = record.timings?.total_program_time || 0;
        const timeClass = this.getTimeClass(totalTime);
        const isSelected = this.selectedRecords.has(record.id);
        
        // Extract memory data (with fallback for old records)
        const peakRam = record.memory?.peak_ram_mb;
        const peakVram = record.memory?.peak_vram_mb;
        
        // Only show checkbox for records from this server
        const canDelete = record.server_hash === this.serverHash;
        
        return `
            <tr class="benchmark-record ${isSelected ? 'selected' : ''}" 
                data-id="${record.id}" 
                data-group-hash="${groupHash}">
                <td>
                    ${canDelete 
                        ? `<input type="checkbox" class="benchmark-checkbox" 
                               data-id="${record.id}" 
                               ${isSelected ? 'checked' : ''}>`
                        : ''}
                </td>
                <td class="benchmark-cell-model" title="${record.model_name}">${record.model_name}</td>
                <td class="benchmark-cell-solver">
                    <span class="solver-badge ${this.getSolverBadgeClass(record.solver_type)}">
                        ${this.formatSolverName(record.solver_type)}
                    </span>
                </td>
                <td>${this.formatNumber(record.model_nodes)}</td>
                <td>${this.formatNumber(record.model_elements)}</td>
                <td class="benchmark-cell-time ${timeClass}">${this.formatTime(totalTime)}</td>
                <td class="benchmark-cell-time">${this.formatTime(record.timings?.assemble_system)}</td>
                <td class="benchmark-cell-time">${this.formatTime(record.timings?.solve_system)}</td>
                <td>${this.formatNumber(record.iterations)}</td>
                <td class="benchmark-cell-memory">${this.formatMemory(peakRam)}</td>
                <td class="benchmark-cell-memory ${this.getVramClass(peakVram)}">${this.formatMemory(peakVram)}</td>
                <td class="benchmark-cell-status">
                    <span class="${record.converged ? 'converged' : 'failed'}">
                        ${record.converged ? '✓' : '✗'}
                    </span>
                </td>
                <td class="benchmark-cell-timestamp">${this.formatTimestamp(record.timestamp)}</td>
            </tr>
        `;
    }
    
    bindTableEvents() {
        const table = this.container.querySelector('.benchmark-table');
        if (!table) return;
        
        // Header click for sorting
        table.querySelectorAll('th[data-column]').forEach(th => {
            if (th.dataset.column === 'select') return;
            
            th.addEventListener('click', () => {
                const column = th.dataset.column;
                if (this.sortColumn === column) {
                    this.sortOrder = this.sortOrder === 'asc' ? 'desc' : 'asc';
                } else {
                    this.sortColumn = column;
                    this.sortOrder = column === 'timestamp' ? 'desc' : 'asc';
                }
                this.renderTable();
            });
        });
        
        // Group header click for expand/collapse
        table.querySelectorAll('.benchmark-group-header').forEach(row => {
            row.addEventListener('click', (e) => {
                // Don't toggle if clicking on a checkbox or button
                if (e.target.closest('input, button')) return;
                
                const hash = row.dataset.groupHash;
                this.toggleGroup(hash);
            });
        });
        
        // Group checkbox click - select/deselect all records in group
        table.querySelectorAll('.benchmark-group-checkbox').forEach(checkbox => {
            // Set indeterminate state (can't be set via HTML attribute)
            const recordIds = checkbox.dataset.recordIds ? checkbox.dataset.recordIds.split(',') : [];
            const allSelected = recordIds.length > 0 && recordIds.every(id => this.selectedRecords.has(id));
            const someSelected = recordIds.some(id => this.selectedRecords.has(id));
            checkbox.indeterminate = someSelected && !allSelected;
            
            checkbox.addEventListener('change', (e) => {
                e.stopPropagation(); // Prevent group expand/collapse
                
                const recordIds = e.target.dataset.recordIds ? e.target.dataset.recordIds.split(',') : [];
                
                if (e.target.checked) {
                    // Select all records in group
                    recordIds.forEach(id => this.selectedRecords.add(id));
                } else {
                    // Deselect all records in group
                    recordIds.forEach(id => this.selectedRecords.delete(id));
                }
                
                this.updateDeleteButton();
                
                // Update individual row checkboxes and highlights
                recordIds.forEach(id => {
                    const rowCheckbox = table.querySelector(`.benchmark-checkbox[data-id="${id}"]`);
                    if (rowCheckbox) {
                        rowCheckbox.checked = e.target.checked;
                        const row = rowCheckbox.closest('tr');
                        if (row) {
                            row.classList.toggle('selected', e.target.checked);
                        }
                    }
                });
            });
        });
        
        // Individual record checkbox click
        table.querySelectorAll('.benchmark-checkbox:not(.benchmark-group-checkbox)').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const id = e.target.dataset.id;
                if (e.target.checked) {
                    this.selectedRecords.add(id);
                } else {
                    this.selectedRecords.delete(id);
                }
                this.updateDeleteButton();
                
                // Update row highlight
                const row = e.target.closest('tr');
                if (row) {
                    row.classList.toggle('selected', e.target.checked);
                }
                
                // Update group checkbox state
                const groupHash = row?.dataset.groupHash;
                if (groupHash) {
                    this.updateGroupCheckboxState(table, groupHash);
                }
            });
        });
    }
    
    /**
     * Update group checkbox checked/indeterminate state based on individual selections
     */
    updateGroupCheckboxState(table, groupHash) {
        const groupCheckbox = table.querySelector(`.benchmark-group-checkbox[data-group-key="${groupHash}"]`);
        if (!groupCheckbox) return;
        
        const recordIds = groupCheckbox.dataset.recordIds ? groupCheckbox.dataset.recordIds.split(',') : [];
        const selectedCount = recordIds.filter(id => this.selectedRecords.has(id)).length;
        
        if (selectedCount === 0) {
            groupCheckbox.checked = false;
            groupCheckbox.indeterminate = false;
        } else if (selectedCount === recordIds.length) {
            groupCheckbox.checked = true;
            groupCheckbox.indeterminate = false;
        } else {
            groupCheckbox.checked = false;
            groupCheckbox.indeterminate = true;
        }
    }
    
    toggleGroup(hash) {
        if (this.expandedGroups.has(hash)) {
            this.expandedGroups.delete(hash);
        } else {
            this.expandedGroups.add(hash);
        }
        this.renderTable();
    }
    
    updateDeleteButton() {
        const deleteBtn = this.container.querySelector('#benchmark-delete-btn');
        if (deleteBtn) {
            deleteBtn.disabled = this.selectedRecords.size === 0;
            deleteBtn.textContent = this.selectedRecords.size > 0 
                ? `Delete Selected (${this.selectedRecords.size})`
                : 'Delete Selected';
        }
    }
    
    async deleteSelected() {
        if (this.selectedRecords.size === 0) return;
        
        const count = this.selectedRecords.size;
        if (!confirm(`Delete ${count} selected record(s)?`)) return;
        
        const deleteBtn = this.container.querySelector('#benchmark-delete-btn');
        if (deleteBtn) {
            deleteBtn.disabled = true;
            deleteBtn.textContent = `Deleting ${count}...`;
        }
        
        try {
            const ids = Array.from(this.selectedRecords);
            
            const response = await fetch(`${this.options.apiBase}/api/benchmark/bulk-delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            
            this.selectedRecords.clear();
            await this.fetchData();
            
            console.log(`[Benchmark] Deleted ${result.deleted} record(s)`);
            
        } catch (error) {
            console.error('[Benchmark] Delete failed:', error);
            alert('Failed to delete records. Please try again.');
        }
        
        this.updateDeleteButton();
    }
    
    renderError(message) {
        const container = this.container.querySelector('#benchmark-table-container');
        if (container) {
            container.innerHTML = `
                <div class="benchmark-empty">
                    <div class="benchmark-empty-icon">⚠️</div>
                    <div class="benchmark-empty-text">
                        Failed to load benchmark data: ${message}
                    </div>
                    <button class="benchmark-btn benchmark-btn-secondary" 
                            style="margin-top: 10px;"
                            onclick="this.closest('.benchmark-container').querySelector('#benchmark-refresh-btn').click()">
                        Retry
                    </button>
                </div>
            `;
        }
    }
    
    // Helper methods
    
    getSortClass(column) {
        if (this.sortColumn !== column) return '';
        return this.sortOrder === 'asc' ? 'sorted-asc' : 'sorted-desc';
    }
    
    getSolverBadgeClass(solver) {
        const classes = {
            'gpu': 'gpu',
            'numba_cuda': 'numba-cuda',
            'numba': 'numba',
            'cpu_multiprocess': 'cpu-multiprocess',
            'cpu_threaded': 'cpu-threaded',
            'cpu': 'cpu'
        };
        return classes[solver] || 'cpu';
    }
    
    formatSolverName(solver) {
        const names = {
            'gpu': 'GPU',
            'numba_cuda': 'Numba CUDA',
            'numba': 'Numba',
            'cpu_multiprocess': 'Multiprocess',
            'cpu_threaded': 'Threaded',
            'cpu': 'CPU'
        };
        return names[solver] || solver;
    }
    
    formatTime(seconds) {
        if (seconds === undefined || seconds === null) return '-';
        if (seconds < 0.01) return '<0.01s';
        if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
        if (seconds < 60) return `${seconds.toFixed(2)}s`;
        const mins = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(1);
        return `${mins}m ${secs}s`;
    }
    
    formatNumber(num) {
        if (num === undefined || num === null) return '-';
        return num.toLocaleString();
    }
    
    formatTimestamp(timestamp) {
        if (!timestamp) return '-';
        const date = new Date(timestamp);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
    
    /**
     * Format memory value in MB to human-readable string
     */
    formatMemory(mb) {
        if (mb === undefined || mb === null || mb === 0) return '-';
        if (mb < 1024) return `${Math.round(mb)} MB`;
        return `${(mb / 1024).toFixed(2)} GB`;
    }
    
    /**
     * Get CSS class for VRAM cell based on usage
     */
    getVramClass(mb) {
        if (mb === undefined || mb === null || mb === 0) return '';
        if (mb > 16000) return 'memory-high';    // > 16 GB
        if (mb > 8000) return 'memory-medium';   // > 8 GB
        return 'memory-low';                      // <= 8 GB
    }
    
    getTimeClass(seconds) {
        if (seconds < 10) return 'fast';
        if (seconds < 60) return 'medium';
        return 'slow';
    }
    
    /**
     * Shorten CPU model name for display
     * "13th Gen Intel(R) Core(TM) i9-13900K" -> "i9-13900K"
     * "AMD Ryzen 9 5900X 12-Core Processor" -> "Ryzen 9 5900X"
     */
    shortenCpuName(cpuModel) {
        if (!cpuModel) return 'Unknown CPU';
        
        // Intel: Extract iX-XXXXX pattern
        const intelMatch = cpuModel.match(/i[3579]-\d{4,5}[A-Z]*/i);
        if (intelMatch) return intelMatch[0];
        
        // AMD Ryzen: Extract "Ryzen X XXXX" pattern
        const ryzenMatch = cpuModel.match(/Ryzen\s+\d+\s+\d{4}[A-Z]*/i);
        if (ryzenMatch) return ryzenMatch[0];
        
        // AMD Threadripper
        const threadripperMatch = cpuModel.match(/Threadripper\s+\d{4}[A-Z]*/i);
        if (threadripperMatch) return threadripperMatch[0];
        
        // Apple Silicon
        const appleMatch = cpuModel.match(/Apple\s+M\d+(\s+\w+)?/i);
        if (appleMatch) return appleMatch[0];
        
        // Fallback: truncate to 20 chars
        if (cpuModel.length > 20) {
            return cpuModel.substring(0, 20) + '...';
        }
        
        return cpuModel;
    }
    
    /**
     * Shorten GPU model name for display
     * "NVIDIA GeForce RTX 4090" -> "RTX 4090"
     * "AMD Radeon RX 7900 XTX" -> "RX 7900 XTX"
     */
    shortenGpuName(gpuModel) {
        if (!gpuModel) return 'No GPU';
        
        // NVIDIA RTX/GTX
        const nvidiaMatch = gpuModel.match(/(RTX|GTX)\s*\d{3,4}(\s*Ti|\s*Super)?/i);
        if (nvidiaMatch) return nvidiaMatch[0];
        
        // NVIDIA Quadro
        const quadroMatch = gpuModel.match(/Quadro\s+\w+\d+/i);
        if (quadroMatch) return quadroMatch[0];
        
        // AMD Radeon RX
        const radeonMatch = gpuModel.match(/RX\s*\d{4}(\s*XT[X]?)?/i);
        if (radeonMatch) return radeonMatch[0];
        
        // Intel Arc
        const arcMatch = gpuModel.match(/Arc\s+A\d{3}/i);
        if (arcMatch) return arcMatch[0];
        
        // Apple Silicon GPU
        if (gpuModel.toLowerCase().includes('apple')) {
            return 'Apple GPU';
        }
        
        // Fallback: truncate to 15 chars
        if (gpuModel.length > 15) {
            return gpuModel.substring(0, 15) + '...';
        }
        
        return gpuModel;
    }

    async loadReportSections() {
        try {
            const response = await fetch(`${this.options.apiBase}/api/benchmark/report/sections`);
            if (!response.ok) return;
            
            const data = await response.json();
            const select = this.container.querySelector('#benchmark-report-section');
            
            if (select && data.sections) {
                data.sections.forEach(section => {
                    const option = document.createElement('option');
                    option.value = section.id;
                    option.textContent = section.title;
                    select.appendChild(option);
                });
            }
        } catch (error) {
            console.error('[Benchmark] Failed to load report sections:', error);
        }
    }

    openReportViewer(sectionId) {
        // Use global ReportViewerPanel if available
        if (window.ReportViewerPanel) {
            // Determine is_automated filter value
            let isAutomated = null;
            if (this.filterTesting === 'automated') {
                isAutomated = true;
            } else if (this.filterTesting === 'manual') {
                isAutomated = false;
            }
            
            const panel = new window.ReportViewerPanel(sectionId, {
                apiBase: this.options.apiBase,
                filters: {
                    solver_type: this.filterSolver || null,
                    model_name: this.filterModel || null,
                    server_hash: this.filterServer || null,
                    is_automated: isAutomated
                }
            });
            panel.open();
        } else {
            console.error('[Benchmark] ReportViewerPanel not available');
        }
    }  

    /**
     * Generate Markdown table from current filtered/sorted records
     */
    generateTableMarkdown() {
        // Get currently displayed records (respecting filters)
        let filteredRecords = this.records;
        
        if (this.filterTesting === 'manual') {
            filteredRecords = filteredRecords.filter(r => r.client_hash !== 'automated');
        } else if (this.filterTesting === 'automated') {
            filteredRecords = filteredRecords.filter(r => r.client_hash === 'automated');
        }
        
        if (this.filterSolver) {
            filteredRecords = filteredRecords.filter(r => r.solver_type === this.filterSolver);
        }
        if (this.filterModel) {
            filteredRecords = filteredRecords.filter(r => r.model_name === this.filterModel);
        }
        if (this.filterServer) {
            filteredRecords = filteredRecords.filter(r => {
                const isAutomated = r.client_hash === 'automated';
                const compositeKey = `${r.server_hash}${isAutomated ? '_automated' : ''}`;
                return compositeKey === this.filterServer;
            });
        }
        
        // Sort records
        filteredRecords = this.sortRecords(filteredRecords);
        
        if (filteredRecords.length === 0) {
            return '*No records to export*';
        }
        
        // Build Markdown table
        let md = '# Benchmark Results\n\n';
        
        // Add filter info
        const filters = [];
        if (this.filterSolver) filters.push(`Solver: ${this.formatSolverName(this.filterSolver)}`);
        if (this.filterModel) filters.push(`Model: ${this.filterModel}`);
        if (this.filterTesting) filters.push(`Testing: ${this.filterTesting}`);
        if (filters.length > 0) {
            md += `**Filters:** ${filters.join(', ')}\n\n`;
        }
        
        md += `**Total Records:** ${filteredRecords.length}\n\n`;
        
        // Table header
        md += '| Model | Solver | Nodes | Elements | Total Time | Assembly | Solve | Iterations | Peak RAM | Peak VRAM | Status | Date |\n';
        md += '|-------|--------|------:|--------:|-----------:|---------:|------:|-----------:|---------:|----------:|:------:|------|\n';
        
        // Table rows
        filteredRecords.forEach(record => {
            const totalTime = record.timings?.total_program_time || 0;
            const assemblyTime = record.timings?.assemble_system || 0;
            const solveTime = record.timings?.solve_system || 0;
            const peakRam = record.memory?.peak_ram_mb;
            const peakVram = record.memory?.peak_vram_mb;
            
            md += `| ${record.model_name} `;
            md += `| ${this.formatSolverName(record.solver_type)} `;
            md += `| ${this.formatNumber(record.model_nodes)} `;
            md += `| ${this.formatNumber(record.model_elements)} `;
            md += `| ${this.formatTime(totalTime)} `;
            md += `| ${this.formatTime(assemblyTime)} `;
            md += `| ${this.formatTime(solveTime)} `;
            md += `| ${this.formatNumber(record.iterations)} `;
            md += `| ${this.formatMemory(peakRam)} `;
            md += `| ${this.formatMemory(peakVram)} `;
            md += `| ${record.converged ? '✓' : '✗'} `;
            md += `| ${this.formatTimestamp(record.timestamp)} |\n`;
        });
        
        return md;
    }

    /**
     * Copy current table to clipboard as Markdown
     */
    copyTableAsMarkdown() {
        const markdown = this.generateTableMarkdown();
        
        navigator.clipboard.writeText(markdown).then(() => {
            // Brief visual feedback
            const btn = this.container.querySelector('#benchmark-copy-table-btn');
            if (btn) {
                const original = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = original, 1500);
            }
        }).catch(err => {
            console.error('[Benchmark] Failed to copy:', err);
        });
    }

    /**
     * Export current table as Markdown file
     */
    exportTableAsMarkdown() {
        const markdown = this.generateTableMarkdown();
        
        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        
        const timestamp = new Date().toISOString().slice(0, 10);
        const filename = `benchmark_table_${timestamp}.md`;
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }    
    
    // Polling
    startPolling() {
        this.stopPolling();
        this.pollTimer = setInterval(() => this.fetchData(), this.options.pollInterval);
    }
    
    stopPolling() {
        if (this.pollTimer) {
            clearInterval(this.pollTimer);
            this.pollTimer = null;
        }
    }
    
    // Public methods
    
    refresh() {
        return this.fetchData();
    }
    
    destroy() {
        this.stopPolling();
        if (this.container) {
            this.container.innerHTML = '';
        }
    }
}

// Auto-initialize if container exists
export function initBenchmarkPanel(containerId = '.benchmark-container', options = {}) {
    const panel = new BenchmarkPanel(containerId, options);
    return panel;
}

export default BenchmarkPanel;
