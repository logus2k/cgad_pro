/**
 * Metrics Catalog Component
 * 
 * Displays available metrics organized by category with toggles to enable/disable each.
 * Manages which metrics are visible in the Runtime tab.
 * 
 * Location: /src/app/client/script/metrics-catalog.js
 */

// Metrics definition with categories
const METRICS_CATALOG = {
    liveMonitor: {
        id: 'live-monitor',
        title: 'Live Solver Monitor',
        icon: '⚡',
        description: 'Real-time solver progress indicators',
        metrics: [
            {
                id: 'progress-ring',
                name: 'Progress Ring',
                description: 'Circular progress indicator with iteration count and ETR',
                type: 'realtime',
                default: true
            },
            {
                id: 'convergence-plot',
                name: 'Convergence Plot',
                description: 'Residual vs iteration chart (log scale)',
                type: 'realtime',
                default: true
            },
            {
                id: 'stage-timeline',
                name: 'Stage Timeline',
                description: 'Horizontal bar showing solver stages progress',
                type: 'realtime',
                default: true
            },
            {
                id: 'residual-display',
                name: 'Residual Display',
                description: 'Current residual norm and relative residual',
                type: 'realtime',
                default: true
            }
        ]
    },
    solutionQuality: {
        id: 'solution-quality',
        title: 'Solution Quality',
        icon: '📊',
        description: 'Post-solve analysis metrics',
        metrics: [
            {
                id: 'solution-range',
                name: 'Solution Range',
                description: 'Min/max potential values with distribution',
                type: 'post-solve',
                default: true
            },
            {
                id: 'velocity-stats',
                name: 'Velocity Field Stats',
                description: 'Min/max/mean velocity magnitude',
                type: 'post-solve',
                default: true
            },
            {
                id: 'pressure-stats',
                name: 'Pressure Distribution',
                description: 'Bernoulli-derived pressure statistics',
                type: 'post-solve',
                default: false
            },
            {
                id: 'convergence-quality',
                name: 'Convergence Quality',
                description: 'Final residual, iterations vs max, convergence status',
                type: 'post-solve',
                default: true
            }
        ]
    },
    performanceBreakdown: {
        id: 'performance-breakdown',
        title: 'Performance Breakdown',
        icon: '⏱️',
        description: 'Timing analysis per solver stage',
        metrics: [
            {
                id: 'timing-waterfall',
                name: 'Timing Waterfall',
                description: 'Visual breakdown of time spent in each stage',
                type: 'post-solve',
                default: true
            },
            {
                id: 'timing-table',
                name: 'Timing Table',
                description: 'Tabular view of all timing metrics',
                type: 'post-solve',
                default: false
            },
            {
                id: 'throughput',
                name: 'Throughput Metrics',
                description: 'Elements/second, iterations/second',
                type: 'post-solve',
                default: false
            }
        ]
    },
    comparativeAnalysis: {
        id: 'comparative-analysis',
        title: 'Comparative Analysis',
        icon: '📈',
        description: 'Cross-run and cross-solver comparisons',
        metrics: [
            {
                id: 'solver-comparison',
                name: 'Solver Comparison',
                description: 'Compare current run against other solver types',
                type: 'comparative',
                default: true
            },
            {
                id: 'historical-best',
                name: 'Historical Best',
                description: 'Compare against best recorded time for this model',
                type: 'comparative',
                default: false
            },
            {
                id: 'speedup-factors',
                name: 'Speedup Factors',
                description: 'Relative speedup vs CPU baseline',
                type: 'comparative',
                default: true
            }
        ]
    },
    meshInfo: {
        id: 'mesh-info',
        title: 'Mesh Information',
        icon: '🔷',
        description: 'Model geometry characteristics',
        metrics: [
            {
                id: 'mesh-stats',
                name: 'Mesh Statistics',
                description: 'Node count, element count, complexity badge',
                type: 'post-solve',
                default: true
            },
            {
                id: 'boundary-info',
                name: 'Boundary Conditions',
                description: 'Robin edges, Dirichlet nodes, unused nodes',
                type: 'post-solve',
                default: false
            }
        ]
    },
    systemInfo: {
        id: 'system-info',
        title: 'System Information',
        icon: '💻',
        description: 'Hardware and environment details',
        metrics: [
            {
                id: 'server-hardware',
                name: 'Server Hardware',
                description: 'CPU, GPU, RAM, CUDA version',
                type: 'system',
                default: false
            },
            {
                id: 'client-hardware',
                name: 'Client Hardware',
                description: 'Browser, WebGL renderer, screen info',
                type: 'system',
                default: false
            },
            {
                id: 'solver-info',
                name: 'Solver Configuration',
                description: 'Solver type, tolerance, max iterations',
                type: 'system',
                default: false
            }
        ]
    }
};

export class MetricsCatalog {
    constructor(containerId, options = {}) {
        this.container = typeof containerId === 'string' 
            ? document.querySelector(containerId) 
            : containerId;
            
        if (!this.container) {
            console.error('[MetricsCatalog] Container not found');
            return;
        }
        
        this.options = {
            storageKey: 'femulator-metrics-config',
            onChange: null,
            ...options
        };
        
        // Load saved state or use defaults
        this.enabledMetrics = this.loadState();
        
        this.render();
        this.bindEvents();
    }
    
    loadState() {
        try {
            const saved = localStorage.getItem(this.options.storageKey);
            if (saved) {
                return new Set(JSON.parse(saved));
            }
        } catch (e) {
            console.warn('[MetricsCatalog] Could not load saved state:', e);
        }
        
        // Build default set from catalog
        const defaults = new Set();
        Object.values(METRICS_CATALOG).forEach(category => {
            category.metrics.forEach(metric => {
                if (metric.default) {
                    defaults.add(metric.id);
                }
            });
        });
        return defaults;
    }
    
    saveState() {
        try {
            localStorage.setItem(
                this.options.storageKey, 
                JSON.stringify([...this.enabledMetrics])
            );
        } catch (e) {
            console.warn('[MetricsCatalog] Could not save state:', e);
        }
    }
    
    render() {
        const enabledCount = this.enabledMetrics.size;
        const totalCount = Object.values(METRICS_CATALOG)
            .reduce((sum, cat) => sum + cat.metrics.length, 0);
        
        this.container.innerHTML = `
            <div class="metrics-quick-toggles">
                <button class="quick-toggle-btn" data-filter="all">All</button>
                <button class="quick-toggle-btn" data-filter="realtime">Real-time</button>
                <button class="quick-toggle-btn" data-filter="post-solve">Post-solve</button>
                <button class="quick-toggle-btn" data-filter="comparative">Comparative</button>
            </div>
            
            ${Object.values(METRICS_CATALOG).map(category => this.renderCategory(category)).join('')}
            
            <div class="metrics-status">
                <span class="metrics-status-count">${enabledCount}</span> of ${totalCount} metrics enabled
            </div>
            
            <div class="metrics-catalog-actions">
                <button class="metrics-catalog-btn secondary" id="metrics-reset-btn">
                    Reset to Defaults
                </button>
                <button class="metrics-catalog-btn primary" id="metrics-apply-btn">
                    Apply Changes
                </button>
            </div>
        `;
    }
    
    renderCategory(category) {
        const enabledInCategory = category.metrics.filter(m => this.enabledMetrics.has(m.id)).length;
        
        return `
            <div class="metrics-category" data-category="${category.id}">
                <div class="metrics-category-header">
                    <span class="metrics-category-icon">${category.icon}</span>
                    <span class="metrics-category-title">${category.title}</span>
                    <span class="metrics-category-count">${enabledInCategory}/${category.metrics.length}</span>
                    <span class="metrics-category-toggle">▼</span>
                </div>
                <div class="metrics-category-items">
                    ${category.metrics.map(metric => this.renderMetric(metric)).join('')}
                </div>
            </div>
        `;
    }
    
    renderMetric(metric) {
        const isEnabled = this.enabledMetrics.has(metric.id);
        
        return `
            <div class="metric-item ${isEnabled ? 'enabled' : ''}" data-metric="${metric.id}">
                <input type="checkbox" 
                       class="metric-checkbox" 
                       id="metric-${metric.id}"
                       data-metric-id="${metric.id}"
                       ${isEnabled ? 'checked' : ''}>
                <div class="metric-info">
                    <div class="metric-name">${metric.name}</div>
                    <div class="metric-description">${metric.description}</div>
                </div>
                <span class="metric-badge ${metric.type}">${this.formatType(metric.type)}</span>
            </div>
        `;
    }
    
    formatType(type) {
        const labels = {
            'realtime': 'Live',
            'post-solve': 'Post',
            'comparative': 'Compare',
            'system': 'System'
        };
        return labels[type] || type;
    }
    
    bindEvents() {
        // Category collapse/expand
        this.container.querySelectorAll('.metrics-category-header').forEach(header => {
            header.addEventListener('click', (e) => {
                // Don't toggle if clicking on checkbox area
                if (e.target.type === 'checkbox') return;
                
                const category = header.closest('.metrics-category');
                category.classList.toggle('collapsed');
            });
        });
        
        // Metric checkboxes
        this.container.querySelectorAll('.metric-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', (e) => {
                const metricId = e.target.dataset.metricId;
                const item = e.target.closest('.metric-item');
                
                if (e.target.checked) {
                    this.enabledMetrics.add(metricId);
                    item.classList.add('enabled');
                } else {
                    this.enabledMetrics.delete(metricId);
                    item.classList.remove('enabled');
                }
                
                this.updateCounts();
            });
        });
        
        // Quick filter buttons
        this.container.querySelectorAll('.quick-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const filter = btn.dataset.filter;
                this.applyQuickFilter(filter);
            });
        });
        
        // Reset button
        const resetBtn = this.container.querySelector('#metrics-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetToDefaults());
        }
        
        // Apply button
        const applyBtn = this.container.querySelector('#metrics-apply-btn');
        if (applyBtn) {
            applyBtn.addEventListener('click', () => this.applyChanges());
        }
    }
    
    applyQuickFilter(filter) {
        // Update button states
        this.container.querySelectorAll('.quick-toggle-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.filter === filter);
        });
        
        // Select metrics based on filter
        if (filter === 'all') {
            // Enable all
            Object.values(METRICS_CATALOG).forEach(category => {
                category.metrics.forEach(metric => {
                    this.enabledMetrics.add(metric.id);
                });
            });
        } else {
            // Enable only matching type, disable others
            this.enabledMetrics.clear();
            Object.values(METRICS_CATALOG).forEach(category => {
                category.metrics.forEach(metric => {
                    if (metric.type === filter) {
                        this.enabledMetrics.add(metric.id);
                    }
                });
            });
        }
        
        // Update UI
        this.container.querySelectorAll('.metric-checkbox').forEach(checkbox => {
            const metricId = checkbox.dataset.metricId;
            const isEnabled = this.enabledMetrics.has(metricId);
            checkbox.checked = isEnabled;
            checkbox.closest('.metric-item').classList.toggle('enabled', isEnabled);
        });
        
        this.updateCounts();
    }
    
    updateCounts() {
        // Update category counts
        this.container.querySelectorAll('.metrics-category').forEach(categoryEl => {
            const categoryId = categoryEl.dataset.category;
            const category = Object.values(METRICS_CATALOG).find(c => c.id === categoryId);
            if (!category) return;
            
            const enabledInCategory = category.metrics.filter(m => this.enabledMetrics.has(m.id)).length;
            const countEl = categoryEl.querySelector('.metrics-category-count');
            if (countEl) {
                countEl.textContent = `${enabledInCategory}/${category.metrics.length}`;
            }
        });
        
        // Update total count
        const totalCount = Object.values(METRICS_CATALOG)
            .reduce((sum, cat) => sum + cat.metrics.length, 0);
        const statusEl = this.container.querySelector('.metrics-status-count');
        if (statusEl) {
            statusEl.textContent = this.enabledMetrics.size;
        }
    }
    
    resetToDefaults() {
        this.enabledMetrics.clear();
        
        Object.values(METRICS_CATALOG).forEach(category => {
            category.metrics.forEach(metric => {
                if (metric.default) {
                    this.enabledMetrics.add(metric.id);
                }
            });
        });
        
        // Update UI
        this.container.querySelectorAll('.metric-checkbox').forEach(checkbox => {
            const metricId = checkbox.dataset.metricId;
            const isEnabled = this.enabledMetrics.has(metricId);
            checkbox.checked = isEnabled;
            checkbox.closest('.metric-item').classList.toggle('enabled', isEnabled);
        });
        
        // Clear quick filter selection
        this.container.querySelectorAll('.quick-toggle-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        this.updateCounts();
        this.saveState();
        
        console.log('[MetricsCatalog] Reset to defaults');
    }
    
    applyChanges() {
        this.saveState();
        
        // Dispatch event for other components to react
        const event = new CustomEvent('metricsConfigChanged', {
            detail: {
                enabledMetrics: [...this.enabledMetrics]
            }
        });
        document.dispatchEvent(event);
        
        // Call onChange callback if provided
        if (this.options.onChange) {
            this.options.onChange([...this.enabledMetrics]);
        }
        
        console.log('[MetricsCatalog] Applied changes:', [...this.enabledMetrics]);
    }
    
    // Public API
    
    getEnabledMetrics() {
        return [...this.enabledMetrics];
    }
    
    isMetricEnabled(metricId) {
        return this.enabledMetrics.has(metricId);
    }
    
    setMetricEnabled(metricId, enabled) {
        if (enabled) {
            this.enabledMetrics.add(metricId);
        } else {
            this.enabledMetrics.delete(metricId);
        }
        
        const checkbox = this.container.querySelector(`#metric-${metricId}`);
        if (checkbox) {
            checkbox.checked = enabled;
            checkbox.closest('.metric-item').classList.toggle('enabled', enabled);
        }
        
        this.updateCounts();
    }
}

// Export catalog definition for use by other components
export { METRICS_CATALOG };

// Auto-initialize if container exists
export function initMetricsCatalog(containerId = '.metrics-catalog-container', options = {}) {
    return new MetricsCatalog(containerId, options);
}

export default MetricsCatalog;
