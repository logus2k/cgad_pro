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
    solver: {
        id: 'solver',
        title: 'Solver',
        description: 'Solver performance and convergence',
        metrics: [
            {
                id: 'convergence-plot',
                name: 'Convergence Iterations',
                description: 'Residual vs iteration chart (log scale)',
                type: 'realtime',
                default: false
            },
            {
                id: 'convergence-quality',
                name: 'Convergence Quality',
                description: 'Final residual, iterations, convergence status',
                type: 'post-solve',
                default: false
            },
            {
                id: 'timing-waterfall',
                name: 'Timing Breakdown',
                description: 'Time spent in each solver stage',
                type: 'post-solve',
                default: false
            },
            {
                id: 'speedup-factors',
                name: 'Speedup Factors',
                description: 'Performance comparison across solver types',
                type: 'post-solve',
                default: false
            }
        ]
    },
    model: {
        id: 'model',
        title: 'Model',
        description: 'Mesh and solution data',
        metrics: [
            {
                id: 'mesh-stats',
                name: 'Mesh Statistics',
                description: 'Node count, element count, complexity',
                type: 'post-solve',
                default: false
            },
            {
                id: 'solution-range',
                name: 'Solution Range',
                description: 'Min/max/mean solution values',
                type: 'post-solve',
                default: false
            }
        ]
    },
    system: {
        id: 'system',
        title: 'System',
        description: 'Hardware information',
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
                description: 'Browser, WebGL renderer, screen',
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
        // Always start with defaults on page load
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
        // No persistence - state resets on page reload
    }
    
    render() {
        this.container.innerHTML = `
            ${Object.values(METRICS_CATALOG).map((category, index) => 
                this.renderCategory(category, index === 0)
            ).join('')}
        `;
    }
    
    renderCategory(category, isOpen = false) {
        const enabledInCategory = category.metrics.filter(m => this.enabledMetrics.has(m.id)).length;
        
        return `
            <div class="metrics-category ${isOpen ? '' : 'collapsed'}" data-category="${category.id}">
                <div class="metrics-category-header">
                    <span class="metrics-category-toggle">${isOpen ? '▼' : '▶'}</span>
                    <span class="metrics-category-title">${category.title}</span>
                    <span class="metrics-category-count">${enabledInCategory}/${category.metrics.length}</span>
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
        // Category collapse/expand (accordion - only one open at a time)
        this.container.querySelectorAll('.metrics-category-header').forEach(header => {
            header.addEventListener('click', (e) => {
                // Don't toggle if clicking on checkbox area
                if (e.target.type === 'checkbox') return;
                
                const category = header.closest('.metrics-category');
                const isCollapsed = category.classList.contains('collapsed');
                
                // Close all categories first
                this.container.querySelectorAll('.metrics-category').forEach(cat => {
                    cat.classList.add('collapsed');
                    cat.querySelector('.metrics-category-toggle').textContent = '▶';
                });
                
                // Open clicked category if it was closed
                if (isCollapsed) {
                    category.classList.remove('collapsed');
                    category.querySelector('.metrics-category-toggle').textContent = '▼';
                }
            });
        });
        
        // Metric checkboxes - apply changes immediately
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
                this.applyChanges(); // Apply immediately
            });
        });
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
        
        this.updateCounts();
        
        console.log('[MetricsCatalog] Reset to defaults');
    }
    
    applyChanges() {
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
