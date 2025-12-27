/**
 * BaseMetric - Abstract base class for all metric panels
 * 
 * Provides:
 * - HUD panel creation and management
 * - ECharts initialization with SVG renderer
 * - Auto-resize via ResizeObserver
 * - Moveable.js integration for drag/resize
 * - Show/hide based on Catalog configuration
 * - Event subscription pattern
 * 
 * Location: /src/app/client/script/metrics/BaseMetric.js
 */

export class BaseMetric {
    /**
     * @param {string} id - Unique metric identifier (matches METRICS_CATALOG id)
     * @param {object} options - Configuration options
     * @param {string} options.title - Panel title
     * @param {number} options.defaultWidth - Default panel width in pixels
     * @param {number} options.defaultHeight - Default panel height in pixels
     * @param {object} options.position - Initial position {top, left, right, bottom}
     * @param {boolean} options.resizable - Whether panel can be resized (default: true)
     */
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            title: options.title || 'Metric',
            defaultWidth: options.defaultWidth || 320,
            defaultHeight: options.defaultHeight || 200,
            position: options.position || { top: '100px', right: '20px' },
            resizable: options.resizable !== false,
            ...options
        };
        
        this.panel = null;
        this.chartContainer = null;
        this.chart = null;
        this.resizeObserver = null;
        this.moveable = null;
        this.moveablePos = { x: 0, y: 0 };
        this.visible = false;
        this.hasData = false;
        this.topZ = 20; // Start above other HUD panels
        
        this._boundHandlers = {};
    }
    
    /**
     * Initialize the metric panel
     * Creates the HUD panel and sets up ECharts
     */
    init() {
        this.createPanel();
        this.initChart();
        this.setupResizeObserver();
        this.bindEvents();
        
        console.log(`[${this.id}] Metric initialized`);
    }
    
    /**
     * Create the HUD panel element
     */
    createPanel() {
        // Check if panel already exists
        const existingPanel = document.getElementById(`hud-metric-${this.id}`);
        if (existingPanel) {
            this.panel = existingPanel;
            this.chartContainer = this.panel.querySelector('.metric-chart-container');
            return;
        }
        
        // Create panel element
        this.panel = document.createElement('div');
        this.panel.id = `hud-metric-${this.id}`;
        this.panel.className = 'hud metric-panel';
        this.panel.style.cssText = `
            z-index: ${this.topZ};
            width: ${this.options.defaultWidth}px;
            height: ${this.options.defaultHeight}px;
            min-width: 200px;
            min-height: 150px;
            position: absolute;
            ${this.getPositionStyle()}
        `;
        
        // Panel content with close button
        this.panel.innerHTML = `
            <h1>${this.options.title.toUpperCase()}</h1>
            <div class="sep"></div>
            <div class="metric-chart-container" style="width: 100%; height: calc(100% - 45px); min-height: 100px;"></div>
            <button class="pm-close">×</button>
        `;
        
        this.chartContainer = this.panel.querySelector('.metric-chart-container');
        
        // Close button handler
        const closeBtn = this.panel.querySelector('.pm-close');
        if (closeBtn) {
            closeBtn.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.hide();
            };
        }
        
        // Add to overlay
        const overlay = document.getElementById('overlay');
        if (overlay) {
            overlay.appendChild(this.panel);
        } else {
            console.error(`[${this.id}] Overlay container not found`);
        }
    }
    
    /**
     * Convert position options to CSS
     */
    getPositionStyle() {
        const pos = this.options.position;
        let style = '';
        if (pos.top) style += `top: ${pos.top};`;
        if (pos.left) style += `left: ${pos.left};`;
        if (pos.right) style += `right: ${pos.right};`;
        if (pos.bottom) style += `bottom: ${pos.bottom};`;
        return style;
    }
    
    /**
     * Initialize Moveable.js for drag and resize
     */
    initMoveable() {
        if (!this.panel || typeof Moveable === 'undefined') {
            console.warn(`[${this.id}] Moveable not available`);
            return;
        }
        
        // Destroy existing moveable if any
        if (this.moveable) {
            this.moveable.destroy();
            this.moveable = null;
        }
        
        // Normalize positioning to top-left (required by Moveable)
        const rect = this.panel.getBoundingClientRect();
        this.panel.style.top = `${rect.top}px`;
        this.panel.style.left = `${rect.left}px`;
        this.panel.style.bottom = 'auto';
        this.panel.style.right = 'auto';
        this.panel.style.margin = '0';
        this.panel.style.transform = 'translate(0px, 0px)';
        
        // Reset position tracker
        this.moveablePos = { x: 0, y: 0 };
        
        const padding = 10;
        const headerEl = this.panel.querySelector('h1');
        
        this.moveable = new Moveable(document.body, {
            target: this.panel,
            draggable: true,
            resizable: this.options.resizable,
            origin: false,
            snappable: true,
            bounds: {
                left: padding,
                top: padding,
                right: window.innerWidth - padding,
                bottom: window.innerHeight - padding
            }
        });
        
        let allowDrag = false;
        
        // Drag events
        this.moveable.on('dragStart', e => {
            const t = e.inputEvent && e.inputEvent.target;
            allowDrag = !!(headerEl && t && (t === headerEl || headerEl.contains(t)));
            if (!allowDrag) {
                e.stop && e.stop();
                return;
            }
            if (e.set) e.set([this.moveablePos.x, this.moveablePos.y]);
            e.inputEvent.stopPropagation();
            
            // Bring to front
            this.topZ += 1;
            this.panel.style.zIndex = String(this.topZ);
            this.syncControlBoxZ();
        })
        .on('drag', e => {
            if (!allowDrag) return;
            const [x, y] = e.beforeTranslate;
            this.moveablePos.x = x;
            this.moveablePos.y = y;
            e.target.style.transform = `translate(${x}px, ${y}px)`;
        })
        .on('dragEnd', () => {
            allowDrag = false;
        });
        
        // Resize events
        this.moveable.on('resizeStart', e => {
            e.setOrigin(['%', '%']);
            if (e.dragStart) e.dragStart.set([this.moveablePos.x, this.moveablePos.y]);
            
            // Bring to front
            this.topZ += 1;
            this.panel.style.zIndex = String(this.topZ);
            this.syncControlBoxZ();
        })
        .on('resize', e => {
            const { target, width, height, drag } = e;
            target.style.width = `${width}px`;
            target.style.height = `${height}px`;
            
            const [x, y] = drag.beforeTranslate;
            target.style.transform = `translate(${x}px, ${y}px)`;
            this.moveablePos.x = x;
            this.moveablePos.y = y;
            
            // Resize chart
            if (this.chart && !this.chart.isDisposed()) {
                this.chart.resize();
            }
        })
        .on('resizeEnd', () => {
            this.moveable.updateRect();
            this.applyControlStyles();
        });
        
        // Apply initial control styles
        this.applyControlStyles();
        this.syncControlBoxZ();
    }
    
    /**
     * Sync Moveable control box z-index with panel
     */
    syncControlBoxZ() {
        if (this.moveable && this.moveable.selfElement) {
            this.moveable.selfElement.style.zIndex = this.panel.style.zIndex;
        }
    }
    
    /**
     * Apply custom styles to Moveable control handles
     */
    applyControlStyles() {
        if (!this.moveable) return;
        
        const controlBox = this.moveable.selfElement;
        if (!controlBox) return;
        
        const controls = controlBox.querySelectorAll('.moveable-control');
        const { width, height } = this.moveable.getRect();
        
        controls.forEach(control => {
            control.classList.add('custom-control');
            
            // Extend horizontal handles to full width
            if (control.classList.contains('moveable-n') || control.classList.contains('moveable-s')) {
                control.style.width = `${width}px`;
                control.style.marginLeft = `-${width / 2}px`;
            }
            // Extend vertical handles to full height
            if (control.classList.contains('moveable-w') || control.classList.contains('moveable-e')) {
                control.style.height = `${height}px`;
                control.style.marginTop = `-${height / 2}px`;
            }
        });
    }
    
    /**
     * Destroy Moveable instance
     */
    destroyMoveable() {
        if (this.moveable) {
            this.moveable.destroy();
            this.moveable = null;
        }
    }
    
    /**
     * Initialize ECharts with SVG renderer
     */
    initChart() {
        if (!this.chartContainer) {
            console.error(`[${this.id}] Chart container not found`);
            return;
        }
        
        // Dispose existing chart if any
        if (this.chart) {
            this.chart.dispose();
        }
        
        // Initialize ECharts with SVG renderer
        this.chart = echarts.init(this.chartContainer, null, {
            renderer: 'svg'
        });
        
        // Set initial empty option (subclass will override)
        this.chart.setOption(this.getDefaultOption());
    }
    
    /**
     * Get default chart option (override in subclass)
     * @returns {object} ECharts option
     */
    getDefaultOption() {
        return {
            title: {
                text: 'No data',
                left: 'center',
                top: 'center',
                textStyle: {
                    color: '#999',
                    fontSize: 14,
                    fontWeight: 'normal'
                }
            }
        };
    }
    
    /**
     * Setup ResizeObserver for auto-resize
     */
    setupResizeObserver() {
        if (!this.panel || !this.chart) return;
        
        this.resizeObserver = new ResizeObserver((entries) => {
            // Debounce resize
            if (this._resizeTimeout) {
                clearTimeout(this._resizeTimeout);
            }
            this._resizeTimeout = setTimeout(() => {
                if (this.chart && !this.chart.isDisposed()) {
                    this.chart.resize();
                }
            }, 50);
        });
        
        this.resizeObserver.observe(this.panel);
    }
    
    /**
     * Bind event listeners (override in subclass to add more)
     */
    bindEvents() {
        // Listen for catalog configuration changes
        this._boundHandlers.onConfigChanged = (e) => this.onConfigChanged(e);
        document.addEventListener('metricsConfigChanged', this._boundHandlers.onConfigChanged);
    }
    
    /**
     * Handle catalog configuration changes
     */
    onConfigChanged(event) {
        const enabledMetrics = event.detail?.enabledMetrics || [];
        const shouldBeVisible = enabledMetrics.includes(this.id);
        
        if (shouldBeVisible && this.hasData) {
            this.show();
        } else {
            this.hide();
        }
    }
    
    /**
     * Show the metric panel
     */
    show() {
        if (this.panel) {
            this.panel.classList.add('visible');
            this.visible = true;
            
            // Initialize Moveable when showing (matches MenuManager pattern)
            if (!this.moveable) {
                // Small delay to ensure panel is visible and has dimensions
                setTimeout(() => {
                    this.initMoveable();
                    
                    // Also resize chart
                    if (this.chart && !this.chart.isDisposed()) {
                        this.chart.resize();
                    }
                }, 50);
            } else {
                this.moveable.updateRect();
                if (this.chart && !this.chart.isDisposed()) {
                    this.chart.resize();
                }
            }
        }
    }
    
    /**
     * Hide the metric panel
     */
    hide() {
        if (this.panel) {
            this.panel.classList.remove('visible');
            this.visible = false;
            
            // Destroy Moveable when hiding (matches MenuManager pattern)
            this.destroyMoveable();
        }
    }
    
    /**
     * Toggle visibility
     */
    toggle() {
        if (this.visible) {
            this.hide();
        } else {
            this.show();
        }
    }
    
    /**
     * Update the chart with new data (override in subclass)
     * @param {object} data - Data to display
     */
    update(data) {
        this.hasData = true;
        // Subclass implements this
    }
    
    /**
     * Reset the metric to initial state
     */
    reset() {
        this.hasData = false;
        if (this.chart && !this.chart.isDisposed()) {
            this.chart.setOption(this.getDefaultOption(), true);
        }
    }
    
    /**
     * Check if this metric is enabled in the catalog
     * @returns {boolean}
     */
    isEnabled() {
        // Check localStorage for saved config
        try {
            const saved = localStorage.getItem('femulator-metrics-config');
            if (saved) {
                const enabled = JSON.parse(saved);
                return enabled.includes(this.id);
            }
        } catch (e) {
            // Ignore errors
        }
        
        // Return default from catalog (subclass should override)
        return true;
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        // Remove event listeners
        if (this._boundHandlers.onConfigChanged) {
            document.removeEventListener('metricsConfigChanged', this._boundHandlers.onConfigChanged);
        }
        
        // Disconnect resize observer
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        
        // Clear timeout
        if (this._resizeTimeout) {
            clearTimeout(this._resizeTimeout);
        }
        
        // Destroy Moveable
        this.destroyMoveable();
        
        // Dispose ECharts
        if (this.chart && !this.chart.isDisposed()) {
            this.chart.dispose();
            this.chart = null;
        }
        
        // Remove panel from DOM
        if (this.panel && this.panel.parentNode) {
            this.panel.parentNode.removeChild(this.panel);
            this.panel = null;
        }
        
        console.log(`[${this.id}] Metric disposed`);
    }
}

export default BaseMetric;
