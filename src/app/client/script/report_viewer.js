/**
 * Report Viewer Panel
 * 
 * Displays rendered Markdown report sections in a panel that integrates
 * with the existing HUD/panel system (Moveable.js, zIndexManager, etc.)
 * 
 * Multiple instances can coexist for different sections.
 * 
 * Location: /src/app/client/script/report_viewer.js
 */

import { getTopZ } from './zIndexManager.js';


export class ReportViewerPanel {

    static instances = new Map();  // Track open panels by section ID
    
    constructor(sectionId, options = {}) {
        this.sectionId = sectionId;
        this.options = {
            apiBase: options.apiBase || '',
            filters: options.filters || {},
            ...options
        };
        
        this.panelId = `hud-report-${sectionId}`;
        this.title = 'Report';
        this.markdown = '';
        this.panel = null;
        this.moveable = null;
        this.position = { x: 0, y: 0 };
    }
    
    async open() {
        // Check if panel already exists - bring to front
        if (ReportViewerPanel.instances.has(this.sectionId)) {
            const existing = ReportViewerPanel.instances.get(this.sectionId);
            existing.bringToFront();
            return existing;
        }
        
        // Fetch report content
        await this.fetchReport();
        
        // Create and show panel
        this.createPanel();
        this.render();
        this.bindEvents();
        this.initMoveable();
        
        // Register instance
        ReportViewerPanel.instances.set(this.sectionId, this);
        
        // Show panel
        this.panel.classList.add('visible');
        this.bringToFront();
        
        return this;
    }
    
    async fetchReport() {
        try {
            // Build query string from filters
            const params = new URLSearchParams();
            if (this.options.filters) {
                const { solver_type, model_name, server_hash, is_automated } = this.options.filters;
                if (solver_type) params.append('solver_type', solver_type);
                if (model_name) params.append('model_name', model_name);
                if (server_hash) params.append('server_hash', server_hash);
                if (is_automated !== null && is_automated !== undefined) {
                    params.append('is_automated', is_automated.toString());
                }
            }
            
            const queryString = params.toString();
            const url = `${this.options.apiBase}/api/benchmark/report/${this.sectionId}${queryString ? '?' + queryString : ''}`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.title = data.title || 'Report';
            this.markdown = data.markdown || '*No content available*';
            
        } catch (error) {
            console.error('[ReportViewer] Failed to fetch report:', error);
            this.title = 'Error';
            this.markdown = `**Error loading report:** ${error.message}`;
        }
    }
    
    createPanel() {
        // Remove existing panel if any
        const existing = document.getElementById(this.panelId);
        if (existing) {
            existing.remove();
        }
        
        // Create panel element using HUD structure
        this.panel = document.createElement('div');
        this.panel.id = this.panelId;
        this.panel.className = 'hud report-viewer';
        
        // Position with offset based on instance count
        const offset = ReportViewerPanel.instances.size * 30;
        this.panel.style.top = `${100 + offset}px`;
        this.panel.style.left = `${150 + offset}px`;
        this.panel.style.width = '700px';
        this.panel.style.height = '550px';
        this.panel.style.position = 'absolute';
        
        const overlay = document.getElementById('overlay');
        if (overlay) {
            overlay.appendChild(this.panel);
        } else {
            document.body.appendChild(this.panel);
        }
    }
    
    render() {
        if (!this.panel) return;
        
        // Render markdown to HTML
        let htmlContent = this.markdown;
        if (window.marked) {
            htmlContent = window.marked.parse(this.markdown);
        }
        
        this.panel.innerHTML = `
            <h1>${this.title.toUpperCase()}</h1>
            <button class="pm-close" title="Close">&times;</button>
            <div class="sep"></div>
            <div class="report-viewer-body">
                <div class="report-viewer-content">
                    ${htmlContent}
                </div>
                <div class="report-viewer-footer">
                    <button class="benchmark-btn benchmark-btn-secondary" id="report-btn-${this.sectionId}-copy">Copy</button>
                    <button class="benchmark-btn benchmark-btn-secondary" id="report-btn-${this.sectionId}-export">Export</button>
                </div>
            </div>
        `;
    }
    
    bindEvents() {
        if (!this.panel) return;
        
        // Close button
        const closeBtn = this.panel.querySelector('.pm-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.close();
            });
        }
        
        // Export button
        const exportBtn = this.panel.querySelector(`#report-btn-${this.sectionId}-export`);
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportMarkdown());
        }

        // Copy button
        const copyBtn = this.panel.querySelector(`#report-btn-${this.sectionId}-copy`);
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyMarkdown());
        }        
        
        // Bring to front on mousedown
        this.panel.addEventListener('mousedown', () => this.bringToFront());
    }
    
    initMoveable() {
        if (typeof Moveable === 'undefined') {
            console.warn('[ReportViewer] Moveable not found, using fallback drag');
            this.initFallbackDrag();
            return;
        }
        
        const headerEl = this.panel.querySelector('h1');
        const padding = 10;
        
        // Normalize positioning
        const rect = this.panel.getBoundingClientRect();
        this.panel.style.top = `${rect.top}px`;
        this.panel.style.left = `${rect.left}px`;
        this.panel.style.bottom = 'auto';
        this.panel.style.right = 'auto';
        this.panel.style.margin = '0';
        this.panel.style.transform = 'translate(0px, 0px)';
        
        this.moveable = new Moveable(document.body, {
            target: this.panel,
            draggable: true,
            resizable: true,
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
        
        this.moveable
            .on('dragStart', e => {
                const t = e.inputEvent && e.inputEvent.target;
                allowDrag = !!(headerEl && t && (t === headerEl || headerEl.contains(t)));
                if (!allowDrag) {
                    e.stop && e.stop();
                    return;
                }
                if (e.set) e.set([this.position.x, this.position.y]);
                e.inputEvent.stopPropagation();
            })
            .on('drag', e => {
                if (!allowDrag) return;
                const [x, y] = e.beforeTranslate;
                this.position.x = x;
                this.position.y = y;
                e.target.style.transform = `translate(${x}px, ${y}px)`;
            })
            .on('dragEnd', () => {
                allowDrag = false;
            })
            .on('resizeStart', e => {
                e.setOrigin(['%', '%']);
                if (e.dragStart) e.dragStart.set([this.position.x, this.position.y]);
            })
            .on('resize', e => {
                const { target, width, height, drag } = e;
                target.style.width = `${width}px`;
                target.style.height = `${height}px`;
                const [x, y] = drag.beforeTranslate;
                target.style.transform = `translate(${x}px, ${y}px)`;
                this.position.x = x;
                this.position.y = y;
            })
            .on('resizeEnd', () => {
                this.moveable.updateRect();
                this.applyControlStyles();
            });
        
        this.moveable.updateRect();
        this.applyControlStyles();
        this.initOcclusionCheck();       
    }

    applyControlStyles() {
        if (!this.moveable) return;
        
        const box = document.querySelectorAll('.moveable-control-box');
        const controlBox = box[box.length - 1];
        
        if (!controlBox) return;
        
        const controls = controlBox.querySelectorAll('.moveable-control');
        const { width, height } = this.moveable.getRect();
        
        controls.forEach(control => {
            control.classList.add('custom-control');
            
            if (control.classList.contains('moveable-n') || control.classList.contains('moveable-s')) {
                control.style.width = `${width}px`;
                control.style.marginLeft = `-${width / 2}px`;
            }
            if (control.classList.contains('moveable-w') || control.classList.contains('moveable-e')) {
                control.style.height = `${height}px`;
                control.style.marginTop = `-${height / 2}px`;
            }
        });
    }    
    
    initFallbackDrag() {
        // Simple fallback drag without Moveable
        const header = this.panel.querySelector('h1');
        if (!header) return;
        
        let isDragging = false;
        let startX, startY, initialX, initialY;
        
        header.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('pm-close')) return;
            
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            initialX = this.panel.offsetLeft;
            initialY = this.panel.offsetTop;
            
            header.style.cursor = 'grabbing';
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            
            this.panel.style.left = `${initialX + dx}px`;
            this.panel.style.top = `${initialY + dy}px`;
        });
        
        document.addEventListener('mouseup', () => {
            if (isDragging) {
                isDragging = false;
                header.style.cursor = 'move';
            }
        });
        
        header.style.cursor = 'move';
    }
    
    bringToFront() {
        if (this.panel) {
            this.panel.style.zIndex = String(getTopZ());
            
            // Sync moveable control box z-index
            if (this.moveable && this.moveable.selfElement) {
                this.moveable.selfElement.style.zIndex = this.panel.style.zIndex;
            }
        }
    }
    
    exportMarkdown() {
        const blob = new Blob([this.markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `benchmark_report_${this.sectionId}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        URL.revokeObjectURL(url);
    }

    copyMarkdown() {
        navigator.clipboard.writeText(this.markdown).then(() => {
            // Brief visual feedback
            const copyBtn = this.panel.querySelector(`#${this.panelId}-copy`);
            if (copyBtn) {
                const original = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => copyBtn.textContent = original, 1500);
            }
        }).catch(err => {
            console.error('[ReportViewer] Failed to copy:', err);
        });
    } 
    
    initOcclusionCheck() {
        // Store the handler so we can remove it on close
        this.occlusionHandler = (e) => {
            if (!this.moveable || !this.panel) return;
            
            const controlBox = this.moveable.selfElement;
            if (!controlBox) return;
            
            const isOccluded = this.isOccludedByHigherPanel(e.clientX, e.clientY);
            
            controlBox.querySelectorAll('.moveable-control').forEach(ctrl => {
                if (isOccluded) {
                    ctrl.style.pointerEvents = 'none';
                    ctrl.style.cursor = 'default';
                } else {
                    ctrl.style.pointerEvents = '';
                    ctrl.style.cursor = '';
                }
            });
        };
        
        document.addEventListener('mousemove', this.occlusionHandler);
    }

    isOccludedByHigherPanel(clientX, clientY) {
        if (!this.panel) return false;
        
        const panelZ = parseInt(this.panel.style.zIndex) || 0;
        const elements = document.elementsFromPoint(clientX, clientY);
        
        for (const el of elements) {
            // If we hit our panel first, we're not occluded
            if (el === this.panel) return false;
            
            // Check if this is another panel with higher z-index
            if (el.classList.contains('hud') && el !== this.panel) {
                const elZ = parseInt(el.style.zIndex) || 0;
                if (elZ > panelZ) {
                    return true;
                }
            }
        }
        return false;
    }    
    
    close() {
        // Remove occlusion handler
        if (this.occlusionHandler) {
            document.removeEventListener('mousemove', this.occlusionHandler);
            this.occlusionHandler = null;
        }
        
        // Destroy moveable
        if (this.moveable) {
            this.moveable.destroy();
            this.moveable = null;
        }
        
        // Remove panel
        if (this.panel) {
            this.panel.classList.remove('visible');
            this.panel.remove();
            this.panel = null;
        }
        
        // Unregister
        ReportViewerPanel.instances.delete(this.sectionId);
    }
    
    static closeAll() {
        for (const panel of ReportViewerPanel.instances.values()) {
            panel.close();
        }
    }
}

// Attach to window for global access
window.ReportViewerPanel = ReportViewerPanel;

export default ReportViewerPanel;
