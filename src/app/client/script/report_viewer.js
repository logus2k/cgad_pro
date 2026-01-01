/**
 * Report Viewer Panel
 * 
 * Displays rendered Markdown report sections in a draggable/resizable panel.
 * Multiple instances can coexist for different sections.
 * 
 * Location: /src/app/client/script/report_viewer.js
 */

export class ReportViewerPanel {
    static instances = new Map();  // Track open panels by section ID
    static zIndexCounter = 1000;   // For bringing panels to front
    
    constructor(sectionId, options = {}) {
        this.sectionId = sectionId;
        this.options = {
            apiBase: options.apiBase || '',
            ...options
        };
        
        this.panelId = `report-panel-${sectionId}`;
        this.title = 'Report';
        this.markdown = '';
        this.panel = null;
    }
    
    async open() {
        // Check if panel already exists
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
        this.makeDraggable();
        this.makeResizable();
        
        // Register instance
        ReportViewerPanel.instances.set(this.sectionId, this);
        
        return this;
    }
    
    async fetchReport() {
        try {
            // Build query string from filters
            const params = new URLSearchParams();
            if (this.options.filters) {
                const { solver_type, model_name, server_hash } = this.options.filters;
                if (solver_type) params.append('solver_type', solver_type);
                if (model_name) params.append('model_name', model_name);
                if (server_hash) params.append('server_hash', server_hash);
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
        
        // Create panel element
        this.panel = document.createElement('div');
        this.panel.id = this.panelId;
        this.panel.className = 'report-viewer-panel';
        
        // Position with slight offset based on instance count
        const offset = ReportViewerPanel.instances.size * 30;
        this.panel.style.left = `${150 + offset}px`;
        this.panel.style.top = `${100 + offset}px`;
        
        document.body.appendChild(this.panel);
        this.bringToFront();
    }
    
    render() {
        if (!this.panel) return;
        
        // Render markdown to HTML
        let htmlContent = this.markdown;
        if (window.marked) {
            htmlContent = window.marked.parse(this.markdown);
        }
        
        this.panel.innerHTML = `
            <div class="report-viewer-header">
                <span class="report-viewer-title">${this.title}</span>
                <button class="report-viewer-close" title="Close">&times;</button>
            </div>
            <div class="report-viewer-content">
                ${htmlContent}
            </div>
            <div class="report-viewer-footer">
                <button class="benchmark-btn benchmark-btn-secondary" id="${this.panelId}-export">
                    Export .md
                </button>
            </div>
            <div class="report-viewer-resize-handle"></div>
        `;
    }
    
    bindEvents() {
        if (!this.panel) return;
        
        // Close button
        const closeBtn = this.panel.querySelector('.report-viewer-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.close());
        }
        
        // Export button
        const exportBtn = this.panel.querySelector(`#${this.panelId}-export`);
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportMarkdown());
        }
        
        // Bring to front on click
        this.panel.addEventListener('mousedown', () => this.bringToFront());
    }
    
    makeDraggable() {
        if (!this.panel) return;
        
        const header = this.panel.querySelector('.report-viewer-header');
        if (!header) return;
        
        let isDragging = false;
        let startX, startY, initialX, initialY;
        
        header.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('report-viewer-close')) return;
            
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
                header.style.cursor = 'grab';
            }
        });
        
        header.style.cursor = 'grab';
    }
    
    makeResizable() {
        if (!this.panel) return;
        
        const handle = this.panel.querySelector('.report-viewer-resize-handle');
        if (!handle) return;
        
        let isResizing = false;
        let startX, startY, startWidth, startHeight;
        
        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            startX = e.clientX;
            startY = e.clientY;
            startWidth = this.panel.offsetWidth;
            startHeight = this.panel.offsetHeight;
            
            e.preventDefault();
            e.stopPropagation();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            
            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            
            const newWidth = Math.max(300, startWidth + dx);
            const newHeight = Math.max(200, startHeight + dy);
            
            this.panel.style.width = `${newWidth}px`;
            this.panel.style.height = `${newHeight}px`;
        });
        
        document.addEventListener('mouseup', () => {
            isResizing = false;
        });
    }
    
    bringToFront() {
        if (this.panel) {
            ReportViewerPanel.zIndexCounter++;
            this.panel.style.zIndex = ReportViewerPanel.zIndexCounter;
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
    
    close() {
        if (this.panel) {
            this.panel.remove();
            this.panel = null;
        }
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
