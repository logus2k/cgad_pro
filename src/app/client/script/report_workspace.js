/**
 * Report Workspace Panel
 * 
 * Simplified: each document is a single .md file.
 * Tree shows documents with headers parsed from markdown.
 * 
 * Location: /src/app/client/script/report_workspace.js
 */

import { getTopZ } from './zIndexManager.js';

export class ReportWorkspace {
    
    constructor(containerSelector, options = {}) {
        this.container = document.querySelector(containerSelector);
        if (!this.container) {
            console.error('[ReportWorkspace] Container not found:', containerSelector);
            return;
        }
        
        this.options = {
            apiBase: options.apiBase || '',
            ...options
        };
        
        // State
        this.mode = 'preview';
        this.documents = [];
        this.content = '';
        this.currentDocument = null;
        this.isDirty = false;
        this.tocVisible = true;
        
        // Components
        this.treeInstance = null;
        this.editorInstance = null;
        this.undockedPanel = null;
        
        // Initialize
        this.init();
    }
    
    async init() {
        // Configure marked with KaTeX extension
        if (window.marked && window.markedKatex) {
            marked.use(markedKatex({
                throwOnError: false
            }));
        }
        
        this.render();
        this.bindEvents();
        await this.loadDocuments();
        this.buildDocumentTree();
    }
    
    render() {
        this.container.innerHTML = `
            <div class="report-workspace-body">
                <div class="report-toc-sidebar">
                    <div class="report-toc-tree" id="report-toc-tree"></div>
                </div>
                <div class="report-main-area">
                    <div class="report-preview-area" id="report-preview-area">
                        <p class="report-placeholder">Select a document to view.</p>
                    </div>
                    <div class="report-editor-area" id="report-editor-area" style="display:none;">
                        <textarea id="report-editor-textarea"></textarea>
                    </div>
                </div>
            </div>
            <div class="report-workspace-footer">
                <button class="report-btn report-btn-toggle-toc" id="report-btn-toggle-toc" title="Toggle Table of Contents">
                    <img src="./icons/left_panel_close.svg" alt="Toggle TOC" id="report-toggle-toc-icon">
                </button>
                <div class="report-footer-spacer"></div>
                <button class="report-btn report-btn-edit" id="report-btn-edit" style="display:none;">Edit</button>
                <button class="report-btn report-btn-preview" id="report-btn-preview" style="display:none;">Preview</button>
                <button class="report-btn report-btn-undock" id="report-btn-undock" style="display:none;">Undock</button>
                <button class="report-btn report-btn-save" id="report-btn-save" style="display:none;">Save</button>
            </div>
        `;
        
        this.btnEdit = this.container.querySelector('#report-btn-edit');
        this.btnPreview = this.container.querySelector('#report-btn-preview');
        this.btnUndock = this.container.querySelector('#report-btn-undock');
        this.btnSave = this.container.querySelector('#report-btn-save');
        this.btnToggleToc = this.container.querySelector('#report-btn-toggle-toc');
        this.toggleTocIcon = this.container.querySelector('#report-toggle-toc-icon');
        this.previewArea = this.container.querySelector('#report-preview-area');
        this.editorArea = this.container.querySelector('#report-editor-area');
        this.tocTree = this.container.querySelector('#report-toc-tree');
        this.tocSidebar = this.container.querySelector('.report-toc-sidebar');
    }
    
    bindEvents() {
        this.btnEdit.addEventListener('click', () => this.enterEditMode());
        this.btnPreview.addEventListener('click', () => this.enterPreviewMode());
        this.btnUndock.addEventListener('click', () => this.undockEditor());
        this.btnSave.addEventListener('click', () => this.saveContent());
        this.btnToggleToc.addEventListener('click', () => this.toggleToc());
    }
    
    toggleToc() {
        this.tocVisible = !this.tocVisible;
        
        if (this.tocVisible) {
            this.tocSidebar.classList.remove('collapsed');
            this.toggleTocIcon.src = './icons/left_panel_close.svg';
            this.btnToggleToc.title = 'Hide Table of Contents';
        } else {
            this.tocSidebar.classList.add('collapsed');
            this.toggleTocIcon.src = './icons/left_panel_open.svg';
            this.btnToggleToc.title = 'Show Table of Contents';
        }
    }
    
    async loadDocuments() {
        try {
            const response = await fetch(`${this.options.apiBase}/api/report/sections`);
            const data = await response.json();
            
            if (data.error) {
                console.error('[ReportWorkspace] Config error:', data.error);
                this.previewArea.innerHTML = `<p class="report-error">${data.error}</p>`;
                return;
            }
            
            this.documents = data.documents || [];
        } catch (error) {
            console.error('[ReportWorkspace] Failed to load documents:', error);
            this.previewArea.innerHTML = `<p class="report-error">Failed to load: ${error.message}</p>`;
        }
    }
    
    buildDocumentTree() {
        const treeData = this.documents.map(doc => ({
            title: doc.title,
            key: doc.id,
            expanded: false,
            folder: true,
            children: []
        }));
        
        this.initTree(treeData);
        
        // Auto-select and load first document (but don't auto-expand)
        if (this.documents.length > 0) {
            const firstDoc = this.documents[0];
            const firstNode = this.treeInstance?.findKey(firstDoc.id);
            if (firstNode) {
                firstNode.setActive(true);
                this.loadDocument(firstDoc.id, firstNode);
            }
        }
    }
    
    initTree(data) {
        if (typeof mar10 === 'undefined' || typeof mar10.Wunderbaum === 'undefined') {
            console.warn('[ReportWorkspace] Wunderbaum not loaded');
            return;
        }
        
        this.treeInstance = new mar10.Wunderbaum({
            element: this.tocTree,
            source: data,
            iconMap: {
                folder: "fa-solid fa-folder",
                folderOpen: "fa-solid fa-folder-open",
                doc: "fa-solid fa-folder",
                expanderExpanded: "fa-solid fa-chevron-down",
                expanderCollapsed: "fa-solid fa-chevron-right",
            },
            header: false,
            connectLines: true,
            click: (e) => this.onTreeClick(e)
        });
    }
    
    async onTreeClick(e) {
        // Ignore clicks on the expander - let Wunderbaum handle expand/collapse
        if (e.targetType === 'expander') {
            return;
        }
        
        const node = e.node;
        const key = node.key;
        
        // Header - scroll to it
        if (key.startsWith('header:')) {
            const parts = key.split(':');
            const docId = parts[1];
            const index = parts[2];
            
            // If different document, load it first
            if (docId !== this.currentDocument) {
                const docNode = this.treeInstance.findKey(docId);
                if (docNode) {
                    await this.loadDocument(docId, docNode);
                }
            }
            
            // Scroll to the header within main area
            this.scrollToHeader(index);
            return;
        }
        
        // Document - load if different
        if (key !== this.currentDocument) {
            await this.loadDocument(key, node);
        }
        
        // Scroll to top
        this.scrollToHeader(0);
    }
    
    scrollToHeader(index) {
        const mainArea = this.container.querySelector('.report-main-area');
        const target = document.getElementById(`report-heading-${index}`);
        
        if (mainArea && target) {
            const containerRect = mainArea.getBoundingClientRect();
            const targetRect = target.getBoundingClientRect();
            const offset = targetRect.top - containerRect.top + mainArea.scrollTop;
            mainArea.scrollTo({ top: offset, behavior: 'smooth' });
        }
    }
    
    async loadDocument(documentId, node) {
        this.currentDocument = documentId;
        
        try {
            const url = `${this.options.apiBase}/api/report/content?document=${documentId}`;
            const response = await fetch(url);
            const data = await response.json();
            this.content = data.content || '';
        } catch (error) {
            console.error('[ReportWorkspace] Failed to load document:', error);
            this.content = `*Error loading: ${error.message}*`;
        }
        
        this.renderPreview();
        this.updateNodeHeaders(node);
        this.showEditButton();
    }
    
    renderPreview() {
        // Dispose any existing charts before replacing content
        if (window.ChartUtils && this.previewArea) {
            window.ChartUtils.disposeCharts(this.previewArea);
        }
        
        let html = this.content;
        if (window.marked) {
            html = window.marked.parse(this.content);
        }
        
        this.previewArea.innerHTML = html;
        
        // Initialize any charts in the preview
        if (window.ChartUtils && this.previewArea) {
            window.ChartUtils.initializeCharts(this.previewArea);
        }
        
        const headers = this.previewArea.querySelectorAll('h1, h2, h3, h4, h5, h6');
        headers.forEach((h, i) => {
            h.id = `report-heading-${i}`;
        });
    }
    
    updateNodeHeaders(node) {
        if (!node) return;
        
        // Skip if headers already loaded for this node
        if (node.children && node.children.length > 0) {
            return;
        }
        
        const tempDiv = document.createElement('div');
        if (window.marked) {
            tempDiv.innerHTML = window.marked.parse(this.content);
        }
        
        const headers = Array.from(tempDiv.querySelectorAll('h1, h2, h3, h4, h5, h6'));
        
        // Build hierarchical tree structure based on header levels
        const headerNodes = this.buildHeaderTree(headers);
        
        if (headerNodes.length > 0) {
            node.addChildren(headerNodes);
        }
    }
    
    buildHeaderTree(headers, docId = null) {
        const documentId = docId || this.currentDocument;
        const root = [];
        const stack = [{ level: 0, children: root }];
        
        headers.forEach((h, i) => {
            const level = parseInt(h.tagName.substring(1));
            const node = {
                title: h.innerText,
                key: `header:${documentId}:${i}`,
                expanded: false,
                children: []
            };
            
            while (stack.length > 1 && stack[stack.length - 1].level >= level) {
                stack.pop();
            }
            
            stack[stack.length - 1].children.push(node);
            stack.push({ level, children: node.children });
        });
        
        // Mark nodes with children as folders so they show expanders
        const markFolders = (nodes) => {
            for (const node of nodes) {
                if (node.children && node.children.length > 0) {
                    node.folder = true;
                    markFolders(node.children);
                }
            }
        };
        markFolders(root);
        
        return root;
    }
    
    showEditButton() {
        this.btnEdit.style.display = '';
    }
    
    hideAllButtons() {
        this.btnEdit.style.display = 'none';
        this.btnPreview.style.display = 'none';
        this.btnUndock.style.display = 'none';
        this.btnSave.style.display = 'none';
    }
    
    enterEditMode() {
        if (!this.currentDocument) return;
        
        this.mode = 'editor';
        this.previewArea.style.display = 'none';
        this.editorArea.style.display = '';
        
        this.btnEdit.style.display = 'none';
        this.btnPreview.style.display = '';
        this.btnUndock.style.display = '';
        this.btnSave.style.display = '';
        
        if (!this.editorInstance) {
            this.initEditor();
        }
        
        this.editorInstance.value(this.content);
    }
    
    enterPreviewMode() {
        this.mode = 'preview';
        
        if (this.editorInstance) {
            this.content = this.editorInstance.value();
        }
        
        this.editorArea.style.display = 'none';
        this.previewArea.style.display = '';
        
        this.btnPreview.style.display = 'none';
        this.btnUndock.style.display = 'none';
        this.btnSave.style.display = 'none';
        this.btnEdit.style.display = '';
        
        this.renderPreview();
        
        const activeNode = this.treeInstance?.getActiveNode();
        if (activeNode && !activeNode.key.startsWith('header:')) {
            this.updateNodeHeaders(activeNode);
        }
    }
    
    initEditor() {
        const textarea = this.container.querySelector('#report-editor-textarea');
        if (!textarea || typeof EasyMDE === 'undefined') return;
        
        this.editorInstance = new EasyMDE({
            element: textarea,
            autoDownloadFontAwesome: false,
            spellChecker: false,
            status: false,
            toolbar: [
                "bold", "italic", "strikethrough", "|",
                "heading-1", "heading-2", "heading-3", "|",
                "code", "quote", "unordered-list", "ordered-list", "|",
                "link", "image", "|",
                "guide"
            ],
            previewRender: (plainText, preview) => {
                const html = marked.parse(plainText);
                setTimeout(() => {
                    if (window.ChartUtils) {
                        // EasyMDE's preview element
                        const previewEl = preview || document.querySelector('.editor-preview-active');
                        if (previewEl) {
                            window.ChartUtils.initializeCharts(previewEl);
                        }
                    }
                }, 10);
                return html;
            }
        });
        
        this.editorInstance.codemirror.on("change", () => {
            this.isDirty = true;
        });
    }
    
    async saveContent() {
        if (!this.currentDocument) return;
        
        const content = this.editorInstance ? this.editorInstance.value() : this.content;
        
        try {
            this.btnSave.disabled = true;
            this.btnSave.textContent = 'Saving...';
            
            const response = await fetch(
                `${this.options.apiBase}/api/report/content?document=${this.currentDocument}`,
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                }
            );
            
            const result = await response.json();
            
            if (result.success) {
                this.content = content;
                this.isDirty = false;
                this.btnSave.textContent = 'Saved!';
                setTimeout(() => {
                    this.btnSave.textContent = 'Save';
                    this.btnSave.disabled = false;
                }, 1500);
                
                if (this.undockedPanel) {
                    this.undockedPanel.setContent(content);
                }
            } else {
                throw new Error(result.error || 'Save failed');
            }
        } catch (error) {
            console.error('[ReportWorkspace] Failed to save:', error);
            this.btnSave.textContent = 'Error!';
            this.btnSave.disabled = false;
            setTimeout(() => {
                this.btnSave.textContent = 'Save';
            }, 2000);
        }
    }
    
    undockEditor() {
        if (!this.currentDocument) return;
        
        const content = this.editorInstance ? this.editorInstance.value() : this.content;
        const doc = this.documents.find(d => d.id === this.currentDocument);
        const title = doc?.title || 'Editor';
        
        this.undockedPanel = new UndockedEditorPanel({
            documentId: this.currentDocument,
            documentTitle: title,
            content: content,
            apiBase: this.options.apiBase,
            onClose: () => {
                this.undockedPanel = null;
            },
            onSave: (newContent) => {
                this.content = newContent;
                if (this.mode === 'preview') {
                    this.renderPreview();
                    const activeNode = this.treeInstance?.getActiveNode();
                    if (activeNode && !activeNode.key.startsWith('header:')) {
                        this.updateNodeHeaders(activeNode);
                    }
                }
            }
        });
        
        this.enterPreviewMode();
    }
    
    refresh() {
        if (this.currentDocument) {
            const activeNode = this.treeInstance?.getActiveNode();
            this.loadDocument(this.currentDocument, activeNode);
        }
    }
    
    destroy() {
        if (this.editorInstance) {
            this.editorInstance.toTextArea();
            this.editorInstance = null;
        }
        if (this.undockedPanel) {
            this.undockedPanel.close();
            this.undockedPanel = null;
        }
    }
}


class UndockedEditorPanel {
    
    constructor(options) {
        this.documentId = options.documentId;
        this.documentTitle = options.documentTitle;
        this.content = options.content;
        this.apiBase = options.apiBase || '';
        this.onCloseCallback = options.onClose;
        this.onSaveCallback = options.onSave;
        
        this.panelId = `undocked-editor-${this.documentId}`;
        this.panel = null;
        this.editorInstance = null;
        this.moveable = null;
        this.position = { x: 0, y: 0 };
        
        this.create();
    }
    
    create() {
        const existing = document.getElementById(this.panelId);
        if (existing) existing.remove();
        
        this.panel = document.createElement('div');
        this.panel.id = this.panelId;
        this.panel.className = 'hud undocked-editor';
        this.panel.style.cssText = 'top:100px;left:200px;width:700px;height:500px;position:absolute;';
        
        this.panel.innerHTML = `
            <h1>EDITOR - ${this.documentTitle}</h1>
            <button class="pm-close" title="Close">&times;</button>
            <div class="undocked-editor-body">
                <div class="undocked-editor-container">
                    <textarea id="${this.panelId}-textarea"></textarea>
                </div>
                <div class="undocked-editor-footer">
                    <button class="report-btn report-btn-save" id="${this.panelId}-save">Save</button>
                </div>
            </div>
        `;
        
        const overlay = document.getElementById('overlay');
        (overlay || document.body).appendChild(this.panel);
        
        this.panel.classList.add('visible');
        this.panel.style.zIndex = String(getTopZ());
        
        this.bindEvents();
        this.initEditor();
        this.initMoveable();
    }
    
    bindEvents() {
        this.panel.querySelector('.pm-close').addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.close();
        });
        
        this.panel.querySelector(`#${this.panelId}-save`).addEventListener('click', () => this.save());
        
        this.panel.addEventListener('mousedown', () => {
            this.panel.style.zIndex = String(getTopZ());
            if (this.moveable?.selfElement) {
                this.moveable.selfElement.style.zIndex = this.panel.style.zIndex;
            }
        });
    }
    
    initEditor() {
        const textarea = this.panel.querySelector(`#${this.panelId}-textarea`);
        if (!textarea || typeof EasyMDE === 'undefined') return;
        
        const self = this;
        this.editorInstance = new EasyMDE({
            element: textarea,
            autoDownloadFontAwesome: false,
            spellChecker: false,
            status: false,
            toolbar: [
                "bold", "italic", "strikethrough", "|",
                "heading-1", "heading-2", "heading-3", "|",
                "code", "quote", "unordered-list", "ordered-list", "|",
                "link", "image", "|",
                "guide"
            ],
            previewRender: (plainText, preview) => {
                const html = marked.parse(plainText);
                setTimeout(() => {
                    if (window.ChartUtils) {
                        const previewEl = preview || self.panel.querySelector('.editor-preview-active');
                        if (previewEl) {
                            window.ChartUtils.initializeCharts(previewEl);
                        }
                    }
                }, 10);
                return html;
            }
        });
        
        this.editorInstance.value(this.content);
    }
    
    initMoveable() {
        if (typeof Moveable === 'undefined') return;
        
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
    
    initOcclusionCheck() {
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
            if (el === this.panel) return false;
            if (el.classList.contains('hud') && el !== this.panel) {
                const elZ = parseInt(el.style.zIndex) || 0;
                if (elZ > panelZ) return true;
            }
        }
        return false;
    }
    
    async save() {
        const content = this.editorInstance?.value() || this.content;
        const saveBtn = this.panel.querySelector(`#${this.panelId}-save`);
        
        try {
            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';
            
            const response = await fetch(
                `${this.apiBase}/api/report/content?document=${this.documentId}`,
                {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content })
                }
            );
            
            const result = await response.json();
            
            if (result.success) {
                this.content = content;
                saveBtn.textContent = 'Saved!';
                this.onSaveCallback?.(content);
                setTimeout(() => {
                    saveBtn.textContent = 'Save';
                    saveBtn.disabled = false;
                }, 1500);
            } else {
                throw new Error(result.error || 'Save failed');
            }
        } catch (error) {
            console.error('[UndockedEditor] Save failed:', error);
            saveBtn.textContent = 'Error!';
            saveBtn.disabled = false;
            setTimeout(() => { saveBtn.textContent = 'Save'; }, 2000);
        }
    }
    
    setContent(content) {
        this.content = content;
        this.editorInstance?.value(content);
    }
    
    close() {
        // Remove occlusion handler
        if (this.occlusionHandler) {
            document.removeEventListener('mousemove', this.occlusionHandler);
            this.occlusionHandler = null;
        }
        
        // Destroy moveable
        this.moveable?.destroy();
        this.moveable = null;
        
        // Cleanup editor
        this.editorInstance?.toTextArea();
        this.editorInstance = null;
        
        // Remove panel
        this.panel?.remove();
        this.panel = null;
        
        this.onCloseCallback?.();
    }
}


window.ReportWorkspace = ReportWorkspace;
export default ReportWorkspace;
