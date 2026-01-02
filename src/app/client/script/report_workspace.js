/**
 * Report Workspace Panel
 * 
 * A panel with TOC sidebar, live preview, and markdown editor modes.
 * Supports multiple documents with sections, and undocking the editor.
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
        this.mode = 'preview'; // 'preview' or 'editor'
        this.currentDocument = null;
        this.currentSection = null; // null means "all sections"
        this.documents = [];
        this.content = '';
        this.editable = false;
        this.isDirty = false;
        
        // Components
        this.treeInstance = null;
        this.editorInstance = null;
        this.undockedPanel = null;
        
        // Initialize
        this.init();
    }
    
    async init() {
        this.render();
        this.bindEvents();
        await this.loadDocuments();
    }
    
    render() {
        this.container.innerHTML = `
            <div class="report-workspace-header">
                <div class="report-workspace-controls">
                    <select class="report-section-select" id="report-section-select">
                        <option value="">Select a document...</option>
                    </select>
                </div>
                <div class="report-workspace-actions">
                    <button class="report-btn report-btn-edit" id="report-btn-edit" style="display:none;">Edit</button>
                    <button class="report-btn report-btn-preview" id="report-btn-preview" style="display:none;">Preview</button>
                    <button class="report-btn report-btn-undock" id="report-btn-undock" style="display:none;">Undock</button>
                    <button class="report-btn report-btn-save" id="report-btn-save" style="display:none;">Save</button>
                </div>
            </div>
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
        `;
        
        // Cache DOM references
        this.sectionSelect = this.container.querySelector('#report-section-select');
        this.btnEdit = this.container.querySelector('#report-btn-edit');
        this.btnPreview = this.container.querySelector('#report-btn-preview');
        this.btnUndock = this.container.querySelector('#report-btn-undock');
        this.btnSave = this.container.querySelector('#report-btn-save');
        this.previewArea = this.container.querySelector('#report-preview-area');
        this.editorArea = this.container.querySelector('#report-editor-area');
        this.tocTree = this.container.querySelector('#report-toc-tree');
    }
    
    bindEvents() {
        // Section selector
        this.sectionSelect.addEventListener('change', () => {
            this.onSelectionChange();
        });
        
        // Edit button
        this.btnEdit.addEventListener('click', () => {
            this.enterEditMode();
        });
        
        // Preview button
        this.btnPreview.addEventListener('click', () => {
            this.enterPreviewMode();
        });
        
        // Undock button
        this.btnUndock.addEventListener('click', () => {
            this.undockEditor();
        });
        
        // Save button
        this.btnSave.addEventListener('click', () => {
            this.saveContent();
        });
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
            
            // Populate dropdown with optgroups
            this.sectionSelect.innerHTML = '<option value="">Select a document...</option>';
            
            this.documents.forEach(doc => {
                const optgroup = document.createElement('optgroup');
                optgroup.label = doc.title;
                
                // Add "All Sections" option for multi-section documents
                if (doc.sections.length > 1) {
                    const allOption = document.createElement('option');
                    allOption.value = `${doc.id}:all`;
                    allOption.textContent = 'All Sections';
                    optgroup.appendChild(allOption);
                }
                
                // Add individual sections
                doc.sections.forEach(section => {
                    const option = document.createElement('option');
                    option.value = `${doc.id}:${section.id}`;
                    option.textContent = section.title;
                    optgroup.appendChild(option);
                });
                
                this.sectionSelect.appendChild(optgroup);
            });
            
        } catch (error) {
            console.error('[ReportWorkspace] Failed to load documents:', error);
            this.previewArea.innerHTML = `<p class="report-error">Failed to load documents: ${error.message}</p>`;
        }
    }
    
    async loadContent() {
        if (!this.currentDocument) {
            this.previewArea.innerHTML = '<p class="report-placeholder">Select a document to view.</p>';
            return;
        }
        
        try {
            let url = `${this.options.apiBase}/api/report/content?document=${this.currentDocument}`;
            if (this.currentSection && this.currentSection !== 'all') {
                url += `&section=${this.currentSection}`;
            }
            
            const response = await fetch(url);
            const data = await response.json();
            
            this.content = data.content || '';
            this.editable = data.editable || false;
            
            // Update UI based on editability
            this.updateEditButton();
            
            // Render preview
            this.renderPreview();
            
            // Update editor if in edit mode
            if (this.mode === 'editor' && this.editorInstance) {
                this.editorInstance.value(this.content);
            }
            
        } catch (error) {
            console.error('[ReportWorkspace] Failed to load content:', error);
            this.previewArea.innerHTML = `<p class="report-error">Failed to load content: ${error.message}</p>`;
        }
    }
    
    async saveContent() {
        if (!this.editable || !this.currentDocument || !this.currentSection || this.currentSection === 'all') {
            return;
        }
        
        const content = this.editorInstance ? this.editorInstance.value() : this.content;
        
        try {
            this.btnSave.disabled = true;
            this.btnSave.textContent = 'Saving...';
            
            const response = await fetch(
                `${this.options.apiBase}/api/report/content?document=${this.currentDocument}&section=${this.currentSection}`,
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
                
                // Also update undocked panel content if exists
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
    
    renderPreview() {
        // Convert markdown to HTML
        let html = this.content;
        if (window.marked) {
            html = window.marked.parse(this.content);
        }
        
        this.previewArea.innerHTML = html;
        
        // Add IDs to headers for TOC navigation
        const headers = this.previewArea.querySelectorAll('h1, h2, h3, h4, h5, h6');
        headers.forEach((h, i) => {
            h.id = `report-heading-${i}`;
        });
        
        // Build TOC tree
        this.buildTocTree(headers);
    }
    
    buildTocTree(headers) {
        const treeData = this.headersToTreeData(headers);
        
        if (!this.treeInstance) {
            this.initTree(treeData);
        } else {
            this.treeInstance.load(treeData);
        }
    }
    
    headersToTreeData(headers) {
        const root = [];
        const stack = [{ level: 0, children: root }];
        
        headers.forEach(h => {
            const level = parseInt(h.tagName.substring(1));
            const node = {
                title: h.innerText,
                key: h.id,
                expanded: true,
                children: []
            };
            
            while (stack.length > 1 && stack[stack.length - 1].level >= level) {
                stack.pop();
            }
            
            stack[stack.length - 1].children.push(node);
            stack.push({ level, children: node.children });
        });
        
        return root;
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
                doc: "fa-regular fa-file",
                expanderExpanded: "fa-solid fa-chevron-down",
                expanderCollapsed: "fa-solid fa-chevron-right",
            },
            header: false,
            connectLines: true,
            fixedCol: false,
            enhance: (e) => {
                if (e.node.children && e.node.children.length > 0) {
                    e.node.folder = true;
                }
            },
            click: (e) => {
                const target = document.getElementById(e.node.key);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        });
    }
    
    updateEditButton() {
        if (this.editable && this.currentSection && this.currentSection !== 'all') {
            this.btnEdit.style.display = '';
            this.btnEdit.disabled = false;
        } else {
            this.btnEdit.style.display = 'none';
        }
    }
    
    onSelectionChange() {
        const value = this.sectionSelect.value;
        
        if (!value) {
            this.currentDocument = null;
            this.currentSection = null;
            this.previewArea.innerHTML = '<p class="report-placeholder">Select a document to view.</p>';
            this.btnEdit.style.display = 'none';
            // Clear TOC
            if (this.treeInstance) {
                this.treeInstance.load([]);
            }
            return;
        }
        
        // Parse "document:section" value
        const [docId, sectionId] = value.split(':');
        this.currentDocument = docId;
        this.currentSection = sectionId;
        
        // If in editor mode and switching to 'all', switch to preview
        if (this.mode === 'editor' && this.currentSection === 'all') {
            this.enterPreviewMode();
        }
        
        this.loadContent();
    }
    
    enterEditMode() {
        if (!this.editable || !this.currentSection || this.currentSection === 'all') {
            return;
        }
        
        this.mode = 'editor';
        
        // Hide preview, show editor
        this.previewArea.style.display = 'none';
        this.editorArea.style.display = '';
        
        // Update buttons
        this.btnEdit.style.display = 'none';
        this.btnPreview.style.display = '';
        this.btnUndock.style.display = '';
        this.btnSave.style.display = '';
        
        // Initialize EasyMDE if not already
        if (!this.editorInstance) {
            this.initEditor();
        }
        
        // Set content
        this.editorInstance.value(this.content);
    }
    
    enterPreviewMode() {
        this.mode = 'preview';
        
        // Sync content from editor
        if (this.editorInstance) {
            this.content = this.editorInstance.value();
        }
        
        // Hide editor, show preview
        this.editorArea.style.display = 'none';
        this.previewArea.style.display = '';
        
        // Update buttons
        this.btnPreview.style.display = 'none';
        this.btnUndock.style.display = 'none';
        this.btnSave.style.display = 'none';
        this.updateEditButton();
        
        // Re-render preview with latest content
        this.renderPreview();
    }
    
    initEditor() {
        const textarea = this.container.querySelector('#report-editor-textarea');
        if (!textarea) return;
        
        if (typeof EasyMDE === 'undefined') {
            console.warn('[ReportWorkspace] EasyMDE not loaded');
            return;
        }
        
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
        });
        
        // Live update TOC on editor change
        this.editorInstance.codemirror.on("change", () => {
            this.isDirty = true;
            this.updateTocFromEditor();
        });
    }
    
    updateTocFromEditor() {
        if (!this.editorInstance) return;
        
        const markdown = this.editorInstance.value();
        
        // Parse markdown to HTML in a temp container
        const tempDiv = document.createElement('div');
        if (window.marked) {
            tempDiv.innerHTML = window.marked.parse(markdown);
        }
        
        // Get headers and build TOC
        const headers = tempDiv.querySelectorAll('h1, h2, h3, h4, h5, h6');
        headers.forEach((h, i) => {
            h.id = `report-heading-${i}`;
        });
        
        this.buildTocTree(headers);
    }
    
    undockEditor() {
        if (!this.editable || !this.currentDocument || !this.currentSection || this.currentSection === 'all') {
            return;
        }
        
        // Get current editor content
        const content = this.editorInstance ? this.editorInstance.value() : this.content;
        
        // Find section title
        const doc = this.documents.find(d => d.id === this.currentDocument);
        const section = doc?.sections.find(s => s.id === this.currentSection);
        const sectionTitle = section?.title || 'Editor';
        
        // Create undocked panel
        this.undockedPanel = new UndockedEditorPanel({
            documentId: this.currentDocument,
            sectionId: this.currentSection,
            sectionTitle: sectionTitle,
            content: content,
            apiBase: this.options.apiBase,
            onClose: () => {
                this.undockedPanel = null;
            },
            onSave: (newContent) => {
                this.content = newContent;
                if (this.mode === 'preview') {
                    this.renderPreview();
                }
            }
        });
        
        // Switch main panel back to preview
        this.enterPreviewMode();
    }
    
    // Public API
    refresh() {
        if (this.currentDocument) {
            this.loadContent();
        }
    }
    
    destroy() {
        if (this.editorInstance) {
            this.editorInstance.toTextArea();
            this.editorInstance = null;
        }
        if (this.treeInstance) {
            this.treeInstance = null;
        }
        if (this.undockedPanel) {
            this.undockedPanel.close();
            this.undockedPanel = null;
        }
    }
}


/**
 * Undocked Editor Panel
 * 
 * A standalone floating panel with just the markdown editor and save button.
 */
class UndockedEditorPanel {
    
    constructor(options) {
        this.documentId = options.documentId;
        this.sectionId = options.sectionId;
        this.sectionTitle = options.sectionTitle;
        this.content = options.content;
        this.apiBase = options.apiBase || '';
        this.onCloseCallback = options.onClose;
        this.onSaveCallback = options.onSave;
        
        this.panelId = `undocked-editor-${this.documentId}-${this.sectionId}`;
        this.panel = null;
        this.editorInstance = null;
        this.moveable = null;
        this.position = { x: 0, y: 0 };
        
        this.create();
    }
    
    create() {
        // Remove existing if any
        const existing = document.getElementById(this.panelId);
        if (existing) existing.remove();
        
        // Create panel
        this.panel = document.createElement('div');
        this.panel.id = this.panelId;
        this.panel.className = 'hud undocked-editor';
        this.panel.style.top = '100px';
        this.panel.style.left = '200px';
        this.panel.style.width = '700px';
        this.panel.style.height = '500px';
        this.panel.style.position = 'absolute';
        
        this.panel.innerHTML = `
            <h1>Editor: ${this.sectionTitle}</h1>
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
        
        // Append to overlay
        const overlay = document.getElementById('overlay');
        if (overlay) {
            overlay.appendChild(this.panel);
        } else {
            document.body.appendChild(this.panel);
        }
        
        // Show panel
        this.panel.classList.add('visible');
        this.panel.style.zIndex = String(getTopZ());
        
        // Bind events
        this.bindEvents();
        
        // Initialize editor
        this.initEditor();
        
        // Initialize moveable
        this.initMoveable();
    }
    
    bindEvents() {
        // Close button
        const closeBtn = this.panel.querySelector('.pm-close');
        closeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.close();
        });
        
        // Save button
        const saveBtn = this.panel.querySelector(`#${this.panelId}-save`);
        saveBtn.addEventListener('click', () => this.save());
        
        // Bring to front on click
        this.panel.addEventListener('mousedown', () => {
            this.panel.style.zIndex = String(getTopZ());
            if (this.moveable && this.moveable.selfElement) {
                this.moveable.selfElement.style.zIndex = this.panel.style.zIndex;
            }
        });
    }
    
    initEditor() {
        const textarea = this.panel.querySelector(`#${this.panelId}-textarea`);
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
        });
        
        this.editorInstance.value(this.content);
    }
    
    initMoveable() {
        if (typeof Moveable === 'undefined') return;
        
        const headerEl = this.panel.querySelector('h1');
        const padding = 10;
        
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
        const content = this.editorInstance ? this.editorInstance.value() : this.content;
        const saveBtn = this.panel.querySelector(`#${this.panelId}-save`);
        
        try {
            saveBtn.disabled = true;
            saveBtn.textContent = 'Saving...';
            
            const response = await fetch(
                `${this.apiBase}/api/report/content?document=${this.documentId}&section=${this.sectionId}`,
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
                
                // Notify main panel
                if (this.onSaveCallback) {
                    this.onSaveCallback(content);
                }
                
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
            setTimeout(() => {
                saveBtn.textContent = 'Save';
            }, 2000);
        }
    }
    
    setContent(content) {
        this.content = content;
        if (this.editorInstance) {
            this.editorInstance.value(content);
        }
    }
    
    close() {
        if (this.occlusionHandler) {
            document.removeEventListener('mousemove', this.occlusionHandler);
        }
        
        if (this.moveable) {
            this.moveable.destroy();
            this.moveable = null;
        }
        
        if (this.editorInstance) {
            this.editorInstance.toTextArea();
            this.editorInstance = null;
        }
        
        if (this.panel) {
            this.panel.remove();
            this.panel = null;
        }
        
        if (this.onCloseCallback) {
            this.onCloseCallback();
        }
    }
}


// Export
window.ReportWorkspace = ReportWorkspace;
export default ReportWorkspace;
