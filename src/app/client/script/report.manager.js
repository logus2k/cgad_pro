// report.manager.js

export class ReportManager {
    constructor() {
        this.editor = document.getElementById('report-editor');
        this.preview = document.getElementById('report-preview');
        this.init();
    }

    init() {
        if (!this.editor) return;
        this.editor.addEventListener('input', () => this.render());
        
        // Initial Template
        this.editor.value = "# Simulation Report\n\n## Summary\n* **Model:** ...\n* **Status:** Complete\n\nInsert findings here...";
        this.render();
    }

    render() {
        const rawText = this.editor.value;
        // Use the global 'marked' object provided by the CDN script
        if (window.marked) {
            this.preview.innerHTML = window.marked.parse(rawText);
        } else {
            // Fallback if library isn't loaded yet
            this.preview.innerText = rawText;
        }
    }
}