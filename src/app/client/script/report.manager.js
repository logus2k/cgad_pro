/**
 * Simple Report Manager to kickstart the area
 */
export class ReportManager {
    constructor() {
        this.editor = document.getElementById('report-editor');
        this.preview = document.getElementById('report-preview');
        this.init();
    }

    init() {
        if (!this.editor) return;

        // Listen for typing to update preview
        this.editor.addEventListener('input', () => {
            this.render();
        });

        // Initial template
        this.editor.value = "# Simulation Report\n\n## Overview\nUpdate your analysis here...";
        this.render();
    }

    render() {
        const rawText = this.editor.value;
        
        // If you include <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        // this.preview.innerHTML = marked.parse(rawText);
        
        // Simple fallback renderer for testing
        this.preview.innerHTML = this.simpleMarkdown(rawText);
    }

    simpleMarkdown(text) {
        return text
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/\*\*(.*)\*\*/gim, '<b>$1</b>')
            .replace(/\*(.*)\*/gim, '<i>$1</i>')
            .replace(/\n/gim, '<br>');
    }
}
