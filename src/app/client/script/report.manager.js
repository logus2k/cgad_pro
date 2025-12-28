import { marked } from "../library/marked.esm.js";

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
        this.editor.value = "";
        this.render();
    }

    render() {
        const rawText = this.editor.value;
        
        this.preview.innerHTML = marked.parse(rawText);
    }
}
