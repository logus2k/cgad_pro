import { ReportManager } from './report.manager.js';

// Initialize and attach to window for global access if needed
window.reportManager = new ReportManager();

// Define the global helper functions used in the HTML buttons
window.clearReport = () => {
    const editor = document.getElementById('report-editor');
    if (editor) {
        editor.value = '';
        window.reportManager.render(); // Update preview
    }
};

window.exportReport = () => {
    const content = document.getElementById('report-preview').innerHTML;
    const blob = new Blob([content], { type: 'text/html' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'simulation_report.html';
    a.click();
};
