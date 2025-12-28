// Add event listeners to tabs
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        const tabId = tab.getAttribute('data-tab');
        
        // 1. Find the specific HUD panel (Settings or Metrics) this tab belongs to
        const parentPanel = tab.closest('.hud');
        if (!parentPanel) return;

        // 2. Remove active class ONLY from tabs and content inside this panel
        parentPanel.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        parentPanel.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        
        // 3. Add active class to the clicked tab
        tab.classList.add('active');
        
        // 4. Find the content within this panel using the ID
        // Note: Using parentPanel.querySelector ensures we don't accidentally grab content from another panel
        const content = parentPanel.querySelector(`#${tabId}-content`);
        if (content) {
            content.classList.add('active');
        }
    });
});
