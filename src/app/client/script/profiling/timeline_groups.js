/**
 * TimelineGroups - Renders group labels sidebar.
 * 
 * Uses HTML for easy styling and interaction.
 * 
 * Location: /src/app/client/script/profiling/timeline_groups.js
 */

export class TimelineGroups {
    
    #container;
    #groups = [];
    #onToggle = null;
    
    // Category definitions for labels
    static CATEGORY_LABELS = {
        cuda_kernel: 'CUDA Kernels',
        cuda_memcpy_h2d: 'MemCpy H-D',
        cuda_memcpy_d2h: 'MemCpy D-H',
        cuda_memcpy_d2d: 'MemCpy D-D',
        cuda_sync: 'CUDA Sync',
        nvtx_range: 'NVTX Ranges'
    };
    
    constructor(container) {
        this.#container = container;
        this.#container.classList.add('timeline-groups');
        this.#applyStyles();
    }
    
    #applyStyles() {
        // Inject styles if not present
        if (!document.getElementById('timeline-groups-styles')) {
            const style = document.createElement('style');
            style.id = 'timeline-groups-styles';
            style.textContent = `
                .timeline-groups {
                    display: flex;
                    flex-direction: column;
                    background: #ffffff;
                    border-right: 1px solid #e0e0e0;
                    overflow: hidden;
                    user-select: none;
                }
                .timeline-group-item {
                    display: flex;
                    align-items: center;
                    padding: 0 14px;
                    border-bottom: 1px solid #f0f0f0;
                    box-sizing: border-box;
                    cursor: pointer;
                    transition: background 0.15s;
                }
                .timeline-group-item:hover {
                    background: #f5f5f5;
                }
                .timeline-group-color {
                    width: 12px;
                    height: 12px;
                    border-radius: 2px;
                    margin-right: 6px;
                    flex-shrink: 0;
                }
                .timeline-group-label {
                    font-size: 11px;
                    color: #333333;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                .timeline-group-item.disabled {
                    opacity: 0.4;
                }
                .timeline-group-item.disabled .timeline-group-color {
                    background: #aaa !important;
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    /**
     * Set groups to display.
     * @param {Array} groups - Array of {id, content, className}
     * @param {number} rowHeight - Height per row in pixels
     */
    setGroups(groups, rowHeight) {
        this.#groups = groups;
        this.#render(rowHeight);
    }
    
    /**
     * Set toggle callback.
     * @param {Function} callback - Called with (groupId, enabled)
     */
    onToggle(callback) {
        this.#onToggle = callback;
    }
    
    #render(rowHeight) {
        this.#container.innerHTML = '';
        
        for (const group of this.#groups) {
            const item = document.createElement('div');
            item.className = 'timeline-group-item';
            item.style.height = `${rowHeight}px`;
            item.dataset.groupId = group.id;
            
            // Color indicator
            const color = document.createElement('div');
            color.className = 'timeline-group-color';
            color.style.backgroundColor = this.#getGroupColor(group);
            item.appendChild(color);
            
            // Label
            const label = document.createElement('span');
            label.className = 'timeline-group-label';
            label.textContent = group.content || TimelineGroups.CATEGORY_LABELS[group.id] || group.id;
            item.appendChild(label);
            
            // Click to toggle
            item.addEventListener('click', () => {
                item.classList.toggle('disabled');
                if (this.#onToggle) {
                    this.#onToggle(group.id, !item.classList.contains('disabled'));
                }
            });
            
            this.#container.appendChild(item);
        }
    }
    
    #getGroupColor(group) {
        // Try to get color from CSS
        const el = document.createElement('div');
        el.className = group.className || `profiling-item-${group.id}`;
        el.style.display = 'none';
        document.body.appendChild(el);
        const color = getComputedStyle(el).backgroundColor;
        document.body.removeChild(el);
        
        if (color && color !== 'rgba(0, 0, 0, 0)') {
            return color;
        }
        
        // Fallback colors
        const defaults = {
            cuda_kernel: 'rgba(231, 76, 60, 0.9)',
            cuda_memcpy_h2d: 'rgba(52, 152, 219, 0.9)',
            cuda_memcpy_d2h: 'rgba(46, 204, 113, 0.9)',
            cuda_memcpy_d2d: 'rgba(243, 156, 18, 0.9)',
            cuda_sync: 'rgba(149, 165, 166, 0.9)',
            nvtx_range: 'rgba(155, 89, 182, 0.9)'
        };
        
        return defaults[group.id] || '#666';
    }
    
    /**
     * Update row heights.
     */
    updateRowHeight(rowHeight) {
        const items = this.#container.querySelectorAll('.timeline-group-item');
        items.forEach(item => {
            item.style.height = `${rowHeight}px`;
        });
    }
    
    /**
     * Cleanup.
     */
    destroy() {
        this.#container.innerHTML = '';
    }
}
