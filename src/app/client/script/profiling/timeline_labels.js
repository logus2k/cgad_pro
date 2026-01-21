/**
 * TimelineLabels - Renders text labels on NVTX range bars.
 * 
 * Uses Canvas2D overlay for crisp text rendering.
 * Labels are shown/hidden based on available bar width.
 * 
 * Location: /src/app/client/script/profiling/timeline_labels.js
 */

export class TimelineLabels {
    
    #canvas;
    #ctx;
    #resolution = { width: 800, height: 600 };
    #timeRange = { start: 0, end: 1000 };
    
    // Label data: array of { name, startMs, durationMs, x, y, width, height }
    #labels = [];
    
    // Styling
    #font = '11px system-ui, -apple-system, sans-serif';
    #fontBold = 'bold 11px system-ui, -apple-system, sans-serif';
    #textColor = '#ffffff';
    #textShadowColor = 'rgba(0, 0, 0, 0.5)';
    #minBarWidthForDuration = 50;  // Minimum bar width to show "[duration]"
    #minBarWidthForName = 100;     // Minimum bar width to show full "name [duration]"
    #padding = 8;                  // Horizontal padding inside bar
    
    constructor(canvas) {
        this.#canvas = canvas;
        this.#ctx = canvas.getContext('2d');
    }
    
    /**
     * Resize canvas to match timeline dimensions.
     */
    resize(width, height) {
        const dpr = window.devicePixelRatio || 1;
        
        this.#canvas.width = Math.round(width * dpr);
        this.#canvas.height = Math.round(height * dpr);
        this.#canvas.style.width = `${width}px`;
        this.#canvas.style.height = `${height}px`;
        
        this.#resolution = { width, height };
    }
    
    /**
     * Update visible time range.
     */
    setTimeRange(start, end) {
        this.#timeRange = { start, end };
    }
    
    /**
     * Set label data for NVTX ranges.
     * @param {Array} labels - Array of { name, startMs, durationMs, rowIndex, rowHeight, stackLevel, maxStackLevels }
     */
    setLabels(labels) {
        this.#labels = labels;
    }
    
    /**
     * Render labels on visible NVTX bars.
     */
    render() {
        const ctx = this.#ctx;
        const dpr = window.devicePixelRatio || 1;
        const width = this.#resolution.width;
        const height = this.#resolution.height;
        
        // Reset transform and clear
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        if (timeSpan <= 0 || this.#labels.length === 0) return;
        
        const pxPerMs = width / timeSpan;
        
        ctx.font = this.#fontBold;
        ctx.textBaseline = 'middle';
        
        for (const label of this.#labels) {
            // Calculate bar position and dimensions in pixels
            const barX = (label.startMs - this.#timeRange.start) * pxPerMs;
            const barWidth = label.durationMs * pxPerMs;
            
            // Skip if bar is outside visible area
            if (barX + barWidth < 0 || barX > width) continue;
            
            // Skip if bar is too narrow for any label
            if (barWidth < this.#minBarWidthForDuration) continue;
            
            // Calculate bar Y position
            const rowHeight = label.rowHeight;
            const maxStackLevels = label.maxStackLevels || 1;
            const barHeight = Math.max((rowHeight * 0.85) / maxStackLevels, 1);
            const barY = label.rowIndex * rowHeight + rowHeight * 0.075 + (label.stackLevel || 0) * barHeight;
            const barCenterY = barY + barHeight / 2;
            
            // Format duration
            const durationStr = this.#formatDuration(label.durationMs);
            
            // Determine what to show based on available width
            const fullLabel = `${label.name} [${durationStr}]`;
            const durationLabel = `[${durationStr}]`;
            
            const fullWidth = ctx.measureText(fullLabel).width;
            const durationWidth = ctx.measureText(durationLabel).width;
            
            let textToRender = null;
            
            if (barWidth >= fullWidth + this.#padding * 2) {
                // Full label fits
                textToRender = fullLabel;
            } else if (barWidth >= this.#minBarWidthForName) {
                // Try truncated name + duration
                const availableWidth = barWidth - this.#padding * 2 - durationWidth - 4;
                if (availableWidth > 20) {
                    const truncatedName = this.#truncateText(ctx, label.name, availableWidth);
                    textToRender = `${truncatedName} [${durationStr}]`;
                } else {
                    textToRender = durationLabel;
                }
            } else if (barWidth >= durationWidth + this.#padding * 2) {
                // Only duration fits
                textToRender = durationLabel;
            }
            
            if (textToRender) {
                // Calculate centered X position, clamped to visible area
                const textWidth = ctx.measureText(textToRender).width;
                let textX = barX + barWidth / 2;
                
                // Clamp text position to keep it visible
                const minX = Math.max(0, barX) + textWidth / 2 + 4;
                const maxX = Math.min(width, barX + barWidth) - textWidth / 2 - 4;
                textX = Math.max(minX, Math.min(maxX, textX));
                
                // Draw text with shadow for readability
                ctx.textAlign = 'center';
                
                // Shadow
                ctx.fillStyle = this.#textShadowColor;
                ctx.fillText(textToRender, textX + 1, barCenterY + 1);
                
                // Main text
                ctx.fillStyle = this.#textColor;
                ctx.fillText(textToRender, textX, barCenterY);
            }
        }
    }
    
    /**
     * Truncate text to fit within maxWidth, adding ellipsis.
     */
    #truncateText(ctx, text, maxWidth) {
        if (ctx.measureText(text).width <= maxWidth) {
            return text;
        }
        
        let truncated = text;
        while (truncated.length > 0 && ctx.measureText(truncated + '...').width > maxWidth) {
            truncated = truncated.slice(0, -1);
        }
        
        return truncated.length > 0 ? truncated + '...' : '';
    }
    
    /**
     * Format duration in appropriate units.
     */
    #formatDuration(ms) {
        if (ms < 0.001) {
            return `${(ms * 1e6).toFixed(0)} ns`;
        } else if (ms < 1) {
            return `${(ms * 1000).toFixed(1)} us`;
        } else if (ms < 1000) {
            return `${ms.toFixed(ms < 10 ? 2 : 1)} ms`;
        } else {
            return `${(ms / 1000).toFixed(3)} s`;
        }
    }
    
    /**
     * Cleanup.
     */
    destroy() {
        // Nothing to cleanup for Canvas2D
    }
}
