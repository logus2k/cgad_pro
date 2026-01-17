/**
 * TimelineAxis - Renders time axis labels.
 * 
 * Uses Canvas2D for crisp text rendering.
 * Auto-scales time units (ns, us, ms, s).
 * 
 * Location: /src/app/client/script/profiling/timeline_axis.js
 */

export class TimelineAxis {
    
    #canvas;
    #ctx;
    #height;
    #timeRange = { start: 0, end: 1000 };
    
    // Styling
    #bgColor = '#ffffff';
    #textColor = '#333333';
    #lineColor = '#cccccc';
    #font = '11px system-ui, sans-serif';
    
    constructor(canvas, height = 30) {
        this.#canvas = canvas;
        this.#ctx = canvas.getContext('2d');
        this.#height = height;
        this.#canvas.height = height;
    }
    
    /**
     * Update visible time range.
     */
    setTimeRange(start, end) {
        this.#timeRange = { start, end };
    }
    
    /**
     * Resize canvas width.
     */
    resize(width) {
        const dpr = window.devicePixelRatio || 1;
        const newWidth = Math.round(width * dpr);
        const newHeight = Math.round(this.#height * dpr);
        
        // Only resize if dimensions actually changed (setting width/height clears canvas)
        if (this.#canvas.width === newWidth && this.#canvas.height === newHeight) {
            return;
        }
        
        // Set canvas buffer size (actual pixels)
        this.#canvas.width = newWidth;
        this.#canvas.height = newHeight;
        
        // Set display size (CSS pixels)
        this.#canvas.style.width = `${width}px`;
        this.#canvas.style.height = `${this.#height}px`;
        
        // Store logical dimensions
        this.#logicalWidth = width;
    }
    
    #logicalWidth = 0;
    
    /**
     * Render axis.
     */
    render() {
        const ctx = this.#ctx;
        const dpr = window.devicePixelRatio || 1;
        const width = this.#logicalWidth || this.#canvas.width / dpr;
        const height = this.#height;
        
        // Reset transform and clear
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        
        // Clear
        ctx.fillStyle = this.#bgColor;
        ctx.fillRect(0, 0, width, height);
        
        // Bottom border
        ctx.strokeStyle = this.#lineColor;
        ctx.beginPath();
        ctx.moveTo(0, height - 0.5);
        ctx.lineTo(width, height - 0.5);
        ctx.stroke();
        
        const timeSpan = this.#timeRange.end - this.#timeRange.start;
        if (timeSpan <= 0) return;
        
        // Determine tick interval and format
        const { interval, unit, format } = this.#computeTickInterval(timeSpan, width);
        
        // Calculate first tick
        const firstTick = Math.ceil(this.#timeRange.start / interval) * interval;
        
        ctx.fillStyle = this.#textColor;
        ctx.font = this.#font;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Draw ticks
        for (let t = firstTick; t <= this.#timeRange.end; t += interval) {
            const x = ((t - this.#timeRange.start) / timeSpan) * width;
            
            // Tick line
            ctx.strokeStyle = this.#lineColor;
            ctx.beginPath();
            ctx.moveTo(x, height - 8);
            ctx.lineTo(x, height);
            ctx.stroke();
            
            // Label
            const label = format(t, unit);
            ctx.fillText(label, x, height / 2);
        }
        
        // Draw minor ticks
        const minorInterval = interval / 5;
        const firstMinorTick = Math.ceil(this.#timeRange.start / minorInterval) * minorInterval;
        
        ctx.strokeStyle = '#e0e0e0';
        for (let t = firstMinorTick; t <= this.#timeRange.end; t += minorInterval) {
            // Skip major ticks
            if (Math.abs(t % interval) < minorInterval * 0.1) continue;
            
            const x = ((t - this.#timeRange.start) / timeSpan) * width;
            ctx.beginPath();
            ctx.moveTo(x, height - 4);
            ctx.lineTo(x, height);
            ctx.stroke();
        }
    }
    
    #computeTickInterval(timeSpanMs, widthPx) {
        // Target ~100px between ticks
        const targetIntervalMs = (timeSpanMs / widthPx) * 100;
        
        // Nice intervals in ms
        const niceIntervals = [
            0.000001, 0.000002, 0.000005,  // nanoseconds
            0.00001, 0.00002, 0.00005,      // 10s of ns
            0.0001, 0.0002, 0.0005,         // 100s of ns
            0.001, 0.002, 0.005,            // microseconds
            0.01, 0.02, 0.05,               // 10s of us
            0.1, 0.2, 0.5,                  // 100s of us
            1, 2, 5,                        // milliseconds
            10, 20, 50,                     // 10s of ms
            100, 200, 500,                  // 100s of ms
            1000, 2000, 5000,               // seconds
            10000, 20000, 50000,            // 10s of s
            100000, 200000, 500000          // 100s of s
        ];
        
        // Find best interval
        let interval = niceIntervals[0];
        for (const ni of niceIntervals) {
            if (ni >= targetIntervalMs) {
                interval = ni;
                break;
            }
            interval = ni;
        }
        
        // Determine unit and format
        let unit, format;
        
        if (interval < 0.001) {
            unit = 'ns';
            format = (t) => `${(t * 1e6).toFixed(0)} ns`;
        } else if (interval < 1) {
            unit = 'us';
            format = (t) => `${(t * 1000).toFixed(interval < 0.01 ? 1 : 0)} us`;
        } else if (interval < 1000) {
            unit = 'ms';
            format = (t) => `${t.toFixed(interval < 10 ? 1 : 0)} ms`;
        } else {
            unit = 's';
            format = (t) => `${(t / 1000).toFixed(interval < 10000 ? 1 : 0)} s`;
        }
        
        return { interval, unit, format };
    }
    
    /**
     * Cleanup.
     */
    destroy() {
        // Nothing to cleanup for Canvas2D
    }
}
