// perlin-horizon.js


class PerlinHorizon {

    constructor(options = {}) {
        this.width = options.width || 1000;
        this.height = options.height || 100;
        this.centerY = options.centerY || 50;
        this.amplitude = options.amplitude || 12.5;
        this.frequency = options.frequency || 0.008;
        this.step = options.step || 5;
        this.seed = options.seed ?? Math.random() * 10000;
        this.strokeColor = options.strokeColor || '#333333';
        this.strokeWidth = options.strokeWidth || 0.2;
        this.fillGradientBelow = options.fillGradientBelow || null;
        this.fillGradientAbove = options.fillGradientAbove || null;
        this.gap = options.gap || null; // { start, end } in pixels from left
        
        this.p = this._initPermutation();
    }

    _initPermutation() {
        const permutation = [];
        const random = this._seededRandom(this.seed);
        
        for (let i = 0; i < 256; i++) permutation[i] = i;
        for (let i = 255; i > 0; i--) {
            const j = Math.floor(random() * (i + 1));
            [permutation[i], permutation[j]] = [permutation[j], permutation[i]];
        }
        return [...permutation, ...permutation];
    }

    _seededRandom(seed) {
        return function() {
            seed = (seed * 9301 + 49297) % 233280;
            return seed / 233280;
        };
    }

    _fade(t) {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    _lerp(a, b, t) {
        return a + t * (b - a);
    }

    _grad(hash, x) {
        return (hash & 1) === 0 ? x : -x;
    }

    noise(x) {
        const xi = Math.floor(x) & 255;
        const xf = x - Math.floor(x);
        const u = this._fade(xf);
        return this._lerp(this._grad(this.p[xi], xf), this._grad(this.p[xi + 1], xf - 1), u);
    }

    _getY(x) {
        let y = this.centerY;
        y += this.noise(x * this.frequency) * this.amplitude;
        y += this.noise(x * this.frequency * 2.3 + 100) * (this.amplitude * 0.5);
        y += this.noise(x * this.frequency * 5.1 + 200) * (this.amplitude * 0.25);
        return y;
    }

    generatePoints() {
        const points = [];
        
        for (let x = 0; x <= this.width; x += this.step) {
            points.push({ x, y: this._getY(x) });
        }
        
        return points;
    }

    generateStrokePath() {
        const points = this.generatePoints();
        
        if (!this.gap) {
            let d = `M${points[0].x},${points[0].y.toFixed(1)}`;
            for (let i = 1; i < points.length; i++) {
                d += ` L${points[i].x},${points[i].y.toFixed(1)}`;
            }
            return d;
        }
        
        // With gap: create two separate path segments for stroke only
        let d = '';
        let inGap = false;
        let started = false;
        
        for (let i = 0; i < points.length; i++) {
            const p = points[i];
            const isInGap = p.x >= this.gap.start && p.x <= this.gap.end;
            
            if (!isInGap) {
                if (!started || inGap) {
                    d += ` M${p.x},${p.y.toFixed(1)}`;
                    started = true;
                } else {
                    d += ` L${p.x},${p.y.toFixed(1)}`;
                }
            }
            inGap = isInGap;
        }
        
        return d.trim();
    }

    generateClosedPathBelow(height) {
        const points = this.generatePoints();
        const bottomY = this.centerY + height;
        
        let d = `M${points[0].x},${points[0].y.toFixed(1)}`;
        for (let i = 1; i < points.length; i++) {
            d += ` L${points[i].x},${points[i].y.toFixed(1)}`;
        }
        d += ` L${this.width},${bottomY} L0,${bottomY} Z`;
        
        return d;
    }

    generateClosedPathAbove(height) {
        const points = this.generatePoints();
        const topY = this.centerY - height;
        
        let d = `M${points[0].x},${points[0].y.toFixed(1)}`;
        for (let i = 1; i < points.length; i++) {
            d += ` L${points[i].x},${points[i].y.toFixed(1)}`;
        }
        d += ` L${this.width},${topY} L0,${topY} Z`;
        
        return d;
    }

    applyTo(elementId) {
        const element = document.getElementById(elementId);
        const svg = element.closest('svg');
        
        // Remove previous elements if re-applying
        svg.querySelector(`#horizon-gradient-below-${this.seed}`)?.remove();
        svg.querySelector(`#horizon-gradient-above-${this.seed}`)?.remove();
        svg.querySelector('.horizon-fill-path-below')?.remove();
        svg.querySelector('.horizon-fill-path-above')?.remove();
        
        let defs = svg.querySelector('defs');
        if (!defs) {
            defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
            svg.insertBefore(defs, svg.firstChild);
        }
        
        // Create gradient below (ground)
        if (this.fillGradientBelow) {
            const gradientId = `horizon-gradient-below-${this.seed}`;
            const gradientHeight = this.fillGradientBelow.height || 100;
            
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.setAttribute('id', gradientId);
            gradient.setAttribute('x1', '0%');
            gradient.setAttribute('y1', '0%');
            gradient.setAttribute('x2', '0%');
            gradient.setAttribute('y2', '100%');
            
            const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop1.setAttribute('offset', '0%');
            stop1.setAttribute('stop-color', this.fillGradientBelow.color);
            stop1.setAttribute('stop-opacity', this.fillGradientBelow.opacity ?? 0.5);
            
            const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop2.setAttribute('offset', '100%');
            stop2.setAttribute('stop-color', this.fillGradientBelow.color);
            stop2.setAttribute('stop-opacity', '0');
            
            gradient.appendChild(stop1);
            gradient.appendChild(stop2);
            defs.appendChild(gradient);
            
            const fillPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            fillPath.setAttribute('class', 'horizon-fill-path-below');
            fillPath.setAttribute('d', this.generateClosedPathBelow(gradientHeight));
            fillPath.setAttribute('fill', `url(#${gradientId})`);
            fillPath.setAttribute('stroke', 'none');
            svg.insertBefore(fillPath, element);
        }
        
        // Create gradient above (sky)
        if (this.fillGradientAbove) {
            const gradientId = `horizon-gradient-above-${this.seed}`;
            const gradientHeight = this.fillGradientAbove.height || 100;
            
            const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
            gradient.setAttribute('id', gradientId);
            gradient.setAttribute('x1', '0%');
            gradient.setAttribute('y1', '100%');
            gradient.setAttribute('x2', '0%');
            gradient.setAttribute('y2', '0%');
            
            const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop1.setAttribute('offset', '0%');
            stop1.setAttribute('stop-color', this.fillGradientAbove.color);
            stop1.setAttribute('stop-opacity', this.fillGradientAbove.opacity ?? 0.5);
            
            const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
            stop2.setAttribute('offset', '100%');
            stop2.setAttribute('stop-color', this.fillGradientAbove.color);
            stop2.setAttribute('stop-opacity', '0');
            
            gradient.appendChild(stop1);
            gradient.appendChild(stop2);
            defs.appendChild(gradient);
            
            const fillPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            fillPath.setAttribute('class', 'horizon-fill-path-above');
            fillPath.setAttribute('d', this.generateClosedPathAbove(gradientHeight));
            fillPath.setAttribute('fill', `url(#${gradientId})`);
            fillPath.setAttribute('stroke', 'none');
            svg.insertBefore(fillPath, element);
        }
        
        // Apply stroke path (with gap if specified)
        element.setAttribute('d', this.generateStrokePath());
        element.setAttribute('stroke', this.strokeColor);
        element.setAttribute('stroke-width', this.strokeWidth);
    }
}

function createHorizon() {
    const screenWidth = window.innerWidth;
    const horizon = new PerlinHorizon({
        width: screenWidth,
        centerY: 150,
        amplitude: 10,
        frequency: 0.008,
        seed: 1,
        strokeColor: '#000000',
        strokeWidth: 2,
        fillGradientAbove: {
            color: '#87CEEB',
            opacity: 0.15,
            height: 600
        },
        fillGradientBelow: {
            color: '#228B22',
            opacity: 0.08,
            height: 800
        },
        gap: {
            start: screenWidth - 226,
            end: screenWidth - 123
        }            
    });
    
    const svg = document.querySelector('.horizon-svg');
    svg.setAttribute('viewBox', `0 0 ${screenWidth} 400`);
    
    horizon.applyTo('horizon-path');
}

createHorizon();
window.addEventListener('resize', createHorizon);
