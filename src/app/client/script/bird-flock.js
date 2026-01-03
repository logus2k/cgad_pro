// bird-flock.js


export class BirdFlock {

    constructor(containerId, options = {}) {

        this.container = document.getElementById(containerId);
        this.top = options.top !== undefined ? options.top : 0;
        this.height = options.height || 400;
        this.speed = options.speed || 20;
        this.count = options.count || 12;
        this.flocking = options.flocking || 0.5;
        this.distance = options.distance || 5;
        this.birdColor = options.birdColor || '#2c3e50';
        this.loop = options.loop !== undefined ? options.loop : 0;
        this.fadeOut = options.fadeOut !== undefined ? options.fadeOut : 1;

        if (this.loop === -1) {
            this.destroy();
        } else {
            this.init();
        }
    }

    destroy() { this.container.innerHTML = ''; }

    init() {
        this.container.style.top = `${this.top}px`;
        this.container.style.height = `${this.height}px`;
        this._injectStyles();
        this.createFlock();
    }

    _injectStyles() {
        const styleId = 'bird-flock-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            const fadeStart = Math.max(0, (this.fadeOut * 100) - 10);
            const fadeEnd = this.fadeOut * 100;

            style.innerHTML = `
                        .bird-wrapper {
                            position: absolute;
                            right: -20%;
                            top: 0;
                            opacity: 0;
                            will-change: transform, opacity;
                        }
                        .bird-svg { transform: scaleX(-1); }
                        .bird-wing { 
                            fill: none; stroke: ${this.birdColor}; 
                            stroke-width: 2; stroke-linecap: round; 
                            animation: flap-wing 0.8s ease-in-out infinite alternate; 
                        }
                        @keyframes fly-left-move {
                            0% { transform: translateX(0); }
                            100% { transform: translateX(-140vw); }
                        }
                        @keyframes fade-out-path {
                            0% { opacity: 0; }
                            1% { opacity: 0.8; }
                            ${fadeStart}% { opacity: 0.8; }
                            ${fadeEnd}% { opacity: 0; }
                            100% { opacity: 0; }
                        }
                        @keyframes flap-wing {
                            0% { d: path("M2,10 L15,18 L28,10"); }
                            100% { d: path("M2,18 L15,10 L28,18"); }
                        }
                    `;
            document.head.appendChild(style);
        }
    }

    createFlock() {
        const loopValue = this.loop === 0 ? 'infinite' : this.loop;
        const variance = 1 - this.flocking;

        for (let i = 0; i < this.count; i++) {
            const bird = document.createElement('div');
            bird.className = 'bird-wrapper';

            // 1. V-FORMATION LOGIC
            const side = i % 2 === 0 ? 1 : -1;
            const orderInArm = Math.ceil(i / 2);

            // Determine the vertical spread per bird so the whole flock fits in 'height'
            // We divide height by (count/2) to ensure the widest part of the V matches height
            const verticalSpacing = (this.height / (this.count / 1.5)) * this.flocking;

            const center = this.height / 2;
            const vShapeY = center + (side * orderInArm * verticalSpacing);
            const randomY = Math.random() * this.height;

            // 2. HEIGHT ENFORCEMENT (Clamping)
            // We blend the values, then ensure the result is between 0 and this.height
            let yOffset = (vShapeY * this.flocking) + (randomY * variance);
            yOffset = Math.max(0, Math.min(yOffset, this.height - 20)); // -20 for bird SVG height

            // 3. SPACING & ANIMATION
            const size = (4 / this.distance) * (0.8 + Math.random() * 0.4);
            const speedMultiplier = 1 + (this.distance * 0.05);
            const duration = (this.speed * speedMultiplier) + (Math.random() * 5 * variance);

            const vShapeX = i * (this.flocking * 40);
            const randomX = Math.random() * 300;
            const xOffset = (vShapeX * this.flocking) + (randomX * variance);

            const delay = (i * 0.2 * this.flocking) + (Math.random() * 5 * variance);
            const flapSpeed = 0.6 + (Math.random() * 0.4);

            bird.style.animation = `
                        fly-left-move ${duration}s linear ${delay}s ${loopValue} both,
                        fade-out-path ${duration}s linear ${delay}s ${loopValue} both
                    `;

            bird.style.top = `${yOffset}px`;
            bird.style.marginRight = `${xOffset}px`;

            bird.innerHTML = `
                        <svg class="bird-svg" width="30" height="30" style="transform: scaleX(-1) scale(${size})">
                            <path class="bird-wing" d="M2,10 L15,18 L28,10" style="animation-duration: ${flapSpeed}s" />
                        </svg>
                    `;

            this.container.appendChild(bird);
        }
    }
}
