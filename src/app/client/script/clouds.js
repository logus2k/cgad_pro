export class CloudGradient {

	constructor(selector, options = {}) {
		this.element = document.querySelector(selector);
		this.count = options.count || 100;
		this.colors = options.colors || ['#ffffff', '#f0fff3', '#f0f0f0', '#e8f8f8'];
		this.blur = options.blur || [10, 25];
		this.spread = options.spread || [1, 15];
		this.x = options.x || [1, 100];
		this.y = options.y || [-20, 120];
	}

	rn(from, to) {
		return ~~(Math.random() * (to - from + 1)) + from;
	}

	rs(arr) {
		return arr[this.rn(0, arr.length - 1)];
	}

	generateShadows() {
		const shadows = [];
		for (let i = 0; i < this.count; ++i) {
			shadows.push(`
        ${this.rn(this.x[0], this.x[1])}vw ${this.rn(this.y[0], this.y[1])}px ${this.rn(this.blur[0], this.blur[1])}px ${this.rn(this.spread[0], this.spread[1])}px
        ${this.rs(this.colors)}
      `);
		}
		return shadows.join(',');
	}

	update() {
		if (this.element) {
			this.element.style.boxShadow = this.generateShadows();
		}
	}

	init() {
		if (document.readyState === 'loading') {
			window.addEventListener('load', () => this.update());
		} else {
			this.update();
		}
	}
}
