const target = document.querySelector("#settings-window");

const moveable = new Moveable(document.body, {
	target,
	draggable: true,
	resizable: true,
	origin: false,
	keepRatio: false,
	bounds: {
		left: 0,
		top: 0,
		right: window.innerWidth,
		bottom: window.innerHeight
	}
});

moveable
	.on("drag", e => {
		e.target.style.transform = e.transform;
	})
	.on("resize", e => {
		e.target.style.width = `${e.width}px`;
		e.target.style.height = `${e.height}px`;
		e.target.style.transform = e.drag.transform;
	});

// moveable.on("dragStart", () => controls.enabled = false);
// moveable.on("dragEnd", () => controls.enabled = true);
