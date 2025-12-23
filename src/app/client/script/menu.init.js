import { MenuManager } from './menu.manager.js';

const menuManager = new MenuManager({
	menuTargetId: "application-menu-container",
	menuPosition: "bottom-center",
	nonResizable: ["about", "metrics"],
	iconSize: 36,
	menuMargin: 20,
	initialVisibility: {
		gallery: true,
		metrics: false,
		benchmark: false,
		report: false,
		settings: false,
		about: false,
	}
});
