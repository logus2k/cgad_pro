import { MenuManager } from './menu.manager.js';

const menuManager = new MenuManager({
	menuTargetId: "application-menu-container",
	menuPosition: "bottom-center",
	iconSize: 36,
	menuMargin: 16,
	initialVisibility: {
		gallery: false,
		metrics: true,
		benchmark: false,
		report: false,
		settings: false,
		about: false,
	}
});
