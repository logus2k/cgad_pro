import { MenuManager } from './menu.manager.js';

const menuManager = new MenuManager({
	menuTargetId: "application-menu-container",
	menuPosition: "bottom-center",
	iconSize: 36,
	menuMargin: 16,
	initialVisibility: {
		settings: false,
		search: false,
		data: false,
		metrics: true,
		about: false
	}
});
