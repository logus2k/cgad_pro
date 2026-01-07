import { MenuManager } from './menu.manager.js';

const menuManager = new MenuManager({
	menuTargetId: "application-menu-container",
	menuPosition: "bottom-center",
	nonResizable: ["gallery", "about", "metrics"], // , "settings"],
	iconSize: 36,
	menuMargin: 20,
	initialVisibility: {
		gallery: false,
		metrics: false,
		benchmark: false,
		report: false,
		//settings: false,
		about: false,
	}
});

window.menuManager = menuManager;
