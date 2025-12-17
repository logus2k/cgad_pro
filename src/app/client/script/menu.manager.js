/*
// Usage: const menuManager = new MenuManager();
// Custom configuration
const menuManager = new MenuManager({
    menuPosition: 'top-left',
    menuLayout: 'vertical',
    menuIconSize: 50,
    initialVisibility: {
        settings: true,  // Show settings panel by default
        data: true       // Show data panel by default
    }
});
*/


export class MenuManager {

    constructor(config = {}) {
        this.cfg = {
            menuTargetId: config.menuTargetId || 'application-menu-container',
            menuPosition: config.menuPosition || 'bottom-center',
            iconSize: config.menuIconSize || 36,
            margin: config.menuMargin || 16,
            panelIds: config.panelIds || ['search', 'data', 'metrics', 'about', 'settings'],
            initialVisibility: config.initialVisibility || {},
        };

        this.menuEl = null;
        this.panels = {};
        this.moveables = new Map();
        this.topZ = 10;
        this.positions = new WeakMap();

        this.tx = 0;
        this.ty = 0;

        this.#initMenu();
        this.#initPanels();
        this.#applyInitialVisibility();

        window.addEventListener('resize', () => this.#handleWindowResize());
    }

    // ------------------ public API ------------------
    showPanel(name) { this.#setPanelDisplay(name, true); }
    hidePanel(name) { this.#setPanelDisplay(name, false); }
    hideAll() { this.cfg.panelIds.forEach(id => this.hidePanel(id)); }
    destroy() {
        this.moveables.forEach(m => m.destroy && m.destroy());
        this.moveables.clear();
        if (this.menuEl && this.menuEl.parentNode) this.menuEl.parentNode.removeChild(this.menuEl);
        this.cfg.panelIds.forEach(id => {
            const p = this.panels[id];
            if (!p) return;
            const btn = p.querySelector('.pm-close');
            if (btn) btn.remove();
            p.style.cursor = '';
            p.onmousedown = null;
        });
    }

    // ------------------ internals ------------------
    #initMenu() {
        const target = document.getElementById(this.cfg.menuTargetId);
        if (!target) throw new Error(`Menu target #${this.cfg.menuTargetId} not found`);

        const wrap = document.createElement('div');
        wrap.className = `pm-menu pm-${this.cfg.menuPosition.replace(/\s+/g, '-')}`;
        wrap.style.setProperty('--pm-icon', `${this.cfg.iconSize}px`);
        wrap.style.setProperty('--pm-m', `${this.cfg.margin}px`);

        const items = [
            { id: 'search', icon: 'search', label: 'Search' },
            { id: 'data', icon: 'description', label: 'Data Explorer' },
            { id: 'metrics', icon: 'equalizer', label: 'Metrics' },
            { id: 'settings', icon: 'settings', label: 'Settings' },
            { id: 'about', icon: 'info', label: 'About' },
        ];

        items.forEach(({ id, icon, label }) => {
            if (!this.cfg.panelIds.includes(id)) return;
            const b = document.createElement('button');
            b.type = 'button';
            b.title = label;
            const i = document.createElement('span');
            i.className = 'material-symbols-outlined';
            i.textContent = icon;
            b.appendChild(i);
            b.addEventListener('click', () => {
                const isVisible = this.#isPanelShown(id);
                this.#setPanelDisplay(id, !isVisible);
                b.classList.toggle('active', !isVisible);
            });
            wrap.appendChild(b);
        });

        target.appendChild(wrap);
        this.menuEl = wrap;
    }

    #initPanels() {
        this.cfg.panelIds.forEach(id => {
            const el = document.getElementById(`hud-${id}`);
            if (!el) return;
            this.panels[id] = el;

            // Ensure we always have a close button with a working listener
            let close = el.querySelector('.pm-close');
            if (!close) {
                close = document.createElement('button');
                close.className = 'pm-close';
                close.textContent = '×';
                el.appendChild(close);
            }

            // Always (re)attach the listener to ensure it works with this instance
            close.onclick = (e) => { 
                e.preventDefault();
                e.stopPropagation(); 
                this.#setPanelDisplay(id, false); 
                this.#syncMenuBtn(id, false);
            };

            this.#makeDraggable(el, id);

            el.addEventListener('mousedown', () => {
                this.topZ += 1; 
                el.style.zIndex = String(this.topZ);
            });
        });
    }

    #makeDraggable(panel, id) {

        if (typeof Moveable === 'undefined') {
            console.warn('Moveable not found: skipping drag/resize for', id);
            return;
        }

        const existingMoveable = this.moveables.get(panel);
        if (existingMoveable) {
            existingMoveable.destroy();
            this.moveables.delete(panel);
        }

        // 1. Capture the current rendered position on screen
        const rect = panel.getBoundingClientRect();

        // 2. Normalize positioning to Top-Left
        // This stops the 'shaking' caused by bottom/right/align-self CSS anchors
        panel.style.top = `${rect.top}px`;
        panel.style.left = `${rect.left}px`;
        panel.style.bottom = 'auto';
        panel.style.right = 'auto';
        panel.style.margin = '0';
        panel.style.position = 'absolute';
        
        // Reset transform to 0,0 since we just moved the element's actual top/left
        panel.style.transform = 'translate(0px, 0px)';

        const isResizable = (id !== 'settings' && id !== 'about');
        const headerEl = panel.querySelector('h1');

        const mv = new Moveable(document.body, {
            target: panel,
            draggable: true,
            resizable: isResizable,
            origin: false,
            // NEW: Constraint properties
            snappable: true,
            bounds: { 
                left: 0, 
                top: 0, 
                right: window.innerWidth, 
                bottom: window.innerHeight 
            }
        });

        let allowDrag = false;
        
        // Initialize or reset position tracker to 0,0 because we normalized the top/left above
        const pos = { x: 0, y: 0 };
        this.positions.set(panel, pos);

        mv.on('dragStart', e => {
            const t = e.inputEvent && e.inputEvent.target;
            allowDrag = !!(headerEl && t && (t === headerEl || headerEl.contains(t)));
            if (!allowDrag) { e.stop && e.stop(); return; }

            if (e.set) e.set([pos.x, pos.y]);
            e.inputEvent.stopPropagation();
        })
        .on('drag', e => {
            if (!allowDrag) return;
            const [x, y] = e.beforeTranslate;
            pos.x = x; pos.y = y;
            e.target.style.transform = `translate(${x}px, ${y}px)`;
        })
        .on('dragEnd', () => { 
            allowDrag = false;
        })
        .on('resizeStart', e => {
            e.setOrigin(['%', '%']);
            if (e.dragStart) e.dragStart.set([pos.x, pos.y]);
        })
        .on('resize', e => {
            const { target, width, height, drag } = e;
            
            target.style.width = `${width}px`;
            target.style.height = `${height}px`;

            const [x, y] = drag.beforeTranslate;
            target.style.transform = `translate(${x}px, ${y}px)`;

            pos.x = x; 
            pos.y = y;
        })
        .on('resizeEnd', () => {
            mv.updateRect();
        });

        this.moveables.set(panel, mv);
        mv.updateRect();
        this.#applyControlStyles(mv);
    }

    #applyControlStyles(mv) {

        const box = document.querySelectorAll('.moveable-control-box');
        const controlBox = box[box.length - 1];

        if (!controlBox) return;

        const controls = controlBox.querySelectorAll('.moveable-control');
        const { width, height } = mv.getRect();

        controls.forEach(control => {
            control.classList.add('custom-control');

            if (control.classList.contains('moveable-n') || control.classList.contains('moveable-s')) {
                control.style.width = `${width}px`;
                control.style.marginLeft = `-${width / 2}px`;
            }
            if (control.classList.contains('moveable-w') || control.classList.contains('moveable-e')) {
                control.style.height = `${height}px`;
                control.style.marginTop = `-${height / 2}px`;
            }
        });
    }
    

    #applyInitialVisibility() {
        Object.entries(this.cfg.initialVisibility).forEach(([id, vis]) => {
            if (!this.panels[id]) return;
            this.#setPanelDisplay(id, !!vis);
            this.#syncMenuBtn(id, !!vis);
        });
    }

    #setPanelDisplay(id, show) {
        const p = this.panels[id];
        if (!p) return;
        
        // Get the current Moveable instance for this panel
        const existingMoveable = this.moveables.get(p);

        if (show) {
            // --- Logic for SHOWING the Panel ---
            p.classList.add('visible');
            this.topZ += 1;
            p.style.zIndex = String(this.topZ);

            // 1. If the panel is being shown, ensure Moveable is active.
            //    Since your #makeDraggable handles creation/recreation, call it here.
            if (!existingMoveable) {
                // If it doesn't exist, create it (this is the key change for visibility)
                this.#makeDraggable(p, id); 
            } else {
                // If it already exists (e.g., if you only deactivated it previously, 
                // though you are using a destroy/recreate pattern) you'd call an activate method.
                // For your current pattern, it's safer to ensure it's created via the call above.
                existingMoveable.updateRect(); // Ensures controls are correctly positioned
            }
            
        } else {
            // --- Logic for HIDING the Panel ---
            p.classList.remove('visible');

            // 2. If the panel is being hidden, DESTROY the Moveable instance.
            if (existingMoveable) {
                existingMoveable.destroy();
                this.moveables.delete(p);
            }
        }
    }

    #isPanelShown(id) {
        const p = this.panels[id];
        return !!p && p.classList.contains('visible');
    }

    #syncMenuBtn(id, active) {
        if (!this.menuEl) return;
        const btns = Array.from(this.menuEl.querySelectorAll('button'));
        const idx = this.cfg.panelIds.indexOf(id);
        const b = btns[idx];
        if (b) b.classList.toggle('active', !!active);
    }

    #handleWindowResize() {
        const newBounds = { 
            left: 0, 
            top: 0, 
            right: window.innerWidth, 
            bottom: window.innerHeight 
        };

        this.moveables.forEach((mv, panel) => {
            // 1. Update the Moveable instance bounds
            mv.bounds = newBounds;
            
            // 2. Get current position data
            const rect = panel.getBoundingClientRect();
            const pos = this.positions.get(panel) || { x: 0, y: 0 };

            let adjustedX = pos.x;
            let adjustedY = pos.y;

            // 3. Logic to "push" the window back inside if the browser shrank
            // Check right edge
            if (rect.right > window.innerWidth) {
                adjustedX -= (rect.right - window.innerWidth);
            }
            // Check bottom edge
            if (rect.bottom > window.innerHeight) {
                adjustedY -= (rect.bottom - window.innerHeight);
            }
            // Ensure it doesn't go past top/left (0,0)
            if (rect.left < 0) adjustedX -= rect.left;
            if (rect.top < 0) adjustedY -= rect.top;

            // 4. Apply adjustments if needed
            if (adjustedX !== pos.x || adjustedY !== pos.y) {
                pos.x = adjustedX;
                pos.y = adjustedY;
                panel.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
            }

            // 5. Refresh Moveable's internal cache
            mv.updateRect();
        });
    }    
}
