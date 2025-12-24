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
            margin: config.menuMargin || 20,
            panelIds: config.panelIds || ['gallery', 'metrics', 'benchmark', 'report', 'settings', 'about'],
            initialVisibility: config.initialVisibility || {},
            nonResizable: config.nonResizable || ['settings', 'about'],
        };

        this.menuEl = null;
        this.panels = {};
        this.moveables = new Map();
        this.topZ = 10;
        this.positions = new WeakMap();
        this.svgCache = new Map();

        this.tx = 0;
        this.ty = 0;

        this.#initMenu();
        this.#initPanels();
        this.#initGlobalHoverCheck();
        this.#applyInitialVisibility();

        window.addEventListener('resize', () => this.#handleWindowResize());
    }

    // ------------------ public API ------------------
    showPanel(name) { 
        this.#setPanelDisplay(name, true); 
        this.#syncMenuBtn(name, true);
    }

    hidePanel(name) { 
        this.#setPanelDisplay(name, false); 
        this.#syncMenuBtn(name, false);
    }
    
    hideAll() { 
        this.cfg.panelIds.forEach(id => this.hidePanel(id)); 
    }

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
            { id: 'gallery', icon: 'deployed_code', label: 'Gallery' },
            { id: 'metrics', icon: 'finance', label: 'Metrics' },
            { id: 'benchmark', icon: 'trophy', label: 'Benchmark' },
            { id: 'report', icon: 'assignment', label: 'Report' },
            { id: 'settings', icon: 'settings', label: 'Settings' },
            { id: 'about', icon: 'info', label: 'About' },
        ];

        items.forEach(({ id, icon, label }) => {
            if (!this.cfg.panelIds.includes(id)) return;
            const b = document.createElement('button');
            b.type = 'button';
            b.title = label;
            const i = document.createElement('span');
            this.getSVGIconByName(i, icon, label);
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

    async getSVGIconByName(element, icon, alt) {

        const url = `./icons/${icon}.svg`;

        if (!this.svgCache.has(url)) {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`SVG load failed: ${url}`);

            const text = await res.text();
            const doc = new DOMParser().parseFromString(text, "image/svg+xml");
            this.svgCache.set(url, doc.documentElement);
        }

        const svg = this.svgCache.get(url).cloneNode(true);

        svg.classList.add("svg-icon");
        if (alt) svg.setAttribute("aria-label", alt);

        element.replaceChildren(svg);
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
                close.innerHTML = '×';
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
        });
    }

    #initGlobalHoverCheck() {
        document.addEventListener('mousemove', (e) => {
            this.moveables.forEach((mv, panel) => {
                const controlBox = mv.selfElement;
                if (!controlBox) return;
                
                const isOccluded = this.#isOccludedByHigherPanel(panel, e.clientX, e.clientY);
                
                controlBox.querySelectorAll('.moveable-control').forEach(ctrl => {
                    if (isOccluded) {
                        ctrl.style.pointerEvents = 'none';
                        ctrl.style.cursor = 'default';
                    } else {
                        ctrl.style.pointerEvents = '';
                        ctrl.style.cursor = '';
                    }
                });
            });
        });
    }

    #isOccludedByHigherPanel(panel, clientX, clientY) {
        const panelZ = parseInt(panel.style.zIndex) || 0;
        const elements = document.elementsFromPoint(clientX, clientY);
        
        for (const el of elements) {
            // If we hit our panel first, we're not occluded
            if (el === panel) return false;
            
            // Check if this is another panel with higher z-index
            if (el.classList.contains('hud') && el !== panel) {
                const elZ = parseInt(el.style.zIndex) || 0;
                if (elZ > panelZ) {
                    return true;
                }
            }
        }
        return false;
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

        const isResizable = !this.cfg.nonResizable.includes(id);
        const headerEl = panel.querySelector('h1');

        const padding = 10;

        const mv = new Moveable(document.body, {
            target: panel,
            draggable: true,
            resizable: isResizable,
            origin: false,
            snappable: true,
            bounds: { 
                left: padding, 
                top: padding, 
                right: window.innerWidth - padding, 
                bottom: window.innerHeight - padding 
            }
        });

        // Sync control box z-index with panel
        const syncZIndex = () => {
            const controlBox = mv.selfElement;
            if (controlBox) {
                controlBox.style.zIndex = panel.style.zIndex;
            }
        };

        // Sync on creation
        syncZIndex();

        // Sync when panel gains focus
        panel.addEventListener('mousedown', () => {
            this.topZ += 1;
            panel.style.zIndex = String(this.topZ);
            syncZIndex();
        });

        let allowDrag = false;
        
        // Initialize or reset position tracker to 0,0 because we normalized the top/left above
        const pos = { x: 0, y: 0 };
        this.positions.set(panel, pos);

        mv.on('dragStart', e => {
            const { clientX, clientY } = e.inputEvent;
            
            // Check if click point is occluded by a higher z-index panel
            if (this.#isOccludedByHigherPanel(panel, clientX, clientY)) {
                e.stop();
                return;
            }
            
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
            const { clientX, clientY } = e.inputEvent;
            
            // Check if click point is occluded by a higher z-index panel
            if (this.#isOccludedByHigherPanel(panel, clientX, clientY)) {
                e.stop();
                return;
            }
            
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
        const padding = 10;
        const newBounds = { 
            left: padding, 
            top: padding, 
            right: window.innerWidth - padding, 
            bottom: window.innerHeight - padding 
        };

        this.moveables.forEach((mv, panel) => {
            // 1. Update the Moveable instance bounds constraint
            mv.bounds = newBounds;
            
            // 2. Get current position data
            const rect = panel.getBoundingClientRect();
            const pos = this.positions.get(panel) || { x: 0, y: 0 };

            let adjustedX = pos.x;
            let adjustedY = pos.y;

            // 3. Logic to "push" the window back inside including the 10px margin
            
            // Check right edge (current right > window width - 10)
            if (rect.right > (window.innerWidth - padding)) {
                adjustedX -= (rect.right - (window.innerWidth - padding));
            }
            
            // Check bottom edge (current bottom > window height - 10)
            if (rect.bottom > (window.innerHeight - padding)) {
                adjustedY -= (rect.bottom - (window.innerHeight - padding));
            }

            // Check left edge (must be at least 10)
            if (rect.left < padding) {
                adjustedX += (padding - rect.left);
            }

            // Check top edge (must be at least 10)
            if (rect.top < padding) {
                adjustedY += (padding - rect.top);
            }

            // 4. Apply adjustments
            if (adjustedX !== pos.x || adjustedY !== pos.y) {
                pos.x = adjustedX;
                pos.y = adjustedY;
                panel.style.transform = `translate(${pos.x}px, ${pos.y}px)`;
            }

            // 5. Refresh Moveable
            mv.updateRect();
        });
    }   
}
