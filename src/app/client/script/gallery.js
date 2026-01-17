/**
 * Mesh Gallery Carousel
 * Handles carousel navigation, item selection, and model loading
 * 
 * Now includes mesh preloading - starts loading mesh data when user
 * selects an item (before confirming), so geometry is ready faster.
 * 
 * Supports two run modes:
 * - Run Simulation: Normal solver execution with visualization
 * - Run Profiling: nsys-wrapped execution for GPU profiling (CUDA solvers only)
 */

import { meshLoader } from './mesh-loader.js';

// CUDA-capable solvers that support profiling
const CUDA_SOLVERS = ['gpu', 'numba_cuda'];

class MeshGallery {
    constructor(options = {}) {
        this.jsonPath = options.jsonPath || 'gallery_files.json';
        this.itemsPerView = options.itemsPerView || 3;
        this.models = [];
        this.solvers = [];
        this.currentIndex = 0;
        this.selectedIndex = 0;
        
        // Preloaded mesh data (ready before solve starts)
        this.preloadedMeshData = null;
        this.preloadingUrl = null;
        
        // DOM elements
        this.track = document.getElementById('carouselTrack');
        this.navDots = document.getElementById('navDots');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.selectBtn = document.getElementById('selectBtn');
        this.profilingBtn = document.getElementById('profilingBtn');
        this.cancelBtn = document.getElementById('cancelBtn');
        this.gallery = document.getElementById('hud-gallery');

        // Clear existing text and load SVG icons
        /*
        if (this.prevBtn) {
            this.prevBtn.textContent = '';
            if (window.menuManager) {
                window.menuManager.getSVGIconByName(this.prevBtn, 'previous', 'Previous');
            }
        }
        if (this.nextBtn) {
            this.nextBtn.textContent = '';
            if (window.menuManager) {
                window.menuManager.getSVGIconByName(this.nextBtn, 'next', 'Next');
            }
        }
        */
        
        // Set up mesh loader progress callback
        meshLoader.setProgressCallback((stage, progress) => {
            this.onMeshLoadProgress(stage, progress);
        });
        
        this.init();
    }
    
    async init() {
        await this.loadModelData();
        this.renderCarousel();
        this.renderNavDots();
        this.bindEvents();
        this.updateNavigation();
        this.selectItem(0);
    }
    
    async loadModelData() {
        try {
            const response = await fetch(this.jsonPath);
            if (!response.ok) throw new Error('Failed to load gallery data');
            const data = await response.json();
            this.models = data.models || [];
            this.solvers = data.solvers || [];
            console.log(`Loaded ${this.models.length} models, ${this.solvers.length} solvers`);
        } catch (error) {
            console.error('Error loading mesh gallery:', error);
            this.models = [];
            this.solvers = [];
        }
    }
    
    /**
     * Get the default mesh for a model, or first mesh if no default specified
     */
    getDefaultMesh(model) {
        if (!model.meshes || model.meshes.length === 0) return null;
        return model.meshes.find(m => m.default) || model.meshes[0];
    }
    
    /**
     * Get the currently selected mesh for a model item
     */
    getSelectedMeshForItem(index) {
        const item = this.track?.querySelector(`.carousel-item[data-index="${index}"]`);
        const meshDropdown = item?.querySelector('.mesh-select');
        const meshIndex = meshDropdown ? parseInt(meshDropdown.value, 10) : 0;
        return this.models[index]?.meshes[meshIndex] || null;
    }
    
    renderCarousel() {
        if (!this.track) return;
        
        this.track.innerHTML = '';
        
        this.models.forEach((model, index) => {
            const item = document.createElement('div');
            item.className = 'carousel-item';
            item.dataset.index = index;
            
            const defaultMesh = this.getDefaultMesh(model);
            const defaultMeshIndex = model.meshes.indexOf(defaultMesh);
            
            // Build mesh options HTML
            const meshOptionsHtml = model.meshes.map((mesh, meshIdx) => {
                const selected = mesh === defaultMesh ? 'selected' : '';
                return `<option value="${meshIdx}" ${selected}>${mesh.label}</option>`;
            }).join('');
            
            // Build solver options HTML
            const solverOptionsHtml = this.solvers.map(solver => {
                const selected = solver.id === model.solver_type ? 'selected' : '';
                return `<option value="${solver.id}" title="${solver.description}" ${selected}>${solver.name}</option>`;
            }).join('');

            // Build extrusion type options HTML
            const extrusionOptionsHtml = `
                <option value="rectangular" selected>Rectangular</option>
                <option value="cylindrical">Cylindrical</option>
            `;            
            
            item.innerHTML = `
                <div class="model-image-container">
                    <img class="model-image" src="${model.thumbnail}" alt="${model.name}" 
                         onerror="this.onerror=null; this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 150%22><rect fill=%22%23f0f0f0%22 width=%22200%22 height=%22150%22/><text x=%22100%22 y=%2275%22 text-anchor=%22middle%22 fill=%22%23999%22 font-family=%22sans-serif%22 font-size=%2214%22>No Preview</text></svg>'">
                    <div class="preload-indicator" style="display: none;">
                        <div class="preload-spinner"></div>
                        <span class="preload-text">Loading mesh...</span>
                    </div>
                </div>
                <div class="model-name">${model.name}</div>
                <div class="model-description">${model.description}</div>
                <div class="model-metadata">
                    <div class="solver-row">
                        <label>Solver:</label>
                        <select class="solver-select" data-index="${index}">
                            ${solverOptionsHtml}
                        </select>
                    </div>
                    <div class="mesh-row">
                        <label>Mesh:</label>
                        <select class="mesh-select" data-index="${index}">
                            ${meshOptionsHtml}
                        </select>
                    </div>
                    <div class="solver-row">
                        <label>Extrusion:</label>
                        <select class="extrusion-select" data-index="${index}">
                            ${extrusionOptionsHtml}
                        </select>
                    </div>                    
                    <div class="complexity-badge ${this.getComplexityClass(defaultMesh?.elements)}" data-complexity>
                        ${this.getComplexityLabel(defaultMesh?.elements)}
                    </div>
                </div>
            `;
            
            // Prevent dropdown clicks from triggering item selection
            const meshSelect = item.querySelector('.mesh-select');
            const solverSelect = item.querySelector('.solver-select');
            const extrusionSelect = item.querySelector('.extrusion-select');
         
            if (meshSelect) {
                meshSelect.addEventListener('click', (e) => e.stopPropagation());
                meshSelect.addEventListener('change', (e) => this.onMeshSelectionChange(index, e.target.value));
            }
            
            if (solverSelect) {
                solverSelect.addEventListener('click', (e) => e.stopPropagation());
                solverSelect.addEventListener('change', () => {
                    this.selectItem(index);
                    this.updateProfilingButtonState();
                });
            }

            if (extrusionSelect) {
                extrusionSelect.addEventListener('click', (e) => e.stopPropagation());
                extrusionSelect.addEventListener('change', () => this.selectItem(index));
            }            
            
            item.addEventListener('click', () => this.selectItem(index));
            this.track.appendChild(item);
        });
    }
    
    /**
     * Handle mesh dropdown change - update complexity badge and trigger preload
     */
    onMeshSelectionChange(modelIndex, meshIndexStr) {

        this.selectItem(modelIndex);

        const meshIndex = parseInt(meshIndexStr, 10);
        const model = this.models[modelIndex];
        const mesh = model?.meshes[meshIndex];
        
        if (!mesh) return;
        
        // Update complexity badge
        const item = this.track.querySelector(`.carousel-item[data-index="${modelIndex}"]`);
        const badge = item?.querySelector('[data-complexity]');
        
        if (badge) {
            badge.className = `complexity-badge ${this.getComplexityClass(mesh.elements)}`;
            badge.textContent = this.getComplexityLabel(mesh.elements);
        }
        
        console.log(`Mesh changed for ${model.name}: ${mesh.label}`);
        
        // Reset preload indicators
        item?.classList.remove('mesh-ready', 'mesh-error');
        const indicator = item?.querySelector('.preload-indicator');
        if (indicator) indicator.style.display = 'none';
        
        // Preload the newly selected mesh if this is the selected model
        if (modelIndex === this.selectedIndex) {
            this.preloadMesh(model, mesh);
        }
    }
    
    getComplexityClass(elements) {
        if (!elements) return 'complexity-unknown';
        if (elements < 500) return 'complexity-low';
        if (elements < 100000) return 'complexity-med';
        if (elements <= 200000) return 'complexity-high';
        if (elements > 200000) return 'complexity-very-high';
    }
    
    getComplexityLabel(elements) {
        if (!elements) return 'Unknown Size';
        if (elements < 500) return 'Small';
        if (elements < 100000) return 'Medium';
        if (elements <= 200000) return 'Large';
        if (elements > 200000) return 'Very Large';
    }
    
    renderNavDots() {
        if (!this.navDots) return;
        
        this.navDots.innerHTML = '';
        const totalPages = Math.ceil(this.models.length / this.itemsPerView);
        
        for (let i = 0; i < totalPages; i++) {
            const dot = document.createElement('span');
            dot.className = 'nav-dot' + (i === 0 ? ' active' : '');
            dot.dataset.page = i;
            dot.addEventListener('click', () => this.goToPage(i));
            this.navDots.appendChild(dot);
        }
    }
    
    bindEvents() {
        if (this.prevBtn) {
            this.prevBtn.addEventListener('click', () => this.navigate(-1));
        }
        
        if (this.nextBtn) {
            this.nextBtn.addEventListener('click', () => this.navigate(1));
        }
        
        if (this.selectBtn) {
            this.selectBtn.addEventListener('click', () => this.confirmSelection());
        }
        
        if (this.profilingBtn) {
            this.profilingBtn.addEventListener('click', () => this.confirmProfiling());
        }
        
        if (this.cancelBtn) {
            this.cancelBtn.addEventListener('click', () => this.close());
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (!this.isVisible()) return;
            
            switch (e.key) {
                case 'ArrowLeft':
                    this.navigate(-1);
                    break;
                case 'ArrowRight':
                    this.navigate(1);
                    break;
                case 'Enter':
                    this.confirmSelection();
                    break;
                case 'Escape':
                    this.close();
                    break;
            }
        });
    }
    
    navigate(direction) {
        const totalPages = Math.ceil(this.models.length / this.itemsPerView);
        const currentPage = Math.floor(this.currentIndex / this.itemsPerView);
        const newPage = Math.max(0, Math.min(totalPages - 1, currentPage + direction));
        
        this.goToPage(newPage);
    }
    
    goToPage(page) {
        this.currentIndex = page * this.itemsPerView;
        this.updateCarouselPosition();
        this.updateNavDots(page);
        this.updateNavigation();
    }
    
    updateCarouselPosition() {
        if (!this.track) return;
        
        const items = this.track.querySelectorAll('.carousel-item');
        if (items.length === 0) return;
        
        // Calculate item width including gap
        const itemStyle = getComputedStyle(items[0]);
        const itemWidth = items[0].offsetWidth;
        const gap = parseInt(getComputedStyle(this.track).gap) || 15;
        
        const offset = this.currentIndex * (itemWidth + gap);
        this.track.style.transform = `translateX(-${offset}px)`;
    }
    
    updateNavDots(activePage) {
        if (!this.navDots) return;
        
        const dots = this.navDots.querySelectorAll('.nav-dot');
        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index === activePage);
        });
    }
    
    updateNavigation() {
        const totalPages = Math.ceil(this.models.length / this.itemsPerView);
        const currentPage = Math.floor(this.currentIndex / this.itemsPerView);
        
        if (this.prevBtn) {
            this.prevBtn.disabled = currentPage === 0;
        }
        
        if (this.nextBtn) {
            this.nextBtn.disabled = currentPage >= totalPages - 1;
        }
    }
    
    /**
     * Check if the currently selected solver supports CUDA profiling
     */
    isProfilingSupported() {
        const solverType = this.getSolverForItem(this.selectedIndex);
        return CUDA_SOLVERS.includes(solverType);
    }
    
    /**
     * Update the profiling button enabled/disabled state
     */
    updateProfilingButtonState() {
        if (!this.profilingBtn) return;
        
        const supported = this.isProfilingSupported();
        this.profilingBtn.disabled = !supported;
        
        if (supported) {
            this.profilingBtn.title = 'Run with NVIDIA Nsight profiling';
        } else {
            this.profilingBtn.title = 'Profiling only available for CUDA solvers (gpu, numba_cuda)';
        }
    }
    
    /**
     * Handle mesh load progress updates
     */
    onMeshLoadProgress(stage, progress) {
        const item = this.track?.querySelector(`.carousel-item[data-index="${this.selectedIndex}"]`);
        if (!item) return;
        
        const indicator = item.querySelector('.preload-indicator');
        const text = item.querySelector('.preload-text');
        
        if (!indicator) return;
        
        switch (stage) {
            case 'init_h5wasm':
                indicator.style.display = 'flex';
                text.textContent = 'Initializing...';
                break;
            case 'downloading':
                indicator.style.display = 'flex';
                if (progress !== null) {
                    text.textContent = `Downloading ${Math.round(progress * 100)}%`;
                } else {
                    text.textContent = 'Downloading...';
                }
                break;
            case 'parsing':
                text.textContent = 'Parsing HDF5...';
                break;
            case 'complete':
                indicator.style.display = 'none';
                // Add a "ready" indicator
                item.classList.add('mesh-ready');
                break;
            case 'error':
                indicator.style.display = 'none';
                item.classList.add('mesh-error');
                break;
        }
    }
    
    /**
     * Select an item and start preloading its mesh
     */
    selectItem(index) {
        if (index < 0 || index >= this.models.length) return;
        
        this.selectedIndex = index;
        
        // Update visual selection
        const items = this.track.querySelectorAll('.carousel-item');
        items.forEach((item, i) => {
            item.classList.toggle('selected-indicator', i === index);
            // Reset preload indicators
            item.classList.remove('mesh-ready', 'mesh-error');
            const indicator = item.querySelector('.preload-indicator');
            if (indicator) indicator.style.display = 'none';
        });
        
        // Ensure selected item is visible
        const page = Math.floor(index / this.itemsPerView);
        if (page !== Math.floor(this.currentIndex / this.itemsPerView)) {
            this.goToPage(page);
        }
        
        const model = this.models[index];
        const selectedMesh = this.getSelectedMeshForItem(index);
        console.log(`Selected: ${model.name} (${selectedMesh?.label})`);
        
        // Start preloading mesh data immediately
        this.preloadMesh(model, selectedMesh);
        
        // Update profiling button state based on selected solver
        this.updateProfilingButtonState();
    }
    
    /**
     * Preload mesh data (non-blocking)
     */
    async preloadMesh(model, mesh) {
        if (!mesh) return;
        
        const meshUrl = mesh.file;
        
        // Skip if already preloading this mesh
        if (this.preloadingUrl === meshUrl) {
            return;
        }
        
        // Skip if already cached
        if (meshLoader.isCached(meshUrl)) {
            console.log(`Mesh already cached: ${model.name} - ${mesh.label}`);
            this.preloadedMeshData = meshLoader.getCached(meshUrl);
            
            // Show ready indicator
            const item = this.track?.querySelector(`.carousel-item[data-index="${this.selectedIndex}"]`);
            if (item) item.classList.add('mesh-ready');
            return;
        }
        
        this.preloadingUrl = meshUrl;
        this.preloadedMeshData = null;
        
        try {
            console.log(`Preloading mesh: ${model.name} - ${mesh.label} (${meshUrl})`);
            const meshData = await meshLoader.preload(meshUrl);
            
            // Only store if still the selected mesh
            const currentMesh = this.getSelectedMeshForItem(this.selectedIndex);
            if (currentMesh?.file === meshUrl) {
                this.preloadedMeshData = meshData;
                console.log(`Mesh preloaded and ready: ${model.name} - ${mesh.label}`);
            }
        } catch (error) {
            console.error(`Failed to preload mesh: ${model.name} - ${mesh.label}`, error);
            this.preloadedMeshData = null;
        } finally {
            if (this.preloadingUrl === meshUrl) {
                this.preloadingUrl = null;
            }
        }
    }
    
    /**
     * Confirm selection and dispatch event with preloaded mesh data
     * This starts a normal simulation run
     */
    confirmSelection() {
        const model = this.models[this.selectedIndex];
        if (!model) {
            console.warn('No model selected');
            return;
        }
        
        // Get selected mesh from dropdown
        const selectedMesh = this.getSelectedMeshForItem(this.selectedIndex);
        if (!selectedMesh) {
            console.warn('No mesh selected');
            return;
        }
        
        // Close gallery
        this.close();
        
        // Auto-open Metrics panel to show solver progress
        if (window.menuManager) {
            window.menuManager.showPanel('metrics');
        }
        
        // Get solver from the selected item's dropdown
        const selectedItem = this.track.querySelector(`.carousel-item[data-index="${this.selectedIndex}"]`);
        const solverDropdown = selectedItem?.querySelector('.solver-select');
        const solverType = solverDropdown?.value || model.solver_type || 'gpu';

        const extrusionDropdown = selectedItem?.querySelector('.extrusion-select');
        const extrusionType = extrusionDropdown?.value || 'rectangular';        
        
        const maxIterations = model.max_iterations || 50000;
        const progressInterval = model.progress_interval || 100;
        
        console.log('=== MODEL SELECTION CONFIRMED ===');
        console.log('Model:', model.name);
        console.log('Mesh:', selectedMesh.label);
        console.log('File:', selectedMesh.file);
        console.log('Solver:', solverType);
        console.log('Extrusion:', extrusionType);        
        console.log('Max Iterations:', maxIterations);
        console.log('=================================');
        
        // Get preloaded data if available (don't wait if not ready)
        const meshData = this.preloadedMeshData;
        
        // Build a combined object for backward compatibility
        const meshInfo = {
            ...model,
            ...selectedMesh,
            extrusion_type: extrusionType,
            solver_type: solverType
        };
        
        // Dispatch event with preloaded data (may be null if still loading)
        const event = new CustomEvent('meshSelected', {
            detail: {
                mesh: meshInfo,
                model: model,
                selectedMesh: selectedMesh,
                index: this.selectedIndex,
                preloadedData: meshData,
                meshLoader: meshLoader,
                extrusionType: extrusionType
            }
        });
        document.dispatchEvent(event);
        
        // Start the solver (uses existing femClient from window)
        if (window.femClient) {
            window.femClient.startSolve({
                mesh_file: selectedMesh.file,
                solver_type: solverType,
                max_iterations: maxIterations,
                progress_interval: progressInterval
            });
        } else {
            console.error('femClient not found on window');
        }
    }
    
    /**
     * Confirm profiling run - starts nsys-wrapped solver execution
     * Opens PROFILING panel instead of SIMULATION panel
     */
    async confirmProfiling() {
        const model = this.models[this.selectedIndex];
        if (!model) {
            console.warn('No model selected');
            return;
        }
        
        // Get selected mesh from dropdown
        const selectedMesh = this.getSelectedMeshForItem(this.selectedIndex);
        if (!selectedMesh) {
            console.warn('No mesh selected');
            return;
        }
        
        // Get solver from the selected item's dropdown
        const selectedItem = this.track.querySelector(`.carousel-item[data-index="${this.selectedIndex}"]`);
        const solverDropdown = selectedItem?.querySelector('.solver-select');
        const solverType = solverDropdown?.value || model.solver_type || 'gpu';
        
        // Verify CUDA solver
        if (!CUDA_SOLVERS.includes(solverType)) {
            console.warn('Profiling only supported for CUDA solvers');
            return;
        }
        
        // Close gallery
        this.close();
        
        // Auto-open PROFILING panel (not metrics/simulation)
        if (window.menuManager) {
            window.menuManager.showPanel('profiling');
        }
        
        console.log('=== PROFILING RUN STARTED ===');
        console.log('Model:', model.name);
        console.log('Mesh:', selectedMesh.label);
        console.log('File:', selectedMesh.file);
        console.log('Solver:', solverType);
        console.log('=============================');
        
        // Start profiling run via API
        try {
            const response = await fetch('/api/profiling/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    solver: solverType,
                    mesh_file: selectedMesh.file,
                    mode: 'timeline'
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to start profiling');
            }
            
            const result = await response.json();
            console.log('Profiling session started:', result.session_id);
            
            // Dispatch event for profiling panel to track progress
            const event = new CustomEvent('profilingStarted', {
                detail: {
                    sessionId: result.session_id,
                    model: model,
                    mesh: selectedMesh,
                    solver: solverType
                }
            });
            document.dispatchEvent(event);
            
        } catch (error) {
            console.error('Failed to start profiling:', error);
            
            // Dispatch error event
            const event = new CustomEvent('profilingError', {
                detail: { error: error.message }
            });
            document.dispatchEvent(event);
        }
    }
    
    // Public methods for external control
    
    open() {
        if (this.gallery) {
            this.gallery.classList.add('visible');
            // Update profiling button state when opening
            this.updateProfilingButtonState();
        }
    }
    
    close() {
        if (this.gallery) {
            // Disable transition for immediate hide
            this.gallery.style.transition = 'none';
            this.gallery.classList.remove('visible');
            // Force reflow to apply the change immediately
            void this.gallery.offsetHeight;
            // Re-enable transition for future animations
            this.gallery.style.transition = '';
            
            // Sync menu button state (deactivate the gallery button)
            if (window.menuManager) {
                const menuEl = window.menuManager.menuEl;
                if (menuEl) {
                    const btns = Array.from(menuEl.querySelectorAll('button'));
                    const galleryIdx = window.menuManager.cfg.panelIds.indexOf('gallery');
                    if (galleryIdx >= 0 && btns[galleryIdx]) {
                        btns[galleryIdx].classList.remove('active');
                    }
                }
            }
        }
    }
    
    isVisible() {
        return this.gallery && this.gallery.classList.contains('visible');
    }
    
    toggle() {
        if (this.isVisible()) {
            this.close();
        } else {
            this.open();
        }
    }
    
    getSelectedModel() {
        return this.models[this.selectedIndex] || null;
    }
    
    getSelectedMesh() {
        return this.getSelectedMeshForItem(this.selectedIndex);
    }
    
    /**
     * Get preloaded mesh data (if available)
     */
    getPreloadedMeshData() {
        return this.preloadedMeshData;
    }
    
    /**
     * Check if mesh is ready (preloaded)
     */
    isMeshReady() {
        return this.preloadedMeshData !== null;
    }
    
    /**
     * Get solver for a specific model item
     */
    getSolverForItem(index) {
        const item = this.track?.querySelector(`.carousel-item[data-index="${index}"]`);
        const dropdown = item?.querySelector('.solver-select');
        return dropdown?.value || this.models[index]?.solver_type || 'gpu';
    }
    
    /**
     * Set solver for a specific model item
     */
    setSolverForItem(index, solverId) {
        const item = this.track?.querySelector(`.carousel-item[data-index="${index}"]`);
        const dropdown = item?.querySelector('.solver-select');
        if (dropdown && this.solvers.some(s => s.id === solverId)) {
            dropdown.value = solverId;
            this.updateProfilingButtonState();
        }
    }
    
    /**
     * Set mesh for a specific model item by mesh index
     */
    setMeshForItem(modelIndex, meshIndex) {
        const item = this.track?.querySelector(`.carousel-item[data-index="${modelIndex}"]`);
        const dropdown = item?.querySelector('.mesh-select');
        if (dropdown && this.models[modelIndex]?.meshes[meshIndex]) {
            dropdown.value = meshIndex;
            this.onMeshSelectionChange(modelIndex, meshIndex.toString());
        }
    }
}

// Initialize gallery when DOM is ready
let meshGallery;

document.addEventListener('DOMContentLoaded', () => {
    meshGallery = new MeshGallery({
        jsonPath: './config/gallery_files.json',
        itemsPerView: 3
    });
    
    // Expose to window for console access
    window.meshGallery = meshGallery;
    window.meshLoader = meshLoader;
});

// Console helper functions
window.openGallery = () => meshGallery?.open();
window.closeGallery = () => meshGallery?.close();
window.toggleGallery = () => meshGallery?.toggle();

export { MeshGallery, meshGallery };
