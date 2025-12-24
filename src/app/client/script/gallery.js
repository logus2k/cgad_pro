/**
 * Mesh Gallery Carousel
 * Handles carousel navigation, item selection, and model loading
 * 
 * Now includes mesh preloading - starts loading mesh data when user
 * selects an item (before confirming), so geometry is ready faster.
 */

import { meshLoader } from './mesh-loader.js';

class MeshGallery {
    constructor(options = {}) {
        this.jsonPath = options.jsonPath || 'gallery_files.json';
        this.itemsPerView = options.itemsPerView || 3;
        this.meshes = [];
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
        this.cancelBtn = document.getElementById('cancelBtn');
        this.gallery = document.getElementById('hud-gallery');

        // Clear existing text
        this.prevBtn.textContent = '';
        this.nextBtn.textContent = '';

        // Load SVG icons
        window.menuManager.getSVGIconByName(this.prevBtn, 'previous', 'Previous');
        window.menuManager.getSVGIconByName(this.nextBtn, 'next', 'Next');        
        
        // Set up mesh loader progress callback
        meshLoader.setProgressCallback((stage, progress) => {
            this.onMeshLoadProgress(stage, progress);
        });
        
        this.init();
    }
    
    async init() {
        await this.loadMeshData();
        this.renderCarousel();
        this.renderNavDots();
        this.bindEvents();
        this.updateNavigation();
        this.selectItem(0);
    }
    
    async loadMeshData() {
        try {
            const response = await fetch(this.jsonPath);
            if (!response.ok) throw new Error('Failed to load gallery data');
            const data = await response.json();
            this.meshes = data.meshes || [];
            console.log(`Loaded ${this.meshes.length} mesh models`);
        } catch (error) {
            console.error('Error loading mesh gallery:', error);
            this.meshes = [];
        }
    }
    
    renderCarousel() {
        if (!this.track) return;
        
        this.track.innerHTML = '';
        
        this.meshes.forEach((mesh, index) => {
            const item = document.createElement('div');
            item.className = 'carousel-item';
            item.dataset.index = index;
            
            item.innerHTML = `
                <div class="model-image-container">
                    <img class="model-image" src="${mesh.thumbnail}" alt="${mesh.name}" 
                         onerror="this.onerror=null; this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 150%22><rect fill=%22%23f0f0f0%22 width=%22200%22 height=%22150%22/><text x=%22100%22 y=%2275%22 text-anchor=%22middle%22 fill=%22%23999%22 font-family=%22sans-serif%22 font-size=%2214%22>No Preview</text></svg>'">
                    <div class="preload-indicator" style="display: none;">
                        <div class="preload-spinner"></div>
                        <span class="preload-text">Loading mesh...</span>
                    </div>
                </div>
                <div class="model-name">${mesh.name}</div>
                <div class="model-description">${mesh.description}</div>
                <div class="model-metadata">
                    <div class="meta-row">
                        <span>Nodes:</span>
                        <span>${mesh.nodes.toLocaleString()}</span>
                    </div>
                    <div class="meta-row">
                        <span>Elements:</span>
                        <span>${mesh.elements.toLocaleString()}</span>
                    </div>
                    <div class="complexity-badge ${this.getComplexityClass(mesh.elements)}">
                        ${this.getComplexityLabel(mesh.elements)}
                    </div>
                </div>
            `;
            
            item.addEventListener('click', () => this.selectItem(index));
            this.track.appendChild(item);
        });
    }
    
    getComplexityClass(elements) {
        if (elements < 5000) return 'complexity-low';
        if (elements < 50000) return 'complexity-med';
        return 'complexity-high';
    }
    
    getComplexityLabel(elements) {
        if (elements < 5000) return 'Low Complexity';
        if (elements < 50000) return 'Medium Complexity';
        return 'High Complexity';
    }
    
    renderNavDots() {
        if (!this.navDots) return;
        
        this.navDots.innerHTML = '';
        const totalPages = Math.ceil(this.meshes.length / this.itemsPerView);
        
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
        const totalPages = Math.ceil(this.meshes.length / this.itemsPerView);
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
        const totalPages = Math.ceil(this.meshes.length / this.itemsPerView);
        const currentPage = Math.floor(this.currentIndex / this.itemsPerView);
        
        if (this.prevBtn) {
            this.prevBtn.disabled = currentPage === 0;
        }
        
        if (this.nextBtn) {
            this.nextBtn.disabled = currentPage >= totalPages - 1;
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
        if (index < 0 || index >= this.meshes.length) return;
        
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
        
        const mesh = this.meshes[index];
        console.log(`Selected: ${mesh.name}`);
        
        // Start preloading mesh data immediately
        this.preloadMesh(mesh);
    }
    
    /**
     * Preload mesh data (non-blocking)
     */
    async preloadMesh(mesh) {
        const meshUrl = mesh.file;
        
        // Skip if already preloading this mesh
        if (this.preloadingUrl === meshUrl) {
            return;
        }
        
        // Skip if already cached
        if (meshLoader.isCached(meshUrl)) {
            console.log(`Mesh already cached: ${mesh.name}`);
            this.preloadedMeshData = meshLoader.getCached(meshUrl);
            
            // Show ready indicator
            const item = this.track?.querySelector(`.carousel-item[data-index="${this.selectedIndex}"]`);
            if (item) item.classList.add('mesh-ready');
            return;
        }
        
        this.preloadingUrl = meshUrl;
        this.preloadedMeshData = null;
        
        try {
            console.log(`Preloading mesh: ${mesh.name} (${meshUrl})`);
            const meshData = await meshLoader.preload(meshUrl);
            
            // Only store if still the selected mesh
            if (this.meshes[this.selectedIndex]?.file === meshUrl) {
                this.preloadedMeshData = meshData;
                console.log(`Mesh preloaded and ready: ${mesh.name}`);
            }
        } catch (error) {
            console.error(`Failed to preload mesh: ${mesh.name}`, error);
            this.preloadedMeshData = null;
        } finally {
            if (this.preloadingUrl === meshUrl) {
                this.preloadingUrl = null;
            }
        }
    }
    
    /**
     * Confirm selection and dispatch event with preloaded mesh data
     */
    async confirmSelection() {
        const selected = this.meshes[this.selectedIndex];
        if (!selected) {
            console.warn('No mesh selected');
            return;
        }
        
        // Determine solver type based on mesh size
        const solverType = selected.elements > 10000 ? 'gpu' : 'gpu';
        
        console.log('=== MESH SELECTION CONFIRMED ===');
        console.log('Name:', selected.name);
        console.log('File:', selected.file);
        console.log('Solver:', solverType);
        console.log('================================');
        
        // Close gallery immediately for better UX
        this.close();
        
        // Wait for mesh preload if still in progress
        let meshData = this.preloadedMeshData;
        if (!meshData && this.preloadingUrl === selected.file) {
            console.log('Waiting for mesh preload to complete...');
            try {
                meshData = await meshLoader.load(selected.file);
                this.preloadedMeshData = meshData;
            } catch (error) {
                console.error('Mesh preload failed:', error);
            }
        }
        
        // Dispatch event with preloaded data for immediate geometry creation
        const event = new CustomEvent('meshSelected', {
            detail: {
                mesh: selected,
                index: this.selectedIndex,
                preloadedData: meshData,  // May be null if preload failed
                meshLoader: meshLoader
            }
        });
        document.dispatchEvent(event);
        
        // Start the solver (uses existing femClient from window)
        if (window.femClient) {
            window.femClient.startSolve({
                mesh_file: selected.file,
                solver_type: solverType,
                max_iterations: 50000,
                progress_interval: 100
            });
        } else {
            console.error('femClient not found on window');
        }
    }
    
    // Public methods for external control
    
    open() {
        if (this.gallery) {
            this.gallery.style.display = 'block';
        }
    }
    
    close() {
        if (this.gallery) {
            this.gallery.style.display = 'none';
        }
    }
    
    isVisible() {
        return this.gallery && this.gallery.style.display !== 'none';
    }
    
    toggle() {
        if (this.isVisible()) {
            this.close();
        } else {
            this.open();
        }
    }
    
    getSelectedMesh() {
        return this.meshes[this.selectedIndex] || null;
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
