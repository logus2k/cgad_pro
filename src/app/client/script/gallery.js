/**
 * Mesh Gallery Carousel
 * Handles carousel navigation, item selection, and model loading
 */

class MeshGallery {
    constructor(options = {}) {
        this.jsonPath = options.jsonPath || 'gallery_files.json';
        this.itemsPerView = options.itemsPerView || 3;
        this.meshes = [];
        this.currentIndex = 0;
        this.selectedIndex = 0;
        
        // DOM elements
        this.track = document.getElementById('carouselTrack');
        this.navDots = document.getElementById('navDots');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.selectBtn = document.getElementById('selectBtn');
        this.cancelBtn = document.getElementById('cancelBtn');
        this.gallery = document.getElementById('hud-gallery');
        
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
    
    selectItem(index) {
        if (index < 0 || index >= this.meshes.length) return;
        
        this.selectedIndex = index;
        
        // Update visual selection
        const items = this.track.querySelectorAll('.carousel-item');
        items.forEach((item, i) => {
            item.classList.toggle('selected-indicator', i === index);
        });
        
        // Ensure selected item is visible
        const page = Math.floor(index / this.itemsPerView);
        if (page !== Math.floor(this.currentIndex / this.itemsPerView)) {
            this.goToPage(page);
        }
        
        console.log(`Selected: ${this.meshes[index].name}`);
    }
    
    confirmSelection() {
        const selected = this.meshes[this.selectedIndex];
        if (!selected) {
            console.warn('No mesh selected');
            return;
        }
        
        console.log('=== MESH SELECTION CONFIRMED ===');
        console.log('Name:', selected.name);
        console.log('File:', selected.file);
        console.log('Nodes:', selected.nodes);
        console.log('Elements:', selected.elements);
        console.log('Description:', selected.description);
        console.log('================================');
        
        // Dispatch custom event for external listeners
        const event = new CustomEvent('meshSelected', {
            detail: {
                mesh: selected,
                index: this.selectedIndex
            }
        });
        document.dispatchEvent(event);
        
        // Close gallery after selection
        this.close();
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
});

// Console helper functions
window.openGallery = () => meshGallery?.open();
window.closeGallery = () => meshGallery?.close();
window.toggleGallery = () => meshGallery?.toggle();
