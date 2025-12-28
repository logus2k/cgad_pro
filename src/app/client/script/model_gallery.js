/**
 * Model Gallery Manager - Phase 2 (Dynamic Metadata & Descriptions)
 * Handles .h5 file listing, rich metadata, and descriptive overlays.
 */
export class ModelGallery {
    
    constructor(serverUrl = null) {
        
        // Dynamic URL resolution
        if (!serverUrl) {
            const hostname = window.location.hostname;
            const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';
            serverUrl = window.location.origin + (isLocal ? '' : '/fem');
        }        
        
        this.serverUrl = serverUrl;
        this.models = [];
        this.originalModelsCount = 0;
        this.visibleItems = 3;
        
        this.currentIndex = 0;
        this.selectedModelIndex = null;
        
        // DOM Elements
        this.modal = document.getElementById('hud-gallery');
        this.carouselTrack = document.getElementById('carouselTrack');
        this.navDots = document.getElementById('navDots');
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.selectBtn = document.getElementById('selectBtn');
        this.cancelBtn = document.getElementById('cancelBtn');
        
        this.setupEventListeners();
    }

    /**
     * Fetch the JSON list of models from the backend
     */
    async init() {
        try {
            console.log('Loading mesh catalog...');
            const response = await fetch(`${this.serverUrl}/meshes`);
            const data = await response.json();
            
            this.models = data.meshes || [];
            this.originalModelsCount = this.models.length;

            if (this.originalModelsCount > 0) {
                this.prepareClones();
                this.render();
            } else {
                this.carouselTrack.innerHTML = '<div class="error-msg">No H5 meshes found.</div>';
            }
        } catch (error) {
            console.error('Gallery Fetch Error:', error);
            this.carouselTrack.innerHTML = '<div class="error-msg">Backend Connection Failed</div>';
        }
    }

    prepareClones() {
        // Create seamless loop clones (3 at each end)
        this.clonedModels = [
            ...this.models.slice(-this.visibleItems),
            ...this.models,
            ...this.models.slice(0, this.visibleItems)
        ];
        this.currentIndex = this.visibleItems;
    }

    setupEventListeners() {
        this.prevBtn.addEventListener('click', () => this.movePrev());
        this.nextBtn.addEventListener('click', () => this.moveNext());
        this.selectBtn.addEventListener('click', () => this.selectModel());
        this.cancelBtn.addEventListener('click', () => this.cancelSelection());
        window.addEventListener('resize', () => this.updatePosition());
    }

    render() {
        this.navDots.innerHTML = '';
        for (let i = 0; i < this.originalModelsCount; i++) {
            const dot = document.createElement('div');
            dot.className = `dot ${i === (this.currentIndex - this.visibleItems) % this.originalModelsCount ? 'active' : ''}`;
            this.navDots.appendChild(dot);
        }

        this.carouselTrack.innerHTML = this.clonedModels.map((model, index) => {
            const isSelected = this.selectedModelIndex === index;
            return `
                <div class="carousel-item ${isSelected ? 'selected-indicator' : ''}" 
                     onclick="window.galleryInstance.handleItemClick(${index})">
                    
                    <div class="model-image-container">
                        <img src="${model.thumbnail || 'assets/placeholder-mesh.png'}" class="model-image">
                    </div>

                    <div class="model-info">
                        <div class="model-name">${model.name}</div>
                        <p class="model-description">${model.description || 'No description available.'}</p>
                        
                        <div class="model-metadata">
                            <div class="meta-row">
                                <span class="label">Nodes:</span>
                                <span class="val">${model.nodes?.toLocaleString()}</span>
                            </div>
                            <div class="meta-row">
                                <span class="label">Elements:</span>
                                <span class="val">${model.elements?.toLocaleString()}</span>
                            </div>
                            <div class="complexity-badge ${this.getComplexityClass(model.nodes)}">
                                ${this.getComplexityLabel(model.nodes)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        window.galleryInstance = this;
        this.updatePosition(false);
    }

    handleItemClick(index) {
        this.selectedModelIndex = index;
        this.render();
    }

    getComplexityLabel(nodes) {
        if (nodes < 5000) return 'Low Intensity';
        if (nodes < 25000) return 'Moderate';
        return 'High Compute';
    }

    getComplexityClass(nodes) {
        if (nodes < 5000) return 'complexity-low';
        if (nodes < 25000) return 'complexity-med';
        return 'complexity-high';
    }

    updatePosition(animate = true) {
        const offset = -this.currentIndex * (100 / this.visibleItems);
        this.carouselTrack.style.transition = animate ? 'transform 0.4s cubic-bezier(0.2, 0, 0.2, 1)' : 'none';
        this.carouselTrack.style.transform = `translateX(${offset}%)`;
        
        const realIndex = (this.currentIndex - this.visibleItems + this.originalModelsCount) % this.originalModelsCount;
        const dots = this.navDots.querySelectorAll('.dot');
        dots.forEach((dot, i) => dot.classList.toggle('active', i === realIndex));
    }

    moveNext() {
        this.currentIndex++;
        this.updatePosition(true);
        if (this.currentIndex >= this.clonedModels.length - this.visibleItems) {
            setTimeout(() => { this.currentIndex = this.visibleItems; this.updatePosition(false); }, 400);
        }
    }

    movePrev() {
        this.currentIndex--;
        this.updatePosition(true);
        if (this.currentIndex < this.visibleItems) {
            setTimeout(() => { this.currentIndex = this.clonedModels.length - (this.visibleItems * 2) + this.visibleItems; this.updatePosition(false); }, 400);
        }
    }

    selectModel() {
        if (this.selectedModelIndex !== null) {
            const selectedModel = this.clonedModels[this.selectedModelIndex];
            window.dispatchEvent(new CustomEvent('modelSelected', { detail: selectedModel }));
            this.hide();
        }
    }

    show() {
        this.modal.style.display = 'block';
        if (this.models.length === 0) this.init();
    }

    hide() { this.modal.style.display = 'none'; }
}
