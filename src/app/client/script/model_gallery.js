export class ModelGallery {
    constructor(models) {
        this.models = models;
        this.originalModelsCount = models.length;
        this.visibleItems = 3;
        
        // Handle single image case
        if (models.length === 1) {
            this.models.push(models[0], models[0]); // Add two more for smooth operation
        } else if (models.length === 2) {
            this.models.push(models[0], models[1]); // Add clones to make it work properly
        }
        
        // Clone first and last items for seamless looping
        this.clonedModels = [
            ...this.models.slice(-this.visibleItems), // Last items at start
            ...this.models,
            ...this.models.slice(0, this.visibleItems)  // First items at end
        ];
        
        this.currentIndex = this.visibleItems; // Start at first real item
        this.selectedModelIndex = null; // No initial selection
        
        this.modal = document.getElementById('hud-gallery');
        this.carouselTrack = document.getElementById('carouselTrack');
        this.navDots = document.getElementById('navDots');
        
        this.prevBtn = document.getElementById('prevBtn');
        this.nextBtn = document.getElementById('nextBtn');
        this.selectBtn = document.getElementById('selectBtn');
        this.cancelBtn = document.getElementById('cancelBtn');
        
        this.setupEventListeners();
        this.render();
    }
    
    setupEventListeners() {
        this.prevBtn.addEventListener('click', () => this.prev());
        this.nextBtn.addEventListener('click', () => this.next());
        this.selectBtn.addEventListener('click', () => this.selectModel());
        this.cancelBtn.addEventListener('click', () => this.cancelSelection());
        
        // Touch/drag events for mobile
        let startX = 0;
        let currentX = 0;
        
        this.carouselTrack.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
        });
        
        this.carouselTrack.addEventListener('touchmove', (e) => {
            currentX = e.touches[0].clientX;
        });
        
        this.carouselTrack.addEventListener('touchend', (e) => {
            const diff = startX - currentX;
            if (Math.abs(diff) > 50) { // Swipe threshold
                if (diff > 0) {
                    this.next();
                } else {
                    this.prev();
                }
            }
        });
        
        // Mouse drag events for desktop
        let isDragging = false;
        let startMouseX = 0;
        
        this.carouselTrack.addEventListener('mousedown', (e) => {
            isDragging = true;
            startMouseX = e.clientX;
            this.carouselTrack.style.transition = 'none';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const diff = e.clientX - startMouseX;
            // Visual feedback during drag (optional)
        });
        
        document.addEventListener('mouseup', (e) => {
            if (!isDragging) return;
            isDragging = false;
            this.carouselTrack.style.transition = 'transform 0.3s ease';
            
            const diff = e.clientX - startMouseX;
            if (Math.abs(diff) > 50) { // Drag threshold
                if (diff > 0) {
                    this.next();
                } else {
                    this.prev();
                }
            } else {
                this.updatePosition();
            }
        });
    }
    
    render() {
        this.carouselTrack.innerHTML = '';
        
        // Create all items including clones
        for (let i = 0; i < this.clonedModels.length; i++) {
            const modelIndex = i >= this.visibleItems && i < this.visibleItems + this.originalModelsCount 
                ? i - this.visibleItems 
                : (i - this.visibleItems + this.originalModelsCount) % this.originalModelsCount;
            
            const item = this.createItem(modelIndex, i);
            this.carouselTrack.appendChild(item);
        }
        
        // Update navigation dots
        this.updateNavDots();
        
        // Position the track with proper infinite loop handling
        this.updatePosition();
    }
    
    createItem(originalIndex, positionIndex) {
        const item = document.createElement('div');
        item.className = 'carousel-item';
        
        // Get the actual model data (this is the correct model index)
        const actualModelIndex = originalIndex % this.originalModelsCount;
        
        // Check if this item corresponds to the selected model
        if (this.selectedModelIndex !== null && actualModelIndex === this.selectedModelIndex) {
            item.classList.add('selected-indicator');
        }
        
        const model = this.models[actualModelIndex];
        
        item.innerHTML = `
            <img src="${model.image}" alt="${model.name}" class="model-image">
            <div class="model-name">${model.name}</div>
        `;
        
        // Click handler for selection
        item.addEventListener('click', () => {
            // Use the actual model index from the parameter
            const clickedModelIndex = actualModelIndex;
            
            // Determine which position this item is in the current view
            const visibleStart = this.currentIndex;
            const visibleEnd = this.currentIndex + this.visibleItems - 1;
            
            // Check if it's in the current visible area
            if (positionIndex >= visibleStart && positionIndex <= visibleEnd) {
                const relativePosition = positionIndex - visibleStart;
                
                if (relativePosition === 0) { // Leftmost item
                    // Temporarily disable transition for instant positioning when wrapping
                    this.carouselTrack.style.transition = 'none';
                    
                    this.currentIndex--;
                    
                    // Handle infinite loop: if we go before the first real item
                    if (this.currentIndex < this.visibleItems) {
                        // Instantly jump to the corresponding real item at the end
                        this.currentIndex = this.clonedModels.length - this.visibleItems * 2;
                    }
                    
                    // Apply the new position instantly
                    this.updatePosition();
                    
                    // Re-enable transition after a brief moment
                    setTimeout(() => {
                        this.carouselTrack.style.transition = 'transform 0.3s ease';
                    }, 10);
                    
                    this.selectedModelIndex = clickedModelIndex;
                } else if (relativePosition === 2) { // Rightmost item
                    // Temporarily disable transition for instant positioning when wrapping
                    this.carouselTrack.style.transition = 'none';
                    
                    this.currentIndex++;
                    
                    // Handle infinite loop: if we go past the last real item
                    if (this.currentIndex >= this.clonedModels.length - this.visibleItems) {
                        // Instantly jump back to the corresponding real item at the start
                        this.currentIndex = this.visibleItems;
                    }
                    
                    // Apply the new position instantly
                    this.updatePosition();
                    
                    // Re-enable transition after a brief moment
                    setTimeout(() => {
                        this.carouselTrack.style.transition = 'transform 0.3s ease';
                    }, 10);
                    
                    this.selectedModelIndex = clickedModelIndex;
                } else if (relativePosition === 1) { // Middle item
                    // Don't move, just select
                    this.selectedModelIndex = clickedModelIndex;
                }
            } else {
                // If not in view, center it
                // Find the nearest occurrence of this model in the real section
                for (let j = this.visibleItems; j < this.visibleItems + this.originalModelsCount; j++) {
                    if ((j - this.visibleItems) % this.originalModelsCount === clickedModelIndex) {
                        this.currentIndex = j - 1; // Position to center this item
                        this.selectedModelIndex = clickedModelIndex;
                        break;
                    }
                }
            }
            
            // Update dots after transition
            setTimeout(() => this.updateNavDots(), 10);
        });
        
        return item;
    }
    
    updatePosition() {
        const translateX = -(this.currentIndex * (100 / this.visibleItems));
        this.carouselTrack.style.transform = `translateX(${translateX}%)`;
    }
    
    updateNavDots() {
        this.navDots.innerHTML = '';
        
        for (let i = 0; i < this.originalModelsCount; i++) {
            const dot = document.createElement('div');
            
            // The active dot is the one matching the selected model
            dot.className = `dot ${i === this.selectedModelIndex ? 'active' : ''}`;
            dot.addEventListener('click', () => {
                // Find the nearest occurrence of the selected model in the real section
                for (let j = this.visibleItems; j < this.visibleItems + this.originalModelsCount; j++) {
                    if ((j - this.visibleItems) % this.originalModelsCount === i) {
                        this.currentIndex = j - 1; // Center this item
                        this.selectedModelIndex = i;
                        this.render();
                        break;
                    }
                }
            });
            this.navDots.appendChild(dot);
        }
    }
    
    next() {
        // Temporarily disable transition for instant positioning when wrapping
        this.carouselTrack.style.transition = 'none';
        
        this.currentIndex++;
        
        // If we've moved past the real items to the cloned ones at the end
        if (this.currentIndex >= this.clonedModels.length - this.visibleItems) {
            // Instantly jump back to the corresponding real item
            this.currentIndex = this.visibleItems;
        }
        
        // Apply the new position instantly
        this.updatePosition();
        
        // Re-enable transition after a brief moment to prevent visible jump
        setTimeout(() => {
            this.carouselTrack.style.transition = 'transform 0.3s ease';
        }, 10);
    }
    
    prev() {
        // Temporarily disable transition for instant positioning when wrapping
        this.carouselTrack.style.transition = 'none';
        
        this.currentIndex--;
        
        // If we've moved before the real items to the cloned ones at the start
        if (this.currentIndex < this.visibleItems) {
            // Instantly jump to the corresponding real item at the end
            this.currentIndex = this.clonedModels.length - this.visibleItems * 2;
        }
        
        // Apply the new position instantly
        this.updatePosition();
        
        // Re-enable transition after a brief moment to prevent visible jump
        setTimeout(() => {
            this.carouselTrack.style.transition = 'transform 0.3s ease';
        }, 10);
    }
    
    selectModel() {
        if (this.selectedModelIndex !== null) {
            const selectedModel = this.models[this.selectedModelIndex % this.originalModelsCount];
            console.log('Selected model:', selectedModel);
            // Add your selection logic here
            this.hide();
        } else {
            alert('Please select a model first');
        }
    }
    
    cancelSelection() {
        this.selectedModelIndex = null; // Clear selection
        this.hide();
    }
    
    show() {
        this.selectedModelIndex = null; // Clear any previous selection
        this.currentIndex = this.visibleItems; // Start at first real item
        this.render();
        this.modal.style.display = 'block';
    }
    
    hide() {
        this.modal.style.display = 'none';
    }
}
