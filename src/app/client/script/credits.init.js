/**
 * Credits Scroll Handler
 * Seamless loop with wheel scroll support on hover
 */

export function initCreditsScroll() {
    const container = document.querySelector('.about-container.credits-mode');
    const creditsScroll = document.querySelector('.credits-scroll');
    
    if (!container || !creditsScroll) return;

    // Clone content for seamless loop
    const loop = document.querySelector('.credits-loop');
    if (loop) {
        const clone = loop.cloneNode(true);
        loop.parentNode.appendChild(clone);
    }

    let isHovering = false;
    let currentY = 0;
    let loopHeight = 0;

    // Get the current translateY from computed style matrix
    const getComputedTranslateY = () => {
        const style = window.getComputedStyle(creditsScroll);
        const matrix = style.transform;
        if (matrix === 'none') return 0;
        
        const match = matrix.match(/matrix.*\((.+)\)/);
        if (match) {
            const values = match[1].split(', ');
            const ty = values.length === 6 ? parseFloat(values[5]) : parseFloat(values[13]);
            return ty || 0;
        }
        return 0;
    };

    // Apply transform with seamless wrapping
    const setTranslateY = (y) => {
        if (loopHeight <= 0) {
            loopHeight = creditsScroll.scrollHeight / 2;
        }
        
        if (loopHeight > 0) {
            while (y <= -loopHeight) y += loopHeight;
            while (y > 0) y -= loopHeight;
        }
        currentY = y;
        creditsScroll.style.transform = `translateY(${y}px)`;
    };

    container.addEventListener('mouseenter', () => {
        isHovering = true;
        
        // Pause animation to freeze position
        creditsScroll.style.animationPlayState = 'paused';
        void creditsScroll.offsetHeight;
        
        // Get frozen position
        const computedY = getComputedTranslateY();
        
        // Recalculate loop height
        loopHeight = creditsScroll.scrollHeight / 2;
        
        // Stop animation and apply exact position
        creditsScroll.style.animation = 'none';
        currentY = computedY;
        creditsScroll.style.transform = `translateY(${currentY}px)`;
    });

    container.addEventListener('mouseleave', () => {
        isHovering = false;
        
        // Recalculate loop height
        loopHeight = creditsScroll.scrollHeight / 2;
        
        // Normalize currentY
        let normalizedY = currentY;
        if (loopHeight > 0) {
            while (normalizedY <= -loopHeight) normalizedY += loopHeight;
            while (normalizedY > 0) normalizedY -= loopHeight;
        }
        
        // Calculate progress (0 to 1)
        let progress = loopHeight > 0 ? -normalizedY / loopHeight : 0;
        
        // Restart animation from position
        creditsScroll.style.transform = '';
        creditsScroll.style.animationPlayState = '';
        creditsScroll.style.animation = 'none';
        void creditsScroll.offsetHeight;
        
        const delay = -(progress * 60); // 60s duration
        creditsScroll.style.animation = `creditsScroll 60s linear infinite`;
        creditsScroll.style.animationDelay = `${delay}s`;
    });

    // Wheel scroll when hovering
    container.addEventListener('wheel', (e) => {
        if (!isHovering) return;
        
        e.preventDefault();
        setTranslateY(currentY - e.deltaY);
    }, { passive: false });
}

// Auto-init when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => setTimeout(initCreditsScroll, 100));
} else {
    setTimeout(initCreditsScroll, 100);
}
