/**
 * Client Hardware Detection Utility
 * 
 * Detects browser, OS, and GPU information for benchmark recording.
 * Should be called once on application startup and stored for use
 * with solve requests.
 * 
 * Location: /src/app/client/script/client-hardware.js
 */

/**
 * Detect client hardware configuration.
 * Uses WebGL to get GPU information.
 * 
 * @returns {Object} Client hardware configuration
 */
export function detectClientHardware() {
    const config = {
        // Browser info
        browser: detectBrowser(),
        browser_version: detectBrowserVersion(),
        
        // OS info
        os: detectOS(),
        os_version: detectOSVersion(),
        
        // Device info
        device_type: detectDeviceType(),
        screen_width: window.screen.width,
        screen_height: window.screen.height,
        device_pixel_ratio: window.devicePixelRatio || 1,
        
        // GPU info (from WebGL)
        gpu_vendor: null,
        gpu_renderer: null,
        webgl_version: null,
        
        // Timestamp
        detected_at: new Date().toISOString()
    };
    
    // Detect GPU via WebGL
    const gpuInfo = detectGPU();
    if (gpuInfo) {
        config.gpu_vendor = gpuInfo.vendor;
        config.gpu_renderer = gpuInfo.renderer;
        config.webgl_version = gpuInfo.webglVersion;
    }
    
    return config;
}

/**
 * Detect browser name
 */
function detectBrowser() {
    const ua = navigator.userAgent;
    
    if (ua.includes('Firefox')) return 'Firefox';
    if (ua.includes('Edg/')) return 'Edge';
    if (ua.includes('Chrome')) return 'Chrome';
    if (ua.includes('Safari')) return 'Safari';
    if (ua.includes('Opera') || ua.includes('OPR')) return 'Opera';
    
    return 'Unknown';
}

/**
 * Detect browser version
 */
function detectBrowserVersion() {
    const ua = navigator.userAgent;
    let match;
    
    if (ua.includes('Firefox')) {
        match = ua.match(/Firefox\/(\d+)/);
    } else if (ua.includes('Edg/')) {
        match = ua.match(/Edg\/(\d+)/);
    } else if (ua.includes('Chrome')) {
        match = ua.match(/Chrome\/(\d+)/);
    } else if (ua.includes('Safari')) {
        match = ua.match(/Version\/(\d+)/);
    } else if (ua.includes('OPR')) {
        match = ua.match(/OPR\/(\d+)/);
    }
    
    return match ? match[1] : 'Unknown';
}

/**
 * Detect operating system
 */
function detectOS() {
    const ua = navigator.userAgent;
    const platform = navigator.platform;
    
    if (ua.includes('Windows')) return 'Windows';
    if (ua.includes('Mac OS')) return 'macOS';
    if (ua.includes('Linux')) return 'Linux';
    if (ua.includes('Android')) return 'Android';
    if (ua.includes('iOS') || ua.includes('iPhone') || ua.includes('iPad')) return 'iOS';
    if (ua.includes('CrOS')) return 'Chrome OS';
    
    return platform || 'Unknown';
}

/**
 * Detect OS version
 */
function detectOSVersion() {
    const ua = navigator.userAgent;
    let match;
    
    if (ua.includes('Windows')) {
        match = ua.match(/Windows NT (\d+\.\d+)/);
        if (match) {
            const version = match[1];
            // Map Windows NT versions to friendly names
            const versions = {
                '10.0': '10/11',
                '6.3': '8.1',
                '6.2': '8',
                '6.1': '7',
                '6.0': 'Vista'
            };
            return versions[version] || version;
        }
    } else if (ua.includes('Mac OS X')) {
        match = ua.match(/Mac OS X (\d+[._]\d+)/);
        if (match) return match[1].replace('_', '.');
    } else if (ua.includes('Android')) {
        match = ua.match(/Android (\d+(\.\d+)?)/);
        if (match) return match[1];
    } else if (ua.includes('iOS') || ua.includes('iPhone') || ua.includes('iPad')) {
        match = ua.match(/OS (\d+[._]\d+)/);
        if (match) return match[1].replace('_', '.');
    }
    
    return 'Unknown';
}

/**
 * Detect device type
 */
function detectDeviceType() {
    const ua = navigator.userAgent;
    
    if (/Tablet|iPad/i.test(ua)) return 'Tablet';
    if (/Mobile|iPhone|Android.*Mobile/i.test(ua)) return 'Mobile';
    
    return 'Desktop';
}

/**
 * Detect GPU information via WebGL
 */
function detectGPU() {
    // Try WebGL2 first, then fall back to WebGL1
    let canvas, gl, debugInfo;
    
    try {
        canvas = document.createElement('canvas');
        
        // Try WebGL2
        gl = canvas.getContext('webgl2');
        let webglVersion = '2.0';
        
        if (!gl) {
            // Fall back to WebGL1
            gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            webglVersion = '1.0';
        }
        
        if (!gl) {
            console.warn('[ClientHardware] WebGL not available');
            return null;
        }
        
        // Get debug info extension
        debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        
        if (debugInfo) {
            const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
            const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
            
            return {
                vendor: vendor,
                renderer: renderer,
                webglVersion: webglVersion
            };
        } else {
            // Fallback to basic info
            return {
                vendor: gl.getParameter(gl.VENDOR),
                renderer: gl.getParameter(gl.RENDERER),
                webglVersion: webglVersion
            };
        }
        
    } catch (e) {
        console.error('[ClientHardware] Error detecting GPU:', e);
        return null;
    } finally {
        // Cleanup
        if (gl) {
            const ext = gl.getExtension('WEBGL_lose_context');
            if (ext) ext.loseContext();
        }
    }
}

/**
 * Generate a hash of the client configuration for grouping.
 * Uses a simple hash of key fields.
 * 
 * @param {Object} config - Client configuration object
 * @returns {string} 12-character hash
 */
export function generateClientHash(config) {
    const hashInput = [
        config.browser || '',
        config.os || '',
        config.gpu_vendor || '',
        config.gpu_renderer || ''
    ].join('|');
    
    // Simple hash function (djb2)
    let hash = 5381;
    for (let i = 0; i < hashInput.length; i++) {
        hash = ((hash << 5) + hash) + hashInput.charCodeAt(i);
        hash = hash & hash; // Convert to 32-bit integer
    }
    
    // Convert to hex string
    return Math.abs(hash).toString(16).padStart(8, '0').slice(0, 12);
}

/**
 * ClientHardware singleton class for managing hardware detection.
 * Detects once and caches the result.
 */
class ClientHardwareManager {
    constructor() {
        this._config = null;
        this._hash = null;
    }
    
    /**
     * Get client hardware configuration (cached).
     * @returns {Object} Client configuration
     */
    getConfig() {
        if (!this._config) {
            this._config = detectClientHardware();
            this._hash = generateClientHash(this._config);
            
            console.log('[ClientHardware] Detected configuration:');
            console.log(`  Browser: ${this._config.browser} ${this._config.browser_version}`);
            console.log(`  OS: ${this._config.os} ${this._config.os_version}`);
            console.log(`  Device: ${this._config.device_type}`);
            console.log(`  GPU: ${this._config.gpu_renderer || 'Unknown'}`);
            console.log(`  Hash: ${this._hash}`);
        }
        return this._config;
    }
    
    /**
     * Get client configuration hash (cached).
     * @returns {string} Configuration hash
     */
    getHash() {
        if (!this._hash) {
            this.getConfig(); // Trigger detection
        }
        return this._hash;
    }
    
    /**
     * Get configuration with hash included.
     * @returns {Object} Configuration with hash
     */
    getConfigWithHash() {
        const config = this.getConfig();
        return {
            ...config,
            hash: this.getHash()
        };
    }
}

// Export singleton instance
export const clientHardware = new ClientHardwareManager();

// Also export for direct use
export default clientHardware;
