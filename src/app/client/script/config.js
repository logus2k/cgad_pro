/**
 * Application Configuration
 * Centralized URL configuration for API endpoints.
 * 
 * Location: /src/app/client/script/config.js
 */

function getApiConfig() {
    const hostname = window.location.hostname;
    const isLocal = hostname === 'localhost' || hostname === '127.0.0.1';
    
    return {
        serverUrl: window.location.origin,
        basePath: isLocal ? '' : '/fem'
    };
}

export const apiConfig = getApiConfig();
