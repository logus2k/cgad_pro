/**
 * Updated loadSession method for ProfilingView - Optimized version.
 * 
 * Uses typed arrays directly from HDF5 when available, skipping
 * the intermediate event object conversion for maximum performance.
 * 
 * Replace the existing loadSession method in profiling.view.js with this.
 * 
 * Location: /src/app/client/script/profiling/profiling.view.js
 */

    /**
     * Load and display a specific session's timeline.
     * Uses optimized typed array path when HDF5 is available.
     * 
     * @param {string} sessionId 
     */
    async loadSession(sessionId) {
        if (!sessionId) return;

        this.#currentSessionId = sessionId;
        this.#showLoading(true, 'Checking data format...');
        this.#updateProgress(0);

        try {
            // Try optimized path first (typed arrays from HDF5)
            let useTypedPath = false;
            let rendererData = null;
            
            try {
                rendererData = await this.#api.getTimelineForRenderer(sessionId, {
                    onProgress: (phase, loaded, total) => {
                        this.#handleLoadProgress(phase, loaded, total);
                    }
                });
                useTypedPath = true;
            } catch (err) {
                // HDF5 not available, fall back to legacy
                console.log('[ProfilingView] HDF5 not available, using JSON fallback');
            }
            
            if (useTypedPath && rendererData) {
                // Optimized path: typed arrays directly to renderer
                this.#showLoading(true, `Rendering ${rendererData.totalEvents.toLocaleString()} events...`);
                this.#updateProgress(90);
                
                if (this.#timeline) {
                    await this.#timeline.loadTyped(rendererData, (progress) => {
                        this.#updateProgress(90 + progress * 0.1);
                    });
                }
                
                this.#updateSummaryCardsFromRendererData(rendererData);
            } else {
                // Legacy path: JSON with event objects
                const timelineData = await this.#api.getTimeline(sessionId, {
                    onProgress: (phase, loaded, total) => {
                        this.#handleLoadProgress(phase, loaded, total);
                    }
                });
                
                const eventCount = timelineData.events?.length || 0;
                this.#showLoading(true, `Rendering ${eventCount.toLocaleString()} events...`);
                this.#updateProgress(90);

                if (this.#timeline) {
                    await this.#timeline.load(timelineData, (progress) => {
                        this.#updateProgress(90 + progress * 0.1);
                    });
                }

                this.#updateSummaryCards(timelineData);
            }

            this.#showLoading(false);
            this.#showStatus('', 'idle');
            
        } catch (error) {
            console.error('[ProfilingView] Failed to load timeline:', error);
            this.#showLoading(false);
            this.#showStatus('Failed to load timeline', 'error');
        }
    }
    
    /**
     * Handle progress updates from API.
     */
    #handleLoadProgress(phase, loaded, total) {
        let message;
        let percent;
        
        switch (phase) {
            case 'check':
                message = 'Checking data format...';
                percent = 0;
                break;
                
            case 'download':
                if (total > 0) {
                    const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
                    const totalMB = (total / (1024 * 1024)).toFixed(1);
                    
                    if (total > 1024 * 1024) {
                        message = `Downloading... ${loadedMB} / ${totalMB} MB`;
                    } else {
                        const loadedKB = (loaded / 1024).toFixed(0);
                        const totalKB = (total / 1024).toFixed(0);
                        message = `Downloading... ${loadedKB} / ${totalKB} KB`;
                    }
                    percent = Math.round((loaded / total) * 70);
                } else {
                    message = `Downloading... ${(loaded / 1024).toFixed(0)} KB`;
                    percent = 30;
                }
                break;
                
            case 'parse':
                message = 'Parsing timeline data...';
                percent = 70 + Math.round((loaded / 100) * 10);
                break;
                
            case 'convert':
                message = 'Preparing events...';
                percent = 80 + Math.round((loaded / 100) * 10);
                break;
                
            default:
                message = 'Loading...';
                percent = 50;
        }
        
        this.#showLoading(true, message);
        this.#updateProgress(percent);
    }
    
    /**
     * Update summary cards from renderer data (typed array format).
     */
    #updateSummaryCardsFromRendererData(rendererData) {
        if (!rendererData) return;

        document.getElementById('prof-total-duration').textContent = 
            this.#formatDuration(rendererData.totalDurationMs);
        
        const kernelData = rendererData.byCategory.cuda_kernel;
        document.getElementById('prof-kernel-count').textContent = 
            kernelData?.count || 0;
        
        const memcpyCount = 
            (rendererData.byCategory.cuda_memcpy_h2d?.count || 0) +
            (rendererData.byCategory.cuda_memcpy_d2h?.count || 0) +
            (rendererData.byCategory.cuda_memcpy_d2d?.count || 0);
        document.getElementById('prof-memcpy-count').textContent = memcpyCount;
        
        const nvtxData = rendererData.byCategory.nvtx_range;
        document.getElementById('prof-nvtx-count').textContent = 
            nvtxData?.count || 0;
    }
