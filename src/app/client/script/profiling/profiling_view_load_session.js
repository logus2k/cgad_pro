/**
 * Updated loadSession method for ProfilingView.
 * 
 * This replaces the existing loadSession method in profiling.view.js
 * to support HDF5 loading with accurate progress display.
 * 
 * Location: /src/app/client/script/profiling/profiling.view.js
 */

    /**
     * Load and display a specific session's timeline.
     * @param {string} sessionId 
     */
    async loadSession(sessionId) {
        if (!sessionId) return;

        this.#currentSessionId = sessionId;
        this.#showLoading(true, 'Checking data format...');
        this.#updateProgress(0);

        try {
            // Fetch timeline data with progress
            const timelineData = await this.#api.getTimeline(sessionId, {
                onProgress: (phase, loaded, total) => {
                    this.#handleLoadProgress(phase, loaded, total);
                }
            });
            
            const eventCount = timelineData.events?.length || 0;
            this.#showLoading(true, `Rendering ${eventCount.toLocaleString()} events...`);
            this.#updateProgress(90);

            // Load into timeline renderer
            if (this.#timeline) {
                await this.#timeline.load(timelineData, (progress) => {
                    // Map render progress to 90-100%
                    this.#updateProgress(90 + progress * 0.1);
                });
            }

            this.#updateSummaryCards(timelineData);
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
     * Maps phases to user-friendly messages and progress percentages.
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
                    const loadedKB = (loaded / 1024).toFixed(0);
                    const totalKB = (total / 1024).toFixed(0);
                    const loadedMB = (loaded / (1024 * 1024)).toFixed(1);
                    const totalMB = (total / (1024 * 1024)).toFixed(1);
                    
                    if (total > 1024 * 1024) {
                        message = `Downloading... ${loadedMB} / ${totalMB} MB`;
                    } else {
                        message = `Downloading... ${loadedKB} / ${totalKB} KB`;
                    }
                    percent = Math.round((loaded / total) * 70); // 0-70%
                } else {
                    message = `Downloading... ${(loaded / 1024).toFixed(0)} KB`;
                    percent = 30; // Unknown total, show indeterminate-ish
                }
                break;
                
            case 'parse':
                message = 'Parsing timeline data...';
                percent = 70 + Math.round((loaded / 100) * 10); // 70-80%
                break;
                
            case 'convert':
                message = 'Preparing events...';
                percent = 80 + Math.round((loaded / 100) * 10); // 80-90%
                break;
                
            default:
                message = 'Loading...';
                percent = 50;
        }
        
        this.#showLoading(true, message);
        this.#updateProgress(percent);
    }
