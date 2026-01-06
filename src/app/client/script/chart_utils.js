/**
 * chart_utils.js - Shared ECharts initialization utilities
 * 
 * Usage in any context:
 *   import { initializeCharts, disposeCharts } from './chart_utils.js';
 *   
 *   // After rendering Markdown to HTML:
 *   const instances = initializeCharts(container);
 *   
 *   // On cleanup:
 *   disposeCharts(container);
 * 
 * For EasyMDE:
 *   previewRender: (text, preview) => {
 *       const html = EasyMDE.prototype.markdown(text);
 *       setTimeout(() => initializeCharts(preview), 0);
 *       return html;
 *   }
 */

// Color palette for charts
const CHART_COLORS = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4'];

// Theme colors (light background: #fff4e5)
const THEME = {
    text: '#333',
    textLight: '#666',
    title: '#1a5a7a',
    axisLine: '#999',
    splitLine: '#ddd',
    background: 'transparent',
    pieBorder: '#fff4e5'
};

/**
 * Build ECharts option from simplified config
 * @param {Object} config - Chart configuration from data-chart attribute
 * @returns {Object} ECharts option object
 */
function buildChartOption(config) {
    const baseOption = {
        backgroundColor: THEME.background,
        textStyle: { color: THEME.text },
        title: {
            text: config.title || '',
            left: 'center',
            textStyle: { color: THEME.title, fontSize: 14, fontWeight: 'bold' }
        },
        tooltip: {
            trigger: config.type === 'pie' ? 'item' : 'axis'
        },
        color: CHART_COLORS
    };

    if (config.type === 'pie') {
        return {
            ...baseOption,
            tooltip: {
                trigger: 'item',
                formatter: '{b}: {c}% ({d}%)'
            },
            legend: {
                orient: 'vertical',
                right: 20,
                top: 'center',
                textStyle: { color: THEME.text }
            },
            series: [{
                type: 'pie',
                radius: ['40%', '70%'],
                center: ['40%', '55%'],
                avoidLabelOverlap: true,
                itemStyle: {
                    borderRadius: 4,
                    borderColor: THEME.pieBorder,
                    borderWidth: 2
                },
                label: {
                    show: true,
                    formatter: '{b}: {c}%',
                    color: THEME.text,
                    fontSize: 12
                },
                labelLine: {
                    show: true,
                    lineStyle: {
                        color: THEME.textLight,
                        width: 1
                    },
                    smooth: 0.2,
                    length: 10,
                    length2: 15
                },
                data: config.data
            }]
        };
    }

    if (config.type === 'bar') {
        return {
            ...baseOption,
            legend: {
                data: config.series.map(s => s.name),
                bottom: 10,
                textStyle: { color: THEME.text }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '15%',
                top: '15%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: config.categories,
                axisLabel: { color: THEME.text, rotate: 30 },
                axisLine: { lineStyle: { color: THEME.axisLine } }
            },
            yAxis: {
                type: 'value',
                name: config.yAxisName || '',
                nameTextStyle: { color: THEME.text },
                axisLabel: { color: THEME.text },
                axisLine: { lineStyle: { color: THEME.axisLine } },
                splitLine: { lineStyle: { color: THEME.splitLine } }
            },
            series: config.series.map(s => ({
                name: s.name,
                type: 'bar',
                data: s.data,
                barGap: '10%'
            }))
        };
    }

    if (config.type === 'stacked-bar') {
        return {
            ...baseOption,
            legend: {
                data: config.series.map(s => s.name),
                bottom: 10,
                textStyle: { color: THEME.text }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '15%',
                top: '15%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: config.categories,
                axisLabel: { color: THEME.text, rotate: 30 },
                axisLine: { lineStyle: { color: THEME.axisLine } }
            },
            yAxis: {
                type: 'value',
                name: config.yAxisName || '',
                nameTextStyle: { color: THEME.text },
                axisLabel: { color: THEME.text },
                axisLine: { lineStyle: { color: THEME.axisLine } },
                splitLine: { lineStyle: { color: THEME.splitLine } }
            },
            series: config.series.map(s => ({
                name: s.name,
                type: 'bar',
                stack: 'total',
                emphasis: { focus: 'series' },
                data: s.data
            }))
        };
    }

    if (config.type === 'line') {
        return {
            ...baseOption,
            legend: {
                data: config.series.map(s => s.name),
                bottom: 10,
                textStyle: { color: THEME.text }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '15%',
                top: '15%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                name: config.xAxisName || '',
                nameTextStyle: { color: THEME.text },
                data: config.categories,
                axisLabel: { color: THEME.text },
                axisLine: { lineStyle: { color: THEME.axisLine } }
            },
            yAxis: {
                type: 'value',
                name: config.yAxisName || '',
                nameTextStyle: { color: THEME.text },
                axisLabel: { color: THEME.text },
                axisLine: { lineStyle: { color: THEME.axisLine } },
                splitLine: { lineStyle: { color: THEME.splitLine } }
            },
            series: config.series.map(s => ({
                name: s.name,
                type: 'line',
                smooth: true,
                data: s.data
            }))
        };
    }

    return baseOption;
}

/**
 * Initialize all ECharts within a container
 * @param {HTMLElement} container - Container element to search for chart placeholders
 * @returns {Array} Array of initialized chart instances
 */
function initializeCharts(container) {
    if (typeof echarts === 'undefined') {
        console.log('[ChartUtils] ECharts not available, skipping initialization');
        return [];
    }
    
    if (!container) {
        console.warn('[ChartUtils] No container provided');
        return [];
    }
    
    const chartElements = container.querySelectorAll('.echart-container[data-chart]');
    if (chartElements.length === 0) {
        return [];
    }
    
    console.log(`[ChartUtils] Initializing ${chartElements.length} chart(s)`);
    
    const instances = [];
    
    chartElements.forEach((el, index) => {
        // Skip if already initialized
        if (el._chartInstance) {
            instances.push(el._chartInstance);
            return;
        }
        
        try {
            const config = JSON.parse(el.dataset.chart);
            const chart = echarts.init(el, null, { renderer: 'svg' });
            
            const option = buildChartOption(config);
            chart.setOption(option);
            
            // Store instance on element for later disposal
            el._chartInstance = chart;
            instances.push(chart);
            
            // Handle resize
            const resizeHandler = () => chart.resize();
            window.addEventListener('resize', resizeHandler);
            el._resizeHandler = resizeHandler;
            
        } catch (err) {
            console.error(`[ChartUtils] Chart ${index} error:`, err);
            el.innerHTML = `<div style="color: #c00; padding: 20px; text-align: center;">Chart error: ${err.message}</div>`;
        }
    });
    
    return instances;
}

/**
 * Dispose all ECharts within a container
 * @param {HTMLElement} container - Container element to search for charts
 */
function disposeCharts(container) {
    if (!container) return;
    
    const chartElements = container.querySelectorAll('.echart-container');
    
    chartElements.forEach(el => {
        // Remove resize handler
        if (el._resizeHandler) {
            window.removeEventListener('resize', el._resizeHandler);
            el._resizeHandler = null;
        }
        
        // Dispose chart instance
        if (el._chartInstance) {
            try {
                el._chartInstance.dispose();
            } catch (e) {
                // Ignore disposal errors
            }
            el._chartInstance = null;
        }
    });
}

/**
 * Resize all ECharts within a container
 * @param {HTMLElement} container - Container element to search for charts
 */
function resizeCharts(container) {
    if (!container) return;
    
    const chartElements = container.querySelectorAll('.echart-container');
    
    chartElements.forEach(el => {
        if (el._chartInstance) {
            el._chartInstance.resize();
        }
    });
}

// Attach to window for global access (works for both module and non-module)
window.ChartUtils = {
    initializeCharts,
    disposeCharts,
    resizeCharts,
    buildChartOption,
    CHART_COLORS,
    THEME
};

// Also export for ES6 module imports
export { initializeCharts, disposeCharts, resizeCharts, buildChartOption, CHART_COLORS, THEME };
