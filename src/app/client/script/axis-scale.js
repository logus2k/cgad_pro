/**
 * AxisScale - Coordinate axes with tick marks and labels
 * 
 * 3D Mode: X, Y, Z axes at grid boundaries with arrows
 * 2D Mode: Chart-style X, Y axes close to mesh bounds
 * 
 * Location: /src/app/client/script/axis-scale.js
 */

import * as THREE from '../library/three.module.min.js';

export class AxisScale {
    constructor(scene, options = {}) {
        this.scene = scene;
        this.options = {
            visible: true,
            axisRadius: 0.08,         // Cylinder radius for thick lines
            tickLength: 0.5,
            tickRadius: 0.04,
            labelSize: 2.0,           // Increased font size
            tickLabelSize: 1.4,       // Increased tick label size
            arrowHeadLength: 1.5,
            arrowHeadRadius: 0.5,
            labelOffset: 1.2,
            gridSize: 100,            // Match millimetric grid size
            ...options
        };
        
        // Axis colors (RGB convention)
        this.colors = {
            x: 0xe63946,  // Red
            y: 0x2a9d8f,  // Green
            z: 0x3a9bdc   // Blue
        };
        
        // Main group containing everything
        this.mainGroup = new THREE.Group();
        this.mainGroup.name = 'AxisScale';
        
        // Separate groups for 3D and 2D modes
        this.group3D = new THREE.Group();
        this.group3D.name = 'AxisScale3D';
        
        this.group2D = new THREE.Group();
        this.group2D.name = 'AxisScale2D';
        
        this.mainGroup.add(this.group3D);
        this.mainGroup.add(this.group2D);
        
        // Initially hidden until mesh loads
        this.mainGroup.visible = false;
        this.group3D.visible = true;
        this.group2D.visible = false;
        
        this.scene.add(this.mainGroup);
        
        // Store mesh transformation info
        this.meshBounds = null;      // Original mesh bounds {xMin, xMax, yMin, yMax}
        this.meshScale = 1;          // Scale applied to mesh
        this.meshOffset = new THREE.Vector3();  // Position offset
        
        // Current mode
        this.is2DMode = false;
        this.enabled = true;
        
        // Cache for label sprites (for disposal)
        this.labelSprites = [];
        
        console.log('AxisScale initialized');
    }
    
    /**
     * Update axes based on mesh bounds and transformation
     * Called when a new mesh is loaded
     */
    updateFromMeshExtruder(meshExtruder) {
        if (!meshExtruder) {
            this.mainGroup.visible = false;
            return;
        }
        
        // Get bounds from mesh extruder
        const bounds = meshExtruder.originalBounds || meshExtruder.bounds;
        if (!bounds) {
            console.warn('[AxisScale] No bounds available from meshExtruder');
            return;
        }
        
        this.meshBounds = {
            xMin: bounds.xMin,
            xMax: bounds.xMax,
            yMin: bounds.yMin,
            yMax: bounds.yMax,
            zMin: bounds.zMin || 0,
            zMax: bounds.zMax || 0
        };
        
        // Get scale and offset from mesh extruder
        this.meshScale = meshExtruder.scale || 1;
        
        // Calculate mesh center in scaled space
        const xRange = this.meshBounds.xMax - this.meshBounds.xMin;
        const yRange = this.meshBounds.yMax - this.meshBounds.yMin;
        
        // The mesh is centered horizontally and rests on floor (Y=0)
        this.meshOffset.set(
            -((this.meshBounds.xMin + this.meshBounds.xMax) / 2) * this.meshScale,
            -this.meshBounds.yMin * this.meshScale,
            0
        );
        
        console.log('[AxisScale] Mesh bounds:', this.meshBounds);
        console.log('[AxisScale] Scale:', this.meshScale);
        console.log('[AxisScale] Offset:', this.meshOffset);
        
        // Rebuild axes
        this.rebuild();
        
        // Show axes
        if (this.enabled) {
            this.mainGroup.visible = true;
        }
    }
    
    /**
     * Rebuild all axis geometry
     */
    rebuild() {
        // Clear existing geometry
        this.clear3D();
        this.clear2D();
        
        if (!this.meshBounds) return;
        
        // Build both representations
        this.build3DAxes();
        this.build2DAxes();
        
        // Set initial visibility based on mode
        this.group3D.visible = !this.is2DMode;
        this.group2D.visible = this.is2DMode;
    }
    
    /**
     * Build 3D axes at grid boundaries with arrows
     */
    build3DAxes() {
        const gridSize = this.options.gridSize;
        const halfGrid = gridSize / 2;
        
        // Origin at front-left corner of grid (-50, 0, +50)
        const origin = new THREE.Vector3(-halfGrid, 0, halfGrid);
        
        // X axis (red) - along front edge, pointing right
        this.createAxis3D(
            origin,
            new THREE.Vector3(1, 0, 0),
            gridSize,
            this.colors.x,
            'X',
            -halfGrid,
            halfGrid
        );
        
        // Y axis (green) - pointing up from front-left corner
        this.createAxis3D(
            origin,
            new THREE.Vector3(0, 1, 0),
            halfGrid,
            this.colors.y,
            'Y',
            0,
            halfGrid
        );
        
        // Z axis (blue) - pointing backward (into the scene)
        this.createAxis3D(
            origin,
            new THREE.Vector3(0, 0, -1),
            gridSize,
            this.colors.z,
            'Z',
            -halfGrid,
            halfGrid
        );
    }
    
    /**
     * Create a single 3D axis with thick cylinder line, arrow, ticks, and labels
     */
    createAxis3D(origin, direction, length, color, label, minVal, maxVal) {
        const { arrowHeadLength, arrowHeadRadius, axisRadius, tickLength, labelOffset, labelSize } = this.options;
        
        // Axis line as cylinder (for thickness)
        const lineLength = length;
        const lineGeom = new THREE.CylinderGeometry(axisRadius, axisRadius, lineLength, 8);
        const lineMat = new THREE.MeshBasicMaterial({ color });
        const line = new THREE.Mesh(lineGeom, lineMat);
        
        // Position and rotate cylinder to align with axis
        const midPoint = origin.clone().add(direction.clone().multiplyScalar(lineLength / 2));
        line.position.copy(midPoint);
        
        // Rotate cylinder to point along direction
        if (direction.x === 1) {
            line.rotation.z = -Math.PI / 2;
        } else if (direction.z === 1) {
            line.rotation.x = Math.PI / 2;
        } else if (direction.z === -1) {
            line.rotation.x = -Math.PI / 2;
        }
        // Y direction is default cylinder orientation
        
        this.group3D.add(line);
        
        // Arrow head (cone)
        const coneGeom = new THREE.ConeGeometry(arrowHeadRadius, arrowHeadLength, 12);
        const coneMat = new THREE.MeshBasicMaterial({ color });
        const cone = new THREE.Mesh(coneGeom, coneMat);
        
        // Position cone at end of axis
        const conePos = origin.clone().add(direction.clone().multiplyScalar(length));
        cone.position.copy(conePos);
        
        // Rotate cone to point along axis direction
        if (direction.y === 1) {
            // Y axis - default cone orientation
        } else if (direction.x === 1) {
            cone.rotation.z = -Math.PI / 2;
        } else if (direction.z === 1) {
            cone.rotation.x = Math.PI / 2;
        } else if (direction.z === -1) {
            cone.rotation.x = -Math.PI / 2;
        }
        
        this.group3D.add(cone);
        
        // Axis label at end
        const labelPos = conePos.clone().add(direction.clone().multiplyScalar(arrowHeadLength + 1.0));
        const axisLabel = this.createTextSprite(label, color, labelSize);
        axisLabel.position.copy(labelPos);
        this.group3D.add(axisLabel);
        this.labelSprites.push(axisLabel);
        
        // Add tick marks and labels
        this.addTickMarks3D(origin, direction, length - arrowHeadLength * 0.5, color, minVal, maxVal, label);
    }
    
    /**
     * Add tick marks along a 3D axis
     */
    addTickMarks3D(origin, direction, length, color, minVal, maxVal, axisName) {
        const { tickLength, tickRadius, labelOffset, tickLabelSize } = this.options;
        
        // Calculate nice tick interval
        const range = maxVal - minVal;
        const tickInterval = this.calculateNiceInterval(range);
        
        // Determine tick perpendicular direction
        let perpDir;
        if (axisName === 'X') {
            perpDir = new THREE.Vector3(0, 0, 1);   // Ticks go forward (toward viewer)
        } else if (axisName === 'Y') {
            perpDir = new THREE.Vector3(1, 0, 1).normalize();  // Ticks go diagonal (visible from front-right)
        } else {
            perpDir = new THREE.Vector3(1, 0, 0);   // Ticks go right
        }
        
        // Generate ticks
        const firstTick = Math.ceil(minVal / tickInterval) * tickInterval;
        
        for (let val = firstTick; val <= maxVal; val += tickInterval) {
            // Position along axis (normalized 0-1)
            const t = (val - minVal) / range;
            const worldPos = t * length;
            
            // Tick position
            const tickPos = origin.clone().add(direction.clone().multiplyScalar(worldPos));
            
            // Tick line as thin cylinder
            const tickGeom = new THREE.CylinderGeometry(tickRadius, tickRadius, tickLength, 6);
            const tickMat = new THREE.MeshBasicMaterial({ color: 0x333333 });
            const tick = new THREE.Mesh(tickGeom, tickMat);
            
            // Position tick
            const tickMid = tickPos.clone().add(perpDir.clone().multiplyScalar(tickLength / 2));
            tick.position.copy(tickMid);
            
            // Rotate tick to align with perpendicular direction
            if (perpDir.x !== 0 && perpDir.z !== 0) {
                tick.rotation.z = Math.PI / 2;
                tick.rotation.y = Math.PI / 4;
            } else if (perpDir.z === -1) {
                tick.rotation.x = Math.PI / 2;
            } else if (perpDir.x === -1) {
                tick.rotation.z = Math.PI / 2;
            }
            
            this.group3D.add(tick);
            
            // Tick label
            const labelPos = tickPos.clone().add(perpDir.clone().multiplyScalar(tickLength + labelOffset));
            const labelText = this.formatTickValue(val);
            const tickLabel = this.createTextSprite(labelText, 0x333333, tickLabelSize);
            tickLabel.position.copy(labelPos);
            this.group3D.add(tickLabel);
            this.labelSprites.push(tickLabel);
        }
    }
    
    /**
     * Build 2D chart-style axes close to mesh
     */
    build2DAxes() {
        const bounds = this.meshBounds;
        const scale = this.meshScale;
        
        // Calculate axis positions in world space
        const xLen = (bounds.xMax - bounds.xMin) * scale;
        const yLen = (bounds.yMax - bounds.yMin) * scale;
        
        // Position axes at mesh bounds with small offset
        const offset = 3.0;  // Offset from mesh
        
        const xAxisY = -offset;  // X axis below mesh
        const yAxisX = -xLen / 2 - offset;  // Y axis to left of mesh
        
        // Z position to match 2D grid (slightly in front of grid at z=-1)
        const zPos = 0;
        
        // X axis (horizontal)
        this.createAxis2D(
            new THREE.Vector3(-xLen / 2, xAxisY, zPos),
            new THREE.Vector3(1, 0, 0),
            xLen,
            this.colors.x,
            'X',
            bounds.xMin,
            bounds.xMax,
            scale,
            'bottom'
        );
        
        // Y axis (vertical)
        this.createAxis2D(
            new THREE.Vector3(yAxisX, 0, zPos),
            new THREE.Vector3(0, 1, 0),
            yLen,
            this.colors.y,
            'Y',
            bounds.yMin,
            bounds.yMax,
            scale,
            'left'
        );
    }
    
    /**
     * Create a single 2D axis with thick line, ticks and labels
     */
    createAxis2D(origin, direction, length, color, label, minVal, maxVal, scale, position) {
        const { tickLength, axisRadius, labelOffset, labelSize } = this.options;
        
        // Axis line as cylinder (for thickness)
        const lineGeom = new THREE.CylinderGeometry(axisRadius * 1.5, axisRadius * 1.5, length, 8);
        const lineMat = new THREE.MeshBasicMaterial({ color });
        const line = new THREE.Mesh(lineGeom, lineMat);
        
        // Position and rotate
        const midPoint = origin.clone().add(direction.clone().multiplyScalar(length / 2));
        line.position.copy(midPoint);
        
        if (direction.x === 1) {
            line.rotation.z = Math.PI / 2;
        }
        // Y direction is default
        
        this.group2D.add(line);
        
        // Axis label at end
        const labelPos = origin.clone().add(direction.clone().multiplyScalar(length + 3));
        const axisLabel = this.createTextSprite(label, color, labelSize);
        axisLabel.position.copy(labelPos);
        this.group2D.add(axisLabel);
        this.labelSprites.push(axisLabel);
        
        // Add tick marks
        this.addTickMarks2D(origin, direction, length, color, minVal, maxVal, scale, position);
    }
    
    /**
     * Add tick marks along a 2D axis
     */
    addTickMarks2D(origin, direction, length, color, minVal, maxVal, scale, position) {
        const { tickLength, tickRadius, labelOffset, tickLabelSize } = this.options;
        
        // Calculate nice tick interval
        const range = maxVal - minVal;
        const tickInterval = this.calculateNiceInterval(range);
        
        // Determine tick perpendicular direction based on position
        let perpDir;
        if (position === 'bottom') {
            perpDir = new THREE.Vector3(0, -1, 0);
        } else if (position === 'left') {
            perpDir = new THREE.Vector3(-1, 0, 0);
        }
        
        // Generate ticks
        const firstTick = Math.ceil(minVal / tickInterval) * tickInterval;
        
        for (let val = firstTick; val <= maxVal + tickInterval * 0.01; val += tickInterval) {
            if (val > maxVal) break;
            
            // Position along axis (normalized 0-1)
            const t = (val - minVal) / range;
            const worldPos = t * length;
            
            // Tick position
            const tickPos = origin.clone().add(direction.clone().multiplyScalar(worldPos));
            
            // Tick line as thin cylinder
            const tickGeom = new THREE.CylinderGeometry(tickRadius * 1.5, tickRadius * 1.5, tickLength, 6);
            const tickMat = new THREE.MeshBasicMaterial({ color: 0x333333 });
            const tick = new THREE.Mesh(tickGeom, tickMat);
            
            // Position and rotate tick
            const tickMid = tickPos.clone().add(perpDir.clone().multiplyScalar(tickLength / 2));
            tick.position.copy(tickMid);
            
            if (position === 'bottom') {
                // Vertical tick, no rotation needed
            } else if (position === 'left') {
                tick.rotation.z = Math.PI / 2;
            }
            
            this.group2D.add(tick);
            
            // Tick label
            const labelPos = tickPos.clone().add(perpDir.clone().multiplyScalar(tickLength + labelOffset));
            const labelText = this.formatTickValue(val);
            const tickLabel = this.createTextSprite(labelText, 0x333333, tickLabelSize);
            tickLabel.position.copy(labelPos);
            this.group2D.add(tickLabel);
            this.labelSprites.push(tickLabel);
        }
    }
    
    /**
     * Calculate a "nice" tick interval for the given range
     */
    calculateNiceInterval(range) {
        if (range <= 0) return 1;
        
        // Target approximately 5-8 ticks
        const roughInterval = range / 6;
        
        // Find the order of magnitude
        const magnitude = Math.pow(10, Math.floor(Math.log10(roughInterval)));
        
        // Normalize to 1-10 range
        const normalized = roughInterval / magnitude;
        
        // Choose a nice number
        let niceNormalized;
        if (normalized <= 1) niceNormalized = 1;
        else if (normalized <= 2) niceNormalized = 2;
        else if (normalized <= 5) niceNormalized = 5;
        else niceNormalized = 10;
        
        return niceNormalized * magnitude;
    }
    
    /**
     * Format tick value for display
     */
    formatTickValue(val) {
        // Handle floating point precision issues
        const rounded = Math.round(val * 1000) / 1000;
        
        // Format based on magnitude
        if (Math.abs(rounded) >= 1000) {
            return rounded.toExponential(1);
        } else if (Math.abs(rounded) >= 1) {
            return rounded.toFixed(1).replace(/\.0$/, '');
        } else if (rounded === 0) {
            return '0';
        } else {
            return rounded.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
        }
    }
    
    /**
     * Create a text sprite for labels
     */
    createTextSprite(text, color, size = 1) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        // Canvas size (power of 2 for texture)
        canvas.width = 256;
        canvas.height = 128;
        
        // Font settings - larger and bolder
        const fontSize = 64;
        context.font = `bold ${fontSize}px Arial, sans-serif`;
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        
        // Convert color to CSS string
        const colorObj = new THREE.Color(color);
        context.fillStyle = `rgb(${Math.floor(colorObj.r * 255)}, ${Math.floor(colorObj.g * 255)}, ${Math.floor(colorObj.b * 255)})`;
        
        // Draw text
        context.fillText(text, canvas.width / 2, canvas.height / 2);
        
        // Create texture and sprite
        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        
        const material = new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthTest: false,
            depthWrite: false
        });
        
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(size * 2.5, size * 1.25, 1);
        
        return sprite;
    }
    
    /**
     * Set 2D mode (during camera transition)
     */
    set2DMode(enabled) {
        this.is2DMode = enabled;
        
        if (!this.enabled || !this.mainGroup.visible) return;
        
        this.group3D.visible = !enabled;
        this.group2D.visible = enabled;
    }
    
    /**
     * Interpolate between 3D and 2D modes (for smooth transition)
     */
    setInterpolation(t) {
        if (!this.enabled || !this.mainGroup.visible) return;
        
        // Crossfade between modes
        if (t < 0.5) {
            this.group3D.visible = true;
            this.group2D.visible = false;
            // Fade out 3D
            this.group3D.traverse(obj => {
                if (obj.material) {
                    obj.material.opacity = 1 - t * 2;
                    obj.material.transparent = true;
                }
            });
        } else {
            this.group3D.visible = false;
            this.group2D.visible = true;
            // Fade in 2D
            this.group2D.traverse(obj => {
                if (obj.material) {
                    obj.material.opacity = (t - 0.5) * 2;
                    obj.material.transparent = true;
                }
            });
        }
    }
    
    /**
     * Toggle visibility
     */
    setVisible(visible) {
        this.enabled = visible;
        this.mainGroup.visible = visible && (this.meshBounds !== null);
        
        if (visible && this.meshBounds) {
            this.group3D.visible = !this.is2DMode;
            this.group2D.visible = this.is2DMode;
        }
    }
    
    /**
     * Toggle visibility and return new state
     */
    toggle() {
        this.setVisible(!this.enabled);
        return this.enabled;
    }
    
    /**
     * Check if visible
     */
    isVisible() {
        return this.enabled;
    }
    
    /**
     * Clear 3D axis geometry
     */
    clear3D() {
        while (this.group3D.children.length > 0) {
            const child = this.group3D.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
            this.group3D.remove(child);
        }
    }
    
    /**
     * Clear 2D axis geometry
     */
    clear2D() {
        while (this.group2D.children.length > 0) {
            const child = this.group2D.children[0];
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (child.material.map) child.material.map.dispose();
                child.material.dispose();
            }
            this.group2D.remove(child);
        }
    }
    
    /**
     * Dispose all resources
     */
    dispose() {
        this.clear3D();
        this.clear2D();
        
        this.labelSprites = [];
        
        if (this.mainGroup.parent) {
            this.mainGroup.parent.remove(this.mainGroup);
        }
    }
}
