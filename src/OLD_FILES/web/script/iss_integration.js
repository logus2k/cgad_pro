// iss_integration.js - Extends your GAIA application

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader';

class ISSOrbitSystem {
  constructor(scene, earth) {
    this.scene = scene;
    this.earth = earth;
    this.issModel = null;
    this.orbitLine = null;
    this.thermalData = new Map(); // element_id -> temperature
    
    this.setupOrbit();
    this.loadISSModel();
    this.setupFEMVisualization();
  }
  
  setupOrbit() {
    // ISS orbital parameters
    const altitude = 408; // km above Earth surface
    const earthRadius = 6371; // km
    const orbitRadius = (earthRadius + altitude) / earthRadius; // Normalized
    
    const inclination = 51.6 * Math.PI / 180; // ISS orbit inclination
    
    // Create orbit line
    const points = [];
    for (let i = 0; i <= 360; i += 2) {
      const theta = i * Math.PI / 180;
      const x = orbitRadius * Math.cos(theta);
      const y = 0;
      const z = orbitRadius * Math.sin(theta);
      points.push(new THREE.Vector3(x, y, z));
    }
    
    const orbitGeometry = new THREE.BufferGeometry().setFromPoints(points);
    const orbitMaterial = new THREE.LineBasicMaterial({ 
      color: 0x00ffff,
      opacity: 0.6,
      transparent: true
    });
    
    this.orbitLine = new THREE.Line(orbitGeometry, orbitMaterial);
    this.orbitLine.rotation.x = inclination;
    this.scene.add(this.orbitLine);
  }
  
  loadISSModel() {
    const loader = new GLTFLoader();
    
    loader.load('/models/iss.glb', (gltf) => {
      this.issModel = gltf.scene;
      
      // Scale appropriately (ISS is ~109m wingspan)
      const issScale = 0.02; // Adjust for visibility
      this.issModel.scale.set(issScale, issScale, issScale);
      
      // Initial position on orbit
      this.issModel.position.set(1.064, 0, 0); // orbitRadius
      this.issModel.rotation.x = 51.6 * Math.PI / 180;
      
      // Prepare for thermal visualization
      this.prepareThermalMesh();
      
      this.scene.add(this.issModel);
    });
  }
  
  prepareThermalMesh() {
    // Convert ISS mesh to support vertex colors for temperature
    this.issModel.traverse((child) => {
      if (child.isMesh) {
        // Store original material
        child.userData.originalMaterial = child.material;
        
        // Create thermal material with vertex colors
        child.material = new THREE.MeshBasicMaterial({
          vertexColors: true,
          side: THREE.DoubleSide
        });
        
        // Initialize color attribute
        const colors = new Float32Array(child.geometry.attributes.position.count * 3);
        colors.fill(0.5); // Default gray
        child.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        // Map mesh vertices to FEM elements
        child.userData.femElementMap = this.createFEMMapping(child);
      }
    });
  }
  
  createFEMMapping(mesh) {
    // Map each mesh vertex to closest FEM element
    // This will be populated when FEM mesh is loaded
    const mapping = new Map();
    const positions = mesh.geometry.attributes.position;
    
    // Store vertex positions for later FEM correlation
    for (let i = 0; i < positions.count; i++) {
      const vertex = new THREE.Vector3(
        positions.getX(i),
        positions.getY(i),
        positions.getZ(i)
      );
      mapping.set(i, { vertex, femElement: null });
    }
    
    return mapping;
  }
  
  setupFEMVisualization() {
    // Socket.IO connection for real-time FEM updates
    this.socket = io('http://localhost:5000');
    
    this.socket.on('fem_update', (data) => {
      this.updateThermalVisualization(data);
    });
    
    this.socket.on('fem_complete', (data) => {
      this.displayResults(data);
    });
    
    this.socket.on('fem_mesh', (meshData) => {
      this.correlateWithFEM(meshData);
    });
  }
  
  correlateWithFEM(meshData) {
    // meshData = { coords: [[x,y,z], ...], connectivity: [...] }
    // Map FEM elements to ISS mesh vertices
    
    this.issModel.traverse((child) => {
      if (child.isMesh && child.userData.femElementMap) {
        const mapping = child.userData.femElementMap;
        
        mapping.forEach((data, vertexIdx) => {
          // Find closest FEM element centroid
          let minDist = Infinity;
          let closestElem = null;
          
          meshData.connectivity.forEach((elemNodes, elemIdx) => {
            // Calculate element centroid
            const centroid = new THREE.Vector3();
            elemNodes.forEach(nodeIdx => {
              const [x, y, z] = meshData.coords[nodeIdx];
              centroid.add(new THREE.Vector3(x, y, z));
            });
            centroid.divideScalar(elemNodes.length);
            
            const dist = data.vertex.distanceTo(centroid);
            if (dist < minDist) {
              minDist = dist;
              closestElem = elemIdx;
            }
          });
          
          data.femElement = closestElem;
        });
      }
    });
  }
  
  updateThermalVisualization(data) {
    // data = { element: int, temperature: float, progress: float }
    this.thermalData.set(data.element, data.temperature);
    
    // Update ISS mesh colors
    this.issModel.traverse((child) => {
      if (child.isMesh && child.userData.femElementMap) {
        const colors = child.geometry.attributes.color;
        const mapping = child.userData.femElementMap;
        
        mapping.forEach((mapData, vertexIdx) => {
          if (mapData.femElement !== null) {
            const temp = this.thermalData.get(mapData.femElement) || 300;
            const color = this.temperatureToColor(temp);
            
            colors.setXYZ(vertexIdx, color.r, color.g, color.b);
          }
        });
        
        colors.needsUpdate = true;
      }
    });
    
    // Update progress bar in UI
    this.updateProgressBar(data.progress);
  }
  
  temperatureToColor(temp) {
    // Temperature range: 200K (dark side) to 400K (sun side)
    const t = (temp - 200) / 200; // Normalize to [0, 1]
    
    // Blue → Cyan → Green → Yellow → Red
    if (t < 0.25) {
      const local = t / 0.25;
      return new THREE.Color().setRGB(0, local, 1);
    } else if (t < 0.5) {
      const local = (t - 0.25) / 0.25;
      return new THREE.Color().setRGB(0, 1, 1 - local);
    } else if (t < 0.75) {
      const local = (t - 0.5) / 0.25;
      return new THREE.Color().setRGB(local, 1, 0);
    } else {
      const local = (t - 0.75) / 0.25;
      return new THREE.Color().setRGB(1, 1 - local, 0);
    }
  }
  
  animate(time) {
    if (this.issModel) {
      // Orbital motion (ISS orbits ~15.5 times per day)
      const orbitalPeriod = 92 * 60; // seconds
      const angle = (time / orbitalPeriod) * 2 * Math.PI;
      
      const orbitRadius = 1.064; // (6371 + 408) / 6371
      this.issModel.position.x = orbitRadius * Math.cos(angle);
      this.issModel.position.z = orbitRadius * Math.sin(angle);
      
      // ISS attitude (simplified - points velocity direction)
      this.issModel.rotation.y = angle + Math.PI / 2;
    }
  }
}
