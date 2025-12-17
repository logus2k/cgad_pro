"""
ISS Mesh Generator: .glb → Quad-8 FEM Surface Mesh

Converts ISS 3D model to second-order quadrilateral surface elements
suitable for thermal FEM analysis.

Pipeline:
1. Load .glb → Triangle mesh
2. Simplify to target density
3. Quad meshing via Gmsh
4. Extract Quad-8 connectivity
5. Validate counter-clockwise orientation
6. Export mesh data + UV mapping for Three.js
"""

import numpy as np
import trimesh
import gmsh
from pathlib import Path
import json
import time
from tqdm import tqdm
import pandas as pd


class ISSMeshGenerator:
    def __init__(self, glb_path, target_elements=20000):
        """
        Parameters
        ----------
        glb_path : str or Path
            Path to ISS .glb model
        target_elements : int
            Target number of Quad-8 elements (actual may vary ±20%)
        """
        self.glb_path = Path(glb_path)
        self.target_elements = target_elements
        
        # Mesh data
        self.coords = None          # (N, 3) nodal coordinates
        self.connectivity = None    # (M, 8) Quad-8 element nodes
        self.element_areas = None   # (M,) element surface areas
        self.normals = None         # (M, 3) element normals
        self.uv_coords = None       # (N, 2) UV mapping for Three.js
        
        # ISS physical scale (meters)
        self.iss_length = 109.0     # Truss length
        self.iss_width = 73.0       # Solar panel span
        
    def generate(self):
        """
        Full pipeline: .glb → Quad-8 mesh
        
        Returns
        -------
        dict with keys:
            - coords: (N, 3) array
            - connectivity: (M, 8) array
            - areas: (M,) array
            - normals: (M, 3) array
            - uv: (N, 2) array
            - metadata: dict
        """
        steps = [
            "Loading ISS model",
            "Exporting to STL", 
            "Quad meshing with Gmsh",
            "Extracting Quad-8 connectivity",
            "Validating orientation",
            "Computing UV mapping"
        ]
        
        start_time = time.time()
        
        with tqdm(total=len(steps), desc="Mesh generation", unit="step", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            
            # Step 1: Load
            pbar.set_description(f"[1/6] {steps[0]}")
            tri_mesh = self._load_and_simplify()
            pbar.update(1)
            
            # Step 2: Export
            pbar.set_description(f"[2/6] {steps[1]}")
            stl_path = self._export_stl(tri_mesh)
            pbar.update(1)
            
            # Step 3: Gmsh quad meshing
            pbar.set_description(f"[3/6] {steps[2]}")
            self._gmsh_quad_remesh(stl_path, tri_mesh)
            pbar.update(1)
            
            # Step 4: Extract
            pbar.set_description(f"[4/6] {steps[3]}")
            self._extract_quad8_from_gmsh()
            pbar.update(1)
            
            # Step 5: Validate
            pbar.set_description(f"[5/6] {steps[4]}")
            self._validate_and_orient()
            pbar.update(1)
            
            # Step 6: UV mapping
            pbar.set_description(f"[6/6] {steps[5]}")
            self._compute_uv_mapping(tri_mesh)
            pbar.update(1)
        
        # Cleanup
        stl_path.unlink()
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"MESH GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Nodes:           {len(self.coords)}")
        print(f"Elements:        {len(self.connectivity)}")
        print(f"Surface area:    {self.element_areas.sum():.2f} m²")
        print(f"Generation time: {elapsed:.1f}s")
        print(f"{'='*60}\n")
        
        return self._package_results()
    
    def _load_and_simplify(self):
        """Load .glb and simplify to target triangle count"""
        scene = trimesh.load(str(self.glb_path))
        
        # Handle scene vs single mesh
        if isinstance(scene, trimesh.Scene):
            mesh = scene.dump(concatenate=True)
        else:
            mesh = scene
        
        # Merge duplicate vertices
        mesh.merge_vertices()
        
        # Target: ~4 triangles per desired quad
        target_faces = self.target_elements * 4
        
        if len(mesh.faces) > target_faces:
            print(f"   Simplifying {len(mesh.faces)} → {target_faces} faces")
            
            try:
                # Try fast quadric decimation first
                mesh = mesh.simplify_quadric_decimation(target_faces)
            except ImportError:
                print("   Fast simplification not available, using alternative method...")
                try:
                    # Fallback to pyfqmr
                    import pyfqmr
                    simplifier = pyfqmr.Simplify()
                    simplifier.setMesh(mesh.vertices, mesh.faces)
                    simplifier.simplify_mesh(target_count=target_faces, preserve_border=True, verbose=False)
                    vertices, faces, _ = simplifier.getMesh()
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                except ImportError:
                    # Final fallback: vertex clustering
                    print("   Using vertex clustering (slower)...")
                    # Calculate target edge length from face count
                    current_area = mesh.area
                    target_edge_length = np.sqrt(current_area / target_faces) * 2
                    mesh = mesh.simplify_vertex_clustering(threshold=target_edge_length)
                    
                    # If still too dense, try again with larger threshold
                    if len(mesh.faces) > target_faces * 1.5:
                        target_edge_length *= 1.5
                        mesh = mesh.simplify_vertex_clustering(threshold=target_edge_length)
        
        print(f"   Loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    def _export_stl(self, mesh):
        """Export to temporary STL for Gmsh"""
        stl_path = Path('/tmp/iss_temp.stl')
        # Export as STL_ASCII for maximum Gmsh compatibility
        with open(stl_path, 'wb') as f:
            mesh.export(f, file_type='stl_ascii')
        return stl_path
    
    def _gmsh_quad_remesh(self, stl_path, original_mesh):
        """
        Use Gmsh to create second-order quad surface mesh
        
        Working approach:
        - Import STL (gets triangle mesh)
        - Recombine existing triangles into quads
        - Elevate to order 2
        """
        # Ensure clean Gmsh state
        if gmsh.isInitialized():
            gmsh.finalize()
        
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        
        # Import STL - this loads a triangulated discrete mesh
        print(f"   Importing STL mesh...")
        gmsh.merge(str(stl_path))
        
        # Get surfaces
        surfaces = gmsh.model.getEntities(2)
        
        if not surfaces:
            raise ValueError("No surfaces found in STL")
        
        print(f"   Found {len(surfaces)} surface(s)")
        
        # Calculate target element size
        iss_area = original_mesh.area
        
        print(f"   Surface area: {iss_area:.2f} m²")
        
        # Set quad recombination options BEFORE recombining
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)  # Blossom (best)
        gmsh.option.setNumber("Mesh.RecombineOptimizeTopology", 5)
        
        # Recombine the EXISTING mesh (triangles → quads)
        # This works on the mesh that was imported from STL
        print(f"   Recombining triangles to quads...")
        gmsh.model.mesh.recombine()
        
        # Check what we got
        elem_types_before, _, _ = gmsh.model.mesh.getElements(2)
        print(f"   After recombination: element types {list(set(elem_types_before))}")
        
        # Elevate to order 2
        print(f"   Elevating to second order...")
        gmsh.model.mesh.setOrder(2)
        
        elem_types_after, _, _ = gmsh.model.mesh.getElements(2)
        print(f"   After elevation: element types {list(set(elem_types_after))}")
        
        # Optimize
        gmsh.model.mesh.optimize("Relocate2D")
        
        print(f"   Gmsh meshing complete")
    
    def _extract_quad8_from_gmsh(self):
        """Extract Quad-8 or Quad-4 nodes and connectivity from Gmsh"""
        
        # Get all nodes
        node_tags, coords_flat, _ = gmsh.model.mesh.getNodes()
        
        # Reshape coordinates
        coords = coords_flat.reshape(-1, 3)
        
        # Create mapping: gmsh_tag → zero-based index
        tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
        
        # Get elements
        elem_types, elem_tags, node_tags_in_elems = gmsh.model.mesh.getElements(2)
        
        # Find Quad elements (type 16=Quad-8, type 10=Quad-9, type 3=Quad-4)
        quad_data = []
        elem_type_found = None
        
        for i, elem_type in enumerate(elem_types):
            if elem_type == 16:  # Quad-8 (serendipity)
                elem_type_found = "Quad-8"
                nodes_per_elem = 8
            elif elem_type == 10:  # Quad-9 (complete second order)
                elem_type_found = "Quad-9" 
                nodes_per_elem = 9
            elif elem_type == 3:   # Quad-4 (linear)
                elem_type_found = "Quad-4"
                nodes_per_elem = 4
            else:
                continue
            
            # Found quad elements
            nodes_flat = node_tags_in_elems[i]
            num_elems = len(nodes_flat) // nodes_per_elem
            nodes_reshaped = nodes_flat.reshape(num_elems, nodes_per_elem)
            quad_data.append((elem_type, nodes_reshaped))
        
        if not quad_data:
            gmsh.finalize()
            raise ValueError("No quad elements found in mesh! Check Gmsh settings.")
        
        # Use the highest order quads found
        quad_data.sort(key=lambda x: x[0], reverse=True)  # Higher type number = higher order
        elem_type, connectivity_gmsh_tags = quad_data[0]
        
        print(f"   Found {len(connectivity_gmsh_tags)} {elem_type_found} elements")
        
        # Convert Gmsh tags to zero-based indices
        connectivity = np.zeros_like(connectivity_gmsh_tags, dtype=int)
        for i in range(connectivity_gmsh_tags.shape[0]):
            for j in range(connectivity_gmsh_tags.shape[1]):
                gmsh_tag = int(connectivity_gmsh_tags[i, j])
                connectivity[i, j] = tag_to_idx[gmsh_tag]
        
        # If Quad-4, pad to Quad-8 format by duplicating corner nodes as mid-side
        if elem_type_found == "Quad-4":
            print(f"   Converting Quad-4 to Quad-8 format...")
            quad8_connectivity = np.zeros((len(connectivity), 8), dtype=int)
            for i in range(len(connectivity)):
                # Quad-4: [0, 1, 2, 3]
                # Quad-8: [0, 1, 2, 3, mid01, mid12, mid23, mid30]
                quad8_connectivity[i, 0:4] = connectivity[i, 0:4]
                # Mid-side nodes (use corners as placeholders)
                quad8_connectivity[i, 4] = connectivity[i, 0]  # mid 0-1
                quad8_connectivity[i, 5] = connectivity[i, 1]  # mid 1-2
                quad8_connectivity[i, 6] = connectivity[i, 2]  # mid 2-3
                quad8_connectivity[i, 7] = connectivity[i, 3]  # mid 3-0
            connectivity = quad8_connectivity
        elif elem_type_found == "Quad-9":
            print(f"   Converting Quad-9 to Quad-8 format...")
            # Quad-9 has center node, Quad-8 doesn't - just drop it
            quad8_connectivity = connectivity[:, :8]
            connectivity = quad8_connectivity
        
        self.coords = coords
        self.connectivity = connectivity
        
        gmsh.finalize()
        
        print(f"   Extracted: {len(coords)} nodes, {len(connectivity)} Quad-8 elements")
    
    def _validate_and_orient(self):
        """
        Validate mesh quality and ensure counter-clockwise orientation
        
        Quad-8 node ordering (counter-clockwise):
        
            3---6---2
            |       |
            7       5
            |       |
            0---4---1
        
        Corner nodes: 0,1,2,3
        Mid-side nodes: 4,5,6,7
        """
        Nels = len(self.connectivity)
        
        self.normals = np.zeros((Nels, 3))
        self.element_areas = np.zeros(Nels)
        
        reoriented_count = 0
        
        for e in range(Nels):
            nodes = self.connectivity[e]
            
            # Corner nodes
            n0, n1, n2, n3 = nodes[0], nodes[1], nodes[2], nodes[3]
            
            # Get coordinates
            p0 = self.coords[n0]
            p1 = self.coords[n1]
            p2 = self.coords[n2]
            p3 = self.coords[n3]
            
            # Compute normal via cross product of diagonals
            diag1 = p2 - p0
            diag2 = p3 - p1
            normal = np.cross(diag1, diag2)
            
            # Area (half the cross product magnitude)
            area = 0.5 * np.linalg.norm(normal)
            
            # Normalize
            if area > 1e-12:
                normal = normal / (2 * area)
            else:
                print(f"   Warning: Element {e} has near-zero area")
                normal = np.array([0, 0, 1])  # Default
            
            # Check orientation: normal should point outward
            # For closed surface, centroid should be "inside"
            centroid = self.coords.mean(axis=0)
            element_center = (p0 + p1 + p2 + p3) / 4
            outward = element_center - centroid
            
            # If normal points inward, flip orientation
            if np.dot(normal, outward) < 0:
                # Reverse node order: [0,1,2,3,4,5,6,7] → [0,3,2,1,7,6,5,4]
                self.connectivity[e] = nodes[[0, 3, 2, 1, 7, 6, 5, 4]]
                normal = -normal
                reoriented_count += 1
            
            self.normals[e] = normal
            self.element_areas[e] = area
        
        print(f"   Reoriented {reoriented_count}/{Nels} elements for consistent normals")
        
        # Quality check
        min_area = self.element_areas.min()
        max_area = self.element_areas.max()
        mean_area = self.element_areas.mean()
        
        print(f"   Element areas: min={min_area:.6f}, max={max_area:.6f}, mean={mean_area:.6f}")
        
        if min_area < mean_area * 0.01:
            print(f"   Warning: {(self.element_areas < mean_area * 0.01).sum()} degenerate elements")
    
    def _compute_uv_mapping(self, original_mesh):
        """
        Compute UV coordinates for texture mapping in Three.js
        
        Uses spherical projection as a simple approach
        """
        # Normalize coordinates to unit sphere
        coords_centered = self.coords - self.coords.mean(axis=0)
        r = np.linalg.norm(coords_centered, axis=1, keepdims=True)
        coords_normalized = coords_centered / (r + 1e-12)
        
        # Spherical coordinates
        x, y, z = coords_normalized.T
        
        # UV mapping: (theta, phi) → (u, v)
        theta = np.arctan2(y, x)        # [-π, π]
        phi = np.arcsin(np.clip(z, -1, 1))  # [-π/2, π/2]
        
        u = (theta + np.pi) / (2 * np.pi)   # [0, 1]
        v = (phi + np.pi/2) / np.pi         # [0, 1]
        
        self.uv_coords = np.column_stack([u, v])
        
        print(f"   UV mapping computed")
    
    def _package_results(self):
        """Package all results into dictionary"""
        metadata = {
            'num_nodes': len(self.coords),
            'num_elements': len(self.connectivity),
            'total_surface_area': float(self.element_areas.sum()),
            'element_type': 'Quad-8 (serendipity)',
            'coordinate_system': 'meters',
            'iss_model': str(self.glb_path.name)
        }
        
        print("\n" + "="*60)
        print("MESH GENERATION COMPLETE")
        print("="*60)
        print(f"Nodes:           {metadata['num_nodes']}")
        print(f"Elements:        {metadata['num_elements']}")
        print(f"Surface area:    {metadata['total_surface_area']:.2f} m²")
        print("="*60 + "\n")
        
        return {
            'coords': self.coords,
            'connectivity': self.connectivity,
            'areas': self.element_areas,
            'normals': self.normals,
            'uv': self.uv_coords,
            'metadata': metadata
        }
    
    def export_to_excel(self, output_path, mesh_data=None):
        """
        Export mesh in format compatible with your FEM solver
        
        Creates Excel file with sheets:
        - coord: Node coordinates (x, y, z)
        - conec: Element connectivity (1-based indexing)
        - areas: Element areas
        - normals: Element normals
        - uv: UV coordinates
        """
        if mesh_data is None:
            mesh_data = {
                'coords': self.coords,
                'connectivity': self.connectivity,
                'areas': self.element_areas,
                'normals': self.normals,
                'uv': self.uv_coords
            }
        
        output_path = Path(output_path)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Coordinates (x, y, z)
            pd.DataFrame(
                mesh_data['coords'],
                columns=['x', 'y', 'z']
            ).to_excel(writer, sheet_name='coord', index=False, header=False)
            
            # Connectivity (1-based for compatibility)
            pd.DataFrame(
                mesh_data['connectivity'] + 1,
                columns=[f'node_{i}' for i in range(8)]
            ).to_excel(writer, sheet_name='conec', index=False, header=False)
            
            # Element areas
            pd.DataFrame(
                mesh_data['areas'],
                columns=['area']
            ).to_excel(writer, sheet_name='areas', index=False, header=False)
            
            # Element normals
            pd.DataFrame(
                mesh_data['normals'],
                columns=['nx', 'ny', 'nz']
            ).to_excel(writer, sheet_name='normals', index=False, header=False)
            
            # UV coordinates
            pd.DataFrame(
                mesh_data['uv'],
                columns=['u', 'v']
            ).to_excel(writer, sheet_name='uv', index=False, header=False)
        
        print(f"Mesh exported to: {output_path}")
    
    def export_to_json(self, output_path, mesh_data=None):
        """
        Export mesh for Three.js frontend
        
        Compact JSON format for Socket.IO transmission
        """
        if mesh_data is None:
            mesh_data = {
                'coords': self.coords,
                'connectivity': self.connectivity,
                'uv': self.uv_coords
            }
        
        output_path = Path(output_path)
        
        # Convert to lists for JSON serialization
        json_data = {
            'coords': mesh_data['coords'].tolist(),
            'connectivity': mesh_data['connectivity'].tolist(),
            'uv': mesh_data['uv'].tolist(),
            'metadata': {
                'num_nodes': len(mesh_data['coords']),
                'num_elements': len(mesh_data['connectivity']),
                'element_type': 'Quad-8'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"JSON mesh exported to: {output_path}")
    
    def visualize_mesh(self, mesh_data=None):
        """
        Quick visualization using trimesh for validation
        
        Returns trimesh scene for inspection
        """
        if mesh_data is None:
            mesh_data = {
                'coords': self.coords,
                'connectivity': self.connectivity,
                'normals': self.normals
            }
        
        # Convert Quad-8 to triangles for visualization
        # Each quad → 2 triangles using corner nodes
        faces = []
        for quad in mesh_data['connectivity']:
            # Corner nodes: 0, 1, 2, 3
            n0, n1, n2, n3 = quad[0], quad[1], quad[2], quad[3]
            faces.append([n0, n1, n2])
            faces.append([n0, n2, n3])
        
        vis_mesh = trimesh.Trimesh(
            vertices=mesh_data['coords'],
            faces=np.array(faces),
            process=False
        )
        
        # Color by normals
        colors = ((mesh_data['normals'] + 1) / 2 * 255).astype(np.uint8)
        colors = np.repeat(colors, 2, axis=0)  # Each quad → 2 faces
        vis_mesh.visual.face_colors = colors
        
        return vis_mesh


def main():
    """
    Example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Quad-8 FEM mesh from ISS .glb')
    parser.add_argument('glb_path', type=str, help='Path to ISS .glb file')
    parser.add_argument('--elements', type=int, default=20000, help='Target number of elements')
    parser.add_argument('--output', type=str, default='iss_mesh', help='Output file prefix')
    parser.add_argument('--visualize', action='store_true', help='Show mesh in viewer')
    
    args = parser.parse_args()
    
    # Generate mesh
    generator = ISSMeshGenerator(args.glb_path, target_elements=args.elements)
    mesh_data = generator.generate()
    
    # Export
    generator.export_to_excel(f'{args.output}.xlsx', mesh_data)
    generator.export_to_json(f'{args.output}.json', mesh_data)
    
    # Visualize
    if args.visualize:
        vis_mesh = generator.visualize_mesh(mesh_data)
        vis_mesh.show()


if __name__ == '__main__':
    main()
