"""
ISS Model Preparation Utility

Validate and prepare ISS .glb model for mesh generation.
"""

from pathlib import Path
import trimesh
import numpy as np


def validate_glb(glb_path):
    """
    Validate .glb file and print diagnostic info
    
    Parameters
    ----------
    glb_path : str or Path
        Path to .glb file
    """
    glb_path = Path(glb_path)
    
    if not glb_path.exists():
        print(f"✗ File not found: {glb_path}")
        return False
    
    print(f"\nValidating: {glb_path}")
    print("-" * 70)
    
    try:
        # Load with trimesh
        scene = trimesh.load(str(glb_path))
        
        # Check if scene or single mesh
        if isinstance(scene, trimesh.Scene):
            print(f"✓ Valid .glb scene with {len(scene.geometry)} components")
            
            # List components
            print("\nComponents:")
            for name, geom in scene.geometry.items():
                if hasattr(geom, 'vertices'):
                    print(f"  - {name}: {len(geom.vertices)} vertices, {len(geom.faces)} faces")
            
            # Dump to single mesh for analysis
            mesh = scene.dump(concatenate=True)
            
        else:
            mesh = scene
            print(f"✓ Valid .glb single mesh")
        
        # Mesh statistics
        print(f"\nCombined Mesh Statistics:")
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces: {len(mesh.faces)}")
        print(f"  Edges: {len(mesh.edges)}")
        print(f"  Is watertight: {mesh.is_watertight}")
        print(f"  Is winding consistent: {mesh.is_winding_consistent}")
        
        # Bounds
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        print(f"\nBounding Box:")
        print(f"  Min: {bounds[0]}")
        print(f"  Max: {bounds[1]}")
        print(f"  Size: {size}")
        
        # Surface area (handle None case)
        try:
            area = mesh.area
            if area is not None and area > 0:
                print(f"  Surface Area: {area:.2f} m²")
            else:
                print(f"  Surface Area: Unable to compute (mesh issues)")
        except:
            print(f"  Surface Area: Unable to compute (mesh issues)")
        
        # Volume
        if mesh.is_watertight:
            print(f"  Volume: {mesh.volume:.2f} m³")
        else:
            print(f"  Volume: N/A (not watertight)")
        
        # Quality checks
        print(f"\nQuality Checks:")
        
        # Check for degenerate faces
        face_areas = mesh.area_faces
        degenerate = face_areas < 1e-10
        if degenerate.any():
            print(f"  ⚠ {degenerate.sum()} degenerate faces (area < 1e-10)")
        else:
            print(f"  ✓ No degenerate faces")
        
        # Check for duplicate vertices
        mesh_clean = mesh.copy()
        merged = mesh_clean.merge_vertices()
        if merged > 0:
            print(f"  ⚠ {merged} duplicate vertices merged")
        else:
            print(f"  ✓ No duplicate vertices")
        
        # Check for unreferenced vertices
        referenced = np.zeros(len(mesh.vertices), dtype=bool)
        referenced[mesh.faces.ravel()] = True
        unreferenced = (~referenced).sum()
        if unreferenced > 0:
            print(f"  ⚠ {unreferenced} unreferenced vertices")
        else:
            print(f"  ✓ All vertices referenced")
        
        # Recommend mesh density
        print(f"\nRecommended Quad-8 Element Counts:")
        for factor in [0.1, 0.25, 0.5, 1.0]:
            target = int(len(mesh.faces) * factor)
            print(f"  {factor*100:3.0f}% density: ~{target:,} elements")
        
        print("-" * 70)
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def inspect_glb_interactive(glb_path):
    """
    Open interactive 3D viewer for inspection
    
    Parameters
    ----------
    glb_path : str or Path
        Path to .glb file
    """
    glb_path = Path(glb_path)
    
    if not glb_path.exists():
        print(f"File not found: {glb_path}")
        return
    
    print(f"Loading {glb_path} in viewer...")
    print("Close the viewer window to continue.")
    
    scene = trimesh.load(str(glb_path))
    scene.show()


def optimize_for_meshing(input_glb, output_glb, target_faces=40000):
    """
    Preprocess .glb for optimal meshing
    
    Operations:
    - Remove small components
    - Merge duplicate vertices
    - Simplify to target face count
    - Fix winding
    
    Parameters
    ----------
    input_glb : str or Path
        Input .glb file
    output_glb : str or Path
        Output optimized .glb file
    target_faces : int
        Target triangle count after simplification
    """
    print(f"\nOptimizing {input_glb} for meshing...")
    
    scene = trimesh.load(str(input_glb))
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    
    print(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # 1. Merge duplicate vertices
    merged = mesh.merge_vertices()
    print(f"Merged {merged} duplicate vertices")
    
    # 2. Remove degenerate faces
    mask = mesh.nondegenerate_faces()
    if not mask.all():
        mesh.update_faces(mask)
        print(f"Removed {(~mask).sum()} degenerate faces")
    
    # 3. Remove unreferenced vertices  
    mesh.remove_unreferenced_vertices()
    print(f"Removed unreferenced vertices")
    
    # 4. Check for scaling issues
    if mesh.area == 0 or mesh.area is None:
        bbox_size = mesh.bounding_box.extents
        print(f"Warning: Zero area detected. Bounding box: {bbox_size}")
        if np.max(bbox_size) < 0.001:
            print("Scaling mesh by 1000x")
            mesh.apply_scale(1000.0)
    
    # 5. Split and filter components
    components = mesh.split(only_watertight=False)
    print(f"Found {len(components)} components")
    
    # Keep components with at least 0.1% of total faces
    if len(components) > 1:
        total_faces = sum(len(c.faces) for c in components)
        min_faces = total_faces * 0.001
        filtered = [c for c in components if len(c.faces) > min_faces]
        print(f"Keeping {len(filtered)} main components (>{min_faces:.0f} faces)")
        
        if len(filtered) > 0:
            mesh = trimesh.util.concatenate(filtered)
        else:
            print("Warning: All components too small, keeping largest")
            mesh = max(components, key=lambda c: len(c.faces))
    
    # 6. Simplify if needed
    if len(mesh.faces) > target_faces:
        print(f"Simplifying {len(mesh.faces)} → {target_faces} faces...")
        try:
            # Use vertex clustering (most reliable)
            current_area = mesh.area if mesh.area and mesh.area > 0 else mesh.bounding_box.volume**(2/3)
            target_edge_length = np.sqrt(current_area / target_faces) * 1.5
            mesh = mesh.simplify_vertex_clustering(threshold=target_edge_length)
            
            # If still too many faces, increase threshold
            iterations = 0
            while len(mesh.faces) > target_faces * 1.2 and iterations < 3:
                target_edge_length *= 1.3
                mesh = mesh.simplify_vertex_clustering(threshold=target_edge_length)
                iterations += 1
                
        except Exception as e:
            print(f"Simplification warning: {e}")
    
    # 7. Fix winding
    if not mesh.is_winding_consistent:
        print("Fixing winding order...")
        mesh.fix_normals()
    
    print(f"Final: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Export
    mesh.export(str(output_glb))
    print(f"✓ Saved to: {output_glb}")


def main():
    """Interactive model preparation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ISS Model Preparation Utility')
    parser.add_argument('--validate', type=str, help='Validate .glb file')
    parser.add_argument('--view', type=str, help='View .glb in interactive viewer')
    parser.add_argument('--optimize', type=str, help='Optimize .glb for meshing')
    parser.add_argument('--output', type=str, default='iss_optimized.glb', help='Output path for optimized file')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_glb(args.validate)
    
    if args.view:
        inspect_glb_interactive(args.view)
    
    if args.optimize:
        output = args.output
        optimize_for_meshing(args.optimize, output)
    
    if not any([args.validate, args.view, args.optimize]):
        print("ISS Model Preparation Utility")
        print("=" * 70)
        print("\nUsage examples:")
        print("  python prepare_iss_model.py --validate ../../data/models/iss.glb")
        print("  python prepare_iss_model.py --view ../../data/models/iss.glb")
        print("  python prepare_iss_model.py --optimize ../../data/models/iss.glb")
        print("\nWorkflow:")
        print("  1. Validate: Check triangle count and quality")
        print("  2. (Optional) Optimize: Simplify if model is too dense")
        print("  3. Generate mesh with iss_mesh_generator.py")


if __name__ == '__main__':
    main()
