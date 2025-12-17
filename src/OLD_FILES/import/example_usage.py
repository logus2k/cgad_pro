"""
Example: Generate ISS Quad-8 mesh for FEM thermal analysis

This script demonstrates the full workflow:
1. Download/load ISS model
2. Generate Quad-8 surface mesh
3. Export for FEM solver
4. Validate mesh quality
"""

from pathlib import Path
import numpy as np
from iss_mesh_generator import ISSMeshGenerator


def example_basic_usage():
    """Basic mesh generation"""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Mesh Generation")
    print("="*70 + "\n")
    
    # Path to your ISS .glb file
    glb_path = "data/models/iss.glb"
    
    # Create generator
    generator = ISSMeshGenerator(
        glb_path=glb_path,
        target_elements=5000  # Start small for testing
    )
    
    # Generate mesh
    mesh_data = generator.generate()
    
    # Export for FEM solver (Excel format)
    generator.export_to_excel("output/iss_mesh_5k.xlsx", mesh_data)
    
    # Export for Three.js (JSON format)
    generator.export_to_json("output/iss_mesh_5k.json", mesh_data)
    
    print("\nMesh files created:")
    print("  - output/iss_mesh_5k.xlsx  (for FEM solver)")
    print("  - output/iss_mesh_5k.json  (for Three.js)")


def example_multi_resolution():
    """Generate meshes at different resolutions for comparison"""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Multi-Resolution Meshes")
    print("="*70 + "\n")
    
    glb_path = "data/models/iss.glb"
    resolutions = [5000, 20000, 50000]
    
    for target_elems in resolutions:
        print(f"\nGenerating {target_elems} element mesh...")
        
        generator = ISSMeshGenerator(glb_path, target_elements=target_elems)
        mesh_data = generator.generate()
        
        # Export
        prefix = f"output/iss_mesh_{target_elems//1000}k"
        generator.export_to_excel(f"{prefix}.xlsx", mesh_data)
        generator.export_to_json(f"{prefix}.json", mesh_data)
        
        # Quality report
        print(f"\nQuality metrics for {target_elems} element mesh:")
        print(f"  Actual elements: {mesh_data['metadata']['num_elements']}")
        print(f"  Surface area: {mesh_data['metadata']['total_surface_area']:.2f} m²")
        
        areas = mesh_data['areas']
        print(f"  Element area range: {areas.min():.6f} - {areas.max():.6f} m²")
        print(f"  Area std/mean: {areas.std()/areas.mean():.3f}")


def example_mesh_validation():
    """Validate mesh quality and check for issues"""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: Mesh Quality Validation")
    print("="*70 + "\n")
    
    glb_path = "data/models/iss.glb"
    
    generator = ISSMeshGenerator(glb_path, target_elements=10000)
    mesh_data = generator.generate()
    
    coords = mesh_data['coords']
    connectivity = mesh_data['connectivity']
    areas = mesh_data['areas']
    normals = mesh_data['normals']
    
    print("Mesh Validation Report")
    print("-" * 70)
    
    # 1. Check for duplicate nodes
    unique_coords = np.unique(coords, axis=0)
    if len(unique_coords) < len(coords):
        print(f"⚠ Warning: {len(coords) - len(unique_coords)} duplicate nodes found")
    else:
        print("✓ No duplicate nodes")
    
    # 2. Check connectivity bounds
    max_node_idx = connectivity.max()
    if max_node_idx >= len(coords):
        print(f"✗ Error: Connectivity references node {max_node_idx} but only {len(coords)} nodes exist")
    else:
        print(f"✓ Connectivity valid (max index: {max_node_idx})")
    
    # 3. Check element quality
    mean_area = areas.mean()
    degenerate = areas < mean_area * 0.01
    if degenerate.any():
        print(f"⚠ Warning: {degenerate.sum()} degenerate elements (area < 1% of mean)")
    else:
        print("✓ No degenerate elements")
    
    # 4. Check normal consistency
    normal_lengths = np.linalg.norm(normals, axis=1)
    if not np.allclose(normal_lengths, 1.0, atol=1e-6):
        print(f"⚠ Warning: {(~np.isclose(normal_lengths, 1.0, atol=1e-6)).sum()} non-unit normals")
    else:
        print("✓ All normals are unit vectors")
    
    # 5. Check for inverted elements (negative area)
    if (areas < 0).any():
        print(f"✗ Error: {(areas < 0).sum()} elements with negative area (inverted)")
    else:
        print("✓ No inverted elements")
    
    # 6. Aspect ratio check (approximate)
    # For each element, check ratio of max to min edge length
    aspect_ratios = []
    for i in range(len(connectivity)):
        nodes = connectivity[i]
        corners = coords[nodes[:4]]  # Corner nodes only
        
        # Compute edge lengths
        edges = [
            np.linalg.norm(corners[1] - corners[0]),
            np.linalg.norm(corners[2] - corners[1]),
            np.linalg.norm(corners[3] - corners[2]),
            np.linalg.norm(corners[0] - corners[3])
        ]
        
        aspect_ratios.append(max(edges) / (min(edges) + 1e-12))
    
    aspect_ratios = np.array(aspect_ratios)
    poor_aspect = aspect_ratios > 5.0
    
    print(f"\nAspect Ratio Statistics:")
    print(f"  Mean: {aspect_ratios.mean():.2f}")
    print(f"  Max: {aspect_ratios.max():.2f}")
    print(f"  Elements with AR > 5: {poor_aspect.sum()} ({poor_aspect.sum()/len(aspect_ratios)*100:.1f}%)")
    
    # 7. Surface closure check
    # All edges should appear exactly twice (shared by 2 elements)
    edge_count = {}
    for elem in connectivity:
        corners = elem[:4]
        edges = [
            tuple(sorted([corners[0], corners[1]])),
            tuple(sorted([corners[1], corners[2]])),
            tuple(sorted([corners[2], corners[3]])),
            tuple(sorted([corners[3], corners[0]]))
        ]
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1
    
    boundary_edges = sum(1 for count in edge_count.values() if count == 1)
    non_manifold = sum(1 for count in edge_count.values() if count > 2)
    
    if boundary_edges > 0:
        print(f"\n⚠ Warning: {boundary_edges} boundary edges (open surface)")
    else:
        print("\n✓ Closed surface (no boundary edges)")
    
    if non_manifold > 0:
        print(f"⚠ Warning: {non_manifold} non-manifold edges")
    else:
        print("✓ Manifold surface")
    
    print("-" * 70)


def example_custom_preprocessing():
    """Demonstrate custom preprocessing before mesh generation"""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom Preprocessing")
    print("="*70 + "\n")
    
    import trimesh
    
    glb_path = "data/models/iss.glb"
    
    # Load and preprocess manually
    print("Loading ISS model...")
    scene = trimesh.load(glb_path)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    
    print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Optional: Remove small disconnected components
    components = mesh.split(only_watertight=False)
    print(f"Found {len(components)} components")
    
    # Keep only large components (e.g., ISS main body, solar panels)
    main_components = [c for c in components if len(c.faces) > 100]
    print(f"Keeping {len(main_components)} main components")
    
    # Merge back
    mesh = trimesh.util.concatenate(main_components)
    
    # Save preprocessed mesh
    mesh.export("data/models/iss_preprocessed.glb")
    
    print("\nGenerating mesh from preprocessed model...")
    generator = ISSMeshGenerator("data/models/iss_preprocessed.glb", target_elements=15000)
    mesh_data = generator.generate()
    
    generator.export_to_excel("output/iss_mesh_preprocessed.xlsx", mesh_data)


def example_integration_test():
    """Test integration with existing FEM solver"""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: FEM Solver Integration Test")
    print("="*70 + "\n")
    
    glb_path = "data/models/iss.glb"
    
    # Generate small mesh for quick testing
    generator = ISSMeshGenerator(glb_path, target_elements=1000)
    mesh_data = generator.generate()
    
    # Export in format compatible with quad8_cpu.py
    output_path = "data/input/mesh_data_iss_test.xlsx"
    generator.export_to_excel(output_path, mesh_data)
    
    print(f"\nMesh exported to: {output_path}")
    print("\nTo test with your FEM solver, modify quad8_cpu.py:")
    print("  1. Change xlsx_path to point to this file")
    print("  2. Adjust boundary conditions for ISS geometry")
    print("  3. Run thermal analysis")
    
    # Show how to load the mesh in FEM solver
    print("\nExample FEM solver adaptation:")
    print("""
    import pandas as pd
    
    xlsx_path = "data/input/mesh_data_iss_test.xlsx"
    
    coord = pd.read_excel(xlsx_path, sheet_name="coord", header=None)
    conec = pd.read_excel(xlsx_path, sheet_name="conec", header=None)
    areas = pd.read_excel(xlsx_path, sheet_name="areas", header=None)
    normals = pd.read_excel(xlsx_path, sheet_name="normals", header=None)
    
    x = coord.iloc[:, 0].to_numpy(dtype=float)
    y = coord.iloc[:, 1].to_numpy(dtype=float)
    z = coord.iloc[:, 2].to_numpy(dtype=float)
    
    quad8 = conec.to_numpy(dtype=int) - 1  # Convert to 0-based
    
    # Now use with your Elem_Quad8_Thermal function
    """)


if __name__ == '__main__':
    # Create output directories
    Path("output").mkdir(exist_ok=True)
    Path("data/models").mkdir(parents=True, exist_ok=True)
    Path("data/input").mkdir(parents=True, exist_ok=True)
    
    # Run examples
    print("\n" + "="*70)
    print("ISS MESH GENERATOR - USAGE EXAMPLES")
    print("="*70)
    
    # Note: You'll need to download the ISS .glb file first
    # Example sources:
    # - NASA 3D Resources: https://nasa3d.arc.nasa.gov/
    # - Sketchfab: https://sketchfab.com/search?q=iss
    
    print("\nIMPORTANT: Place your ISS .glb file at: data/models/iss.glb")
    print("\nPress Enter to continue with examples (or Ctrl+C to exit)...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\nExiting...")
        exit(0)
    
    # Run each example
    try:
        example_basic_usage()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_mesh_validation()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Review generated meshes in output/")
    print("  2. Visualize with trimesh: python -c \"import trimesh; trimesh.load('output/iss_mesh_5k.json').show()\"")
    print("  3. Integrate with FEM solver (see example_integration_test)")
    print("  4. Set up Socket.IO server for real-time visualization")
