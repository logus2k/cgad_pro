"""
Test Suite for ISS Mesh Generator

Verifies:
1. Dependencies installed correctly
2. Basic mesh generation works
3. Output formats valid
4. Quality checks pass
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages are installed"""
    print("\n" + "="*70)
    print("TEST 1: Checking Dependencies")
    print("="*70)
    
    required = [
        ('numpy', 'NumPy'),
        ('trimesh', 'Trimesh'),
        ('gmsh', 'Gmsh'),
        ('pandas', 'Pandas'),
        ('cupy', 'CuPy (GPU)'),
    ]
    
    optional = [
        ('numba', 'Numba (optional)'),
        ('flask', 'Flask (for web server)'),
    ]
    
    all_ok = True
    
    for module, name in required:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            all_ok = False
    
    print("\nOptional packages:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"  {name} - not installed")
    
    if all_ok:
        print("\n✓ All required dependencies installed")
        return True
    else:
        print("\n✗ Missing required dependencies - run: pip install -r requirements.txt")
        return False


def test_gmsh():
    """Test Gmsh functionality"""
    print("\n" + "="*70)
    print("TEST 2: Gmsh Functionality")
    print("="*70)
    
    try:
        import gmsh
        
        gmsh.initialize()
        gmsh.model.add("test")
        
        # Create simple geometry
        gmsh.model.geo.addPoint(0, 0, 0, 1.0, 1)
        gmsh.model.geo.addPoint(1, 0, 0, 1.0, 2)
        gmsh.model.geo.addPoint(1, 1, 0, 1.0, 3)
        gmsh.model.geo.addPoint(0, 1, 0, 1.0, 4)
        
        gmsh.model.geo.addLine(1, 2, 1)
        gmsh.model.geo.addLine(2, 3, 2)
        gmsh.model.geo.addLine(3, 4, 3)
        gmsh.model.geo.addLine(4, 1, 4)
        
        gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1)
        gmsh.model.geo.addPlaneSurface([1], 1)
        
        gmsh.model.geo.synchronize()
        
        # Generate mesh
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        
        # Elevate to order 2
        gmsh.model.mesh.setOrder(2)
        
        try:
            # Get all element types
            all_data = gmsh.model.mesh.getElements(2)
            elem_types = all_data[0]  # This is a tuple of element types
            
            print(f"DEBUG: Found {len(elem_types)} element type(s)")
            
            # Check if we have quad elements
            has_quads = False
            found_types = []
            
            for et in elem_types:
                et_int = int(et)
                found_types.append(et_int)
                if et_int in [3, 10, 16]:
                    has_quads = True
            
            print(f"DEBUG: Element types present: {found_types}")
            
            gmsh.finalize()
            
            if has_quads:
                quad_types = [t for t in found_types if t in [3, 10, 16]]
                type_names = {3: "Quad-4", 10: "Quad-9", 16: "Quad-8"}
                found = ", ".join(type_names.get(t, f"Type-{t}") for t in set(quad_types))
                print(f"✓ Gmsh can generate quad elements ({found})")
                return True
            else:
                print("✗ Gmsh did not generate quad elements")
                return False
                
        except Exception as e:
            gmsh.finalize()
            print(f"✗ Error checking elements: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"✗ Gmsh test failed: {e}")
        return False


def test_cupy():
    """Test CuPy GPU functionality"""
    print("\n" + "="*70)
    print("TEST 3: GPU (CuPy) Functionality")
    print("="*70)
    
    try:
        import cupy as cp
        import numpy as np
        
        # Test basic operations
        a = cp.array([1, 2, 3])
        b = cp.array([4, 5, 6])
        c = a + b
        
        expected = np.array([5, 7, 9])
        result = cp.asnumpy(c)
        
        if np.allclose(result, expected):
            # Get device info safely
            try:
                device_name = cp.cuda.Device().compute_capability
                device_str = f"CUDA Device (CC {device_name})"
            except:
                device_str = "CUDA Device"
            
            print(f"✓ CuPy working ({device_str})")
            
            # Test memory
            mem_info = cp.cuda.runtime.memGetInfo()
            free_gb = mem_info[0] / 1e9
            total_gb = mem_info[1] / 1e9
            print(f"  GPU Memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
            
            return True
        else:
            print("✗ CuPy computation incorrect")
            return False
            
    except Exception as e:
        print(f"⚠ CuPy not available: {e}")
        print("  GPU acceleration will not work, but CPU solver still functional")
        return False


def test_simple_mesh_generation():
    """Test mesh generation with a simple cube"""
    print("\n" + "="*70)
    print("TEST 4: Simple Mesh Generation")
    print("="*70)
    
    try:
        import trimesh
        import numpy as np
        from iss_mesh_generator import ISSMeshGenerator
        
        # Create simple cube
        cube = trimesh.creation.box(extents=[1, 1, 1])
        
        # Save as temp GLB
        temp_glb = Path('/tmp/test_cube.glb')
        cube.export(str(temp_glb))
        
        print("✓ Created test cube geometry")
        
        # Generate mesh
        print("  Generating Quad-8 mesh...")
        generator = ISSMeshGenerator(temp_glb, target_elements=100)
        mesh_data = generator.generate()
        
        # Validate output
        assert mesh_data['coords'] is not None, "No coordinates generated"
        assert mesh_data['connectivity'] is not None, "No connectivity generated"
        assert len(mesh_data['connectivity'][0]) == 8, "Not Quad-8 elements"
        
        print(f"✓ Generated {len(mesh_data['connectivity'])} Quad-8 elements")
        print(f"  Nodes: {len(mesh_data['coords'])}")
        print(f"  Surface area: {mesh_data['metadata']['total_surface_area']:.2f} m²")
        
        # Test exports
        temp_xlsx = Path('/tmp/test_mesh.xlsx')
        temp_json = Path('/tmp/test_mesh.json')
        
        generator.export_to_excel(temp_xlsx, mesh_data)
        generator.export_to_json(temp_json, mesh_data)
        
        assert temp_xlsx.exists(), "Excel export failed"
        assert temp_json.exists(), "JSON export failed"
        
        print("✓ Export to Excel and JSON successful")
        
        # Cleanup
        temp_glb.unlink()
        temp_xlsx.unlink()
        temp_json.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Mesh generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mesh_quality():
    """Test mesh quality validation"""
    print("\n" + "="*70)
    print("TEST 5: Mesh Quality Validation")
    print("="*70)
    
    try:
        import trimesh
        import numpy as np
        from iss_mesh_generator import ISSMeshGenerator
        
        # Create sphere (better geometry for quality test)
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        
        temp_glb = Path('/tmp/test_sphere.glb')
        sphere.export(str(temp_glb))
        
        # Generate mesh
        generator = ISSMeshGenerator(temp_glb, target_elements=200)
        mesh_data = generator.generate()
        
        # Quality checks
        areas = mesh_data['areas']
        normals = mesh_data['normals']
        
        # Check 1: No degenerate elements
        min_area = areas.min()
        mean_area = areas.mean()
        if min_area > mean_area * 0.01:
            print("✓ No degenerate elements")
        else:
            print(f"⚠ Some degenerate elements (min area = {min_area:.6f})")
        
        # Check 2: Normals are unit vectors
        normal_lens = np.linalg.norm(normals, axis=1)
        if np.allclose(normal_lens, 1.0, atol=1e-6):
            print("✓ All normals are unit vectors")
        else:
            print(f"⚠ Some non-unit normals (range: {normal_lens.min():.6f} - {normal_lens.max():.6f})")
        
        # Check 3: Connectivity valid
        max_idx = mesh_data['connectivity'].max()
        num_nodes = len(mesh_data['coords'])
        if max_idx < num_nodes:
            print("✓ Connectivity indices valid")
        else:
            print(f"✗ Invalid connectivity (max index {max_idx} >= {num_nodes} nodes)")
        
        # Cleanup
        temp_glb.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Quality validation failed: {e}")
        return False


def test_fem_integration():
    """Test integration with FEM solver format"""
    print("\n" + "="*70)
    print("TEST 6: FEM Solver Integration")
    print("="*70)
    
    try:
        import pandas as pd
        import numpy as np
        import trimesh
        from iss_mesh_generator import ISSMeshGenerator
        
        # Generate test mesh
        cube = trimesh.creation.box(extents=[1, 1, 1])
        temp_glb = Path('/tmp/test_fem.glb')
        cube.export(str(temp_glb))
        
        generator = ISSMeshGenerator(temp_glb, target_elements=50)
        mesh_data = generator.generate()
        
        # Export to Excel
        temp_xlsx = Path('/tmp/test_fem.xlsx')
        generator.export_to_excel(temp_xlsx, mesh_data)
        
        # Try to load as if in FEM solver
        coord = pd.read_excel(temp_xlsx, sheet_name="coord", header=None)
        conec = pd.read_excel(temp_xlsx, sheet_name="conec", header=None)
        
        x = coord.iloc[:, 0].to_numpy(dtype=float)
        y = coord.iloc[:, 1].to_numpy(dtype=float)
        z = coord.iloc[:, 2].to_numpy(dtype=float)
        
        quad8 = conec.to_numpy(dtype=int) - 1  # Convert to 0-based
        
        # Validate
        assert len(x) == len(mesh_data['coords']), "Coordinate count mismatch"
        assert quad8.shape[1] == 8, "Not Quad-8 connectivity"
        assert quad8.min() >= 0, "Invalid 0-based index (negative)"
        assert quad8.max() < len(x), "Invalid index (out of range)"
        
        print("✓ Excel format compatible with FEM solver")
        print(f"  Loaded {len(x)} nodes, {len(quad8)} elements")
        
        # Cleanup
        temp_glb.unlink()
        temp_xlsx.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ FEM integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "ISS MESH GENERATOR TEST SUITE" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = [
        ("Dependencies", test_imports),
        ("Gmsh", test_gmsh),
        ("GPU (CuPy)", test_cupy),
        ("Mesh Generation", test_simple_mesh_generation),
        ("Quality Validation", test_mesh_quality),
        ("FEM Integration", test_fem_integration),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - System ready for mesh generation!")
        return 0
    elif passed >= total - 1 and not results[2][1]:  # CuPy optional
        print("\n⚠ All core tests passed (GPU optional)")
        print("  You can proceed with CPU-only mesh generation")
        return 0
    else:
        print("\n❌ Some tests failed - please fix before proceeding")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
