"""
Mesh Generator for 2D Quad-8 FEM Solver

Generates HDF5 mesh files compatible with the FEM solver and web visualization.
Supports multiple pre-defined geometries with configurable mesh density.

Dependencies:
    pip install gmsh h5py numpy

Usage:
    from mesh_generator import generate_mesh
    
    # Using preset sizes
    generate_mesh("venturi", "venturi_medium.h5", size="medium")
    
    # Using custom target node count
    generate_mesh("channel_with_cylinder", "cylinder.h5", target_nodes=100000)
    
    # Using explicit element size
    generate_mesh("straight_channel", "channel.h5", element_size=0.02)

Available geometries:
    - straight_channel
    - converging_nozzle
    - diverging_diffuser
    - venturi
    - s_bend
    - t_junction
    - elbow_90
    - backward_step
    - channel_with_cylinder
    - channel_with_rectangle
"""

import gmsh
import h5py
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import math


# =============================================================================
# Size Presets (matching Y-tube mesh sizes)
# =============================================================================
SIZE_PRESETS = {
    "small": {"target_nodes": 200},
    "medium": {"target_nodes": 195_000},
    "large": {"target_nodes": 772_000},
    "xlarge": {"target_nodes": 1_300_000},
}


# =============================================================================
# Geometry Definitions
# =============================================================================

def _create_straight_channel(length: float = 3.0, width: float = 1.0, **kwargs) -> Tuple[float, Dict]:
    """
    Simple rectangular channel.
    
    Inlet: left edge (x=0)
    Outlet: right edge (x=length)
    """
    rect = gmsh.model.occ.addRectangle(0, 0, 0, length, width)
    gmsh.model.occ.synchronize()
    
    area = length * width
    return area, {"length": length, "width": width}


def _create_converging_nozzle(
    length: float = 3.0,
    inlet_width: float = 1.0,
    outlet_width: float = 0.4,
    **kwargs
) -> Tuple[float, Dict]:
    """
    Nozzle that narrows toward outlet.
    
    Inlet: left edge (wide)
    Outlet: right edge (narrow)
    """
    # Create trapezoid using points
    y_offset_top = (inlet_width - outlet_width) / 2
    y_offset_bottom = (inlet_width - outlet_width) / 2
    
    p1 = gmsh.model.occ.addPoint(0, 0, 0)
    p2 = gmsh.model.occ.addPoint(length, y_offset_bottom, 0)
    p3 = gmsh.model.occ.addPoint(length, y_offset_bottom + outlet_width, 0)
    p4 = gmsh.model.occ.addPoint(0, inlet_width, 0)
    
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)
    
    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    
    area = length * (inlet_width + outlet_width) / 2
    return area, {"length": length, "inlet_width": inlet_width, "outlet_width": outlet_width}


def _create_diverging_diffuser(
    length: float = 3.0,
    inlet_width: float = 0.4,
    outlet_width: float = 1.0,
    **kwargs
) -> Tuple[float, Dict]:
    """
    Diffuser that widens toward outlet.
    
    Inlet: left edge (narrow)
    Outlet: right edge (wide)
    """
    return _create_converging_nozzle(
        length=length,
        inlet_width=inlet_width,
        outlet_width=outlet_width
    )


def _create_venturi(
    length: float = 3.0,
    width: float = 1.0,
    throat_width: float = 0.4,
    throat_position: float = 0.5,
    **kwargs
) -> Tuple[float, Dict]:
    """
    Venturi tube - converges then diverges.
    
    Inlet: left edge
    Outlet: right edge
    Throat: narrowest point at throat_position (0-1)
    """
    throat_x = length * throat_position
    y_offset = (width - throat_width) / 2
    
    # Create shape with smooth curves using splines
    n_points = 20
    top_points = []
    bottom_points = []
    
    for i in range(n_points + 1):
        x = length * i / n_points
        # Smooth contraction/expansion using cosine
        if x <= throat_x:
            t = x / throat_x
            y_shrink = y_offset * (1 - math.cos(t * math.pi)) / 2
        else:
            t = (x - throat_x) / (length - throat_x)
            y_shrink = y_offset * (1 + math.cos(t * math.pi)) / 2
        
        top_points.append(gmsh.model.occ.addPoint(x, width - y_shrink, 0))
        bottom_points.append(gmsh.model.occ.addPoint(x, y_shrink, 0))
    
    # Create splines for top and bottom
    top_spline = gmsh.model.occ.addSpline(top_points)
    bottom_spline = gmsh.model.occ.addSpline(bottom_points)
    
    # Close the ends
    inlet_line = gmsh.model.occ.addLine(bottom_points[0], top_points[0])
    outlet_line = gmsh.model.occ.addLine(top_points[-1], bottom_points[-1])
    
    loop = gmsh.model.occ.addCurveLoop([bottom_spline, outlet_line, -top_spline, -inlet_line])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    
    # Approximate area
    area = length * (width + throat_width) / 2
    return area, {"length": length, "width": width, "throat_width": throat_width}


def _create_s_bend(
    length: float = 4.0,
    width: float = 0.5,
    amplitude: float = 1.0,
    **kwargs
) -> Tuple[float, Dict]:
    """
    S-shaped curved channel.
    
    Inlet: left edge
    Outlet: right edge (offset by amplitude)
    """
    n_points = 40
    top_points = []
    bottom_points = []
    
    for i in range(n_points + 1):
        x = length * i / n_points
        # S-curve using sine
        y_center = amplitude * math.sin(x / length * math.pi)
        
        top_points.append(gmsh.model.occ.addPoint(x, y_center + width/2, 0))
        bottom_points.append(gmsh.model.occ.addPoint(x, y_center - width/2, 0))
    
    top_spline = gmsh.model.occ.addSpline(top_points)
    bottom_spline = gmsh.model.occ.addSpline(bottom_points)
    
    inlet_line = gmsh.model.occ.addLine(bottom_points[0], top_points[0])
    outlet_line = gmsh.model.occ.addLine(top_points[-1], bottom_points[-1])
    
    loop = gmsh.model.occ.addCurveLoop([bottom_spline, outlet_line, -top_spline, -inlet_line])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    
    area = length * width
    return area, {"length": length, "width": width, "amplitude": amplitude}


def _create_t_junction(
    main_length: float = 3.0,
    main_width: float = 0.5,
    branch_length: float = 1.0,
    branch_width: float = 0.5,
    branch_position: float = 0.5,
    **kwargs
) -> Tuple[float, Dict]:
    """
    T-junction - main channel with perpendicular branch.
    
    Inlet: left edge of main channel
    Outlets: right edge of main channel + top of branch
    """
    branch_x = main_length * branch_position
    
    # Main channel rectangle
    main_rect = gmsh.model.occ.addRectangle(0, 0, 0, main_length, main_width)
    
    # Branch rectangle
    branch_rect = gmsh.model.occ.addRectangle(
        branch_x - branch_width/2, 
        main_width, 
        0, 
        branch_width, 
        branch_length
    )
    
    # Fuse them together
    result = gmsh.model.occ.fuse([(2, main_rect)], [(2, branch_rect)])
    gmsh.model.occ.synchronize()
    
    area = main_length * main_width + branch_length * branch_width
    return area, {
        "main_length": main_length, 
        "main_width": main_width,
        "branch_length": branch_length,
        "branch_width": branch_width
    }


def _create_elbow_90(
    width: float = 0.5,
    inner_radius: float = 0.5,
    inlet_length: float = 1.0,
    outlet_length: float = 1.0,
    **kwargs
) -> Tuple[float, Dict]:
    """
    90-degree elbow with rounded corner.
    
    Inlet: bottom (negative y direction)
    Outlet: right (positive x direction)
    """
    outer_radius = inner_radius + width
    
    # Create the elbow arc region
    # Inner arc
    inner_arc = gmsh.model.occ.addCircle(0, 0, 0, inner_radius, angle1=0, angle2=math.pi/2)
    outer_arc = gmsh.model.occ.addCircle(0, 0, 0, outer_radius, angle1=0, angle2=math.pi/2)
    
    # Inlet section (vertical, going down from arc)
    inlet_rect = gmsh.model.occ.addRectangle(inner_radius, -inlet_length, 0, width, inlet_length)
    
    # Outlet section (horizontal, going right from arc)
    outlet_rect = gmsh.model.occ.addRectangle(0, inner_radius, 0, outlet_length, width)
    
    # Create arc section
    p1 = gmsh.model.occ.addPoint(inner_radius, 0, 0)
    p2 = gmsh.model.occ.addPoint(outer_radius, 0, 0)
    p3 = gmsh.model.occ.addPoint(0, outer_radius, 0)
    p4 = gmsh.model.occ.addPoint(0, inner_radius, 0)
    
    center = gmsh.model.occ.addPoint(0, 0, 0)
    
    arc_inner = gmsh.model.occ.addCircleArc(p1, center, p4)
    arc_outer = gmsh.model.occ.addCircleArc(p2, center, p3)
    line_bottom = gmsh.model.occ.addLine(p1, p2)
    line_top = gmsh.model.occ.addLine(p3, p4)
    
    loop = gmsh.model.occ.addCurveLoop([line_bottom, arc_outer, line_top, -arc_inner])
    arc_surface = gmsh.model.occ.addPlaneSurface([loop])
    
    # Fuse all parts
    result = gmsh.model.occ.fuse(
        [(2, arc_surface)], 
        [(2, inlet_rect), (2, outlet_rect)]
    )
    gmsh.model.occ.synchronize()
    
    area = (math.pi * (outer_radius**2 - inner_radius**2) / 4 + 
            inlet_length * width + outlet_length * width)
    return area, {
        "width": width, 
        "inner_radius": inner_radius,
        "inlet_length": inlet_length,
        "outlet_length": outlet_length
    }


def _create_backward_step(
    inlet_length: float = 1.0,
    inlet_height: float = 0.5,
    outlet_length: float = 3.0,
    outlet_height: float = 1.0,
    **kwargs
) -> Tuple[float, Dict]:
    """
    Backward-facing step - classic CFD benchmark.
    
    Inlet: left edge (narrow)
    Outlet: right edge (full height)
    Step at junction of inlet and outlet sections.
    """
    step_height = outlet_height - inlet_height
    
    # Create L-shaped region
    p1 = gmsh.model.occ.addPoint(0, step_height, 0)
    p2 = gmsh.model.occ.addPoint(inlet_length, step_height, 0)
    p3 = gmsh.model.occ.addPoint(inlet_length, 0, 0)
    p4 = gmsh.model.occ.addPoint(inlet_length + outlet_length, 0, 0)
    p5 = gmsh.model.occ.addPoint(inlet_length + outlet_length, outlet_height, 0)
    p6 = gmsh.model.occ.addPoint(0, outlet_height, 0)
    
    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p5)
    l5 = gmsh.model.occ.addLine(p5, p6)
    l6 = gmsh.model.occ.addLine(p6, p1)
    
    loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4, l5, l6])
    surface = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()
    
    area = inlet_length * inlet_height + outlet_length * outlet_height
    return area, {
        "inlet_length": inlet_length,
        "inlet_height": inlet_height,
        "outlet_length": outlet_length,
        "outlet_height": outlet_height
    }


def _create_channel_with_cylinder(
    length: float = 4.0,
    width: float = 1.5,
    obstacle_radius: float = 0.25,
    obstacle_x: Optional[float] = None,
    obstacle_y: Optional[float] = None,
    **kwargs
) -> Tuple[float, Dict]:
    """
    Rectangular channel with circular obstacle.
    
    Inlet: left edge
    Outlet: right edge
    Obstacle: cylinder at specified position (default: center)
    """
    actual_obstacle_x = obstacle_x if obstacle_x is not None else length / 3
    actual_obstacle_y = obstacle_y if obstacle_y is not None else width / 2
    
    # Create channel
    rect = gmsh.model.occ.addRectangle(0, 0, 0, length, width)
    
    # Create cylinder (disk in 2D)
    disk = gmsh.model.occ.addDisk(actual_obstacle_x, actual_obstacle_y, 0, obstacle_radius, obstacle_radius)
    
    # Subtract cylinder from channel
    result = gmsh.model.occ.cut([(2, rect)], [(2, disk)])
    gmsh.model.occ.synchronize()
    
    area = length * width - math.pi * obstacle_radius**2
    return area, {
        "length": length,
        "width": width,
        "obstacle_radius": obstacle_radius,
        "obstacle_x": actual_obstacle_x,
        "obstacle_y": actual_obstacle_y
    }


def _create_channel_with_rectangle(
    length: float = 4.0,
    width: float = 1.5,
    obstacle_width: float = 0.3,
    obstacle_height: float = 0.5,
    obstacle_x: Optional[float] = None,
    obstacle_y: Optional[float] = None,
    **kwargs
) -> Tuple[float, Dict]:
    """
    Rectangular channel with rectangular obstacle (bluff body).
    
    Inlet: left edge
    Outlet: right edge
    Obstacle: rectangle at specified position (default: center)
    """
    actual_obstacle_x = obstacle_x if obstacle_x is not None else length / 3
    actual_obstacle_y = obstacle_y if obstacle_y is not None else (width - obstacle_height) / 2
    
    # Create channel
    rect = gmsh.model.occ.addRectangle(0, 0, 0, length, width)
    
    # Create obstacle
    obstacle = gmsh.model.occ.addRectangle(
        actual_obstacle_x, actual_obstacle_y, 0, 
        obstacle_width, obstacle_height
    )
    
    # Subtract obstacle from channel
    result = gmsh.model.occ.cut([(2, rect)], [(2, obstacle)])
    gmsh.model.occ.synchronize()
    
    area = length * width - obstacle_width * obstacle_height
    return area, {
        "length": length,
        "width": width,
        "obstacle_width": obstacle_width,
        "obstacle_height": obstacle_height,
        "obstacle_x": actual_obstacle_x,
        "obstacle_y": actual_obstacle_y
    }


# =============================================================================
# Geometry Registry
# =============================================================================
GEOMETRIES = {
    "straight_channel": _create_straight_channel,
    "converging_nozzle": _create_converging_nozzle,
    "diverging_diffuser": _create_diverging_diffuser,
    "venturi": _create_venturi,
    "s_bend": _create_s_bend,
    "t_junction": _create_t_junction,
    "elbow_90": _create_elbow_90,
    "backward_step": _create_backward_step,
    "channel_with_cylinder": _create_channel_with_cylinder,
    "channel_with_rectangle": _create_channel_with_rectangle,
}


# =============================================================================
# Mesh Generation
# =============================================================================

def _estimate_element_size(target_nodes: int | float, area: float) -> float:
    """
    Estimate element size to achieve target node count.
    
    For Quad-8 elements:
    - Each element has 8 nodes
    - Shared nodes reduce total count
    - Roughly: nodes ≈ 3 * elements (for well-structured mesh)
    - elements ≈ area / (element_size^2)
    
    So: element_size ≈ sqrt(factor * area / target_nodes)
    """
    # Empirical factor adjusted for Quad-8
    # Calibrated: target 195K with factor 2.5 gave 292K nodes
    # Adjusted: 2.5 * (292014/195000) ≈ 3.75
    factor = 3.75
    element_size = math.sqrt(factor * area / target_nodes)
    return element_size


def _extract_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract mesh data from Gmsh.
    
    Returns:
        x: Node X coordinates
        y: Node Y coordinates
        quad8: Element connectivity (0-indexed)
    """
    # Get all nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    
    # Create mapping from Gmsh node tags to 0-indexed
    tag_to_idx = {int(tag): idx for idx, tag in enumerate(node_tags)}
    
    # Extract coordinates
    n_nodes = len(node_tags)
    x = np.zeros(n_nodes, dtype=np.float64)
    y = np.zeros(n_nodes, dtype=np.float64)
    
    for i, tag in enumerate(node_tags):
        idx = tag_to_idx[int(tag)]
        x[idx] = node_coords[i * 3]
        y[idx] = node_coords[i * 3 + 1]
    
    # Get 2D elements
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    
    # Gmsh element types for quadrilaterals:
    # Type 3: 4-node quad (linear)
    # Type 10: 9-node quad (second order, serendipity)
    # Type 16: 8-node quad (second order, serendipity) - what we want
    # Type 36: 16-node quad (third order)
    
    quad8_connectivity = None
    
    # Debug: print what element types we found
    print(f"  Found element types: {list(elem_types)}")
    
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        # Get element type info
        name, dim, order, num_nodes, local_coords, num_primary = gmsh.model.mesh.getElementProperties(etype)
        print(f"  Element type {etype}: {name}, {num_nodes} nodes, order {order}")
        
        if num_nodes == 8 and dim == 2:
            # 8-node quadrilateral
            n_elements = len(etags)
            quad8_connectivity = np.zeros((n_elements, 8), dtype=np.int32)
            
            for i in range(n_elements):
                for j in range(8):
                    gmsh_tag = int(enodes[i * 8 + j])
                    quad8_connectivity[i, j] = tag_to_idx[gmsh_tag]
            break
        elif num_nodes == 9 and dim == 2:
            # 9-node quadrilateral - convert to 8-node by dropping center node
            print(f"  Converting 9-node quads to 8-node (dropping center node)")
            n_elements = len(etags)
            quad8_connectivity = np.zeros((n_elements, 8), dtype=np.int32)
            
            # Gmsh 9-node quad node ordering: corners (0-3), mid-edges (4-7), center (8)
            # We keep nodes 0-7, drop node 8
            for i in range(n_elements):
                for j in range(8):
                    gmsh_tag = int(enodes[i * 9 + j])
                    quad8_connectivity[i, j] = tag_to_idx[gmsh_tag]
            break
    
    if quad8_connectivity is None:
        raise RuntimeError("No Quad-8 or Quad-9 elements found in mesh. Check meshing options.")
    
    return x, y, quad8_connectivity


def _save_mesh(filepath: Path, x: np.ndarray, y: np.ndarray, quad8: np.ndarray, 
               geometry_type: str, params: Dict[str, Any]):
    """Save mesh to HDF5 file."""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('x', data=x, dtype=np.float64)
        f.create_dataset('y', data=y, dtype=np.float64)
        f.create_dataset('quad8', data=quad8, dtype=np.int32)
        
        # Store metadata
        f.attrs['geometry_type'] = geometry_type
        f.attrs['nodes'] = len(x)
        f.attrs['elements'] = len(quad8)
        
        for key, value in params.items():
            f.attrs[key] = value


def generate_mesh(
    geometry: str,
    output_path: str | Path,
    size: Optional[str] = None,
    target_nodes: Optional[int] = None,
    element_size: Optional[float] = None,
    verbose: bool = True,
    **geometry_params
) -> Dict[str, Any]:
    """
    Generate a 2D Quad-8 mesh for the specified geometry.
    
    Args:
        geometry: Geometry type (see GEOMETRIES.keys())
        output_path: Path to save HDF5 mesh file
        size: Preset size ("small", "medium", "large", "xlarge")
        target_nodes: Target number of nodes (alternative to size)
        element_size: Explicit element size (alternative to size/target_nodes)
        verbose: Print progress information
        **geometry_params: Geometry-specific parameters
        
    Returns:
        Dictionary with mesh statistics
    """
    if geometry not in GEOMETRIES:
        available = ", ".join(GEOMETRIES.keys())
        raise ValueError(f"Unknown geometry '{geometry}'. Available: {available}")
    
    # Determine element size
    if element_size is None:
        if target_nodes is not None:
            pass  # Will calculate after geometry creation
        elif size is not None:
            if size not in SIZE_PRESETS:
                available = ", ".join(SIZE_PRESETS.keys())
                raise ValueError(f"Unknown size '{size}'. Available: {available}")
            target_nodes = SIZE_PRESETS[size]["target_nodes"]
        else:
            # Default to medium
            target_nodes = SIZE_PRESETS["medium"]["target_nodes"]
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize Gmsh
    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)
    
    try:
        gmsh.model.add("mesh")
        
        # Create geometry
        if verbose:
            print(f"Creating geometry: {geometry}")
        
        geometry_func = GEOMETRIES[geometry]
        area, params = geometry_func(**geometry_params)
        
        if verbose:
            print(f"  Area: {area:.4f}")
            print(f"  Parameters: {params}")
        
        # Calculate element size if needed
        if element_size is None:
            assert target_nodes is not None, "target_nodes must be set"
            element_size = _estimate_element_size(target_nodes, area)
            if verbose:
                print(f"  Target nodes: {target_nodes:,}")
                print(f"  Estimated element size: {element_size:.6f}")
        
        # Configure mesh options for Quad-8
        gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Second order (8-node)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine into quads
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # Blossom algorithm
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)  # Serendipity elements (Quad-8, not Quad-9)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size * 1.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)
        
        # Set element size on all points
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), element_size)
        
        # Generate mesh
        if verbose:
            print("Generating mesh...")
        
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.recombine()
        
        # Extract mesh data
        if verbose:
            print("Extracting mesh data...")
        
        x, y, quad8 = _extract_mesh()
        
        # Save to HDF5
        if verbose:
            print(f"Saving to {output_path}")
        
        _save_mesh(output_path, x, y, quad8, geometry, params)
        
        stats = {
            "geometry": geometry,
            "nodes": len(x),
            "elements": len(quad8),
            "element_size": element_size,
            "area": area,
            "parameters": params,
            "output_path": str(output_path)
        }
        
        if verbose:
            print(f"\nMesh generated successfully:")
            print(f"  Nodes: {stats['nodes']:,}")
            print(f"  Elements: {stats['elements']:,}")
            print(f"  Output: {output_path}")
        
        return stats
        
    finally:
        gmsh.finalize()


def list_geometries():
    """Print available geometries and their parameters."""
    print("Available geometries:")
    print("=" * 60)
    
    for name, func in GEOMETRIES.items():
        doc = func.__doc__ or "No description available."
        # Get first paragraph of docstring
        desc = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
        print(f"\n{name}:")
        print(f"  {desc}")
    
    print("\n" + "=" * 60)
    print("\nSize presets:")
    for name, preset in SIZE_PRESETS.items():
        print(f"  {name}: ~{preset['target_nodes']:,} nodes")


def generate_all_samples(output_dir: str | Path, size: str = "small"):
    """Generate sample meshes for all geometries."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating all sample meshes ({size} size)")
    print("=" * 60)
    
    results = []
    
    for geometry in GEOMETRIES.keys():
        try:
            output_path = output_dir / f"{geometry}_{size}.h5"
            stats = generate_mesh(geometry, output_path, size=size, verbose=False)
            results.append(stats)
            print(f"  {geometry}: {stats['nodes']:,} nodes, {stats['elements']:,} elements")
        except Exception as e:
            print(f"  {geometry}: FAILED - {e}")
    
    print("=" * 60)
    print(f"Generated {len(results)} meshes in {output_dir}")
    
    return results


# =============================================================================
# Command Line Interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate 2D Quad-8 meshes for FEM solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mesh_generator.py venturi output/venturi.h5 --size medium
  python mesh_generator.py channel_with_cylinder output/cyl.h5 --target-nodes 50000
  python mesh_generator.py straight_channel output/channel.h5 --element-size 0.02
  python mesh_generator.py --list
  python mesh_generator.py --generate-all output/samples --size small
        """
    )
    
    parser.add_argument("geometry", nargs="?", help="Geometry type")
    parser.add_argument("output", nargs="?", help="Output HDF5 file path")
    parser.add_argument("--size", choices=list(SIZE_PRESETS.keys()), 
                        help="Preset size")
    parser.add_argument("--target-nodes", type=int, 
                        help="Target number of nodes")
    parser.add_argument("--element-size", type=float, 
                        help="Explicit element size")
    parser.add_argument("--list", action="store_true", 
                        help="List available geometries")
    parser.add_argument("--generate-all", metavar="DIR",
                        help="Generate all geometries to directory")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress output")
    
    args = parser.parse_args()
    
    if args.list:
        list_geometries()
    elif args.generate_all:
        generate_all_samples(args.generate_all, size=args.size or "small")
    elif args.geometry and args.output:
        generate_mesh(
            args.geometry,
            args.output,
            size=args.size,
            target_nodes=args.target_nodes,
            element_size=args.element_size,
            verbose=not args.quiet
        )
    else:
        parser.print_help()
