import gmsh
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_mesh(x_coords, y_coords, config):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("airfoil_domain")

    # --- 1. GEOMETRY ---
    min_x = min(x_coords)
    le_index = list(x_coords).index(min_x)
    lc_airfoil = 0.001 # proper x^+
    
    pts_lower = []
    for x, y in zip(x_coords[:le_index+1], y_coords[:le_index+1]):
        pts_lower.append(gmsh.model.geo.addPoint(x, y, 0, lc_airfoil))
        
    pts_upper = [pts_lower[-1]] 
    for x, y in zip(x_coords[le_index+1:], y_coords[le_index+1:]):
        pts_upper.append(gmsh.model.geo.addPoint(x, y, 0, lc_airfoil))
        
    # Creating the side curves.
    curve_lower = gmsh.model.geo.addSpline(pts_lower)
    curve_upper = gmsh.model.geo.addSpline(pts_upper)
    
    # Center of the closing arc of the trailing edge (X=1.0, Y=0.0).
    center_te = gmsh.model.geo.addPoint(1.0, 0.0, 0.0, lc_airfoil)
    
    # Creation of the closing arc (CCW from bottom to top). Ensures convexity on the outside.
    curve_te = gmsh.model.geo.addCircleArc(pts_lower[0], center_te, pts_upper[-1]) 

    R = 15.0
    lc_far = 1.5
    center = gmsh.model.geo.addPoint(0.5, 0, 0)
    
    p1 = gmsh.model.geo.addPoint(0.5 + R, 0, 0, lc_far)
    p2 = gmsh.model.geo.addPoint(0.5, R, 0, lc_far)
    p3 = gmsh.model.geo.addPoint(0.5 - R, 0, 0, lc_far)
    p4 = gmsh.model.geo.addPoint(0.5, -R, 0, lc_far)
    
    arc1 = gmsh.model.geo.addCircleArc(p1, center, p2)
    arc2 = gmsh.model.geo.addCircleArc(p2, center, p3)
    arc3 = gmsh.model.geo.addCircleArc(p3, center, p4)
    arc4 = gmsh.model.geo.addCircleArc(p4, center, p1)
    
    farfield_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
    
    # Closing the airfoil loop. The '-' modifier reverses the direction of the curve_te from top to bottom.
    airfoil_loop = gmsh.model.geo.addCurveLoop([curve_lower, curve_upper, -curve_te])
    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])
    gmsh.model.geo.synchronize()
    
    # Forcing 20 nodes on the trailing edge curve. This ensures a smooth representation 
    # of the curvature instead of a sharp corner with 3 points. 
    gmsh.model.mesh.setTransfiniteCurve(curve_te, 20)

    # --- 2. REFINEMENT BOX ---
    # Mesh refinement box behind the airfoil. The mesh will be finer in this area, which is important for capturing the wake.
    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", 0.002)
    gmsh.model.mesh.field.setNumber(1, "VOut", lc_far)
    gmsh.model.mesh.field.setNumber(1, "XMin", 0.9)
    gmsh.model.mesh.field.setNumber(1, "XMax", 1.5)
    gmsh.model.mesh.field.setNumber(1, "YMin", -0.15)
    gmsh.model.mesh.field.setNumber(1, "YMax", 0.15)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 2.0)

    # --- 3. BOUNDARY LAYER  ---
    gmsh.model.mesh.field.add("BoundaryLayer", 2)
    gmsh.model.mesh.field.setNumbers(2, "CurvesList", [curve_lower, curve_upper, curve_te])
    gmsh.model.mesh.field.setNumber(2, "Size", 0.00001)
    gmsh.model.mesh.field.setNumber(2, "Ratio", 1.10)
    gmsh.model.mesh.field.setNumber(2, "Thickness", 0.03)
    gmsh.model.mesh.field.setNumber(2, "Quads", 1)
    gmsh.model.mesh.field.setAsBoundaryLayer(2)

    # --- 4. MERGING FIELDS ---
    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])
    gmsh.model.mesh.field.setAsBackgroundMesh(3)

    # --- 5. SIZE CONSTRAINTS ---
    # Smallest mesh size on the airfoil and in the boundary layer, largest in the far field.
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.00001)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)

    # Saving groups
    farfield_group = gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4])
    gmsh.model.setPhysicalName(1, farfield_group, "farfield")
    airfoil_group = gmsh.model.addPhysicalGroup(1, [curve_lower, curve_upper, curve_te])
    gmsh.model.setPhysicalName(1, airfoil_group, "airfoil")
    fluid_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, fluid_group, "fluid")

    gmsh.model.mesh.generate(2)
    
    # --- 6. QUALITY REPORT AND VISUALIZATION ---
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    num_tris = 0
    num_quads = 0
    sicn_min = 1.0
    sicn_sum = 0.0
    total_2d = 0
    
    for e_type, tags in zip(elem_types, elem_tags):
        num_elements = len(tags)
        if e_type == 2: num_tris += num_elements
        elif e_type == 3: num_quads += num_elements
        qualities = gmsh.model.mesh.getElementQualities(tags, "minSICN")
        sicn_min = min(sicn_min, min(qualities))
        sicn_sum += sum(qualities)
        total_2d += num_elements
        
    sicn_avg = sicn_sum / total_2d if total_2d > 0 else 0
    total_nodes = gmsh.model.mesh.getNodes()[0].size
    
    report_path = os.path.join(config["workspace_dir"], "mesh_report.txt")
    with open(report_path, "w") as f:
        f.write("--- QUALITY REPORT OF THE COMPUTATIONAL MESH ---\n")
        f.write(f"Total nodes: {total_nodes}\n")
        f.write(f"2D elements: {total_2d}\n")
        f.write(f"  - Triangles (background): {num_tris}\n")
        f.write(f"  - Quads (boundary layer): {num_quads}\n\n")
        f.write("--- minSICN METRIC ---\n")
        f.write(f"Minimum minSICN: {sicn_min:.5f}\n")
        f.write(f"Average minSICN: {sicn_avg:.5f}\n")

    try:
        _, node_coords, _ = gmsh.model.mesh.getNodes()
        node_x, node_y = node_coords[0::3], node_coords[1::3]
        plt.figure(figsize=(12, 6))
        plt.plot(node_x, node_y, 'k.', markersize=0.3, alpha=0.3)
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.4, 0.4)
        plt.gca().set_aspect('equal')
        plt.title('Mesh density distribution')
        plt.savefig(os.path.join(config["workspace_dir"], "mesh_preview_wsl.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except: pass

    output_path = os.path.join(config["workspace_dir"], config["mesh_filename"])
    gmsh.write(output_path)
    gmsh.finalize()