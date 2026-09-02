import gmsh
import os

def generate_mesh(x_coords, y_coords, config):
    gmsh.initialize()
    # Wyłączenie GUI; logi pozostawione (1) do diagnozy.
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.add("airfoil_domain")

    # Geometria profilu.
    # Znalezienie indeksu krawedzi natarcia (minimalna wartosc X).
    # Znalezienie indeksu krawedzi natarcia.
    min_x = min(x_coords)
    le_index = list(x_coords).index(min_x)
    
    lc_airfoil = 0.005
    
    # Punkty dolnej powierzchni.
    pts_lower = []
    for x, y in zip(x_coords[:le_index+1], y_coords[:le_index+1]):
        pts_lower.append(gmsh.model.geo.addPoint(x, y, 0, lc_airfoil))
        
    # Punkty gornej powierzchni.
    pts_upper = [pts_lower[-1]] 
    for x, y in zip(x_coords[le_index+1:-1], y_coords[le_index+1:-1]):
        pts_upper.append(gmsh.model.geo.addPoint(x, y, 0, lc_airfoil))
        
    # Zszycie krawedzi splywu - uzycie ID pierwszego punktu krzywej dolnej
    pts_upper.append(pts_lower[0])
    
    # Tworzenie krzywych bocznych (profil zamkniety na koncu punktem).
    curve_lower = gmsh.model.geo.addSpline(pts_lower)
    curve_upper = gmsh.model.geo.addSpline(pts_upper)
    # curve_te = gmsh.model.geo.addLine(pts_upper[-1], pts_lower[0])

    # Definicja domeny zewnetrznej (O-grid).
    R = 20.0
    lc_far = 2.0
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
    
    # Zbudowanie petli profilu z dwoch krzywych.
    airfoil_loop = gmsh.model.geo.addCurveLoop([curve_lower, curve_upper])
    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])
    gmsh.model.geo.synchronize()

    # Definicja warstwy przyściennej dla RANS.
    gmsh.model.mesh.field.add("BoundaryLayer", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [curve_lower, curve_upper])
    gmsh.model.mesh.field.setNumber(1, "Size", 0.00001)
    gmsh.model.mesh.field.setNumber(1, "Ratio", 1.15)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 0.03)
    gmsh.model.mesh.field.setNumber(1, "Quads", 1)
    gmsh.model.mesh.field.setAsBoundaryLayer(1)

    # 1. Wewnetrzny, drobny box blisko splywu.
    gmsh.model.mesh.field.add("Box", 2)
    gmsh.model.mesh.field.setNumber(2, "VIn", 0.003)
    gmsh.model.mesh.field.setNumber(2, "VOut", lc_far)
    gmsh.model.mesh.field.setNumber(2, "XMin", 0.9)
    gmsh.model.mesh.field.setNumber(2, "XMax", 1.5)
    gmsh.model.mesh.field.setNumber(2, "YMin", -0.15)
    gmsh.model.mesh.field.setNumber(2, "YMax", 0.15)

    # 2. Zewnetrzny, szerszy box dla odchylonego sladu.
    gmsh.model.mesh.field.add("Box", 3)
    gmsh.model.mesh.field.setNumber(3, "VIn", 0.01)
    gmsh.model.mesh.field.setNumber(3, "VOut", lc_far)
    gmsh.model.mesh.field.setNumber(3, "XMin", 1.5)
    gmsh.model.mesh.field.setNumber(3, "XMax", 6.0)
    gmsh.model.mesh.field.setNumber(3, "YMin", -0.5)
    gmsh.model.mesh.field.setNumber(3, "YMax", 1.0)
    
    # 3. Zlaczenie obu obszarow (Gmsh wybierze mniejszy rozmiar elementu).
    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
    
    # Ustawienie połączonego obszaru MIN jako docelowego tła.
    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    # Parametryzacja zageszczenia siatki 2D.
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
    
    # Grupowanie elementow dla oznaczen w SU2 (MARKER).
    farfield_group = gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4])
    gmsh.model.setPhysicalName(1, farfield_group, "farfield")
    
    # Definicja grupy fizycznej profilu.
    airfoil_group = gmsh.model.addPhysicalGroup(1, [curve_lower, curve_upper])
    gmsh.model.setPhysicalName(1, airfoil_group, "airfoil")
    
    fluid_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, fluid_group, "fluid")

    # Pojedyncza, wlasciwa generacja siatki.
    gmsh.model.mesh.generate(2)
    
    # Ekstrakcja statystyk i weryfikacja udzialu elementow czworokatnych.
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    num_tris = 0
    num_quads = 0
    
    for e_type, tags in zip(elem_types, elem_tags):
        if e_type == 2:    # Element trojkatny (3-wezlowy)
            num_tris += len(tags)
        elif e_type == 3:  # Element czworokatny (4-wezlowy)
            num_quads += len(tags)
            
    print(f"   [GMSH] Raport siatki - Trójkąty: {num_tris}, Czworokąty: {num_quads}")

    output_path = os.path.join(config["workspace_dir"], config["mesh_filename"])
    gmsh.write(output_path)
    gmsh.finalize()