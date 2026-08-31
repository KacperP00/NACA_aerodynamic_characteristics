import gmsh
import os

def generate_mesh(x_coords, y_coords, config):
    gmsh.initialize()
    # Wyłączenie GUI; logi pozostawione (1) do diagnozy.
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.add("airfoil_domain")

    # Geometria profilu.
    airfoil_pts = []
    lc_airfoil = 0.005  # Wymuszenie mniejszego rozmiaru elementu na ściance profilu.
    
    # Pominięcie ostatniego punktu (duplikatu krawędzi spływu).
    for x, y in zip(x_coords[:-1], y_coords[:-1]):
        airfoil_pts.append(gmsh.model.geo.addPoint(x, y, 0, lc_airfoil))
    
    # Domknięcie topologiczne - ponowne dodanie pierwszego ID.
    airfoil_pts.append(airfoil_pts[0])
    airfoil_curve = gmsh.model.geo.addSpline(airfoil_pts)

    # Definicja domeny zewnętrznej (O-grid).
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
    
    # Tworzenie pętli i powierzchni obszaru płynu.
    farfield_loop = gmsh.model.geo.addCurveLoop([arc1, arc2, arc3, arc4])
    airfoil_loop = gmsh.model.geo.addCurveLoop([airfoil_curve])
    
    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])
    gmsh.model.geo.synchronize()

    # Definicja warstwy przyściennej dla RANS.
    gmsh.model.mesh.field.add("BoundaryLayer", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [airfoil_curve])
    
    # Parametry dla y+ ~ 1 (przy Re = 3e6).
    gmsh.model.mesh.field.setNumber(1, "Size", 0.00001)
    gmsh.model.mesh.field.setNumber(1, "Ratio", 1.15)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 0.03)
    gmsh.model.mesh.field.setNumber(1, "Quads", 1) # wymuszenie kształtu elementów w warstwie przyściennej (quad).
    gmsh.model.mesh.field.setAsBoundaryLayer(1)

    # Definicja prostokątnego obszaru zagęszczenia w śladzie torowym (Wake Refinement).
    gmsh.model.mesh.field.add("Box", 2)
    gmsh.model.mesh.field.setNumber(2, "VIn", 0.03)      # Rozmiar elementu wewnątrz śladu
    gmsh.model.mesh.field.setNumber(2, "VOut", lc_far)   # Rozmiar elementu poza śladem (dalekie pole)
    gmsh.model.mesh.field.setNumber(2, "XMin", 0.9)      # Początek obszaru (nieco przed krawędzią spływu)
    gmsh.model.mesh.field.setNumber(2, "XMax", 10.0)     # Zasięg obszaru w dół rzeki
    gmsh.model.mesh.field.setNumber(2, "YMin", -0.2)     # Dolna granica
    gmsh.model.mesh.field.setNumber(2, "YMax", 0.2)      # Górna granica
    gmsh.model.mesh.field.setNumber(2, "Thickness", 0.3) # Szerokość strefy płynnego przejścia siatki
    
    # Ustawienie obszaru Box jako tła dla siatki.
    # Uwaga: Warstwa przyścienna (Field 1) aplikowana jest niezależnie przez setAsBoundaryLayer.
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Parametryzacja zagęszczenia siatki 2D.
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
    
    # Grupowanie elementów dla oznaczeń w SU2 (MARKER).
    farfield_group = gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4])
    gmsh.model.setPhysicalName(1, farfield_group, "farfield")
    
    airfoil_group = gmsh.model.addPhysicalGroup(1, [airfoil_curve])
    gmsh.model.setPhysicalName(1, airfoil_group, "airfoil")
    
    fluid_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, fluid_group, "fluid")

    # Generacja i zapis siatki (format .su2).
    gmsh.model.mesh.generate(2)
    output_path = os.path.join(config["workspace_dir"], config["mesh_filename"])

    # Generacja siatki.
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