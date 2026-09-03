import gmsh
import os
import matplotlib
matplotlib.use('Agg')  # Wymusza silnik "headless" - pomija X11 i OpenGL w WSL
import matplotlib.pyplot as plt

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
    # 1. Górna powierzchnia - UWAGA: bierzemy wszystkie punkty aż do końca (usuwamy [:-1])
    pts_upper = [pts_lower[-1]] 
    for x, y in zip(x_coords[le_index+1:], y_coords[le_index+1:]):
        pts_upper.append(gmsh.model.geo.addPoint(x, y, 0, lc_airfoil))
        
    # USUNIĘTE ZSZYWANIE W PUNKT: pts_upper.append(pts_lower[0])
    
    # 2. Tworzenie krzywych bocznych i TĘPEGO SPŁYWU
    curve_lower = gmsh.model.geo.addSpline(pts_lower)
    curve_upper = gmsh.model.geo.addSpline(pts_upper)
    # Dodanie pionowego odcinka na krawędzi spływu
    curve_te = gmsh.model.geo.addLine(pts_upper[-1], pts_lower[0]) 

    # 3. Definicja domeny (O-grid) - bez zmian
    R = 17.0
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
    
    # 4. Zbudowanie pętli profilu z TRZECH krzywych
    airfoil_loop = gmsh.model.geo.addCurveLoop([curve_lower, curve_upper, curve_te])
    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])
    gmsh.model.geo.synchronize()

    # 5. Definicja warstwy przyściennej 
    # UWAGA: Do CurvesList wrzucamy tylko profil. Krawędź spływu (curve_te) zostaje 
    # wolna, dzięki czemu warstwa zejdzie płasko do śladu bez zakrzywiania się!
    gmsh.model.mesh.field.add("BoundaryLayer", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [curve_lower, curve_upper])
    gmsh.model.mesh.field.setNumber(1, "Size", 0.00001)
    gmsh.model.mesh.field.setNumber(1, "Ratio", 1.15)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 0.03)
    gmsh.model.mesh.field.setNumber(1, "Quads", 1)
    gmsh.model.mesh.field.setAsBoundaryLayer(1)

    # 6. Płynne boxy ze strefami przejścia (Thickness)
    # Wewnętrzny, drobny box blisko spływu
    gmsh.model.mesh.field.add("Box", 2)
    gmsh.model.mesh.field.setNumber(2, "VIn", 0.003)
    gmsh.model.mesh.field.setNumber(2, "VOut", 0.02) # Płynny krok do Boxa 3
    gmsh.model.mesh.field.setNumber(2, "XMin", 0.95)
    gmsh.model.mesh.field.setNumber(2, "XMax", 1.5)
    gmsh.model.mesh.field.setNumber(2, "YMin", -0.05)
    gmsh.model.mesh.field.setNumber(2, "YMax", 0.05)
    gmsh.model.mesh.field.setNumber(2, "Thickness", 0.15) # Gradient 15 cm

    # Zewnętrzny, szerszy box dla odchylonego śladu
    gmsh.model.mesh.field.add("Box", 3)
    gmsh.model.mesh.field.setNumber(3, "VIn", 0.02)
    gmsh.model.mesh.field.setNumber(3, "VOut", lc_far) # Koniec gradientu, przejście w tło 2.0
    gmsh.model.mesh.field.setNumber(3, "XMin", 1.0)
    gmsh.model.mesh.field.setNumber(3, "XMax", 6.0)
    gmsh.model.mesh.field.setNumber(3, "YMin", -0.3)
    gmsh.model.mesh.field.setNumber(3, "YMax", 0.3)
    gmsh.model.mesh.field.setNumber(3, "Thickness", 1.0) # Łagodne rozszerzenie na 1 metr

    # Łączenie boxów 
    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2, 3])
    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    # Parametryzacja zageszczenia siatki 2D.
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
    gmsh.option.setNumber("Mesh.MeshSizeMax", lc_far)
    
    # Grupowanie elementow dla oznaczen w SU2 (MARKER).
    farfield_group = gmsh.model.addPhysicalGroup(1, [arc1, arc2, arc3, arc4])
    gmsh.model.setPhysicalName(1, farfield_group, "farfield")
    
    # Definicja grupy fizycznej profilu.
    airfoil_group = gmsh.model.addPhysicalGroup(1, [curve_lower, curve_upper, curve_te])
    gmsh.model.setPhysicalName(1, airfoil_group, "airfoil")
    
    fluid_group = gmsh.model.addPhysicalGroup(2, [surface])
    gmsh.model.setPhysicalName(2, fluid_group, "fluid")

    # Pojedyncza, wlasciwa generacja siatki.
    gmsh.model.mesh.generate(2)
    
    # Ekstrakcja statystyk i weryfikacja udzialu elementow 2D.
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(2)
    num_tris = 0
    num_quads = 0
    
    # Inicjalizacja metryki minSICN.
    sicn_min = 1.0
    sicn_sum = 0.0
    total_2d = 0
    
    for e_type, tags in zip(elem_types, elem_tags):
        num_elements = len(tags)
        if e_type == 2:    # Trojkat
            num_tris += num_elements
        elif e_type == 3:  # Czworokat
            num_quads += num_elements
            
        # Obliczenie jakosci elementow (minSICN).
        qualities = gmsh.model.mesh.getElementQualities(tags, "minSICN")
        sicn_min = min(sicn_min, min(qualities))
        sicn_sum += sum(qualities)
        total_2d += num_elements
        
    sicn_avg = sicn_sum / total_2d if total_2d > 0 else 0
    total_nodes = gmsh.model.mesh.getNodes()[0].size
    
    # Zapis raportu do pliku tekstowego.
    report_path = os.path.join(config["workspace_dir"], "mesh_report.txt")
    with open(report_path, "w") as f:
        f.write("--- RAPORT JAKOSCI SIATKI OBLICZENIOWEJ ---\n")
        f.write(f"Wezly ogolem: {total_nodes}\n")
        f.write(f"Elementy 2D: {total_2d}\n")
        f.write(f"  - Trojkaty (tlo): {num_tris}\n")
        f.write(f"  - Czworokaty (warstwa przyscienna): {num_quads}\n\n")
        f.write("--- METRYKA minSICN (1.0 = ideal, <= 0.0 = zdegenerowany) ---\n")
        f.write(f"Minimum minSICN: {sicn_min:.5f}\n")
        f.write(f"Srednia minSICN: {sicn_avg:.5f}\n\n")
        f.write("--- WNIOSKI ---\n")
        if sicn_min <= 0:
            f.write("[BLAD] Wykryto zdegenerowane elementy (minSICN <= 0). Symulacja moze byc niestabilna.\n")
        elif sicn_min < 0.1:
            f.write("[OSTRZEZENIE] Bardzo niska jakosc niektorych elementow (0 < minSICN < 0.1).\n")
        else:
            f.write("[OK] Brak zdegenerowanych komorek. Siatka prawidlowa dla solvera RANS.\n")

    print(f"   [GMSH] Raport jakosci siatki zapisano w: {report_path}")

    # Szybki podglad siatki w Matplotlib (tryb headless dla WSL).
    try:
        # Ekstrakcja wezlow.
        _, node_coords, _ = gmsh.model.mesh.getNodes()
        node_x = node_coords[0::3]
        node_y = node_coords[1::3]
        
        # Generowanie wykresu rozkladu wezlow.
        plt.figure(figsize=(12, 6))
        plt.plot(node_x, node_y, 'k.', markersize=0.3, alpha=0.3)
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.4, 0.4)
        plt.gca().set_aspect('equal')
        plt.title('Rozklad gestosci siatki wokol profilu (WSL Headless)')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        
        preview_path = os.path.join(config["workspace_dir"], "mesh_preview_wsl.png")
        plt.savefig(preview_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   [PODGLAD] Zapisano podglad siatki (bez GUI) jako: {preview_path}")
    except Exception as e:
        print(f"   [BLAD PODGLADU] Nie udalo sie wygenerowac wykresu siatki: {e}")

    # Zapis i zakonczenie.
    output_path = os.path.join(config["workspace_dir"], config["mesh_filename"])
    gmsh.write(output_path)
    gmsh.finalize()