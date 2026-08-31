import os
import numpy as np
from geometry import generate_naca4
from meshing import generate_mesh
from simulation import run_su2_cases
from postprocessing import plot_polar

def main():
    # Konfiguracja parametrów globalnych analizy.
    config = {
        "naca_code": "0012",
        "mach": 0.15,
        "reynolds": 3000000,
        "alpha_start": -5,
        "alpha_end": 15,
        "alpha_step": 1,
        "workspace_dir": "workspace",
        "mesh_filename": "mesh.su2"
    }

    # Przygotowanie struktury katalogów na pliki tymczasowe i wyniki.
    workspace_path = config["workspace_dir"]
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path)

    print(f"--- Rozpoczęcie analizy zautomatyzowanej NACA {config['naca_code']} ---")

    # Etap 1: Generacja współrzędnych profilu.
    x_coords, y_coords = generate_naca4(config["naca_code"], n_points=100)
    print(f"Wygenerowano geometrię profilu: {len(x_coords)} punktów.")

    # Etap 2: Generacja domeny i siatki obliczeniowej w Gmsh.
    generate_mesh(x_coords, y_coords, config)
    print("Siatka obliczeniowa została wygenerowana i zapisana.")

    # Etap 3: Uruchomienie pętli symulacji w SU2 (z warm-startem).
    run_su2_cases(config)
    print("Symulacje CFD zakończone.")

    # Etap 4: Zbieranie danych i rysowanie biegunowej.
    plot_polar(config)
    print("Biegunowa została wygenerowana.")

if __name__ == "__main__":
    main()