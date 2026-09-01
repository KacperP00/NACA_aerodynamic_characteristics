import os
import time
import datetime
import numpy as np
import shutil
from geometry import generate_naca4
from meshing import generate_mesh
from simulation_parallel import run_su2_cases
from postprocessing import plot_polar

def main():
    start_time = time.time()
    # Konfiguracja parametrów globalnych analizy.
    config = {
        "naca_code": "0012",
        "mach": 0.15,
        "reynolds": 3000000,
        "alpha_start": -5,
        "alpha_end": 12,
        "alpha_step": 1,
        "workspace_dir": "workspace",
        "mesh_filename": "mesh.su2"
    }

    # Przygotowanie struktury katalogów na pliki tymczasowe i wyniki.
    workspace_path = config["workspace_dir"]
    # Czyszczenie starych wynikow symulacji.
    if os.path.exists(workspace_path):
        for item in os.listdir(workspace_path):
            if item.startswith("run_alpha_"):
                dir_to_remove = os.path.join(workspace_path, item)
                shutil.rmtree(dir_to_remove)
    else:
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

    elapsed_time = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
    print(f"--- Czas całkowity: {formatted_time} ---")

if __name__ == "__main__":
    main()