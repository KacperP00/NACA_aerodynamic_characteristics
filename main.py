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
    start_time = time.time() # Time initialization for performance measurement

    # Global parameters configuration
    # Definition of the AoA range and step

    config = {
        "naca_code": "0012", # which NACA airfoil to analyze
        "mach": 0.15,
        "reynolds": 3000000,
        "alpha_start": -16,
        "alpha_end": 16,
        "alpha_step": 2,
        "turb_model": "SST", # k-omega SST turbulence model
        "trans_model": "LM",  # Transition model
        "workspace_dir": "workspace",
        "mesh_filename": "mesh.su2"
    }

    # Preparation of the directory structure for temporary files and results.
    workspace_path = config["workspace_dir"]
    # Deleting old simulation results before new analysis
    if os.path.exists(workspace_path):
        for item in os.listdir(workspace_path):
            if item.startswith("run_alpha_"):
                dir_to_remove = os.path.join(workspace_path, item)
                shutil.rmtree(dir_to_remove)
    else:
        os.makedirs(workspace_path)

    print(f"--- Start of analysis for NACA {config['naca_code']} ---")

    # Step 1: Airfoil geometry generation
    x_coords, y_coords = generate_naca4(config["naca_code"], n_points=100)
    print(f"Airfoil geometry generated: {len(x_coords)} points.")

    # Step 2: Computational domain and mesh generation in Gmsh
    generate_mesh(x_coords, y_coords, config)
    print("Computational mesh has been generated and saved.")

    # Step 3: Running the simulation loop in SU2.
    run_su2_cases(config)
    print("CFD simulations completed.")

    # Step 4: Collecting data and plotting the polar curve.
    plot_polar(config)
    print("Polar curve has been generated.")

    elapsed_time = time.time() - start_time
    formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
    print(f"--- Total time: {formatted_time} ---")

if __name__ == "__main__":
    main()