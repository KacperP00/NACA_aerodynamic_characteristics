import os
import subprocess
import concurrent.futures

def run_single_case(alpha, config, template_content, mesh_path):
    # Formatting the run directory name and creating it if it doesn't exist.
    run_dir_name = f"run_alpha_{alpha:02d}"
    run_dir_path = os.path.join(config["workspace_dir"], run_dir_name)
    
    if not os.path.exists(run_dir_path):
        os.makedirs(run_dir_path)

    # Preparing the configuration file for the given angle of attack.
    cfg_content = template_content.replace("%MACH%", str(config["mach"]))
    cfg_content = cfg_content.replace("%AOA%", str(alpha))
    cfg_content = cfg_content.replace("%REYNOLDS%", str(config["reynolds"]))
    cfg_content = cfg_content.replace("%INPUT_MESH%", mesh_path)
    cfg_content = cfg_content.replace("%TURB_MODEL%", config["turb_model"])
    cfg_content = cfg_content.replace("%TRANS_MODEL%", config["trans_model"])
    cfg_content = cfg_content.replace("%RESTART%", "NO")
    
    cfg_path = os.path.join(run_dir_path, "config.cfg")
    with open(cfg_path, "w") as file:
        file.write(cfg_content)

    print(f"-> SU2 Run for {alpha}°...")
    
    # Solver enabled with pipe redirection to log file.
    log_file_path = os.path.join(run_dir_path, "su2.log")
    try:
        with open(log_file_path, "w") as log_file:
            subprocess.run(
                ["SU2_CFD", "config.cfg"], 
                cwd=run_dir_path, 
                check=True, 
                stdout=log_file, 
                stderr=subprocess.STDOUT
            )
        print(f"   [SUCCESS] Completed for {alpha}°")
    except subprocess.CalledProcessError:
        print(f"   [ERROR] Problem in simulation for {alpha}°. Check su2.log.")

def run_su2_cases(config):
    # Reading the template and setting up global paths.
    with open("template.cfg", "r") as file:
        template_content = file.read()
        
    workspace_dir = config["workspace_dir"]
    mesh_path = os.path.abspath(os.path.join(workspace_dir, config["mesh_filename"]))
    alphas = range(config["alpha_start"], config["alpha_end"] + 1, config["alpha_step"])
    
    # Number of parallel processes.
    max_workers = 6
    print(f"--- Running parallel computations: ({max_workers} processes) ---")
    
    # Use of ProcessPoolExecutor to run simulations in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_case, alpha, config, template_content, mesh_path)
            for alpha in alphas
        ]
        # Waiting for all tasks in the pool to complete.
        concurrent.futures.wait(futures)