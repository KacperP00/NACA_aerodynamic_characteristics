import os
import subprocess
import concurrent.futures

def run_single_case(alpha, config, template_content, mesh_path):
    # Formatowanie nazwy i tworzenie katalogu roboczego.
    run_dir_name = f"run_alpha_{alpha:02d}"
    run_dir_path = os.path.join(config["workspace_dir"], run_dir_name)
    
    if not os.path.exists(run_dir_path):
        os.makedirs(run_dir_path)

    # Przygotowanie pliku konfiguracyjnego dla zadanego kata natarcia.
    cfg_content = template_content.replace("%MACH%", str(config["mach"]))
    cfg_content = cfg_content.replace("%AOA%", str(alpha))
    cfg_content = cfg_content.replace("%REYNOLDS%", str(config["reynolds"]))
    cfg_content = cfg_content.replace("%INPUT_MESH%", mesh_path)
    cfg_content = cfg_content.replace("%RESTART%", "NO")
    
    cfg_path = os.path.join(run_dir_path, "config.cfg")
    with open(cfg_path, "w") as file:
        file.write(cfg_content)

    print(f"-> Uruchamiono SU2 dla {alpha}°...")
    
    # Wywolanie solvera z przekierowaniem strumieni do pliku logu.
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
        print(f"   [SUKCES] Zbiegnieto dla {alpha}°")
    except subprocess.CalledProcessError:
        print(f"   [BLAD] Problem w symulacji dla {alpha}°. Sprawdz su2.log.")

def run_su2_cases(config):
    # Wczytanie szablonu i konfiguracja sciezek globalnych.
    with open("template.cfg", "r") as file:
        template_content = file.read()
        
    workspace_dir = config["workspace_dir"]
    mesh_path = os.path.abspath(os.path.join(workspace_dir, config["mesh_filename"]))
    alphas = range(config["alpha_start"], config["alpha_end"] + 1, config["alpha_step"])
    
    # Ustalenie limitu jednoczesnych procesow (4 dla i7-4790, mozna zwiekszyc dla G15).
    max_workers = 2
    print(f"--- Uruchamianie obliczen rownoleglych ({max_workers} procesy) ---")
    
    # Wykorzystanie puli procesow do asynchronicznego uruchamiania przypadkow.
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_case, alpha, config, template_content, mesh_path)
            for alpha in alphas
        ]
        # Oczekiwanie na zakonczenie wszystkich zadan w puli.
        concurrent.futures.wait(futures)