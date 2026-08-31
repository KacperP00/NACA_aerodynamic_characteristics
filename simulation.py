import os
import shutil
import subprocess

def run_su2_cases(config):
    # Wczytanie bazowego szablonu SU2.
    with open("template.cfg", "r") as file:
        template_content = file.read()

    workspace_dir = config["workspace_dir"]
    mesh_path = os.path.abspath(os.path.join(workspace_dir, config["mesh_filename"]))
    
    alphas = range(config["alpha_start"], config["alpha_end"] + 1, config["alpha_step"])
    prev_run_dir = None

    for alpha in alphas:
        # Formatowanie nazwy katalogu roboczego (np. run_alpha_02).
        run_dir_name = f"run_alpha_{alpha:02d}"
        run_dir_path = os.path.join(workspace_dir, run_dir_name)
        
        if not os.path.exists(run_dir_path):
            os.makedirs(run_dir_path)

        # Wymuszenie cold-startu dla każdego kąta.
        # Zapewnia to pełne przeliczenie zadanej liczby iteracji niezależnie od historii.
        restart_flag = "NO"

        # Podmiana markerów w konfiguracji.
        cfg_content = template_content.replace("%MACH%", str(config["mach"]))
        cfg_content = cfg_content.replace("%AOA%", str(alpha))
        cfg_content = cfg_content.replace("%REYNOLDS%", str(config["reynolds"]))
        cfg_content = cfg_content.replace("%INPUT_MESH%", mesh_path)
        cfg_content = cfg_content.replace("%RESTART%", restart_flag)

        # Zapis gotowego pliku konfiguracyjnego dla bieżącego kąta.
        cfg_path = os.path.join(run_dir_path, "config.cfg")
        with open(cfg_path, "w") as file:
            file.write(cfg_content)

        print(f"-> Uruchamianie SU2 dla kąta {alpha} stopni...")
        
        # Wywołanie procesu SU2_CFD.
        # Wymaga dodania ścieżki SU2 do zmiennej środowiskowej PATH w WSL.
        # Przekierowanie wyjścia do pliku logu dla celów diagnostycznych.
        log_file_path = os.path.join(run_dir_path, "su2.log")
        
        try:
            with open(log_file_path, "w") as log_file:
                subprocess.run(["SU2_CFD", "config.cfg"], cwd=run_dir_path, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            print(f"   Zakończono obliczenia dla alfa = {alpha}")
        except subprocess.CalledProcessError:
            print(f"   [BŁĄD] SU2 napotkał problem dla alfa = {alpha}. Przerwanie pętli.")
            print(f"   Sprawdź szczegóły w pliku: {log_file_path}")
            
            # Automatyczne wypisanie ostatnich 15 linijek logu do konsoli.
            print("   --- Ostatnie linie logu SU2 ---")
            with open(log_file_path, "r") as log_file:
                lines = log_file.readlines()
                for line in lines[-15:]:
                    print("   " + line.strip())
            break
        
        # Aktualizacja wskaźnika poprzedniego katalogu dla kolejnej iteracji.
        prev_run_dir = run_dir_path