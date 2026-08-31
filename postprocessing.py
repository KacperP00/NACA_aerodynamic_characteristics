import os
import csv
import matplotlib.pyplot as plt

def plot_polar(config):
    workspace_dir = config["workspace_dir"]
    alphas = range(config["alpha_start"], config["alpha_end"] + 1, config["alpha_step"])
    
    cl_list = []
    cd_list = []
    alpha_list = []

    for alpha in alphas:
        run_dir_name = f"run_alpha_{alpha:02d}"
        history_file = os.path.join(workspace_dir, run_dir_name, "history.csv")
        
        if not os.path.exists(history_file):
            print(f"[OSTRZEŻENIE] Brak pliku history.csv dla alfa = {alpha}. Pomijanie.")
            continue
            
        # Ekstrakcja ostatnich wartości współczynników sił aerodynamicznych.
        with open(history_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            # Zabezpieczenie na wypadek białych znaków w nagłówkach z SU2.
            headers = [h.strip().replace('"', '') for h in reader.fieldnames]
            reader.fieldnames = headers
            
            last_row = None
            for row in reader:
                last_row = row
                
            if last_row:
                cl_list.append(float(last_row.get("CL", 0)))
                cd_list.append(float(last_row.get("CD", 0)))
                alpha_list.append(alpha)

    # Generowanie wykresu biegunowej i zapis do pliku (omijanie problemów z GUI na WSL).
    if not cl_list:
        print("[BŁĄD] Brak danych do narysowania wykresu.")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(cd_list, cl_list, marker='o', linestyle='-', color='b', label=f'NACA {config["naca_code"]}')
    
    # Adnotacje z wartościami kątów przy punktach.
    for i, txt in enumerate(alpha_list):
        plt.annotate(f"{txt}°", (cd_list[i], cl_list[i]), textcoords="offset points", xytext=(5,5), ha='left')

    plt.title("Biegunowa profilu (Cl od Cd)")
    plt.xlabel("Współczynnik oporu (Cd)")
    plt.ylabel("Współczynnik siły nośnej (Cl)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_plot = os.path.join(workspace_dir, "polar_plot.png")
    plt.savefig(output_plot, dpi=300)
    print(f"-> Wykres biegunowej został zapisany jako: {output_plot}")