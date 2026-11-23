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
            print(f"[WARNING] No history.csv for alpha = {alpha}. Skipping.")
            continue
            
        with open(history_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = [h.strip().replace('"', '') for h in reader.fieldnames]
            reader.fieldnames = headers
            
            cl_history = []
            cd_history = []
            
            for row in reader:
                cl_history.append(float(row.get("CL", 0)))
                cd_history.append(float(row.get("CD", 0)))
                
            if cl_history:
                window = min(100, len(cl_history))
                cl_avg = sum(cl_history[-window:]) / window
                cd_avg = sum(cd_history[-window:]) / window
                
                cl_list.append(cl_avg)
                cd_list.append(cd_avg)
                alpha_list.append(alpha)

    if not cl_list:
        print("[ERROR] No data available for plotting.")
        return

    # Writing the tabular data to a CSV file.
    csv_output = os.path.join(workspace_dir, "polar_data.csv")
    with open(csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Alpha", "Cd", "Cl"])
        for a, cd, cl in zip(alpha_list, cd_list, cl_list):
            writer.writerow([a, cd, cl])
    print(f"-> Tabular data saved as: {csv_output}")

    # Plot creation
    plt.figure(figsize=(8, 6))
    plt.plot(cd_list, cl_list, marker='o', linestyle='-', color='b', label=f'NACA {config["naca_code"]}')
    
    for i, txt in enumerate(alpha_list):
        plt.annotate(f"{txt}°", (cd_list[i], cl_list[i]), textcoords="offset points", xytext=(5,5), ha='left')

    plt.title("Drag polar of the airfoil (Cl vs. Cd)")
    plt.xlabel("Drag coefficient (Cd)")
    plt.ylabel("Lift coefficient (Cl)")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    exp_file = "naca0012_exp.csv"
    if os.path.exists(exp_file):
        exp_cl = []
        exp_cd = []
        with open(exp_file, 'r') as exp_csv:
            reader = csv.DictReader(exp_csv)
            for row in reader:
                exp_cl.append(float(row["CL"]))
                exp_cd.append(float(row["CD"]))
        
        plt.plot(exp_cd, exp_cl, 'r--', marker='s', label='Experiment (Re=3e6)')
    
    plt.legend()
    output_plot = os.path.join(workspace_dir, "polar_plot.png")
    plt.savefig(output_plot, dpi=300)
    print(f"-> Drag polar has been saved as: {output_plot}")