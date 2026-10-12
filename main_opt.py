import os
import sys
import time
import datetime
import shutil
import subprocess
import numpy as np
import pandas as pd
import nlopt
from geometry import generate_naca4
from meshing import generate_mesh
from postprocessing_opt import generate_summary

CORES = 6
N_VARS = 16

def build_ffd_definition():
    ffd_def = "FFD_DEFINITION= (MAIN_BOX, -0.1, -0.2, 0.0, 1.1, -0.2, 0.0, 1.1, 0.2, 0.0, -0.1, 0.2, 0.0)\n"
    ffd_deg = "FFD_DEGREE= (9, 1, 0)\n"
    return ffd_def + ffd_deg

def generate_dv_strings(x):
    # Rozbicie na osobne wektory chroni przed przekroczeniem limitu znakow MAX_STRING_SIZE w parserze SU2.
    dv_kind, dv_marker, dv_param, dv_value = [], [], [], []
    var_idx = 0
    for i in range(1, 9):
        for j in [0, 1]:
            dv_kind.append("FFD_CONTROL_POINT_2D")
            dv_marker.append("( airfoil )")
            dv_param.append(f"(MAIN_BOX,{i},{j},0.0,1.0)")
            dv_value.append(f"{x[var_idx]:.6f}")
            var_idx += 1
            
    return "\n".join([
        "DV_KIND= " + ", ".join(dv_kind),
        "DV_MARKER= " + ", ".join(dv_marker),
        "DV_PARAM= " + ", ".join(dv_param),
        "DV_VALUE= " + ", ".join(dv_value)
    ]) + "\n"

def run_su2_module(module, cwd, cores=1, use_mpi=False):
    if use_mpi:
        cmd = f"export OMPI_MCA_osc=pt2pt && mpirun -n {cores} {module} config_opt.cfg > {module}.log 2>&1"
    else:
        cmd = f"{module} config_opt.cfg > {module}.log 2>&1"
        
    res = subprocess.run(cmd, shell=True, cwd=cwd, executable='/bin/bash')
    if res.returncode != 0:
        raise RuntimeError(f"Blad modulu {module}. Sprawdz log w {cwd}")

def prepare_config(module, x, config_params, work_dir):
    with open("template_opt.cfg", "r") as f:
        tpl = f.read()
        
    tpl = tpl.replace("%MACH%", str(config_params["mach"]))
    tpl = tpl.replace("%AOA%", str(config_params["aoa"]))
    tpl = tpl.replace("%REYNOLDS%", str(config_params["reynolds"]))
    tpl = tpl.replace("%TURB_MODEL%", config_params["turb_model"])
    tpl = tpl.replace("%TRANS_MODEL%", config_params["trans_model"])
    tpl = tpl.replace("%INPUT_MESH%", config_params["mesh_filename"])
    tpl = tpl.replace("%ITER%", "150" if config_params["debug"] else "1500")

    with open(os.path.join(work_dir, "config_opt.cfg"), "w") as f:
        f.write(tpl)
        f.write("\n% --- ZMIENNE PROJEKTOWE FFD ---\n")
        
        if module == "SU2_DEF":
            f.write("MATH_PROBLEM= DIRECT\n")
            f.write("DEFORM_MESH= YES\n")
            f.write("MARKER_DEFORM_MESH= ( airfoil )\n")
            f.write("DEFORM_LINEAR_SOLVER= FGMRES\n")
            f.write("DEFORM_LINEAR_SOLVER_ITER= 1000\n")
            f.write("DEFORM_NONLINEAR_ITER= 1\n")
            f.write("DEFORM_STIFFNESS_TYPE= WALL_DISTANCE\n")
            f.write(build_ffd_definition())
            f.write(generate_dv_strings(x))
            
        elif module == "SU2_CFD":
            f.write("MATH_PROBLEM= DIRECT\n")
            f.write("DEFORM_MESH= NO\n")
            
        elif module == "SU2_CFD_AD":
            f.write("MATH_PROBLEM= DISCRETE_ADJOINT\n")
            f.write("OBJECTIVE_FUNCTION= DRAG\n")
            f.write("DEFORM_MESH= NO\n")
            f.write("SOLUTION_FILENAME= solution_flow.dat\n")
            
        elif module == "SU2_DOT_AD":
            f.write("MATH_PROBLEM= DISCRETE_ADJOINT\n")
            f.write("OBJECTIVE_FUNCTION= DRAG\n")
            f.write("DEFORM_MESH= YES\n")
            f.write("MARKER_DEFORM_MESH= ( airfoil )\n")
            f.write(build_ffd_definition())
            f.write(generate_dv_strings(x))
            f.write("SOLUTION_FILENAME= solution_flow.dat\n")
            f.write("SOLUTION_ADJ_FILENAME= solution_adj.dat\n")

def main():
    config = {
        "naca_code": "0012",
        "mach": 0.6,
        "reynolds": 3000000,
        "aoa": 2.0,
        "turb_model": "SST",
        "trans_model": "LM",
        "workspace_dir": "workspace_opt",
        "mesh_filename": "mesh.su2",
        "debug": True
    }

    wk_dir = config["workspace_dir"]
    hist_dir = os.path.join(wk_dir, "optimization_history")
    
    if os.path.exists(wk_dir):
        shutil.rmtree(wk_dir)
    os.makedirs(hist_dir)

    x_c, y_c = generate_naca4(config["naca_code"], n_points=100)
    generate_mesh(x_c, y_c, config)
    shutil.copy(os.path.join(wk_dir, config["mesh_filename"]), os.path.join(wk_dir, "mesh_base.su2"))

    opt_history = {"iter": [], "cd": [], "cl": [], "obj": []}
    iter_count = 0

    def objective(x, grad):
        nonlocal iter_count
        iter_count += 1
        print(f"\n--- Iteracja {iter_count} ---")
        
        shutil.copy(os.path.join(wk_dir, "mesh_base.su2"), os.path.join(wk_dir, config["mesh_filename"]))
        
        prepare_config("SU2_DEF", x, config, wk_dir)
        run_su2_module("SU2_DEF", wk_dir, cores=1, use_mpi=False)
        shutil.move(os.path.join(wk_dir, "mesh_out.su2"), os.path.join(wk_dir, config["mesh_filename"]))
        
        prepare_config("SU2_CFD", x, config, wk_dir)
        run_su2_module("SU2_CFD", wk_dir, cores=CORES, use_mpi=True)
        
        hist = pd.read_csv(os.path.join(wk_dir, "history.csv"))
        hist.columns = hist.columns.str.strip()
        cd = hist['"CD"'].iloc[-1]
        cl = hist['"CL"'].iloc[-1]
        
        penalty = 0.0
        if cl < 0.25:
            penalty = 10.0 * (0.25 - cl)**2
        obj_val = cd + penalty

        shutil.copy(os.path.join(wk_dir, "surface_flow.csv"), os.path.join(hist_dir, f"surf_{iter_count:03d}.csv"))
        
        opt_history["iter"].append(iter_count)
        opt_history["cd"].append(cd)
        opt_history["cl"].append(cl)
        opt_history["obj"].append(obj_val)

        if grad.size > 0:
            shutil.copy(os.path.join(wk_dir, "restart_flow.dat"), os.path.join(wk_dir, "solution_flow.dat"))
            
            prepare_config("SU2_CFD_AD", x, config, wk_dir)
            run_su2_module("SU2_CFD_AD", wk_dir, cores=1, use_mpi=False)
            
            shutil.copy(os.path.join(wk_dir, "restart_adj.dat"), os.path.join(wk_dir, "solution_adj.dat"))
            
            prepare_config("SU2_DOT_AD", x, config, wk_dir)
            run_su2_module("SU2_DOT_AD", wk_dir, cores=1, use_mpi=False)
            
            grad_df = pd.read_csv(os.path.join(wk_dir, "of_grad.csv"))
            grad[:] = grad_df['Gradient'].values[:N_VARS]
            
        print(f"Wynik -> CD: {cd:.5f} | CL: {cl:.5f} | Cel: {obj_val:.5f}")
        return obj_val

    opt = nlopt.opt(nlopt.LD_MMA, N_VARS)
    opt.set_lower_bounds(np.full(N_VARS, -0.05))
    opt.set_upper_bounds(np.full(N_VARS, 0.05))
    opt.set_min_objective(objective)
    opt.set_maxeval(50)
    opt.set_ftol_rel(1e-4)
    
    start_time = time.time()
    try:
        x_opt = opt.optimize(np.zeros(N_VARS))
        print(f"\n--- Zbieznosc! Min opor CD: {opt.last_optimum_value():.5f} ---")
    except Exception as e:
        print(f"\nProces przerwany: {e}")
        
    pd.DataFrame(opt_history).to_csv(os.path.join(wk_dir, "opt_metrics.csv"), index=False)
    
    generate_summary(wk_dir, hist_dir)
    print(f"Czas calkowity: {datetime.timedelta(seconds=int(time.time() - start_time))}")

if __name__ == "__main__":
    main()