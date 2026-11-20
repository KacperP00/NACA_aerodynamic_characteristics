import os, sys, shutil
import numpy as np
import pandas as pd
import main_opt as m

it, iters = int(sys.argv[1]), int(sys.argv[2])
BASE, SRC = "reference/mesh_base_ffd.su2", "reference/restart_flow.dat"
CFG = {"mach": 0.15, "reynolds": 3000000, "aoa": 2.0, "turb_model": "SST",
       "trans_model": "NONE", "mesh_filename": "mesh.su2", "debug": False,
       "lift_weight": 0.0, "drag_weight": 1.0}

x = np.load(f"workspace_opt/optimization_history/x_{it:03d}.npy")
d = f"final_check/it{it}"
os.makedirs(d, exist_ok=True)
shutil.copy(BASE, f"{d}/mesh.su2")
m.prepare_config("SU2_DEF", x, CFG, d)
m.run_su2_module("SU2_DEF", d, cores=1)
shutil.move(f"{d}/mesh_out.su2", f"{d}/mesh.su2")
shutil.copy(SRC, f"{d}/solution_flow.dat")
m.prepare_config("SU2_CFD", x, dict(CFG, restart=True, iters=iters), d)
m.run_su2_module("SU2_CFD", d, cores=m.CORES)

h = pd.read_csv(f"{d}/history.csv")
h.columns = h.columns.str.strip().str.strip('"')
print("evaluation", it, "| iterations run:", len(h), "| rms[Rho] last: %.2f" % h['rms[Rho]'].iloc[-1])
for s in range(0, len(h) - 99, 500):
    w = h.iloc[s:s + 500]
    print("  iters %4d-%4d: CD %.6f  CL %.5f  L/D %.2f" % (s, s + 500, w['CD'].mean(), w['CL'].mean(), w['CL'].mean() / w['CD'].mean()))
cd, cl = h['CD'].iloc[-300:].mean(), h['CL'].iloc[-300:].mean()
print("last 300 iterations: CD %.6f  CL %.5f  L/D %.2f" % (cd, cl, cl / cd))