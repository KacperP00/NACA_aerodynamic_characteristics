import os, re, sys, shutil
import numpy as np
import pandas as pd
import main_opt as m

BASE, SRC = "reference/mesh_base_ffd.su2", "reference/restart_flow.dat"
CD0, CL0 = 0.009821, 0.22409          # x = 0 under the same protocol (fd2/base)
it = int(sys.argv[1]) if len(sys.argv) > 1 else 15
npy = f"workspace_opt/optimization_history/x_{it:03d}.npy"
if os.path.exists(npy):
    x = np.load(npy)
elif it == 15:
    txt = open("workspace_opt/config_opt.cfg").read()
    x = np.array([float(v) for v in re.search(r"DV_VALUE=(.*)", txt).group(1).split(",")])
else:
    sys.exit(f"no saved design vector for iteration {it}")
print("design vector:", np.array2string(x, precision=5))

CFG = {"mach": 0.15, "reynolds": 3000000, "aoa": 2.0, "turb_model": "SST",
       "trans_model": "NONE", "mesh_filename": "mesh.su2", "debug": False,
       "lift_weight": 0.0}
d = f"ref_check/it{it}"
os.makedirs(d, exist_ok=True)
shutil.copy(BASE, f"{d}/mesh.su2")
m.prepare_config("SU2_DEF", x, CFG, d)
m.run_su2_module("SU2_DEF", d, cores=1)
shutil.move(f"{d}/mesh_out.su2", f"{d}/mesh.su2")
shutil.copy(SRC, f"{d}/solution_flow.dat")
m.prepare_config("SU2_CFD", x, dict(CFG, restart=True, iters=1000), d)
p = f"{d}/config_opt.cfg"
txt = open(p).read()
open(p, "w").write(txt.replace("CONV_RESIDUAL_MINVAL= -8", "CONV_RESIDUAL_MINVAL= -14"))
m.run_su2_module("SU2_CFD", d, cores=m.CORES)

h = pd.read_csv(f"{d}/history.csv")
h.columns = h.columns.str.strip().str.strip('"')
print("iterations run:", len(h), "| rms[Rho] first/last: %.2f / %.2f" % (h['rms[Rho]'].iloc[0], h['rms[Rho]'].iloc[-1]))
for s in range(400, len(h) - 99, 200):
    w = h.iloc[s:s + 200]
    print("  iters %4d-%4d: CD %.6f  CL %.5f" % (s, s + 200, w['CD'].mean(), w['CL'].mean()))
cd, cl = h['CD'].iloc[-300:].mean(), h['CL'].iloc[-300:].mean()
print("x = 0 (same protocol): CD %.6f  CL %.5f" % (CD0, CL0))
print("iteration %d design:   CD %.6f  CL %.5f   dCD = %+.2e (%+.1f%%)   dCL = %+.4f"
      % (it, cd, cl, cd - CD0, 100 * (cd - CD0) / CD0, cl - CL0))