import numpy as np, glob
for f in sorted(glob.glob("workspace_opt/optimization_history/x_*.npy")):
    x = np.load(f)
    print(f[-7:-4], "max|x| %.2e  norm %.2e" % (np.abs(x).max(), np.linalg.norm(x)))