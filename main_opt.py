import argparse
import datetime
import glob
import math
import os
import re
import shutil
import subprocess
import time
import nlopt
import numpy as np
import pandas as pd

from geometry import generate_naca4
from meshing import generate_mesh
from postprocessing_opt import generate_summary

N_VARS = 16
FFD_DEGREE_I = 9
THICK_STATIONS = np.linspace(0.10, 0.90, 9)


def build_ffd_definition():
    # Definicja klatki FFD uzywana przy generowaniu siatki bazowej
    ffd_def = (
        "FFD_DEFINITION= (MAIN_BOX, -0.1, -0.2, 0.0, 1.1, -0.2, 0.0, "
        "1.1, 0.2, 0.0, -0.1, 0.2, 0.0)\n"
    )
    ffd_deg = f"FFD_DEGREE= ({FFD_DEGREE_I}, 1, 0)\n"
    return ffd_def + ffd_deg


def build_dv_ffd_setting():
    # Dyrektywa osadzajaca klatke FFD w pliku siatki .su2
    return (
        "DV_KIND= FFD_SETTING\n"
        "DV_MARKER= ( airfoil )\n"
        "DV_PARAM= ( 1.0 )\n"
        "DV_VALUE= 0.0\n"
    )


def build_dv_definition():
    # Skladnia wektorow zmiennych FFD zgodna z parserem SU2 8.5.0
    dv_kind = ["FFD_CONTROL_POINT_2D"] * N_VARS
    dv_param = []
    for i in range(1, FFD_DEGREE_I):
        for j in (0, 1):
            dv_param.append(f"( MAIN_BOX, {i}, {j}, 0.0, 1.0 )")
    return (
        "DV_KIND= " + ", ".join(dv_kind) + "\n"
        "DV_MARKER= ( airfoil )\n"
        "DV_PARAM= " + " ; ".join(dv_param) + "\n"
    )


def build_dv_value(x):
    # Wektor wartosci zmiennych projektowych
    return "DV_VALUE= " + ", ".join(f"{v:.9f}" for v in x) + "\n"


def read_ffd_bounds(mesh_path):
    # Odczyt granic klatki FFD z pliku siatki .su2
    default_bounds = (-0.1, 1.1, -0.2, 0.2)
    if not os.path.exists(mesh_path):
        return default_bounds
    with open(mesh_path, "r") as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("FFD_CORNER_POINTS="):
            pts = []
            for k in range(1, 5):
                if idx + k < len(lines):
                    parts = lines[idx + k].split()
                    if len(parts) >= 2:
                        pts.append((float(parts[0]), float(parts[1])))
            if len(pts) == 4:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return (min(xs), max(xs), min(ys), max(ys))
    return default_bounds


def build_thickness_matrix(stations, ffd_bounds):
    # Macierz Bernsteina A: delta_t(s)/t0(s) = A @ x dla ruchu pionowego FFD
    x_min, x_max, y_min, y_max = ffd_bounds
    dy_box = y_max - y_min
    dx_box = x_max - x_min
    mat = np.zeros((len(stations), N_VARS))
    for row, xc in enumerate(stations):
        u = np.clip((xc - x_min) / dx_box, 0.0, 1.0)
        for i in range(1, FFD_DEGREE_I):
            b_i = math.comb(FFD_DEGREE_I, i) * (u ** i) * ((1.0 - u) ** (FFD_DEGREE_I - i))
            idx_lo = 2 * (i - 1)
            idx_hi = idx_lo + 1
            mat[row, idx_lo] = -b_i / dy_box
            mat[row, idx_hi] = b_i / dy_box
    return mat


def measure_surface_geometry(csv_path, stations=THICK_STATIONS):
    # Pomiar grubosci i linii szkieletowej profilu ze wspolrzednych powierzchni
    s = pd.read_csv(csv_path)
    s.columns = [c.strip().strip('"') for c in s.columns]
    x = s["x"].values if "x" in s.columns else s["PointX"].values
    y = s["y"].values if "y" in s.columns else s["PointY"].values

    i_le = int(np.argmin(x))
    i_te = int(np.argmax(x))
    chord_y = np.interp(x, [x[i_le], x[i_te]], [y[i_le], y[i_te]])
    up_mask = y >= chord_y
    lo_mask = y <= chord_y

    ux, uy = x[up_mask], y[up_mask]
    lx, ly = x[lo_mask], y[lo_mask]
    u_ord = np.argsort(ux)
    l_ord = np.argsort(lx)

    yu = np.interp(stations, ux[u_ord], uy[u_ord])
    yl = np.interp(stations, lx[l_ord], ly[l_ord])
    thick = yu - yl
    camber = 0.5 * (yu + yl)
    return thick, camber, float(y[i_te])


def clean_adjoint_files(cwd):
    # Usuwa pozostalosci poprzednich przebiegow adjointowych
    patterns = ["restart_adj*.dat", "solution_adj*.dat", "of_grad", "of_grad.csv"]
    for pat in patterns:
        for p in glob.glob(os.path.join(cwd, pat)):
            os.remove(p)


def parse_of_grad(cwd, n_vars=N_VARS):
    # Odczyt gradientu z pliku of_grad generowanego przez SU2_DOT_AD
    grad_path = os.path.join(cwd, "of_grad")
    if not os.path.exists(grad_path):
        raise FileNotFoundError(f"Brak pliku {grad_path}")
    vals = []
    with open(grad_path, "r") as f:
        for line in f:
            tok = line.strip()
            if not tok:
                continue
            try:
                vals.append(float(tok))
            except ValueError:
                continue
    if len(vals) < n_vars:
        raise ValueError(f"Niekompletny gradient w {grad_path}: {len(vals)}/{n_vars}")
    arr = np.array(vals[:n_vars], dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Nieprawidlowe wartosci (NaN/Inf) w {grad_path}")
    return arr


def read_cfd_metrics(history_path, avg_window=100):
    # Odczyt usrednionych wspolczynnikow i koncowych residuow z history.csv
    df = pd.read_csv(history_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    n = len(df)
    w = min(avg_window, max(1, n // 5))
    tail = df.iloc[-w:]
    metrics = {
        "iters_run": n,
        "cd": float(tail["CD"].mean()),
        "cl": float(tail["CL"].mean()),
        "cmz": float(tail["CMz"].mean()) if "CMz" in df.columns else 0.0,
        "cd_std": float(tail["CD"].std()) if w > 1 else 0.0,
        "cl_std": float(tail["CL"].std()) if w > 1 else 0.0,
        "rms_rho_first": float(df["rms[Rho]"].iloc[0]),
        "rms_rho_last": float(df["rms[Rho]"].iloc[-1]),
        "rms_w_last": float(df["rms[w]"].iloc[-1]) if "rms[w]" in df.columns else np.nan,
    }
    return metrics


def read_adj_residuals(history_path):
    # Odczyt koncowych residuow adjointu z history.csv
    if not os.path.exists(history_path):
        return np.nan, np.nan
    df = pd.read_csv(history_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    a_rho = float(df["rms[A_Rho]"].iloc[-1]) if "rms[A_Rho]" in df.columns else np.nan
    a_k = float(df["rms[A_k]"].iloc[-1]) if "rms[A_k]" in df.columns else np.nan
    return a_rho, a_k


def assert_ffd_applied(cwd, x, log_name="SU2_DEF.log"):
    # Sprawdza, czy SU2_DEF faktycznie wykonal deformacje FFD dla niezerowego x
    if np.linalg.norm(x) <= 1e-12:
        return
    log_path = os.path.join(cwd, log_name)
    txt = open(log_path).read() if os.path.exists(log_path) else ""
    if "Update cartesian coord" not in txt:
        raise RuntimeError(f"SU2_DEF nie zastosowal deformacji FFD (sprawdz {log_path})")


def run_su2_module(module, cwd, cores=1, log_name=None):
    # Uruchomienie binarki SU2 przez OpenMPI z bezpiecznym srodowiskiem
    log_file = log_name or f"{module}.log"
    print(f"   -> {module} (MPI: {cores})... ", end="", flush=True)
    t0 = time.time()

    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["OMPI_MCA_osc"] = "pt2pt"

    cmd = f"ulimit -s unlimited && mpirun -n {cores} {module} config_opt.cfg > {log_file} 2>&1"
    res = subprocess.run(cmd, shell=True, cwd=cwd, env=env, executable="/bin/bash")
    dt = time.time() - t0

    log_path = os.path.join(cwd, log_file)
    log_text = ""
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log_text = f.read()

    if res.returncode != 0:
        # Obejscie bledu destruktora C++ (Signal 6) wystepujacego po zapisie plikow
        if "Finalize Solver" in log_text or "Exit Success" in log_text:
            print(f"({dt:.1f}s) [zignorowano blad zamykania]")
            return
        tail = "\n".join(log_text.splitlines()[-25:])
        raise RuntimeError(f"Blad modulu {module} (rc={res.returncode}). Ostatnie logi:\n{tail}")

    print(f"({dt:.1f}s)")


def prepare_config(module, x, cfg, work_dir):
    # Budowa deterministycznego config_opt.cfg z oczyszczonego szablonu
    with open("template_opt.cfg", "r") as f:
        tpl_lines = f.readlines()

    strip_prefixes = (
        "MATH_PROBLEM", "DEFORM_", "MARKER_DEFORM_MESH", "FFD_", "DV_", "DEFINITION_DV",
        "OBJECTIVE_FUNCTION", "OBJECTIVE_WEIGHT", "SOLUTION_FILENAME", "RESTART_FILENAME",
        "SOLUTION_ADJ_FILENAME", "RESTART_SOL", "INNER_ITER", "CONV_RESIDUAL_MINVAL",
        "CFL_NUMBER", "CFL_ADAPT", "SLOPE_LIMITER_FLOW", "FROZEN_LIMITER_DISC",
        "FREESTREAM_TURBULENCEINTENSITY", "FREESTREAM_TURB2LAMVISCRATIO",
    )
    clean_lines = [l for l in tpl_lines if not l.strip().startswith(strip_prefixes)]
    tpl = "".join(clean_lines)

    tpl = tpl.replace("%MACH%", str(cfg["mach"]))
    tpl = tpl.replace("%AOA%", str(cfg["aoa"]))
    tpl = tpl.replace("%REYNOLDS%", str(cfg["reynolds"]))
    tpl = tpl.replace("%TURB_MODEL%", cfg["turb_model"])
    tpl = tpl.replace("%TRANS_MODEL%", cfg["trans_model"])
    tpl = tpl.replace("%INPUT_MESH%", cfg["mesh_filename"])

    lines = [
        tpl.rstrip(),
        "",
        "% --- PARAMETRY NUMERYCZNE I TURBULENCJI ---",
        "FREESTREAM_TURBULENCEINTENSITY= 0.001",
        "FREESTREAM_TURB2LAMVISCRATIO= 10.0",
        "SLOPE_LIMITER_FLOW= NONE",
        "SOLUTION_FILENAME= solution_flow.dat",
        "RESTART_FILENAME= restart_flow.dat",
    ]

    if module == "SU2_DEF_FFD":
        lines += [
            "MATH_PROBLEM= DIRECT",
            "DEFORM_MESH= NO",
            "RESTART_SOL= NO",
            "INNER_ITER= 1",
            "CONV_RESIDUAL_MINVAL= -8",
            "CFL_NUMBER= 10.0",
            "CFL_ADAPT= YES",
            "OBJECTIVE_FUNCTION= DRAG",
            build_ffd_definition().rstrip(),
            build_dv_ffd_setting().rstrip(),
        ]

    elif module == "SU2_DEF":
        lines += [
            "MATH_PROBLEM= DIRECT",
            "INNER_ITER= 1",
            "CONV_RESIDUAL_MINVAL= -8",
            "CFL_NUMBER= 10.0",
            "CFL_ADAPT= YES",
            "OBJECTIVE_FUNCTION= DRAG",
            "DEFORM_LINEAR_SOLVER= FGMRES",
            "DEFORM_LINEAR_SOLVER_ITER= 1000",
            "DEFORM_NONLINEAR_ITER= 1",
            "DEFORM_STIFFNESS_TYPE= WALL_DISTANCE",
            build_dv_definition().rstrip(),
            build_dv_value(x).rstrip(),
        ]

    elif module == "SU2_CFD":
        restart_flag = "YES" if cfg.get("restart", False) else "NO"
        lines += [
            "MATH_PROBLEM= DIRECT",
            "DEFORM_MESH= NO",
            f"RESTART_SOL= {restart_flag}",
            f"INNER_ITER= {cfg['iters']}",
            f"CONV_RESIDUAL_MINVAL= {cfg.get('conv_minval', -14)}",
            "CFL_NUMBER= 10.0",
            "CFL_ADAPT= YES",
            "CFL_ADAPT_PARAM= ( 0.1, 2.0, 10.0, 100.0 )",
            "OBJECTIVE_FUNCTION= DRAG",
        ]

    elif module == "SU2_CFD_AD":
        obj_fn = cfg["adj_objective"]
        lines += [
            "MATH_PROBLEM= DISCRETE_ADJOINT",
            "DEFORM_MESH= NO",
            "RESTART_SOL= NO",
            f"INNER_ITER= {cfg['adj_iters']}",
            "CONV_RESIDUAL_MINVAL= -12",
            f"CFL_NUMBER= {cfg.get('adj_cfl', 50.0)}",
            "CFL_ADAPT= NO",
            "FROZEN_LIMITER_DISC= YES",
            f"OBJECTIVE_FUNCTION= {obj_fn}",
        ]
        if "adj_weights" in cfg and cfg["adj_weights"] is not None:
            w_str = ", ".join(f"{w:.8f}" for w in cfg["adj_weights"])
            lines.append(f"OBJECTIVE_WEIGHT= {w_str}")

    elif module == "SU2_DOT_AD":
        obj_fn = cfg["adj_objective"]
        lines += [
            "MATH_PROBLEM= DISCRETE_ADJOINT",
            "DEFORM_MESH= YES",
            "RESTART_SOL= YES",
            "INNER_ITER= 1",
            "CONV_RESIDUAL_MINVAL= -8",
            "CFL_NUMBER= 10.0",
            "CFL_ADAPT= NO",
            "FROZEN_LIMITER_DISC= YES",
            "MARKER_DEFORM_MESH= ( airfoil )",
            "DEFORM_LINEAR_SOLVER= FGMRES",
            "DEFORM_LINEAR_SOLVER_ITER= 1000",
            "DEFORM_NONLINEAR_ITER= 1",
            "DEFORM_STIFFNESS_TYPE= WALL_DISTANCE",
            f"OBJECTIVE_FUNCTION= {obj_fn}",
            build_dv_definition().rstrip(),
            build_dv_value(x).rstrip(),
        ]
        if "adj_weights" in cfg and cfg["adj_weights"] is not None:
            w_str = ", ".join(f"{w:.8f}" for w in cfg["adj_weights"])
            lines.append(f"OBJECTIVE_WEIGHT= {w_str}")

    else:
        raise ValueError(f"Nieznany modul SU2: {module}")

    with open(os.path.join(work_dir, "config_opt.cfg"), "w") as f:
        f.write("\n".join(lines) + "\n")


def ensure_base_mesh(cfg, wk_dir, remesh=False):
    # Przygotowanie siatki bazowej z osadzona klatka FFD
    mesh_base = os.path.join(wk_dir, "mesh_base.su2")
    ref_mesh = "reference/mesh_base_ffd.su2"

    if not remesh and os.path.exists(ref_mesh):
        shutil.copy(ref_mesh, mesh_base)
        print(f"Uzyto zweryfikowanej siatki bazowej: {ref_mesh}")
    else:
        x_c, y_c = generate_naca4(cfg["naca_code"], n_points=100)
        generate_mesh(x_c, y_c, cfg)
        prepare_config("SU2_DEF_FFD", np.zeros(N_VARS), cfg, wk_dir)
        run_su2_module("SU2_DEF", wk_dir, cores=1, log_name="SU2_DEF_FFD.log")
        shutil.move(os.path.join(wk_dir, "mesh_out.su2"), mesh_base)

    with open(mesh_base, "r") as f:
        if "FFD_NBOX" not in f.read():
            raise RuntimeError("Siatka mesh_base.su2 nie zawiera definicji FFD_NBOX.")
    return mesh_base


def run_adjoint_pair(x, cfg, wk_dir, objective_str, weights, cores, tag):
    # Oblicza pojedynczy gradient (SU2_CFD_AD + SU2_DOT_AD) dla zadanego celu
    clean_adjoint_files(wk_dir)
    ad_cfg = dict(cfg, adj_objective=objective_str, adj_weights=weights)

    prepare_config("SU2_CFD_AD", x, ad_cfg, wk_dir)
    run_su2_module("SU2_CFD_AD", wk_dir, cores=cores, log_name=f"SU2_CFD_AD_{tag}.log")
    a_rho, a_k = read_adj_residuals(os.path.join(wk_dir, "history.csv"))

    adj_restarts = sorted(glob.glob(os.path.join(wk_dir, "restart_adj*.dat")))
    if not adj_restarts:
        raise RuntimeError(f"Brak pliku restart_adj*.dat po SU2_CFD_AD ({tag})")
    for src in adj_restarts:
        dst_name = os.path.basename(src).replace("restart_adj", "solution_adj")
        shutil.copy(src, os.path.join(wk_dir, dst_name))

    # SU2_DOT_AD rzutuje wrazliwosc na klatke FFD zapisana w siatce bazowej
    dot_cfg = dict(ad_cfg, mesh_filename="mesh_base.su2")
    prepare_config("SU2_DOT_AD", x, dot_cfg, wk_dir)
    run_su2_module("SU2_DOT_AD", wk_dir, cores=1, log_name=f"SU2_DOT_AD_{tag}.log")
    grad = parse_of_grad(wk_dir, N_VARS)
    return grad, a_rho, a_k


class AirfoilEvaluator:
    # Cache ewaluacji dla wspoldzielenia CFD miedzy funkcja celu i ograniczeniami NLopt
    def __init__(self, cfg, wk_dir, hist_dir, cores, cmz_max, min_thick_ratio, ffd_bounds):
        self.cfg = cfg
        self.wk_dir = wk_dir
        self.hist_dir = hist_dir
        self.cores = cores
        self.cmz_max = cmz_max
        self.min_thick_ratio = min_thick_ratio
        self.thick_mat = build_thickness_matrix(THICK_STATIONS, ffd_bounds)

        self.iter_count = 0
        self.last_x = None
        self.state = None
        self.base_thick = None
        self.last_good_restart = os.path.join(wk_dir, "last_good_flow.dat")
        self.history = []

        if os.path.exists("reference/restart_flow.dat"):
            shutil.copy("reference/restart_flow.dat", self.last_good_restart)

    def _ensure_primal(self, x):
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return

        self.iter_count += 1
        print(f"\n--- Ewaluacja {self.iter_count} ---")
        mesh_path = os.path.join(self.wk_dir, self.cfg["mesh_filename"])
        shutil.copy(os.path.join(self.wk_dir, "mesh_base.su2"), mesh_path)

        # Deformacja siatki na 1 rdzeniu
        prepare_config("SU2_DEF", x, self.cfg, self.wk_dir)
        run_su2_module("SU2_DEF", self.wk_dir, cores=1)
        assert_ffd_applied(self.wk_dir, x)
        shutil.move(os.path.join(self.wk_dir, "mesh_out.su2"), mesh_path)

        # Warm-start z poprzedniej zbieglej iteracji (lub ze stanu referencyjnego)
        use_restart = os.path.exists(self.last_good_restart)
        if use_restart:
            shutil.copy(self.last_good_restart, os.path.join(self.wk_dir, "solution_flow.dat"))

        cfd_cfg = dict(self.cfg, restart=use_restart, iters=self.cfg["cfd_iters"])
        prepare_config("SU2_CFD", x, cfd_cfg, self.wk_dir)
        run_su2_module("SU2_CFD", self.wk_dir, cores=self.cores)

        m = read_cfd_metrics(os.path.join(self.wk_dir, "history.csv"))
        shutil.copy(
            os.path.join(self.wk_dir, "restart_flow.dat"),
            os.path.join(self.wk_dir, "solution_flow.dat"),
        )
        if np.isfinite(m["rms_rho_last"]) and m["rms_rho_last"] < -5.5:
            shutil.copy(os.path.join(self.wk_dir, "restart_flow.dat"), self.last_good_restart)

        surf_src = os.path.join(self.wk_dir, "surface_flow.csv")
        surf_dst = os.path.join(self.hist_dir, f"surf_{self.iter_count:03d}.csv")
        shutil.copy(surf_src, surf_dst)
        np.save(os.path.join(self.hist_dir, f"x_{self.iter_count:03d}.npy"), x)

        thick, camber, te_y = measure_surface_geometry(surf_src)
        if self.base_thick is None:
            self.base_thick = thick.copy()
        actual_min_ratio = float(np.min(thick / self.base_thick))
        max_camber = float(np.max(np.abs(camber)))

        ld = m["cl"] / m["cd"]
        obj = m["cd"] / m["cl"]
        feasible = (abs(m["cmz"]) <= self.cmz_max + 1e-4) and (
            actual_min_ratio >= self.min_thick_ratio - 1e-3
        )

        self.last_x = x.copy()
        self.state = {
            "iter": self.iter_count,
            "cd": m["cd"],
            "cl": m["cl"],
            "cmz": m["cmz"],
            "ld": ld,
            "obj": obj,
            "cd_std": m["cd_std"],
            "rms_rho": m["rms_rho_last"],
            "rms_w": m["rms_w_last"],
            "min_thick_ratio": actual_min_ratio,
            "max_camber": max_camber,
            "te_y": te_y,
            "feasible": int(feasible),
            "rms_a_rho_obj": np.nan,
            "rms_a_k_obj": np.nan,
            "rms_a_rho_cmz": np.nan,
            "grad_obj": None,
            "grad_cmz": None,
        }
        self.history.append(self.state)
        self._save_metrics()

        print(
            f"   Wynik -> CD: {m['cd']:.6f} (std {m['cd_std']:.1e}) | CL: {m['cl']:.5f} | "
            f"L/D: {ld:.2f} | CMz: {m['cmz']:+.5f} | t/t0 min: {actual_min_ratio:.3f} | "
            f"rms[Rho]: {m['rms_rho_last']:.2f}"
        )

    def _save_metrics(self):
        # Zapis historii metryk na biezaco po kazdej ewaluacji
        keys = [
            "iter", "cd", "cl", "cmz", "ld", "obj", "cd_std", "rms_rho", "rms_w",
            "min_thick_ratio", "max_camber", "te_y", "feasible",
            "rms_a_rho_obj", "rms_a_k_obj", "rms_a_rho_cmz",
        ]
        rows = [{k: row[k] for k in keys} for row in self.history]
        pd.DataFrame(rows).to_csv(os.path.join(self.wk_dir, "opt_metrics.csv"), index=False)

    def objective(self, x, grad):
        self._ensure_primal(x)
        if grad.size > 0:
            if self.state["grad_obj"] is None:
                cd, cl = self.state["cd"], self.state["cl"]
                weights = [1.0 / cl, -cd / (cl ** 2)]
                g_obj, a_rho, a_k = run_adjoint_pair(
                    x, self.cfg, self.wk_dir, "DRAG, LIFT", weights, self.cores, "obj"
                )
                self.state["grad_obj"] = g_obj
                self.state["rms_a_rho_obj"] = a_rho
                self.state["rms_a_k_obj"] = a_k
                self._save_metrics()
                print(f"   Adjoint(CD/CL) -> rms[A_Rho]: {a_rho:.2f}, rms[A_k]: {a_k:.2f}")
            grad[:] = self.state["grad_obj"]
        return self.state["obj"]

    def _ensure_cmz_grad(self, x):
        if self.state["grad_cmz"] is None:
            g_cmz, a_rho, _ = run_adjoint_pair(
                x, self.cfg, self.wk_dir, "MOMENT_Z", None, self.cores, "cmz"
            )
            self.state["grad_cmz"] = g_cmz
            self.state["rms_a_rho_cmz"] = a_rho
            self._save_metrics()
            print(f"   Adjoint(CMz)   -> rms[A_Rho]: {a_rho:.2f}")

    def cmz_upper(self, x, grad):
        # Ograniczenie CMz(x) - cmz_max <= 0
        self._ensure_primal(x)
        if grad.size > 0:
            self._ensure_cmz_grad(x)
            grad[:] = self.state["grad_cmz"]
        return self.state["cmz"] - self.cmz_max

    def cmz_lower(self, x, grad):
        # Ograniczenie -CMz(x) - cmz_max <= 0
        self._ensure_primal(x)
        if grad.size > 0:
            self._ensure_cmz_grad(x)
            grad[:] = -self.state["grad_cmz"]
        return -self.state["cmz"] - self.cmz_max

    def thickness_constraints(self, result, x, grad):
        # Liniowe ograniczenia FFD: (min_thick_ratio - 1) - (A @ x) <= 0
        result[:] = (self.min_thick_ratio - 1.0) - (self.thick_mat @ x)
        if grad.size > 0:
            grad[:, :] = -self.thick_mat


def run_fd_check(cfg, args):
    # Weryfikacja gradientow adjointowych (CD/CL i CMz) roznicami skonczonymi centralnymi
    wk_dir = args.work_dir
    if os.path.exists(wk_dir):
        shutil.rmtree(wk_dir)
    os.makedirs(wk_dir)

    mesh_base = ensure_base_mesh(cfg, wk_dir, remesh=args.remesh)
    ffd_bounds = read_ffd_bounds(mesh_base)
    thick_mat = build_thickness_matrix(THICK_STATIONS, ffd_bounds)

    var_indices = [int(v.strip()) for v in args.vars.split(",") if v.strip()]
    step = args.fd_step
    x0 = np.zeros(N_VARS)

    def eval_single(x_vec, tag):
        sub = os.path.join(wk_dir, tag)
        os.makedirs(sub, exist_ok=True)
        shutil.copy(mesh_base, os.path.join(sub, "mesh_base.su2"))
        shutil.copy(mesh_base, os.path.join(sub, cfg["mesh_filename"]))
        prepare_config("SU2_DEF", x_vec, cfg, sub)
        run_su2_module("SU2_DEF", sub, cores=1)
        assert_ffd_applied(sub, x_vec if "x_vec" in locals() else x)
        shutil.move(os.path.join(sub, "mesh_out.su2"), os.path.join(sub, cfg["mesh_filename"]))

        use_rst = os.path.exists("reference/restart_flow.dat")
        if use_rst:
            shutil.copy("reference/restart_flow.dat", os.path.join(sub, "solution_flow.dat"))
        prepare_config("SU2_CFD", x_vec, dict(cfg, restart=use_rst, iters=cfg["cfd_iters"]), sub)
        run_su2_module("SU2_CFD", sub, cores=args.cores)
        m = read_cfd_metrics(os.path.join(sub, "history.csv"))
        thick, _, _ = measure_surface_geometry(os.path.join(sub, "surface_flow.csv"))
        return m, thick, sub

    print(f"\n=== FD-CHECK (step={step:.1e}, cfd_iters={cfg['cfd_iters']}, adj_iters={cfg['adj_iters']}) ===")
    m0, t0, sub0 = eval_single(x0, "base")
    shutil.copy(os.path.join(sub0, "restart_flow.dat"), os.path.join(sub0, "solution_flow.dat"))
    weights0 = [1.0 / m0["cl"], -m0["cd"] / (m0["cl"] ** 2)]
    g_obj, a_rho_o, a_k_o = run_adjoint_pair(x0, cfg, sub0, "DRAG, LIFT", weights0, args.cores, "obj")
    g_cmz, a_rho_m, _ = run_adjoint_pair(x0, cfg, sub0, "MOMENT_Z", None, args.cores, "cmz")
    print(f"Adjoint base residua: obj(A_Rho={a_rho_o:.2f}, A_k={a_k_o:.2f}) | cmz(A_Rho={a_rho_m:.2f})")

    rows = []
    for k in var_indices:
        xp = x0.copy()
        xm = x0.copy()
        xp[k] += step
        xm[k] -= step
        mp, tp, _ = eval_single(xp, f"v{k}_p")
        mm, tm, _ = eval_single(xm, f"v{k}_m")

        fd_obj = (mp["cd"] / mp["cl"] - mm["cd"] / mm["cl"]) / (2.0 * step)
        fd_cmz = (mp["cmz"] - mm["cmz"]) / (2.0 * step)
        rel_obj = 100.0 * abs(g_obj[k] - fd_obj) / (abs(fd_obj) + 1e-12)
        rel_cmz = 100.0 * abs(g_cmz[k] - fd_cmz) / (abs(fd_cmz) + 1e-12)

        # Sprawdzenie zgodnosci analitycznego modelu grubosci (x/c = 0.50 -> indeks 4)
        fd_dt_rel = ((tp[4] - tm[4]) / t0[4]) / (2.0 * step)
        ana_dt_rel = thick_mat[4, k]

        rows.append({
            "var": k,
            "adj_obj": g_obj[k], "fd_obj": fd_obj, "err_obj_%": rel_obj,
            "adj_cmz": g_cmz[k], "fd_cmz": fd_cmz, "err_cmz_%": rel_cmz,
            "ana_dt50": ana_dt_rel, "fd_dt50": fd_dt_rel,
        })

    df = pd.DataFrame(rows)
    print("\n--- Porownanie Adjoint vs FD ---")
    print(df.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    df.to_csv(os.path.join(wk_dir, "fd_check_results.csv"), index=False)


def run_verify(cfg, wk_dir, iters_to_check, verify_iters, cores):
    # Niezalezna weryfikacja CFD dla wybranych iteracji z historii optymalizacji
    hist_dir = os.path.join(wk_dir, "optimization_history")
    mesh_base = os.path.join(wk_dir, "mesh_base.su2")
    if not os.path.exists(mesh_base):
        mesh_base = "reference/mesh_base_ffd.su2"

    ver_root = os.path.join(wk_dir, "verification")
    os.makedirs(ver_root, exist_ok=True)
    summary = []

    print(f"\n=== WERYFIKACJA KONCOWA (INNER_ITER={verify_iters}, iteracje={iters_to_check}) ===")
    for it in iters_to_check:
        npy = os.path.join(hist_dir, f"x_{it:03d}.npy")
        if not os.path.exists(npy):
            print(f"Pominieto iteracje {it}: brak {npy}")
            continue
        x = np.load(npy)
        sub = os.path.join(ver_root, f"it{it}")
        os.makedirs(sub, exist_ok=True)
        shutil.copy(mesh_base, os.path.join(sub, cfg["mesh_filename"]))

        prepare_config("SU2_DEF", x, cfg, sub)
        run_su2_module("SU2_DEF", sub, cores=1)
        assert_ffd_applied(sub, x_vec if "x_vec" in locals() else x)
        shutil.move(os.path.join(sub, "mesh_out.su2"), os.path.join(sub, cfg["mesh_filename"]))

        use_rst = os.path.exists("reference/restart_flow.dat")
        if use_rst:
            shutil.copy("reference/restart_flow.dat", os.path.join(sub, "solution_flow.dat"))

        ver_cfg = dict(cfg, restart=use_rst, iters=verify_iters, conv_minval=-14)
        prepare_config("SU2_CFD", x, ver_cfg, sub)
        run_su2_module("SU2_CFD", sub, cores=cores)

        h = pd.read_csv(os.path.join(sub, "history.csv"))
        h.columns = [c.strip().strip('"') for c in h.columns]
        print(f"\n-- Weryfikacja it{it} (n={len(h)}, rms[Rho]={h['rms[Rho]'].iloc[-1]:.2f}) --")
        for s in range(0, len(h), 500):
            w = h.iloc[s : s + 500]
            cd_w, cl_w = w["CD"].mean(), w["CL"].mean()
            print(f"   {s:4d}-{s+len(w):4d}: CD={cd_w:.6f} (std {w['CD'].std():.1e}) | CL={cl_w:.5f} | L/D={cl_w/cd_w:.2f}")

        m = read_cfd_metrics(os.path.join(sub, "history.csv"), avg_window=300)
        thick, camber, te_y = measure_surface_geometry(os.path.join(sub, "surface_flow.csv"))
        summary.append({
            "iter": it,
            "cd": m["cd"],
            "cl": m["cl"],
            "cmz": m["cmz"],
            "ld": m["cl"] / m["cd"],
            "cd_std_300": m["cd_std"],
            "rms_rho": m["rms_rho_last"],
            "rms_w": m["rms_w_last"],
            "max_thick": float(thick.max()),
            "max_camber": float(np.max(np.abs(camber))),
            "te_y": te_y,
        })

    if summary:
        df = pd.DataFrame(summary)
        print("\n--- Podsumowanie weryfikacji ---")
        print(df.to_string(index=False))
        df.to_csv(os.path.join(ver_root, "verify_summary.csv"), index=False)


def run_inspect(wk_dir):
    # Podglad metryk i zapisanych wektorow zmiennych projektowych
    metrics_path = os.path.join(wk_dir, "opt_metrics.csv")
    if os.path.exists(metrics_path):
        print(f"\n=== {metrics_path} ===")
        print(pd.read_csv(metrics_path).to_string(index=False))
    xs = sorted(glob.glob(os.path.join(wk_dir, "optimization_history", "x_*.npy")))
    if xs:
        print("\n=== Zapisane wektory x ===")
        for p in xs:
            v = np.load(p)
            print(f"{os.path.basename(p)}: min={v.min():+.5f}, max={v.max():+.5f}, norm={np.linalg.norm(v):.5f}")


def select_best_feasible_iter(wk_dir):
    # Wybor najlepszej dopuszczalnej iteracji z opt_metrics.csv
    metrics_path = os.path.join(wk_dir, "opt_metrics.csv")
    if not os.path.exists(metrics_path):
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None
    if "feasible" in df.columns and (df["feasible"] == 1).any():
        sub = df[df["feasible"] == 1]
    else:
        sub = df
    return int(sub.loc[sub["obj"].idxmin(), "iter"])


def run_optimization(cfg, args):
    # Glowna petla optymalizacji NLopt LD_MMA
    wk_dir = args.work_dir
    hist_dir = os.path.join(wk_dir, "optimization_history")

    if os.path.exists(wk_dir):
        shutil.rmtree(wk_dir)
    os.makedirs(hist_dir)

    mesh_base = ensure_base_mesh(cfg, wk_dir, remesh=args.remesh)
    ffd_bounds = read_ffd_bounds(mesh_base)

    evaluator = AirfoilEvaluator(
        cfg=cfg,
        wk_dir=wk_dir,
        hist_dir=hist_dir,
        cores=args.cores,
        cmz_max=args.cmz_max,
        min_thick_ratio=args.min_thick_ratio,
        ffd_bounds=ffd_bounds,
    )

    opt = nlopt.opt(nlopt.LD_MMA, N_VARS)
    opt.set_lower_bounds(np.full(N_VARS, -args.box_bound))
    opt.set_upper_bounds(np.full(N_VARS, args.box_bound))
    opt.set_initial_step(0.005)
    opt.set_min_objective(evaluator.objective)

    if args.cmz_max > 0:
        opt.add_inequality_constraint(evaluator.cmz_upper, 1e-4)
        opt.add_inequality_constraint(evaluator.cmz_lower, 1e-4)
    if args.min_thick_ratio > 0:
        opt.add_inequality_mconstraint(
            evaluator.thickness_constraints, np.full(len(THICK_STATIONS), 1e-4)
        )

    opt.set_maxeval(args.max_eval)
    opt.set_ftol_rel(1e-4)
    opt.set_xtol_rel(1e-4)

    print(
        f"\nStart optymalizacji NLopt (LD_MMA): max(L/D), |CMz|<={args.cmz_max}, "
        f"t/t0>={args.min_thick_ratio}, x in [{-args.box_bound}, {args.box_bound}]"
    )
    t0 = time.time()
    try:
        opt.optimize(np.zeros(N_VARS))
        print(f"\n--- Zakonczono optymalizacje (min CD/CL: {opt.last_optimum_value():.5f}) ---")
    except Exception as e:
        print(f"\nProces przerwany: {e}")

    best_it = select_best_feasible_iter(wk_dir)
    if args.verify_final and best_it is not None:
        to_check = sorted(set([1, best_it]))
        run_verify(cfg, wk_dir, to_check, args.verify_iters, args.cores)

    try:
        generate_summary(wk_dir, hist_dir, cmz_max=args.cmz_max, flow_cfg=cfg)
    except Exception as e:
        print(f"Blad postprocessingu: {e}")

    print(f"Czas calkowity: {datetime.timedelta(seconds=int(time.time() - t0))}")


def parse_args():
    p = argparse.ArgumentParser(description="Optymalizacja ksztaltu profilu NACA 0012 (SU2 + NLopt)")
    p.add_argument("--mode", choices=["opt", "fd-check", "verify", "inspect"], default="opt")
    p.add_argument("--work-dir", default="workspace_opt")
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--max-eval", type=int, default=20)
    p.add_argument("--cfd-iters", type=int, default=1500)
    p.add_argument("--adj-iters", type=int, default=600)
    p.add_argument("--adj-cfl", type=float, default=50.0)
    p.add_argument("--verify-iters", type=int, default=3500)
    p.add_argument("--cmz-max", type=float, default=0.05)
    p.add_argument("--min-thick-ratio", type=float, default=0.90)
    p.add_argument("--box-bound", type=float, default=0.06)
    p.add_argument("--remesh", action="store_true")
    p.add_argument("--verify-final", action="store_true")
    p.add_argument("--vars", default="2,12,14", help="Indeksy zmiennych dla fd-check")
    p.add_argument("--fd-step", type=float, default=1e-3)
    p.add_argument("--iters", type=int, nargs="+", help="Lista iteracji dla trybu verify")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        "naca_code": "0012",
        "mach": 0.15,
        "reynolds": 3000000,
        "aoa": 2.0,
        "temperature": 288.15,
        "turb_model": "SST",
        "trans_model": "NONE",
        "workspace_dir": args.work_dir,
        "mesh_filename": "mesh.su2",
        "cfd_iters": args.cfd_iters,
        "adj_iters": args.adj_iters,
        "adj_cfl": args.adj_cfl,
    }

    if args.mode == "opt":
        run_optimization(cfg, args)
    elif args.mode == "fd-check":
        run_fd_check(cfg, args)
    elif args.mode == "verify":
        iters = args.iters
        if not iters:
            best_it = select_best_feasible_iter(args.work_dir)
            iters = sorted(set([1, best_it])) if best_it else [1]
        run_verify(cfg, args.work_dir, iters, args.verify_iters, args.cores)
    elif args.mode == "inspect":
        run_inspect(args.work_dir)


if __name__ == "__main__":
    main()