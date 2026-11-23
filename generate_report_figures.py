import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd

OUT_DIR = "report_figures"
FLOW_CFG = {"mach": 0.15, "reynolds": 3000000.0, "temperature": 288.15}
THICK_STATIONS = np.linspace(0.10, 0.90, 9)


def configure_style():
    # Ustawienia typografii zblizone do LaTeX (Computer Modern / STIX)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })


def save_fig(fig, name):
    # Zapis w formacie wektorowym PDF (do LaTeX) oraz pogladowym PNG
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, dpi=350, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def find_run_dir(candidates):
    # Wybiera pierwszy istniejacy katalog wynikowy z opt_metrics.csv
    for c in candidates:
        if os.path.exists(os.path.join(c, "opt_metrics.csv")):
            return c
    return None


def load_surface(csv_path):
    # Wczytanie danych powierzchniowych z pliku SU2
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    x = df["x"].values if "x" in df.columns else df["PointX"].values
    y = df["y"].values if "y" in df.columns else df["PointY"].values
    return df, x, y


def split_surfaces(x, y):
    # Podzial na gorna i dolna powierzchnie wzgledem cieciwy (posortowane po x)
    i_le = int(np.argmin(x))
    i_te = int(np.argmax(x))
    chord_y = np.interp(x, [x[i_le], x[i_te]], [y[i_le], y[i_te]])
    up = np.where(y >= chord_y)[0]
    lo = np.where(y <= chord_y)[0]
    up = up[np.argsort(x[up])]
    lo = lo[np.argsort(x[lo])]
    return up, lo


def order_loop(x, y):
    # Zamkniety obwod profilu do rysowania konturu
    up, lo = split_surfaces(x, y)
    idx = np.concatenate([up, lo[::-1], up[:1]])
    return x[idx], y[idx]


def compute_cp(df, flow_cfg=FLOW_CFG):
    # Wspolczynnik cisnienia Cp wyliczony ze zmiennych zachowawczych SU2
    if "Pressure_Coefficient" in df.columns:
        return df["Pressure_Coefficient"].values
    gamma = 1.4
    r_gas = 287.058
    t_inf = flow_cfg["temperature"]
    mach = flow_cfg["mach"]
    reynolds = flow_cfg["reynolds"]

    mu_inf = 1.716e-5 * ((t_inf / 273.15) ** 1.5) * ((273.15 + 110.4) / (t_inf + 110.4))
    a_inf = math.sqrt(gamma * r_gas * t_inf)
    v_inf = mach * a_inf
    rho_inf = reynolds * mu_inf / v_inf
    p_inf = rho_inf * r_gas * t_inf
    q_inf = 0.5 * rho_inf * (v_inf ** 2)

    rho = df["Density"].values
    mx = df["Momentum_x"].values
    my = df["Momentum_y"].values
    en = df["Energy"].values
    p = (gamma - 1.0) * (en - 0.5 * (mx ** 2 + my ** 2) / np.maximum(rho, 1e-12))
    return (p - p_inf) / q_inf


def compute_camber_thickness(x, y, grid):
    # Interpolacja linii szkieletowej i rozkladu grubosci wzdluz cieciwy
    up, lo = split_surfaces(x, y)
    yu = np.interp(grid, x[up], y[up])
    yl = np.interp(grid, x[lo], y[lo])
    return 0.5 * (yu + yl), yu - yl


def pick_best_iter(df):
    sub = df[df["feasible"] == 1] if ("feasible" in df.columns and (df["feasible"] == 1).any()) else df
    return int(sub.loc[sub["obj"].idxmin(), "iter"])


def load_su2_mesh(mesh_path):
    # Parsowanie wezlow i elementow 2D (trojkaty/czworokaty) z pliku .su2
    with open(mesh_path, "r") as f:
        lines = f.readlines()

    ie = next(i for i, l in enumerate(lines) if l.strip().startswith("NELEM="))
    ne = int(lines[ie].split("=")[1].split()[0])
    ip = next(i for i, l in enumerate(lines) if l.strip().startswith("NPOIN="))
    npn = int(lines[ip].split("=")[1].split()[0])

    pts = np.empty((npn, 2), dtype=float)
    for k in range(npn):
        toks = lines[ip + 1 + k].split()
        pts[k, 0] = float(toks[0])
        pts[k, 1] = float(toks[1])

    tris, quads = [], []
    for k in range(ne):
        toks = lines[ie + 1 + k].split()
        etype = int(toks[0])
        if etype == 5:
            tris.append((int(toks[1]), int(toks[2]), int(toks[3])))
        elif etype == 9:
            quads.append((int(toks[1]), int(toks[2]), int(toks[3]), int(toks[4])))

    return pts, np.array(tris, dtype=np.int32), np.array(quads, dtype=np.int32)


def plot_fig01_ffd(surf_base_path):
    # Rysunek 1: Parametryzacja FFD, zmienne projektowe i przekroje ograniczen
    _, x, y = load_surface(surf_base_path)
    xo, yo = order_loop(x, y)

    x_min, x_max, y_min, y_max = -0.1, 1.1, -0.2, 0.2
    xs_cp = np.linspace(x_min, x_max, 10)
    ys_cp = [y_min, y_max]

    fig, ax = plt.subplots(figsize=(8.2, 3.3))
    # Klatka FFD
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
            "k--", linewidth=1.0, alpha=0.7, label="FFD box (MAIN_BOX)")
    for xv in xs_cp[1:-1]:
        ax.plot([xv, xv], [y_min, y_max], color="0.82", linewidth=0.6, zorder=1)

    # Kontur profilu
    ax.fill(xo, yo, facecolor="#d9e6f2", edgecolor="#1f4e79", linewidth=1.4, zorder=3,
            label="Baseline NACA 0012")

    # Przekroje kontroli grubosci
    up, lo = split_surfaces(x, y)
    for k, st in enumerate(THICK_STATIONS):
        yu = np.interp(st, x[up], y[up])
        yl = np.interp(st, x[lo], y[lo])
        lbl = "Thickness stations ($t/t_0 \\geq 0.90$)" if k == 0 else None
        ax.plot([st, st], [yl, yu], color="#2ca02c", linewidth=1.1, alpha=0.85, zorder=4, label=lbl)

    # Punkty kotwiczenia LE/TE
    ax.scatter([0.0, 1.0], [0.0, 0.0], marker="D", s=34, color="#d62728", zorder=6,
               label="Pinned LE/TE ($|\\Delta y| \\leq 5\\times 10^{-4}c$)")

    # Punkty kontrolne FFD: unieruchomione i aktywne
    fx_x = [xs_cp[0], xs_cp[0], xs_cp[-1], xs_cp[-1]]
    fx_y = [ys_cp[0], ys_cp[1], ys_cp[0], ys_cp[1]]
    ax.scatter(fx_x, fx_y, marker="s", s=36, color="0.35", zorder=5, label="Fixed FFD corners ($i=0,9$)")

    act_x = [xv for xv in xs_cp[1:-1] for _ in ys_cp]
    act_y = [yv for _ in xs_cp[1:-1] for yv in ys_cp]
    ax.scatter(act_x, act_y, marker="o", s=34, color="#1f77b4", zorder=5,
               label="Active vertical DVs ($i=1\\dots 8$, $n=16$)")

    # Strzalki kierunku ruchu dla reprezentatywnej pary punktow
    ax.annotate("", xy=(xs_cp[5], y_max + 0.055), xytext=(xs_cp[5], y_max + 0.008),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.0))
    ax.text(xs_cp[5] + 0.02, y_max + 0.028, r"$\Delta y_{i,j}$", color="#1f77b4", fontsize=8.5)

    ax.set_aspect("equal")
    ax.set_xlim(-0.18, 1.18)
    ax.set_ylim(-0.27, 0.29)
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("FFD Parameterization and Geometric Constraints")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=True)
    save_fig(fig, "fig01_ffd_parameterization")


def plot_fig02_mesh(mesh_path, surf_base_path):
    # Rysunek 2: Topologia siatki (strefa przyprofilowa, LE, TE)
    if not os.path.exists(mesh_path):
        print(f"Skipped fig02: missing {mesh_path}")
        return

    print(f"Reading mesh for fig02 ({mesh_path})...")
    pts, tris, quads = load_su2_mesh(mesh_path)
    _, sx, sy = load_surface(surf_base_path)
    sxo, syo = order_loop(sx, sy)

    tri_c = pts[tris].mean(axis=1) if len(tris) else np.empty((0, 2))
    quad_c = pts[quads].mean(axis=1) if len(quads) else np.empty((0, 2))

    views = [
        ("(a) Near-field & wake refinement", (-0.25, 1.45), (-0.42, 0.42), 0.14),
        ("(b) Leading-edge boundary layer", (-0.03, 0.11), (-0.055, 0.055), 0.25),
        ("(c) Rounded trailing edge (20 nodes)", (0.955, 1.035), (-0.032, 0.032), 0.30),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6))
    for ax, (title, xlim, ylim, lw) in zip(axes, views):
        pad_x = 0.08 * (xlim[1] - xlim[0])
        pad_y = 0.08 * (ylim[1] - ylim[0])
        if len(tris):
            m_t = (
                (tri_c[:, 0] >= xlim[0] - pad_x) & (tri_c[:, 0] <= xlim[1] + pad_x) &
                (tri_c[:, 1] >= ylim[0] - pad_y) & (tri_c[:, 1] <= ylim[1] + pad_y)
            )
            if m_t.any():
                pc_t = PolyCollection(pts[tris[m_t]], facecolor="none", edgecolor="0.45", linewidths=lw)
                pc_t.set_rasterized(True)
                ax.add_collection(pc_t)
        if len(quads):
            m_q = (
                (quad_c[:, 0] >= xlim[0] - pad_x) & (quad_c[:, 0] <= xlim[1] + pad_x) &
                (quad_c[:, 1] >= ylim[0] - pad_y) & (quad_c[:, 1] <= ylim[1] + pad_y)
            )
            if m_q.any():
                pc_q = PolyCollection(pts[quads[m_q]], facecolor="none", edgecolor="#1f4e79", linewidths=lw)
                pc_q.set_rasterized(True)
                ax.add_collection(pc_q)

        ax.plot(sxo, syo, "k-", linewidth=0.9, zorder=5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x/c$")
        ax.set_ylabel("$y/c$")
        ax.set_title(title)
        ax.grid(False)

    plt.tight_layout()
    save_fig(fig, "fig02_mesh_topology")


def _find_col(df, candidates):
    # Wyszukanie nazwy kolumny niezaleznie od wielkosci liter i spacji
    norm = {c.strip().strip('"').lower().replace(" ", "_"): c for c in df.columns}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    return None


def plot_fig03_validation():
    # Rysunek 3: Walidacja polar NACA 0012 (CFD SST+LM vs dane eksperymentalne)
    polar_path = "workspace/polar_data.csv"
    exp_path = "naca0012_exp.csv"
    if not os.path.exists(polar_path):
        print(f"Skipped fig03: missing {polar_path}")
        return

    df_cfd = pd.read_csv(polar_path, sep=None, engine="python")
    ca = _find_col(df_cfd, ["alpha", "aoa", "angle", "angle_of_attack"])
    cl = _find_col(df_cfd, ["cl", "c_l", "lift"])
    cd = _find_col(df_cfd, ["cd", "c_d", "drag"])
    if not (ca and cl and cd):
        print(f"Skipped fig03: unexpected columns in {polar_path}: {list(df_cfd.columns)}")
        return
    df_cfd = df_cfd.sort_values(ca)

    df_exp = None
    if os.path.exists(exp_path):
        df_exp = pd.read_csv(exp_path, sep=None, engine="python")
        ea = _find_col(df_exp, ["alpha", "aoa", "angle", "angle_of_attack"])
        ecl = _find_col(df_exp, ["cl", "c_l", "lift"])
        ecd = _find_col(df_exp, ["cd", "c_d", "drag"])
        if ea:
            df_exp = df_exp.sort_values(ea)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))

    # Panel (a): CL vs alpha
    ax = axes[0]
    ax.plot(df_cfd[ca], df_cfd[cl], "o-", color="#1f77b4", ms=4.2, lw=1.5, label="SU2 (SST + LM)")
    if df_exp is not None and ea and ecl:
        ax.scatter(df_exp[ea], df_exp[ecl], marker="s", s=26, facecolor="none", edgecolor="#d62728",
                   linewidths=1.2, zorder=5, label="Experimental data")
    ax.set_xlabel(r"Angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"Lift coefficient $C_L$")
    ax.set_title("(a) Lift curve")
    ax.legend(loc="upper left")

    # Panel (b): CD vs alpha
    ax = axes[1]
    ax.plot(df_cfd[ca], df_cfd[cd], "o-", color="#1f77b4", ms=4.2, lw=1.5, label="SU2 (SST + LM)")
    if df_exp is not None and ea and ecd:
        ax.scatter(df_exp[ea], df_exp[ecd], marker="s", s=26, facecolor="none", edgecolor="#d62728",
                   linewidths=1.2, zorder=5, label="Experimental data")
    ax.set_xlabel(r"Angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"Drag coefficient $C_D$")
    ax.set_title("(b) Drag curve")
    ax.legend(loc="upper center")

    # Panel (c): L/D vs alpha
    ax = axes[2]
    ax.plot(df_cfd[ca], df_cfd[cl] / df_cfd[cd], "o-", color="#1f77b4", ms=4.2, lw=1.5, label="SU2 (SST + LM)")
    if df_exp is not None and ea and ecl and ecd:
        ax.scatter(df_exp[ea], df_exp[ecl] / df_exp[ecd], marker="s", s=26, facecolor="none",
                   edgecolor="#d62728", linewidths=1.2, zorder=5, label="Experimental data")
    ax.axvline(2.0, color="0.4", linestyle=":", linewidth=1.0, label=r"Opt. design point ($\alpha = 2^\circ$)")
    ax.set_xlabel(r"Angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"Lift-to-drag ratio $L/D$")
    ax.set_title("(c) Aerodynamic efficiency")
    ax.legend(loc="lower right")

    plt.tight_layout()
    save_fig(fig, "fig03_naca0012_validation")


def plot_fig04_opt_convergence(pinned_dir):
    # Rysunek 4: Zbieznosc optymalizacji, ograniczenia i residua
    df = pd.read_csv(os.path.join(pinned_dir, "opt_metrics.csv"))
    best_it = pick_best_iter(df)
    b_row = df[df["iter"] == best_it].iloc[0]

    ver_path = os.path.join(pinned_dir, "verification", "verify_summary.csv")
    df_ver = pd.read_csv(ver_path) if os.path.exists(ver_path) else None

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))

    # Panel (a): L/D
    ax = axes[0, 0]
    ax.plot(df["iter"], df["ld"], "o-", color="#1f77b4", ms=4.5, lw=1.6, label="Optimization loop (1200 it)")
    ax.plot(best_it, b_row["ld"], "o", color="#d62728", ms=7.0, label=f"Best feasible (it{best_it}: {b_row['ld']:.2f})")
    if df_ver is not None and best_it in df_ver["iter"].values:
        v_ld = float(df_ver[df_ver["iter"] == best_it]["ld"].iloc[0])
        ax.scatter([best_it], [v_ld], marker="*", s=110, color="#2ca02c", zorder=6,
                   label=f"Verified 4500 it ({v_ld:.2f})")
    ax.set_xlabel("Optimization evaluation")
    ax.set_ylabel(r"Lift-to-drag ratio $L/D$")
    ax.set_title("(a) Objective improvement ($L/D$)")
    ax.set_xticks(df["iter"])
    ax.legend(loc="upper left")

    # Panel (b): CD i CL
    ax = axes[0, 1]
    l1 = ax.plot(df["iter"], df["cd"] * 1e4, "o-", color="#1f77b4", ms=4.2, lw=1.5, label=r"$C_D$ [counts]")
    ax.set_xlabel("Optimization evaluation")
    ax.set_ylabel(r"Drag coefficient $C_D$ [counts, $\times 10^{-4}$]", color="#1f77b4")
    ax.set_xticks(df["iter"])
    ax_r = ax.twinx()
    ax_r.grid(False)
    l2 = ax_r.plot(df["iter"], df["cl"], "s-", color="#2ca02c", ms=4.2, lw=1.5, label=r"$C_L$")
    ax_r.set_ylabel(r"Lift coefficient $C_L$", color="#2ca02c")
    lines = l1 + l2
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax.set_title("(b) Lift and drag coefficients")

    # Panel (c): Ograniczenia CMz i grubosc (bez nakladania legendy)
    ax = axes[1, 0]
    h_leg = []
    h_leg += ax.plot(df["iter"], df["cmz"], "o-", color="#9467bd", ms=4.2, lw=1.5, label=r"$C_{Mz}$")
    h_leg += [ax.axhline(0.05, color="#d62728", linestyle="--", lw=1.2, label=r"Limit $|C_{Mz}| \leq 0.05$")]
    ax.set_ylim(-0.008, 0.058)
    ax.set_xlabel("Optimization evaluation")
    ax.set_ylabel(r"Pitching moment $C_{Mz}$", color="#9467bd")
    ax.set_xticks(df["iter"])

    ax_t = ax.twinx()
    ax_t.grid(False)
    h_leg += ax_t.plot(df["iter"], df["min_thick_ratio"], "s--", color="0.2", ms=3.8, lw=1.2, label=r"$\min(t/t_0)$")
    h_leg += [ax_t.axhline(0.90, color="0.35", linestyle=":", lw=1.2, label=r"Limit $t/t_0 \geq 0.90$")]
    ax_t.set_ylim(0.885, 1.015)
    ax_t.set_ylabel(r"Minimum thickness ratio $\min(t/t_0)$", color="0.2")
    ax.legend(h_leg, [h.get_label() for h in h_leg], loc="center left", bbox_to_anchor=(0.01, 0.56))
    ax.set_title("(c) Pitching-moment and thickness constraints")

    # Panel (d): Residua prymalne i adjointowe
    ax = axes[1, 1]
    ax.plot(df["iter"], df["rms_rho"], "o-", color="#1f77b4", ms=4.0, lw=1.5, label=r"Primal $\log_{10}(\mathrm{rms}[\rho])$")
    if "rms_a_rho_obj" in df.columns:
        ax.plot(df["iter"], df["rms_a_rho_obj"], "s-", color="#d62728", ms=3.8, lw=1.3, label=r"Adj($C_D/C_L$) $\mathrm{rms}[A_\rho]$")
    if "rms_a_k_obj" in df.columns:
        ax.plot(df["iter"], df["rms_a_k_obj"], "^--", color="#ff7f0e", ms=3.8, lw=1.2, label=r"Adj($C_D/C_L$) $\mathrm{rms}[A_k]$")
    if "rms_a_rho_cmz" in df.columns:
        ax.plot(df["iter"], df["rms_a_rho_cmz"], "d:", color="#9467bd", ms=3.8, lw=1.2, label=r"Adj($C_{Mz}$) $\mathrm{rms}[A_\rho]$")
    ax.set_xlabel("Optimization evaluation")
    ax.set_ylabel(r"Final residual $\log_{10}(\mathrm{RMS})$")
    ax.set_xticks(df["iter"])
    ax.set_title("(d) Primal and discrete-adjoint residuals")
    ax.set_ylim(-9.15, -4.80)
    ax.legend(loc="lower left", ncol=2, fontsize=7.5)

    plt.tight_layout(w_pad=2.2)
    fig.subplots_adjust(right=0.92)
    save_fig(fig, "fig04_opt_convergence")


def plot_fig05_geometry(pinned_dir):
    # Rysunek 5: Kontury, linia szkieletowa i rozklad grubosci (it1 vs best)
    df = pd.read_csv(os.path.join(pinned_dir, "opt_metrics.csv"))
    best_it = pick_best_iter(df)

    p_base = os.path.join(pinned_dir, "verification", "it1", "surface_flow.csv")
    p_opt = os.path.join(pinned_dir, "verification", f"it{best_it}", "surface_flow.csv")
    _, x1, y1 = load_surface(p_base)
    _, x2, y2 = load_surface(p_opt)

    x1o, y1o = order_loop(x1, y1)
    x2o, y2o = order_loop(x2, y2)

    grid = np.linspace(0.01, 0.99, 250)
    c1, t1 = compute_camber_thickness(x1, y1, grid)
    c2, t2 = compute_camber_thickness(x2, y2, grid)

    fig = plt.figure(figsize=(10.5, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    # Gorny szeroki panel: kontur 1:1
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(x1o, y1o, "k--", lw=1.3, label="Baseline NACA 0012")
    ax0.plot(x2o, y2o, "-", color="#1f77b4", lw=1.9, label=f"Optimized pinned LE/TE (it{best_it})")
    ax0.plot(grid, c2, ":", color="#d62728", lw=1.2, label=f"Optimized camber line (max {c2.max()*100:.2f}%c)")
    ax0.set_aspect("equal")
    ax0.set_xlim(-0.03, 1.03)
    ax0.set_ylim(-0.09, 0.13)
    ax0.set_xlabel("$x/c$")
    ax0.set_ylabel("$y/c$")
    ax0.set_title("(a) Airfoil profile comparison (true aspect ratio)")
    ax0.legend(loc="upper right", ncol=3)

    # Lewy dolny: linia szkieletowa
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(grid, c1 * 100.0, "k--", lw=1.2, label="Baseline NACA 0012")
    ax1.plot(grid, c2 * 100.0, "-", color="#1f77b4", lw=1.7, label="Optimized")
    ax1.set_xlabel("$x/c$")
    ax1.set_ylabel("Camber $y_c/c$ [%]")
    ax1.set_title("(b) Mean camber line")
    ax1.legend(loc="upper right")

    # Prawy dolny: rozklad grubosci + stosunek t/t0
    ax2 = fig.add_subplot(gs[1, 1])
    l1 = ax2.plot(grid, t1 * 100.0, "k--", lw=1.2, label="Baseline $t/c$")
    l2 = ax2.plot(grid, t2 * 100.0, "-", color="#1f77b4", lw=1.6, label="Optimized $t/c$")
    ax2.set_xlabel("$x/c$")
    ax2.set_ylabel("Thickness $t/c$ [%]")

    st_mask = (grid >= 0.10) & (grid <= 0.90)
    ax2_r = ax2.twinx()
    ax2_r.grid(False)
    l3 = ax2_r.plot(grid[st_mask], t2[st_mask] / t1[st_mask], "-", color="#2ca02c", lw=1.2, alpha=0.85, label="Ratio $t/t_0$")
    l4 = [ax2_r.axhline(0.90, color="#d62728", linestyle=":", lw=1.1, label="Limit $t/t_0 = 0.90$")]
    ax2_r.set_ylim(0.88, 1.03)
    ax2_r.set_ylabel("Local thickness ratio $t/t_0$", color="#2ca02c")
    lines = l1 + l2 + l3 + l4
    ax2.legend(lines, [l.get_label() for l in lines], loc="center", bbox_to_anchor=(0.40, 0.42), fontsize=7.6)
    ax2.set_title("(c) Thickness distribution and structural margin")

    plt.tight_layout()
    save_fig(fig, "fig05_geometry_analysis")


def plot_fig06_cp(pinned_dir):
    # Rysunek 6: Ciagle krzywe rozkladu wspolczynnika cisnienia -Cp(x/c)
    df = pd.read_csv(os.path.join(pinned_dir, "opt_metrics.csv"))
    best_it = pick_best_iter(df)

    df1, x1, y1 = load_surface(os.path.join(pinned_dir, "verification", "it1", "surface_flow.csv"))
    df2, x2, y2 = load_surface(os.path.join(pinned_dir, "verification", f"it{best_it}", "surface_flow.csv"))
    cp1, cp2 = compute_cp(df1), compute_cp(df2)
    u1, l1 = split_surfaces(x1, y1)
    u2, l2 = split_surfaces(x2, y2)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(x1[u1], cp1[u1], "k--", lw=1.4, label="Baseline NACA 0012 (upper/lower)")
    ax.plot(x1[l1], cp1[l1], "k--", lw=1.4)
    ax.plot(x2[u2], cp2[u2], "-", color="#1f77b4", lw=1.9, label=f"Optimized pinned LE/TE, it{best_it} (upper)")
    ax.plot(x2[l2], cp2[l2], "-", color="#d62728", lw=1.6, label=f"Optimized pinned LE/TE, it{best_it} (lower)")

    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("$x/c$")
    ax.set_ylabel(r"Pressure coefficient $C_p$")
    ax.set_title(r"Surface Pressure Distribution Comparison ($M=0.15$, $Re=3\times 10^6$, $\alpha=2^\circ$)")
    ax.legend(loc="lower right")
    save_fig(fig, "fig06_cp_distribution")


def plot_fig07_verification(pinned_dir):
    # Rysunek 7: Zbieznosc weryfikacji CFD na 4500 iteracji
    df = pd.read_csv(os.path.join(pinned_dir, "opt_metrics.csv"))
    best_it = pick_best_iter(df)

    h1 = pd.read_csv(os.path.join(pinned_dir, "verification", "it1", "history.csv"))
    h2 = pd.read_csv(os.path.join(pinned_dir, "verification", f"it{best_it}", "history.csv"))
    h1.columns = [c.strip().strip('"') for c in h1.columns]
    h2.columns = [c.strip().strip('"') for c in h2.columns]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

    # Panel (a): Residua
    ax = axes[0]
    ax.plot(h1["Inner_Iter"], h1["rms[Rho]"], "k--", lw=1.3, label=r"Baseline $\mathrm{rms}[\rho]$")
    ax.plot(h2["Inner_Iter"], h2["rms[Rho]"], "-", color="#1f77b4", lw=1.5, label=r"Optimized $\mathrm{rms}[\rho]$")
    if "rms[w]" in h1.columns:
        ax.plot(h1["Inner_Iter"], h1["rms[w]"], "k:", lw=1.1, label=r"Baseline $\mathrm{rms}[\omega]$")
        ax.plot(h2["Inner_Iter"], h2["rms[w]"], "-", color="#ff7f0e", lw=1.2, alpha=0.85, label=r"Optimized $\mathrm{rms}[\omega]$")
    ax.set_xlabel("CFD iteration")
    ax.set_ylabel(r"Residual $\log_{10}(\mathrm{RMS})$")
    ax.set_title("(a) Verification residual history")
    ax.legend(loc="center right")

    # Panel (b): Stabilizacja L/D i CD (od iteracji 500 dla czytelnosci skali)
    ax = axes[1]
    s_idx = 500
    ax.plot(h1["Inner_Iter"].iloc[s_idx:], (h1["CL"] / h1["CD"]).iloc[s_idx:], "k--", lw=1.3, label="Baseline $L/D$")
    ax.plot(h2["Inner_Iter"].iloc[s_idx:], (h2["CL"] / h2["CD"]).iloc[s_idx:], "-", color="#1f77b4", lw=1.6, label="Optimized $L/D$")
    ax.axvspan(len(h2) - 300, len(h2), color="#2ca02c", alpha=0.14, label="Averaging window (300 it)")
    ax.set_xlabel("CFD iteration")
    ax.set_ylabel(r"Lift-to-drag ratio $L/D$")
    ax.set_title("(b) Aerodynamic efficiency stabilization")
    ax.legend(loc="center right", bbox_to_anchor=(0.90, 0.45))

    plt.tight_layout()
    save_fig(fig, "fig07_verification_cfd")


def plot_fig08_pinned_vs_unpinned(pinned_dir, free_dir):
    # Rysunek 8: Porownanie wariantu bez kotwiczenia LE/TE i z kotwiczeniem LE/TE
    if not free_dir:
        print("Skipped fig08: unpinned run directory not found.")
        return

    df_p = pd.read_csv(os.path.join(pinned_dir, "opt_metrics.csv"))
    df_f = pd.read_csv(os.path.join(free_dir, "opt_metrics.csv"))
    it_p = pick_best_iter(df_p)
    it_f = pick_best_iter(df_f)

    vp = pd.read_csv(os.path.join(pinned_dir, "verification", "verify_summary.csv"))
    vf = pd.read_csv(os.path.join(free_dir, "verification", "verify_summary.csv"))
    r_base = vp[vp["iter"] == 1].iloc[0]
    r_pin = vp[vp["iter"] == it_p].iloc[0]
    r_free = vf[vf["iter"] == it_f].iloc[0]

    df_b, xb, yb = load_surface(os.path.join(pinned_dir, "verification", "it1", "surface_flow.csv"))
    df_s_p, xp, yp = load_surface(os.path.join(pinned_dir, "verification", f"it{it_p}", "surface_flow.csv"))
    df_s_f, xf, yf = load_surface(os.path.join(free_dir, "verification", f"it{it_f}", "surface_flow.csv"))

    fig = plt.figure(figsize=(10.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.15])

    # Panel (a): Szeroki kontur w skali 1:1 z legenda w wolnym pasie nad profilem
    ax = fig.add_subplot(gs[0, :])
    ax.plot(*order_loop(xb, yb), "k--", lw=1.3, label="Baseline NACA 0012")
    ax.plot(*order_loop(xf, yf), "-", color="#ff7f0e", lw=1.7, label=r"Unpinned LE/TE ($\Delta\alpha_{\mathrm{eff}} \approx +2.7^\circ$)")
    ax.plot(*order_loop(xp, yp), "-", color="#1f77b4", lw=1.9, label=r"Pinned LE/TE ($\Delta\alpha_{\mathrm{eff}} = +0.05^\circ$)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.085, 0.195)
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.set_title("(a) Profile shape & pitch rotation (true aspect ratio)")
    ax.legend(loc="upper center", ncol=3, fontsize=8.2)

    # Panel (b): Rozklad Cp
    ax = fig.add_subplot(gs[1, 0])
    for d, x, y, style, col, lbl in [
        (df_b, xb, yb, "--", "k", "Baseline"),
        (df_s_f, xf, yf, "-", "#ff7f0e", "Unpinned LE/TE"),
        (df_s_p, xp, yp, "-", "#1f77b4", "Pinned LE/TE"),
    ]:
        cp = compute_cp(d)
        up, lo = split_surfaces(x, y)
        ax.plot(x[up], cp[up], style, color=col, lw=1.4, label=lbl)
        ax.plot(x[lo], cp[lo], style, color=col, lw=1.4)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$C_p$")
    ax.set_title("(b) Pressure distribution ($C_p$)")
    ax.legend(loc="lower right", fontsize=7.8)

    # Panel (c): Zestawienie L/D i CL/CMz
    ax = fig.add_subplot(gs[1, 1])
    labels = ["Baseline", "Pinned\nLE/TE", "Unpinned\nLE/TE"]
    lds = [r_base["ld"], r_pin["ld"], r_free["ld"]]
    colors = ["0.55", "#1f77b4", "#ff7f0e"]
    bars = ax.bar(labels, lds, color=colors, width=0.52, edgecolor="k", linewidth=0.7)
    for b, val, cmz in zip(bars, lds, [r_base["cmz"], r_pin["cmz"], r_free["cmz"]]):
        ax.text(b.get_x() + b.get_width() / 2.0, val + 1.4,
                f"L/D={val:.1f}\n$C_{{Mz}}$={cmz:.3f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_ylim(0, max(lds) * 1.24)
    ax.set_ylabel(r"Verified $L/D$")
    ax.set_title("(c) Verified performance vs. trim moment")

    plt.tight_layout()
    save_fig(fig, "fig08_pinned_vs_unpinned")


def main():
    configure_style()
    pinned_dir = find_run_dir(["results/LD_cmz005_pinnedLE", "workspace_opt"])
    free_dir = find_run_dir(["results/LD_cmz005_freeLE"])
    if not pinned_dir:
        raise RuntimeError("Nie znaleziono katalogu wynikowego (results/LD_cmz005_pinnedLE ani workspace_opt).")
    print(f"Using pinned run: {pinned_dir}")
    if free_dir:
        print(f"Using unpinned run: {free_dir}")

    surf_base = os.path.join(pinned_dir, "verification", "it1", "surface_flow.csv")
    plot_fig01_ffd(surf_base)
    plot_fig02_mesh("reference/mesh_base_ffd.su2", surf_base)
    plot_fig03_validation()
    plot_fig04_opt_convergence(pinned_dir)
    plot_fig05_geometry(pinned_dir)
    plot_fig06_cp(pinned_dir)
    plot_fig07_verification(pinned_dir)
    plot_fig08_pinned_vs_unpinned(pinned_dir, free_dir)
    print(f"\nAll figures exported to: {OUT_DIR}/")


if __name__ == "__main__":
    main()