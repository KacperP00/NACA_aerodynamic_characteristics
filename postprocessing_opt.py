import glob
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd


def _load_surface(csv_path):
    # Wczytanie wspolrzednych i pol ze zrzutu powierzchniowego SU2
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    x_col = "x" if "x" in df.columns else "PointX"
    y_col = "y" if "y" in df.columns else "PointY"
    return df, df[x_col].values, df[y_col].values


def _order_airfoil_loop(x, y):
    # Porzadkuje punkty wzdluz obwodu (gorna od LE do TE, dolna od TE do LE)
    i_le = int(np.argmin(x))
    i_te = int(np.argmax(x))
    chord_y = np.interp(x, [x[i_le], x[i_te]], [y[i_le], y[i_te]])

    up = np.where(y >= chord_y)[0]
    lo = np.where(y < chord_y)[0]
    up = up[np.argsort(x[up])]
    lo = lo[np.argsort(x[lo])[::-1]]

    idx = np.concatenate([up, lo, up[:1]])
    return x[idx], y[idx]


def _compute_cp(df, flow_cfg):
    # Wylicza wspolczynnik cisnienia Cp z pol zachowawczych SU2 (gaz doskonaly)
    if "Pressure_Coefficient" in df.columns:
        return df["Pressure_Coefficient"].values

    gamma = 1.4
    r_gas = 287.058
    t_inf = flow_cfg.get("temperature", 288.15)
    mach = flow_cfg.get("mach", 0.15)
    reynolds = flow_cfg.get("reynolds", 3000000.0)

    # Lepkosc dynamiczna ze wzoru Sutherlanda (identycznie z SU2)
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


def _pick_best_iter(df):
    # Wybor najlepszej dopuszczalnej iteracji (minimalne CD/CL)
    if "feasible" in df.columns and (df["feasible"] == 1).any():
        sub = df[df["feasible"] == 1]
    else:
        sub = df
    return int(sub.loc[sub["obj"].idxmin(), "iter"])


def generate_summary(work_dir, hist_dir, cmz_max=0.05, flow_cfg=None):
    # Generuje raport wykonawczy, porownanie rozkladu Cp oraz animacje GIF
    flow_cfg = flow_cfg or {"mach": 0.15, "reynolds": 3000000.0, "temperature": 288.15}
    metrics_path = os.path.join(work_dir, "opt_metrics.csv")
    best_it = None

    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        if not df.empty:
            if "ld" not in df.columns:
                df["ld"] = df["cl"] / df["cd"]
            best_it = _pick_best_iter(df)

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))

            # Panel 1: Doskonalosc L/D
            ax = axes[0, 0]
            ax.plot(df["iter"], df["ld"], "b.-", linewidth=1.8, label="L/D")
            b_row = df[df["iter"] == best_it].iloc[0]
            ax.plot(best_it, b_row["ld"], "ro", markersize=8, label=f"Best dopuszczalny (it{best_it}: {b_row['ld']:.2f})")
            ax.set_title("Doskonalosc aerodynamiczna (L/D)")
            ax.set_xlabel("Iteracja")
            ax.set_ylabel("L/D")
            ax.grid(True, alpha=0.4)
            ax.legend()

            # Panel 2: CL i CD
            ax = axes[0, 1]
            l1 = ax.plot(df["iter"], df["cd"], "b.-", label="CD")
            ax.set_xlabel("Iteracja")
            ax.set_ylabel("CD", color="b")
            ax.grid(True, alpha=0.4)
            ax_r = ax.twinx()
            l2 = ax_r.plot(df["iter"], df["cl"], "g.-", label="CL")
            ax_r.set_ylabel("CL", color="g")
            lines = l1 + l2
            ax.legend(lines, [l.get_label() for l in lines], loc="best")
            ax.set_title("Wspolczynniki oporu (CD) i sily nosnej (CL)")

            # Panel 3: Ograniczenia (CMz i min t/t0)
            ax = axes[1, 0]
            if "cmz" in df.columns:
                ax.plot(df["iter"], df["cmz"], "m.-", label="CMz")
                if cmz_max and cmz_max > 0:
                    ax.axhline(cmz_max, color="r", linestyle="--", label=f"Limit +CMz ({cmz_max})")
                    ax.axhline(-cmz_max, color="r", linestyle="--")
                ax.set_ylabel("CMz", color="m")
            if "min_thick_ratio" in df.columns:
                ax_t = ax.twinx()
                ax_t.plot(df["iter"], df["min_thick_ratio"], "k.--", alpha=0.7, label="min t/t0")
                ax_t.set_ylabel("min t/t0", color="k")
            ax.set_title("Kontrola ograniczen (moment CMz i grubosc)")
            ax.set_xlabel("Iteracja")
            ax.grid(True, alpha=0.4)
            ax.legend(loc="upper left")

            # Panel 4: Porownanie geometrii (it1 vs best)
            ax = axes[1, 1]
            s1 = os.path.join(hist_dir, "surf_001.csv")
            sb = os.path.join(hist_dir, f"surf_{best_it:03d}.csv")
            if os.path.exists(s1) and os.path.exists(sb):
                _, x1, y1 = _load_surface(s1)
                _, xb, yb = _load_surface(sb)
                x1o, y1o = _order_airfoil_loop(x1, y1)
                xbo, ybo = _order_airfoil_loop(xb, yb)
                ax.plot(x1o, y1o, "k--", linewidth=1.4, label="Bazowy NACA 0012 (it1)")
                ax.plot(xbo, ybo, "b-", linewidth=2.0, label=f"Zoptymalizowany (it{best_it})")
                ax.set_aspect("equal")
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.15, 0.20)
                ax.legend()
            ax.set_title("Porownanie geometrii profilu")
            ax.set_xlabel("x/c")
            ax.set_ylabel("y/c")
            ax.grid(True, alpha=0.4)

            plt.tight_layout()
            out_png = os.path.join(work_dir, "executive_summary.png")
            plt.savefig(out_png, dpi=300)
            plt.close()
            print(f"-> Wygenerowano {out_png}")

    # Porownanie rozkladu Cp (preferuje zweryfikowane przebiegi, fallback na optimization_history)
    if best_it is not None:
        cand_base = [
            os.path.join(work_dir, "verification", "it1", "surface_flow.csv"),
            os.path.join(hist_dir, "surf_001.csv"),
        ]
        cand_best = [
            os.path.join(work_dir, "verification", f"it{best_it}", "surface_flow.csv"),
            os.path.join(hist_dir, f"surf_{best_it:03d}.csv"),
        ]
        p_base = next((p for p in cand_base if os.path.exists(p)), None)
        p_best = next((p for p in cand_best if os.path.exists(p)), None)
        if p_base and p_best:
            df1, x1, y1 = _load_surface(p_base)
            dfb, xb, yb = _load_surface(p_best)
            cp1 = _compute_cp(df1, flow_cfg)
            cpb = _compute_cp(dfb, flow_cfg)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(x1, cp1, s=6, c="k", alpha=0.5, label="Bazowy (it1)")
            ax.scatter(xb, cpb, s=6, c="b", alpha=0.6, label=f"Zoptymalizowany (it{best_it})")
            ax.invert_yaxis()
            ax.set_xlabel("x/c")
            ax.set_ylabel("Cp")
            ax.set_title("Porownanie rozkladu wspolczynnika cisnienia Cp")
            ax.grid(True, alpha=0.4)
            ax.legend()
            cp_png = os.path.join(work_dir, "cp_comparison.png")
            plt.tight_layout()
            plt.savefig(cp_png, dpi=300)
            plt.close()
            print(f"-> Wygenerowano {cp_png}")

    # Animacja GIF ewolucji ksztaltu
    surf_files = sorted(glob.glob(os.path.join(hist_dir, "surf_*.csv")))
    if not surf_files:
        return

    _, x0, y0 = _load_surface(surf_files[0])
    x0o, y0o = _order_airfoil_loop(x0, y0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 0.20)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.plot(x0o, y0o, "k--", linewidth=1.0, alpha=0.5, label="Bazowy")
    line, = ax.plot([], [], "b-", linewidth=2.0, label="Aktualny")
    ax.legend(loc="upper right")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        _, x, y = _load_surface(frame)
        xo, yo = _order_airfoil_loop(x, y)
        line.set_data(xo, yo)
        it_num = os.path.basename(frame).split("_")[1].split(".")[0]
        ax.set_title(f"Ewolucja ksztaltu FFD - Iteracja {it_num}")
        return (line,)

    ani = FuncAnimation(fig, update, frames=surf_files, init_func=init, blit=True, repeat=False)
    gif_path = os.path.join(work_dir, "airfoil_evolution.gif")
    ani.save(gif_path, writer="pillow", fps=3)
    plt.close()
    print(f"-> Wygenerowano animacje: {gif_path}")