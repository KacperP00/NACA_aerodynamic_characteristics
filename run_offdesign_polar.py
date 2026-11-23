import argparse
import datetime
import os
import shutil
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main_opt as m

DEFAULT_ANGLES = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
OUT_FIG_DIR = "report_figures"


def find_run_dir(explicit_dir=None):
    # Lokalizuje katalog wynikowy finalnej optymalizacji
    if explicit_dir:
        if os.path.exists(os.path.join(explicit_dir, "verification")):
            return explicit_dir
        raise FileNotFoundError(f"Brak katalogu verification w: {explicit_dir}")
    for cand in ("results/LD_cmz005_pinnedLE", "workspace_opt"):
        if os.path.exists(os.path.join(cand, "verification")):
            return cand
    raise FileNotFoundError("Nie znaleziono katalogu z podkatalogiem verification/.")


def aoa_tag(aoa):
    # Nazwa podkatalogu dla kata natarcia
    sign = "m" if aoa < 0 else "p"
    return f"aoa_{sign}{abs(aoa):.1f}"


def read_polar_metrics(history_path, avg_window=200):
    # Odczyt wspolczynnikow aerodynamicznych, residuow i dryfu CD
    df = pd.read_csv(history_path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    n = len(df)
    w = min(avg_window, max(1, n // 5))
    tail = df.iloc[-w:]

    cd = float(tail["CD"].mean())
    cl = float(tail["CL"].mean())
    cmz = float(tail["CMz"].mean()) if "CMz" in df.columns else 0.0
    cd_std = float(tail["CD"].std()) if w > 1 else 0.0

    # Dryf miedzy dwoma ostatnimi oknami jako miara stabilizacji
    if n >= 2 * w:
        prev_cd = float(df.iloc[-2 * w : -w]["CD"].mean())
        cd_drift = abs(cd - prev_cd)
    else:
        cd_drift = np.nan

    return {
        "iters_run": n,
        "cd": cd,
        "cl": cl,
        "cmz": cmz,
        "ld": cl / cd if cd > 0 else np.nan,
        "cd_std": cd_std,
        "cd_drift": cd_drift,
        "rms_rho": float(df["rms[Rho]"].iloc[-1]),
        "rms_w": float(df["rms[w]"].iloc[-1]) if "rms[w]" in df.columns else np.nan,
    }


def save_table(rows, csv_path):
    # Zapis posortowanej tabeli wynikow po kazdym kroku
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["case", "aoa"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)


def run_polar_sweep(run_dir, out_dir, angles, iters, cores, force=False):
    # Przelicza biegunowa off-design z dwukierunkowym warm-startem od AoA=2.0
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "offdesign_polar.csv")

    existing = {}
    if os.path.exists(csv_path) and not force:
        df_old = pd.read_csv(csv_path)
        for _, r in df_old.iterrows():
            existing[(str(r["case"]), round(float(r["aoa"]), 3))] = r.to_dict()

    best_it = m.select_best_feasible_iter(run_dir) or 16
    cases = [
        ("baseline", 1, os.path.join(run_dir, "verification", "it1")),
        ("optimized", best_it, os.path.join(run_dir, "verification", f"it{best_it}")),
    ]

    base_cfg = {
        "mach": 0.15,
        "reynolds": 3000000,
        "turb_model": "SST",
        "trans_model": "NONE",
        "mesh_filename": "mesh.su2",
        "conv_minval": -14,
    }

    t_start = time.time()
    for case_name, it_num, ver_dir in cases:
        mesh_src = os.path.join(ver_dir, "mesh.su2")
        rst_ref = os.path.join(ver_dir, "restart_flow.dat")
        hist_ref = os.path.join(ver_dir, "history.csv")

        if not (os.path.exists(mesh_src) and os.path.exists(rst_ref)):
            raise FileNotFoundError(f"Brak mesh.su2 lub restart_flow.dat w {ver_dir}")

        print(f"\n================ PROFIL: {case_name.upper()} (it{it_num}) ================")

        # Punkt projektowy AoA=2.0 pobierany z gotowej weryfikacji 4500 it
        if 2.0 in angles and os.path.exists(hist_ref):
            m2 = read_polar_metrics(hist_ref, avg_window=300)
            existing[(case_name, 2.0)] = {
                "case": case_name,
                "iter_ref": it_num,
                "aoa": 2.0,
                "status": "verified",
                **m2,
            }
            save_table(list(existing.values()), csv_path)
            print(
                f"AoA=+2.0 deg (verified): CL={m2['cl']:.5f} | CD={m2['cd']:.6f} | "
                f"L/D={m2['ld']:.2f} | CMz={m2['cmz']:+.5f}"
            )

        # Dwie galezie startujace od zbieglego pola przy AoA=2.0
        down_branch = sorted([a for a in angles if a < 2.0], reverse=True)
        up_branch = sorted([a for a in angles if a > 2.0])

        for branch_name, branch in (("down", down_branch), ("up", up_branch)):
            prev_rst = rst_ref
            for aoa in branch:
                key = (case_name, round(float(aoa), 3))
                sub = os.path.join(out_dir, case_name, aoa_tag(aoa))
                sub_rst = os.path.join(sub, "restart_flow.dat")

                if key in existing and existing[key].get("status") == "ok" and not force:
                    print(f"AoA={aoa:+5.1f} deg ({branch_name}): pominieto (juz policzone)")
                    if os.path.exists(sub_rst):
                        prev_rst = sub_rst
                    continue

                os.makedirs(sub, exist_ok=True)
                shutil.copy(mesh_src, os.path.join(sub, "mesh.su2"))
                use_rst = os.path.exists(prev_rst)
                if use_rst:
                    shutil.copy(prev_rst, os.path.join(sub, "solution_flow.dat"))

                cfg = dict(base_cfg, aoa=aoa, restart=use_rst, iters=iters)
                print(f"\n--- {case_name} | AoA={aoa:+5.1f} deg ({branch_name}, iters={iters}) ---")
                m.prepare_config("SU2_CFD", np.zeros(m.N_VARS), cfg, sub)

                try:
                    m.run_su2_module("SU2_CFD", sub, cores=cores)
                    met = read_polar_metrics(os.path.join(sub, "history.csv"), avg_window=200)
                    stable = np.isfinite(met["rms_rho"]) and (met["rms_rho"] < -5.0)
                    status = "ok" if stable else "poor_conv"
                    existing[key] = {
                        "case": case_name,
                        "iter_ref": it_num,
                        "aoa": float(aoa),
                        "status": status,
                        **met,
                    }
                    if stable and os.path.exists(sub_rst):
                        prev_rst = sub_rst
                    print(
                        f"   Wynik -> CL={met['cl']:.5f} | CD={met['cd']:.6f} "
                        f"(drift {met['cd_drift']:.1e}, std {met['cd_std']:.1e}) | "
                        f"L/D={met['ld']:.2f} | CMz={met['cmz']:+.5f} | "
                        f"rms[Rho]={met['rms_rho']:.2f} [{status}]"
                    )
                except Exception as exc:
                    print(f"   BLAD dla {case_name} AoA={aoa}: {exc}")
                    existing[key] = {
                        "case": case_name,
                        "iter_ref": it_num,
                        "aoa": float(aoa),
                        "status": "failed",
                        "iters_run": 0,
                        "cd": np.nan,
                        "cl": np.nan,
                        "cmz": np.nan,
                        "ld": np.nan,
                        "cd_std": np.nan,
                        "cd_drift": np.nan,
                        "rms_rho": np.nan,
                        "rms_w": np.nan,
                    }
                    # Przerwanie galezi w przypadku rozbieznosci wysokiego kata natarcia
                    break
                finally:
                    save_table(list(existing.values()), csv_path)

    print(f"\nZapisano wyniki biegunowej: {csv_path}")
    print(f"Czas calkowity: {datetime.timedelta(seconds=int(time.time() - t_start))}")
    return csv_path


def plot_offdesign_polar(csv_path, out_fig_dir=OUT_FIG_DIR):
    # Generuje 4-panelowy wykres biegunowej off-design w stylu raportu LaTeX
    df = pd.read_csv(csv_path)
    df = df[df["status"].isin(["ok", "verified"])].copy()
    if df.empty:
        print("Brak poprawnych punktow do wyrysowania.")
        return

    b = df[df["case"] == "baseline"].sort_values("aoa")
    o = df[df["case"] == "optimized"].sort_values("aoa")

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
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6))

    # Panel (a): Krzywa sily nosnej CL(alpha)
    ax = axes[0, 0]
    ax.plot(b["aoa"], b["cl"], "o--", color="0.25", ms=4.5, lw=1.4, label="Baseline NACA 0012")
    ax.plot(o["aoa"], o["cl"], "s-", color="#1f77b4", ms=4.8, lw=1.7, label="Optimized (pinned LE/TE)")
    ax.axvline(2.0, color="#d62728", linestyle=":", lw=1.1, label=r"Design point ($\alpha = 2^\circ$)")
    ax.set_xlabel(r"Angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"Lift coefficient $C_L$")
    ax.set_title(r"(a) Lift curve $C_L(\alpha)$")
    ax.legend(loc="upper left")

    # Panel (b): Biegunowa oporu CL vs CD (porownanie przy rownym CL)
    ax = axes[0, 1]
    ax.plot(b["cd"] * 1e4, b["cl"], "o--", color="0.25", ms=4.5, lw=1.4, label="Baseline NACA 0012")
    ax.plot(o["cd"] * 1e4, o["cl"], "s-", color="#1f77b4", ms=4.8, lw=1.7, label="Optimized (pinned LE/TE)")
    b2 = b[np.isclose(b["aoa"], 2.0)]
    o2 = o[np.isclose(o["aoa"], 2.0)]
    if not b2.empty and not o2.empty:
        ax.scatter(
            [b2["cd"].iloc[0] * 1e4, o2["cd"].iloc[0] * 1e4],
            [b2["cl"].iloc[0], o2["cl"].iloc[0]],
            s=55, facecolor="none", edgecolor="#d62728", linewidths=1.3, zorder=6,
            label=r"Design point ($\alpha = 2^\circ$)",
        )
    ax.set_xlabel(r"Drag coefficient $C_D$ [counts, $\times 10^{-4}$]")
    ax.set_ylabel(r"Lift coefficient $C_L$")
    ax.set_title(r"(b) Drag polar $C_L(C_D)$")
    ax.legend(loc="lower right")

    # Panel (c): Doskonalosc L/D(alpha)
    ax = axes[1, 0]
    ax.plot(b["aoa"], b["ld"], "o--", color="0.25", ms=4.5, lw=1.4, label="Baseline NACA 0012")
    ax.plot(o["aoa"], o["ld"], "s-", color="#1f77b4", ms=4.8, lw=1.7, label="Optimized (pinned LE/TE)")
    ax.axvline(2.0, color="#d62728", linestyle=":", lw=1.1, label=r"Design point ($\alpha = 2^\circ$)")
    ax.set_xlabel(r"Angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"Lift-to-drag ratio $L/D$")
    ax.set_title(r"(c) Aerodynamic efficiency $L/D(\alpha)$")
    ax.legend(loc="lower right")

    # Panel (d): Moment pochylajacy CMz(alpha)
    ax = axes[1, 1]
    ax.plot(b["aoa"], b["cmz"], "o--", color="0.25", ms=4.5, lw=1.4, label="Baseline NACA 0012")
    ax.plot(o["aoa"], o["cmz"], "s-", color="#9467bd", ms=4.8, lw=1.7, label="Optimized (pinned LE/TE)")
    ax.axhline(0.05, color="#d62728", linestyle="--", lw=1.1, label=r"Design limit $|C_{Mz}| \leq 0.05$")
    ax.axhline(-0.05, color="#d62728", linestyle="--", lw=1.1)
    ax.axvline(2.0, color="0.45", linestyle=":", lw=1.0, label=r"Design point ($\alpha = 2^\circ$)")
    ax.set_xlabel(r"Angle of attack $\alpha$ [deg]")
    ax.set_ylabel(r"Pitching-moment coefficient $C_{Mz}$")
    ax.set_ylim(-0.058, 0.060)
    ax.set_title(r"(d) Pitching moment $C_{Mz}(\alpha)$")
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.66))

    plt.tight_layout()
    os.makedirs(out_fig_dir, exist_ok=True)
    pdf_path = os.path.join(out_fig_dir, "fig09_offdesign_polar.pdf")
    png_path = os.path.join(out_fig_dir, "fig09_offdesign_polar.png")
    fig.savefig(pdf_path, dpi=350)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Wygenerowano wykres: {pdf_path} oraz {png_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Biegunowa off-design dla profilu bazowego i zoptymalizowanego")
    p.add_argument("--run-dir", default=None, help="Katalog optymalizacji z verification/")
    p.add_argument("--out-dir", default=None, help="Katalog wyjsciowy (domyslnie <run-dir>/offdesign_polar)")
    p.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES)
    p.add_argument("--iters", type=int, default=1500, help="Liczba iteracji CFD na punkt")
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--force", action="store_true", help="Przelicz od nowa juz istniejace punkty")
    p.add_argument("--plot-only", action="store_true", help="Tylko regeneracja wykresu z CSV")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = find_run_dir(args.run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "offdesign_polar")
    csv_path = os.path.join(out_dir, "offdesign_polar.csv")

    if not args.plot_only:
        run_polar_sweep(run_dir, out_dir, args.angles, args.iters, args.cores, force=args.force)

    if os.path.exists(csv_path):
        print("\n=== Podsumowanie offdesign_polar.csv ===")
        print(pd.read_csv(csv_path).to_string(index=False))
        plot_offdesign_polar(csv_path)


if __name__ == "__main__":
    main()