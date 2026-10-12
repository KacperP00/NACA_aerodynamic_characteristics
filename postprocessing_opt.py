import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_summary(work_dir, hist_dir):
    # Generuje raport wykonawczy (Executive Summary) i animacje GIF na podstawie logow z optymalizacji.
    metrics_path = os.path.join(work_dir, "opt_metrics.csv")
    
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(df['iter'], df['cd'], 'b.-', label='Opor (CD)')
        ax1.set_title('Zbieznosc Wspolczynnika Oporu')
        ax1.set_xlabel('Iteracja')
        ax1.set_ylabel('CD')
        ax1.grid(True)
        ax1.legend()
        
        ax2.plot(df['iter'], df['cl'], 'g.-', label='Sila nosna (CL)')
        ax2.axhline(y=0.25, color='r', linestyle='--', label='Limit Kary (CL=0.25)')
        ax2.set_title('Utrzymanie Sily Nosnej')
        ax2.set_xlabel('Iteracja')
        ax2.set_ylabel('CL')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(work_dir, "executive_summary.png"), dpi=300)
        plt.close()
        print("-> Wygenerowano executive_summary.png")

    surf_files = sorted(glob.glob(os.path.join(hist_dir, "surf_*.csv")))
    if not surf_files:
        return

    # Inicjalizacja narzedzi animacji (GIF)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.25, 0.25)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_title("Ewolucja ksztaltu - Optymalizacja FFD")
    
    line, = ax.plot([], [], 'b-', linewidth=2)
    
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        surf = pd.read_csv(frame)
        surf.columns = surf.columns.str.strip()
        # Wyciaganie kolumn PointX i PointY zdefiniowanych przez logi SU2
        x = surf['"PointX"'].values
        y = surf['"PointY"'].values
        
        # Sortowanie punktow po obwodzie profilu
        angle = np.arctan2(y, x - 0.5)
        idx = np.argsort(angle)
        x, y = x[idx], y[idx]
        
        line.set_data(x, y)
        ax.set_title(f"Iteracja: {os.path.basename(frame).split('_')[1].split('.')[0]}")
        return line,

    ani = FuncAnimation(fig, update, frames=surf_files, init_func=init, blit=True, repeat=False)
    gif_path = os.path.join(work_dir, "airfoil_evolution.gif")
    ani.save(gif_path, writer='pillow', fps=5)
    plt.close()
    print(f"-> Wygenerowano animacje: {gif_path}")