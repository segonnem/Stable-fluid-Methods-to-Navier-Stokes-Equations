import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmasher as cmr

N_TIME_STEPS = 1000  # À ajuster
N_POINTS = 2**7-1  # À ajuster
DOMAIN_SIZE = 1  # Votre valeur de taille de domaine

x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
X_plot, Y_plot = np.meshgrid(x, y, indexing='ij')

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=160)

# Charger et prétraiter les données
curls_data1 = np.loadtxt("gpu_simu1.csv", delimiter=',').reshape(-1, N_POINTS, N_POINTS)
curls_data2 = np.loadtxt("gpu_simu_spin.csv", delimiter=',').reshape(-1, N_POINTS, N_POINTS)

def update_frame(i):
    axes[0].clear()
    axes[1].clear()
    axes[0].contourf(X_plot, Y_plot, curls_data1[i], cmap=cmr.redshift, levels=100)
    axes[1].contourf(X_plot, Y_plot, curls_data2[i], cmap=cmr.redshift, levels=100)

# Ajustement de la plage des frames
frames = range(0, min(curls_data1.shape[0], curls_data2.shape[0], N_TIME_STEPS - 200), 1)

ani = animation.FuncAnimation(fig, update_frame, frames=frames, interval=0.01, blit=True)

ani.save("simulation2.gif", writer='pillow', fps=120)




