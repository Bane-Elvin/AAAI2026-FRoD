import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_landscapes_init_trained():
    x = np.linspace(-0.2, 0.2, 21)
    y = np.linspace(-0.2, 0.2, 21)

    files_and_titles = [
        ("initial_frod_0.01_surface_1.npy", "Initial FRoD"),
        ("trained_frod_0.01_surface_1.npy", "Trained FRoD"),
    ]

    fig = plt.figure(figsize=(8, 4))
    n_rows, n_cols = 1, 2

    plt.rcParams.update({'font.size': 10})
    label_id = ["a", "b"]
    for idx, (filename, title) in enumerate(files_and_titles):

        try:
            z = np.load(filename)
        except FileNotFoundError:
            print(f"File {filename} not found. Creating a dummy array.")

            if "initial" in filename:
                z = np.sin(X * 5) + np.cos(Y * 5) + np.random.rand(*X.shape) * 0.1 + 5.5
            else:
                z = (X ** 2 + Y ** 2) * 5 + np.random.rand(*X.shape) * 0.01

        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, z.T, cmap='viridis', linewidth=0, antialiased=True)

        ax.set_xlabel('Direction 1 (α)', fontsize=12)
        ax.set_ylabel('Direction 2 (β)', fontsize=12)

        ax.set_title("(" + label_id[idx] + ") " + title, fontsize=14, ha='center', va='top', y=-0.18)

        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)

        ax.zaxis.set_tick_params(pad=5)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.15, wspace=0.05, hspace=0.12)

    plt.savefig("plot_Frod_landscapes_0.01_init_trained.pdf")
    plt.show()

def plot_landscapes_in_all():
    x = np.linspace(-0.2, 0.2, 21)
    y = np.linspace(-0.2, 0.2, 21)

    files_and_titles = [
        ("initial_frod_0.1_surface_0.npy", "Initial FRoD"),
        ("initial_frod_0.1_surface_1.npy", "Initial FRoD"),
        ("initial_frod_0.1_surface_2.npy", "Initial FRoD"),
        ("initial_frod_0.1_surface_3.npy", "Initial FRoD"),
        ("trained_frod_0.1_surface_0.npy", "Trained FRoD"),
        ("trained_frod_0.1_surface_1.npy", "Trained FRoD"),
        ("trained_frod_0.1_surface_2.npy", "Trained FRoD"),
        ("trained_frod_0.1_surface_3.npy", "Trained FRoD"),
    ]


    fig = plt.figure(figsize=(16, 8))
    n_rows, n_cols = 2, 4


    plt.rcParams.update({'font.size': 10})
    label_id = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for idx, (filename, title) in enumerate(files_and_titles):

        try:
            z = np.load(filename)
        except FileNotFoundError:
            print(f"File {filename} not found. Creating a dummy array.")

            if "initial" in filename:
                z = np.sin(X*5) + np.cos(Y*5) + np.random.rand(*X.shape)*0.1 + 5.5
            else:
                z = (X**2 + Y**2) * 5 + np.random.rand(*X.shape)*0.01

        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        X, Y = np.meshgrid(x, y)
        surf = ax.plot_surface(X, Y, z.T, cmap='viridis', linewidth=0, antialiased=True)


        ax.set_xlabel('Direction 1 (α)', fontsize=14)
        ax.set_ylabel('Direction 2 (β)', fontsize=14)

        ax.set_title("(" + label_id[idx] + ") " + title, fontsize=16, ha='center', va='top', y=-0.18)


        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)

        ax.zaxis.set_tick_params(pad=5)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.15, wspace=0.05, hspace=0.12)

    plt.savefig("plot_Frod_landscapes_0.1_in_all.pdf")
    plt.show()

if __name__ == "__main__":
    plot_landscapes_init_trained()
    # plot_landscapes_in_all()
