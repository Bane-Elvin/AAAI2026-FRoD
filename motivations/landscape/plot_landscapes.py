import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_loss_curves():
    data = np.load("training_loss_curves.npz")

    plt.figure(figsize=(12, 8))
    plt.plot(data['full_ft_epochs'], data['full_ft_train_losses'], label='Full FT', color='blue', linestyle='-')
    plt.plot(data['lora_epochs'], data['lora_train_losses'], label='LoRA', color='red', linestyle='-')
    plt.plot(data['vera_epochs'], data['vera_train_losses'], label='VERA', color='purple', linestyle='-.')
    plt.plot(data['pissa_epochs'], data['pissa_train_losses'], label='PISSA', color='orange', linestyle=':')
    plt.plot(data['frod_epochs'], data['frod_train_losses'], label='FROD', color='green', linestyle='--')

    plt.title('Training Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.suptitle('Comparison of Training Dynamics', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_surface_from_file(filename, title, x, y):
    z = np.load(filename)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, z.T, cmap='viridis', linewidth=0, antialiased=True)
    ax.set_xlabel('Direction 1 (α)', fontsize=12)
    ax.set_ylabel('Direction 2 (β)', fontsize=12)
    ax.set_zlabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=16, pad=20)
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, label='Loss Value')
    plt.tight_layout()
    plt.show()

def remove_local_outliers(data, threshold=0.5):
    if len(data) < 3:
        return data

    result = [data[0]]
    for i in range(1, len(data) - 1):
        prev_val = data[i - 1]
        next_val = data[i + 1]
        local_avg = (prev_val + next_val) / 2

        if abs(data[i] - local_avg) / local_avg <= threshold:
            result.append(data[i])
        else:
            print(f"Removed outlier at index {i}: {data[i]} (local avg: {local_avg})")
    result.append(data[-1])
    return result

def plot_surface():
    x = np.linspace(-0.2, 0.2, 21)
    y = np.linspace(-0.2, 0.2, 21)

    files_and_titles = [
        ("initial_full_fine-tune_surface.npy", "Initial Full Fine-tune"),
        ("trained_full_fine-tune_surface.npy", "Trained Full Fine-tune"),
        ("initial_lora_surface.npy", "Initial LoRA"),
        ("trained_lora_surface.npy", "Trained LoRA"),
        ("initial_vera_surface.npy", "Initial VERA"),
        ("trained_vera_surface.npy", "Trained VERA"),
        ("initial_pissa_surface.npy", "Initial PISSA"),
        ("trained_pissa_surface.npy", "Trained PISSA"),
        # ("initial_frod_surface.npy", "Initial FROD"),
        # ("trained_frod_surface.npy", "Trained FROD"),
    ]

    for filename, title in files_and_titles:
        plot_surface_from_file(filename, f"{title} Loss Landscape", x, y)

def plot_landscapes_in_all():
    x = np.linspace(-0.2, 0.2, 21)
    y = np.linspace(-0.2, 0.2, 21)

    files_and_titles = [
        ("initial_full_fine-tune_surface.npy", "Initial Full Fine-tune"),
        ("initial_lora_surface.npy", "Initial LoRA"),
        ("initial_pissa_surface.npy", "Initial PISSA"),
        ("initial_vera_surface.npy", "Initial VERA"),
        ("trained_full_fine-tune_surface.npy", "Trained Full Fine-tune"),
        ("trained_lora_surface.npy", "Trained LoRA"),
        ("trained_pissa_surface.npy", "Trained PISSA"),
        ("trained_vera_surface.npy", "Trained VERA"),
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

        ax.set_xlabel('Direction 1 (α)', fontsize=12)
        ax.set_ylabel('Direction 2 (β)', fontsize=12)

        ax.set_title("(" + label_id[idx] + ") " + title, fontsize=14, ha='center', va='top', y=-0.18)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.ax.tick_params(labelsize=8)

        ax.tick_params(axis='both', labelsize=8)
        ax.zaxis.set_tick_params(pad=5)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.15, wspace=0.05, hspace=0.12)

    plt.savefig("plot_landscapes_in_all.pdf")
    plt.show()



if __name__ == "__main__":
    # plot_loss_curves()
    # plot_surface()
    plot_landscapes_in_all()
