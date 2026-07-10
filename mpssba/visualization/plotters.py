from math import comb

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from mpssba.masks import build_boolean_mask, calculate_correlation_length
from mpssba.visualization.data_loader import BatchedData

plt.style.use(["science", "no-latex"])

HW_PADDING = 3
CORR_PADDING = 3

MASKING_TYPE_TO_LABEL = {
    "hamming": "Hamming Weight",
    "correlation": "Correlation Length",
    "corr_2body": "Correlation Length (2-body)",
}


def _plot_mean_std(ax, df_mean, df_std, label_prefix=""):
    for col in df_mean.columns:
        if col == "Epoch" or col == "Exact":
            continue

        p = ax.plot(df_mean.index, df_mean[col], label=f"{label_prefix}{col}")
        if df_std is not None and col in df_std:
            ax.fill_between(
                df_mean.index,
                df_mean[col] - df_std[col],
                df_mean[col] + df_std[col],
                alpha=0.2,
                color=p[0].get_color(),
            )


def plot_learning_curves(runs, labels):
    fig, ax = plt.subplots(figsize=(8, 5))

    for run, label in zip(runs, labels):
        batch_data = run.get_learning_curves()
        _plot_mean_std(ax, batch_data.mean, batch_data.std, label_prefix=f"{label} - ")

    ax.set_title("MPS Learning Curves", fontsize=14)
    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    return fig


def plot_masses(runs, labels, masking_type):
    fig, ax = plt.subplots(figsize=(8, 5))

    if runs:
        batch_data = runs[0].get_masses_data()
        df_mean = batch_data.mean
        df_std = batch_data.std

        if df_std is not None:
            ax.errorbar(
                df_mean.index,
                df_mean["Exact"],
                yerr=df_std["Exact"],
                label="Ideal Distribution",
                color="black",
                linewidth=2,
                marker="o",
                capsize=3,
            )
        else:
            ax.plot(
                df_mean["Exact"].index,
                df_mean["Exact"],
                label="Ideal Distribution",
                color="black",
                marker="o",
            )

    for run, label in zip(runs, labels):
        batch_data = run.get_masses_data()
        df_mean = batch_data.mean
        df_std = batch_data.std

        for col in df_mean.columns:
            if col != "Exact":
                if df_std is not None:
                    ax.errorbar(
                        df_mean.index,
                        df_mean[col],
                        yerr=df_std[col],
                        label=f"{label} - {col}",
                        linestyle="--",
                        marker="x",
                        capsize=3,
                    )
                else:
                    ax.plot(
                        df_mean.index,
                        df_mean[col],
                        label=f"{label} - {col}",
                        linestyle="--",
                        marker="x",
                    )

    ax.set_title("Masses of Fourier Characters", fontsize=14)
    ax.set_xlabel(
        MASKING_TYPE_TO_LABEL.get(masking_type, "NO_LABEL_FOUND"), fontsize=12
    )
    ax.set_ylabel("Mass $M_h(p)$", fontsize=12)
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_deltas(runs, labels, masking_type):
    for run, label in zip(runs, labels):
        n_qubits = run.config.model.n_qubits
        _, mask_name = build_boolean_mask(run.config, n_qubits)

        try:
            delta_batch = run.get_wht_deltas_history(mask_name)
        except FileNotFoundError:
            print(f"Skipping plot_deltas for {label} - not found.")
            continue

        fig, axis = plt.subplots(figsize=(8, 6))
        df_mean = delta_batch.mean
        df_std = delta_batch.std

        limit = run.config.training.masking.value
        padding = HW_PADDING if masking_type == "hamming" else CORR_PADDING
        
        epochs = df_mean.index.values
        for col in df_mean.columns:
            # col format: "h=1" or "crl=12"
            try:
                val = int(col.split("=")[1])
            except (IndexError, ValueError):
                continue
                
            if val > limit + padding:
                continue
                
            try:
                color = plt.cm.viridis(val / n_qubits)
            except Exception:
                color = None

            p = axis.plot(epochs, df_mean[col], label=col.replace("=", " = "), color=color)
            if df_std is not None:
                axis.fill_between(
                    epochs,
                    df_mean[col] - df_std[col],
                    df_mean[col] + df_std[col],
                    alpha=0.2,
                    color=p[0].get_color(),
                )

        axis.set_title(f"$\Delta_h$ vs Epoch ({label})")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("$\Delta_h$")
        axis.grid(True, linestyle="--", alpha=0.7)
        axis.axhline(0, color="black", linewidth=1)
        axis.legend(
            title="Fourier Weight $h$", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        fig.tight_layout()


def plot_covariance(runs, labels, masking_type):
    n_runs = len(runs)
    if n_runs == 0:
        return

    cols = min(2, n_runs)
    rows = (n_runs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, (run, label) in enumerate(zip(runs, labels)):
        ax = axes[idx]
        n_qubits = run.config.model.n_qubits
        _, mask_name = build_boolean_mask(run.config, n_qubits)

        try:
            cov_err_batch = run.get_cov_error_history(mask_name)
        except FileNotFoundError:
            print(
                f"Skipping covariance plot for {label} - Covariance error history not found."
            )
            ax.set_visible(False)
            continue

        df_mean = cov_err_batch.mean

        epochs = df_mean.index.values
        distances = df_mean.columns.values

        # We need a 2D array of shape (len(distances), len(epochs))
        error_matrix = df_mean.values.T

        # --- Figure 1: Relative Error ---
        c = ax.pcolormesh(
            epochs,
            distances,
            error_matrix,
            shading="nearest",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
        )
        fig.colorbar(c, ax=ax, label="Mean Relative Covariance Error", extend="max")

        ax.set_title(f"Covariance Error Lightcone ({label})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Distance $|i - j|$")
        ax.set_yticks(distances)

    for i in range(n_runs, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()


def plot_learning_time(runs, labels, threshold=0.1):
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted_any = False
    for run, label in zip(runs, labels):
        n_qubits = run.config.model.n_qubits
        _, mask_name = build_boolean_mask(run.config, n_qubits)

        try:
            cov_err_batch = run.get_cov_error_history(mask_name)
        except FileNotFoundError:
            print(
                f"Skipping learning time plot for {label} - Covariance error history not found."
            )
            continue

        df_mean = cov_err_batch.mean
        distances = df_mean.columns.values

        learning_epochs = []
        valid_distances = []

        for d in distances:
            learned = df_mean[d] <= threshold
            if learned.any():
                epoch_learned = df_mean[learned].index[0]
                learning_epochs.append(epoch_learned)
                valid_distances.append(d)

        if valid_distances:
            ax.plot(
                valid_distances,
                learning_epochs,
                marker="o",
                linestyle="-",
                label=f"{label}",
            )
            plotted_any = True

    if plotted_any:
        ax.set_title(
            f"Correlation Learning Time (Threshold = {threshold * 100}%)", fontsize=14
        )
        ax.set_xlabel("Distance $|i - j|$", fontsize=12)
        ax.set_ylabel("Epoch Learned", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
    else:
        print(
            f"Warning: No runs reached the threshold {threshold}. Try a higher threshold."
        )
        plt.close(fig)
        return None

    return fig
