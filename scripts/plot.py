import argparse
import os

import matplotlib.pyplot as plt
from mpssba.visualization.data_loader import RunDataLoader
from mpssba.visualization.plotters import (
    plot_deltas,
    plot_learning_curves,
    plot_masses,
    plot_covariance,
    plot_learning_time,
)


def main():
    parser = argparse.ArgumentParser(description="Unified Plotting for MPS Pipeline")
    parser.add_argument(
        "--runs", nargs="+", required=True, help="List of run directories to plot"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels for the runs in the legend (must match the number of runs)",
    )
    parser.add_argument(
        "--plot_type",
        choices=["masses", "learning_curves", "deltas", "gradients", "covariance", "learning_time", "all"],
        default="all",
        help="What to plot",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-1,
        help="Error threshold for determining when a correlation is learned",
    )

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError("Number of labels must match number of runs.")

    labels = (
        args.labels
        if args.labels
        else [os.path.basename(os.path.normpath(r)) for r in args.runs]
    )

    runs = [RunDataLoader(r) for r in args.runs]
    masking_types = set(r.config.training.masking.type for r in runs)

    if len(masking_types) > 1:
        raise ValueError(
            "All runs must have the same masking type for combined plotting."
        )

    masking_type = masking_types.pop()

    

    if args.plot_type in ["masses", "all"]:
        plot_masses(runs, labels, masking_type)

    if args.plot_type in ["learning_curves", "all"]:
        plot_learning_curves(runs, labels)

    if args.plot_type in ["deltas", "all"]:
        plot_deltas(runs, labels, masking_type)

    if args.plot_type in ["covariance", "all"]:
        plot_covariance(runs, labels, masking_type)

    if args.plot_type in ["learning_time", "all"]:
        plot_learning_time(runs, labels, args.threshold)

    plt.show()


if __name__ == "__main__":
    main()
