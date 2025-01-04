import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from os import listdir
from os.path import isfile, join

##############################################################################
# 1) Data Extraction
##############################################################################

def extract_data(source_dir, data_name, print_data=False):
    """
    Reads all .json files in <source_dir>/<data_name>, extracting relevant
    information into a pandas DataFrame.

    Parameters
    ----------
    source_dir : str
        Directory containing subdirectories named after datasets.
    data_name : str
        Name of the dataset folder to look into.
    print_data : bool
        Whether to print debug info.

    Returns
    -------
    pd.DataFrame
        DataFrame containing:
        (dataset, test runs, optimizer, activation, vderiv, accuracy, loss, time).
    """
    d = []
    source_path = Path(source_dir) / data_name
    files = [f for f in listdir(source_path) if isfile(source_path / f)]
    for file in files:
        full_path = source_path / file
        with open(full_path, "r") as json_file:
            json_data = json.load(json_file)
            if print_data:
                print("Opened", str(full_path))

        # If the JSON is a list, handle the first element if present
        if isinstance(json_data, list) and len(json_data) > 0:
            json_data = json_data[0]

        # Fallback to 'relu' if activation not found
        activation = json_data.get("activation", "relu")

        # If top-level "results" exist, rename to 'vderivs' for consistency
        if 'results' in json_data:
            json_data['vderivs'] = json_data.pop('results')

        # For each vderiv entry, retrieve the metrics
        for v in json_data['vderivs']:
            data_row = {
                'dataset': json_data['dataset'],
                'test runs': json_data['test runs'],
                'optimizer': json_data['optimizer'],
                'activation': activation,
                'vderiv': float(v['vderiv']),
                'accuracy': float(v['avg accuracy']),
            }
            # If extra metrics exist, store them
            if 'avg time' in v:
                data_row['time'] = float(v['avg time'])
            if 'avg loss' in v:
                data_row['loss'] = float(v['avg loss'])

            d.append(data_row)

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(d)
    # Sort by vderiv ascending (for cleaner plots)
    df.sort_values(by='vderiv', inplace=True, ignore_index=True)

    return df


##############################################################################
# 2) Plotting Utilities
##############################################################################

def _save_plot(fig, plot_path_base):
    """
    Saves the Matplotlib figure both as .png and .eps in the given path base.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    plot_path_base : str
        The path (excluding extension) where the figure will be saved.
    """
    # Save as PNG
    fig.savefig(plot_path_base + ".png", bbox_inches="tight", dpi=300)
    # Save as EPS
    fig.savefig(plot_path_base + ".eps", bbox_inches="tight", dpi=300)
    print(f"Plots saved: {plot_path_base}.png and .eps")


def plot_acc_vs_dataset(df, output_dir):
    """
    For each (optimizer, activation) pair, find the maximum accuracy across
    all derivative orders for each dataset. Plot these maxima vs. dataset.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns:
        [dataset, test runs, optimizer, activation, vderiv, accuracy].
    output_dir : str
        Base directory to save the plots.
    """
    # Ensure the top-level output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # List all datasets in df
    all_datasets = df['dataset'].unique()
    # Group by (optimizer, activation)
    group_oa = df.groupby(['optimizer', 'activation'])

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    # We'll pick a distinct color for each (optimizer, activation) pair
    color_cycle = plt.get_cmap('tab20')
    idx = 0

    for (optimizer, activation), group_df in group_oa:
        # We want a single point (the maximum accuracy) per dataset
        max_acc_list = []
        ds_list = []

        for ds in all_datasets:
            sub_df = group_df[group_df['dataset'] == ds]
            if len(sub_df) == 0:
                continue
            max_acc = sub_df['accuracy'].max()
            ds_list.append(ds)
            max_acc_list.append(max_acc)

        c = color_cycle(idx % 20)
        idx += 1
        # Plot the line
        ax.plot(ds_list, max_acc_list, label=f"{optimizer}, {activation}", color=c, marker='o')

    # Final formatting
    ax.set_xticklabels(all_datasets, rotation=20, ha='right', fontsize=8)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Max Accuracy Over All vderiv")
    ax.set_title("Max Accuracy per (Optimizer, Activation) across Datasets")
    ax.legend(fontsize=8)

    fig.tight_layout()
    # Save
    plot_path_base = str(Path(output_dir) / "max_accuracy_vs_dataset")
    _save_plot(fig, plot_path_base)
    #plt.show()
    plt.close(fig)


def plot_vderiv_vs_metric(df, output_dir, metric="accuracy"):
    """
    Plots 'metric' (accuracy, loss, time, etc.) as a function of 'vderiv'
    for each combination of (optimizer, activation, dataset).

    For fractional optimizers (opt_name starts with 'F'), we also attempt to
    find a matching non-fractional optimizer to overlay on the same plot
    for visual comparison.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns:
        [dataset, test runs, optimizer, activation, vderiv, accuracy, (loss/time optional)]
    output_dir : str
        Base directory for saving plots. Figures are saved under:
        <output_dir>/<dataset>/<optimizer>/<activation>/
    metric : str
        Name of the metric to plot (e.g., "accuracy", "loss", "time").
    """
    # Unique datasets in the DataFrame
    datasets = df["dataset"].unique()

    # Ensure the top-level output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # For color cycling
    color_cycle = plt.get_cmap("tab20")

    # For each dataset
    for ds in datasets:
        df_ds = df[df["dataset"] == ds]
        if df_ds.empty:
            continue

        # For each (optimizer, activation) pair
        group_oa = df_ds.groupby(["optimizer", "activation"])
        for (optimizer, activation), group_df in group_oa:
            # We will create a line plot of metric vs. vderiv
            fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

            # Sort by vderiv for a smooth line
            group_df_sorted = group_df.sort_values(by="vderiv")

            # If the chosen metric doesn't exist, skip
            if metric not in group_df_sorted.columns:
                plt.close(fig)
                continue

            # Plot fractional or regular line
            clr = color_cycle(0)
            mrk = 'D' if optimizer.startswith('F') else 'x'
            ax.plot(group_df_sorted["vderiv"], group_df_sorted[metric],
                    marker=mrk, color=clr,
                    label=f"{optimizer}, {activation}, vderiv",
                    linewidth=1, markersize=6)

            # If fractional, see if there's a matching "standard" optimizer:
            if optimizer.startswith('F'):
                std_opt = optimizer[1:].lower()  # e.g. "adam"
                std_group_df = df_ds[
                    (df_ds["optimizer"].str.lower() == std_opt) &
                    (df_ds["activation"] == activation)
                ].sort_values(by="vderiv")
                if not std_group_df.empty and metric in std_group_df.columns:
                    clr2 = color_cycle(1)
                    ax.plot(std_group_df["vderiv"], std_group_df[metric],
                            marker='x', color=clr2,
                            label=f"{std_opt}, {activation}, vderiv",
                            linewidth=1, markersize=8)

            ax.set_xlabel("Fractional Derivative Order (vderiv)")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{ds} - {optimizer}, {activation} [{metric}]")
            ax.legend(fontsize=8)
            ax.grid(True)

            # Save figure in dataset/optimizer/activation subfolder
            out_folder = Path(output_dir) / ds / optimizer / activation
            out_folder.mkdir(parents=True, exist_ok=True)

            plot_file_base = str(out_folder / f"{metric}_vs_vderiv")
            fig.tight_layout()
            _save_plot(fig, plot_file_base)
            #plt.show()
            plt.close(fig)


##############################################################################
# 3) Demonstration of Usage
##############################################################################

if __name__ == "__main__":
    # Example usage:

    # 1) Define source directory and output folder for plots
    source_dir = "results_10_runs"  # or wherever your merged JSON files live
    output_plots_dir = "plots"      # We'll place generated figures here

    # Ensure the top-level plotting directory exists
    Path(output_plots_dir).mkdir(parents=True, exist_ok=True)

    # 2) Define your list of datasets (subfolders in source_dir)
    datasets = [
        "diabetes",
        "blood-transfusion-service-center",
        "tae",
        "pendigits",
        "balance-scale",
        "tic-tac-toe",
        "vertebra-column",
        "vehicle",
        "climate-model-simulation-crashes"
    ]

    # 3) Extract Data + Plot
    df_all = pd.DataFrame()
    for ds in datasets:
        df_ds = extract_data(source_dir, ds, print_data=False)
        df_all = pd.concat([df_all, df_ds], ignore_index=True)

    # Now we have a single df_all containing every dataset’s results

    # Plot 1: Maximum accuracy vs. dataset
    plot_acc_vs_dataset(df_all, output_dir=output_plots_dir)

    # Plot 2: metric vs. derivative order for each (optimizer, activation),
    # separated by dataset. Possible metrics: "accuracy", "loss", "time" (if present).
    plot_vderiv_vs_metric(df_all, output_dir=output_plots_dir, metric="accuracy")
    if "loss" in df_all.columns:
        plot_vderiv_vs_metric(df_all, output_dir=output_plots_dir, metric="loss")
    if "time" in df_all.columns:
        plot_vderiv_vs_metric(df_all, output_dir=output_plots_dir, metric="time")
