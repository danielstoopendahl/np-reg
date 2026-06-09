from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    (
        REPO_ROOT / "image/results/CIFAR test.csv",
        "CIFAR-10",
    ),
    (
        REPO_ROOT / "language/results/IMDb test.csv",
        "IMDb",
    ),
    (
        REPO_ROOT / "tabular/results/UCI HAR test.csv",
        "UCI HAR",
    ),
]

METHODS = ["Vanilla", "Weight Decay", "Dropout", "Layer Norm", "Batch Norm", "NP-reg"]


@dataclass
class Block:
    size: int
    method_values: dict[str, float]
    method_std: dict[str, float]


def parse_percent(value: str) -> float:
    value = value.strip().replace("%", "")
    return float(value)


def parse_model_size(label: str) -> int:
    label = label.strip()
    slash_match = re.search(r"/(\d+)$", label)
    if slash_match:
        return int(slash_match.group(1))

    number_match = re.search(r"(\d+)$", label)
    if not number_match:
        raise ValueError(f"Could not parse model size from '{label}'")

    return int(number_match.group(1))


def parse_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))

    header_sizes = [cell.strip() for cell in rows[0]]
    data_rows = rows[2:]
    mean_row = next(row for row in data_rows if row and row[0] == "Mean")
    std_row = next(row for row in data_rows if row and row[0] == "Std")

    blocks: list[Block] = []
    block_count = sum(1 for cell in header_sizes[::8] if cell)
    for block_idx in range(block_count):
        offset = block_idx * 8
        method_cols = [offset + i + 1 for i in range(len(METHODS))]
        size = parse_model_size(header_sizes[offset])

        # Extract the repeated block-specific mean/std values from the right columns.
        mean_values = {method: parse_percent(mean_row[col]) for method, col in zip(METHODS, method_cols)}
        std_values = {method: parse_percent(std_row[col]) for method, col in zip(METHODS, method_cols)}

        blocks.append(Block(size=size, method_values=mean_values, method_std=std_values))

    mean_records = []
    std_records = []
    for block in blocks:
        for method in METHODS:
            mean_records.append(
                {
                    "model_size": block.size,
                    "method": method,
                    "accuracy": block.method_values[method],
                }
            )
            std_records.append(
                {
                    "model_size": block.size,
                    "method": method,
                    "std": block.method_std[method],
                }
            )

    return pd.DataFrame(mean_records), pd.DataFrame(std_records)


def plot_csv(csv_path: Path, title: str) -> Path:
    means, stds = parse_csv(csv_path)
    plot_df = means.merge(stds, on=["model_size", "method"])
    plot_df["method"] = plot_df["method"].replace({"Np-reg": "NP-reg"})

    output_path = csv_path.with_name(
        f"{re.sub(r'[^a-z0-9]+', '_', csv_path.stem.lower()).strip('_')}_accuracy_by_model_size.pdf"
    )

    palette = {
        "Vanilla": "#C37238",
        "Weight Decay": "#926942",
        "Dropout": "#386463",
        "Layer Norm": "#C0B76F",
        "Batch Norm": "#829750",
        "NP-reg": "#789EB8",
    }

    # Enable markers for all lines, then explicitly set NP-reg to triangle.

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)

    legend_methods = ["Vanilla", "Weight Decay", "Dropout", "Layer Norm", "Batch Norm", "NP-reg"]

    sns.lineplot(
        data=plot_df,
        x="model_size",
        y="accuracy",
        hue="method",
        style="method",
        hue_order=legend_methods,
        style_order=legend_methods,
        markers=True,
        dashes=False,
        linewidth=2.7,
        alpha=0.9,
        markersize=10,
        palette=palette,
        ax=ax,
    )

    # Robustly set markers on plotted lines by matching label or color.
    custom_markers = {
        "NP-reg": "^",
        "Layer Norm": "s",
        "Dropout": "P",
    }
    target_rgbas = {}
    for cl in custom_markers:
        try:
            target_rgbas[cl] = mcolors.to_rgba(palette[cl])
        except Exception:
            pass

    for line in ax.get_lines():
        lab = (line.get_label() or "")
        color = line.get_color()
        matched_method = None
        if lab in custom_markers:
            matched_method = lab
        else:
            try:
                rgba = mcolors.to_rgba(color)
                for cl, trgba in target_rgbas.items():
                    if np.allclose(rgba[:3], trgba[:3], atol=1e-2):
                        matched_method = cl
                        break
            except Exception:
                pass

        if matched_method:
            line.set_marker(custom_markers[matched_method])
            if matched_method == "Layer Norm":
                line.set_markersize(8)
            else:
                line.set_markersize(10)
            line.set_markeredgewidth(0.8)
            line.set_markerfacecolor(line.get_color())

    for method, method_data in plot_df.groupby("method", sort=False):
        method_data = method_data.sort_values("model_size")
        x = method_data["model_size"].to_numpy()
        y = method_data["accuracy"].to_numpy()
        s = method_data["std"].to_numpy()
        ax.fill_between(x, y - s, y + s, color=palette[method], alpha=0.18)

    if "Vanilla" not in plot_df["method"].unique():
        raise ValueError("Could not find 'Vanilla' in method names.")

    vanilla_lookup = plot_df[plot_df["method"] == "Vanilla"][
        ["model_size", "accuracy", "std"]
    ].rename(columns={"accuracy": "vanilla_mean", "std": "vanilla_std"})

    diff_df = plot_df.merge(vanilla_lookup, on="model_size", how="left")
    diff_df["pct_increase"] = (
        (diff_df["accuracy"] - diff_df["vanilla_mean"]) / diff_df["vanilla_mean"]
    ) * 100.0
    diff_df["pct_increase_std"] = 100.0 * np.sqrt(
        (diff_df["std"] / diff_df["vanilla_mean"]) ** 2
        + (((diff_df["accuracy"] - diff_df["vanilla_mean"]) * diff_df["vanilla_std"]) ** 2)
        / (diff_df["vanilla_mean"] ** 4)
    )

    sns.lineplot(
        data=diff_df,
        x="model_size",
        y="pct_increase",
        hue="method",
        style="method",
        hue_order=legend_methods,
        style_order=legend_methods,
        markers=True,
        dashes=False,
        linewidth=2.7,
        alpha=0.9,
        markersize=10,
        palette=palette,
        ax=ax2,
    )

    for line in ax2.get_lines():
        lab = (line.get_label() or "")
        color = line.get_color()
        matched_method = None
        if lab in custom_markers:
            matched_method = lab
        else:
            try:
                rgba = mcolors.to_rgba(color)
                for cl, trgba in target_rgbas.items():
                    if np.allclose(rgba[:3], trgba[:3], atol=1e-2):
                        matched_method = cl
                        break
            except Exception:
                pass

        if matched_method:
            line.set_marker(custom_markers[matched_method])
            if matched_method == "Layer Norm":
                line.set_markersize(8)
            else:
                line.set_markersize(10)
            line.set_markeredgewidth(0.8)
            line.set_markerfacecolor(line.get_color())

    for method, method_data in diff_df.groupby("method", sort=False):
        method_data = method_data.sort_values("model_size")
        x = method_data["model_size"].to_numpy()
        y = method_data["pct_increase"].to_numpy()
        s = method_data["pct_increase_std"].to_numpy()
        ax2.fill_between(x, y - s, y + s, color=palette[method], alpha=0.18)

    # Determine the N_train x position for each dataset
    if "CIFAR" in csv_path.name:
        x_ntrain = 16.223
    elif "IMDb" in csv_path.name:
        x_ntrain = 11.314
    else:  # UCI
        x_ntrain = 12.966

    def reduce_y_ticks(axis: plt.Axes) -> None:
        ticks = axis.get_yticks()
        if len(ticks) > 1:
            axis.set_yticks(ticks[::2])

    ax.set_xscale("log", base=2)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", length=4)
    ax.set_xlabel("Model Size (#Parameters)")
    ax.set_ylabel("Accuracy (%)")
    def legend_rowwise(labels: list[str], ncol: int) -> list[str]:
        nrow = int(np.ceil(len(labels) / ncol))
        ordered = []
        for col in range(ncol):
            for row in range(nrow):
                idx = row * ncol + col
                if idx < len(labels):
                    ordered.append(labels[idx])
        return ordered

    legend_labels = legend_rowwise(legend_methods, ncol=3)
    handles, labels = ax.get_legend_handles_labels()
    handle_map = {label: handle for handle, label in zip(handles, labels)}
    ordered_labels = [label for label in legend_labels if label in handle_map]
    ordered_handles = [handle_map[label] for label in ordered_labels]
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=3,
        frameon=True,
    )

    ax2.set_xscale("log", base=2)
    ax2.grid(False)
    ax2.tick_params(axis="both", which="both", direction="out", length=4)
    ax2.set_xlabel("Model Size (#Parameters)")
    ax2.set_ylabel("% Increase Over Vanilla")
    handles, labels = ax2.get_legend_handles_labels()
    handle_map = {label: handle for handle, label in zip(handles, labels)}
    ordered_labels = [label for label in legend_labels if label in handle_map]
    ordered_handles = [handle_map[label] for label in ordered_labels]
    ax2.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        ncol=3,
        frameon=True,
    )

    # Set custom x-axis labels based on dataset
    if "CIFAR" in csv_path.name:
        size_labels = {8192: "25M", 2048: "6.3M", 512: "1.6M", 128: "390k", 32: "99k", 8: "25k"}
    elif "IMDb" in csv_path.name:
        size_labels = {256: "13M", 128: "3.2M", 64: "800k", 32: "200k", 16: "50k", 8: "13k"}
    else:  # UCI
        size_labels = {8192: "4.6M", 2048: "1.2M", 512: "290k", 128: "73k", 32: "18k", 8: "4.5k"}

    unique_sizes = sorted(plot_df["model_size"].unique())
    unique_sizes = sorted(plot_df["model_size"].unique())
    all_ticks = sorted(unique_sizes + [x_ntrain])
    tick_labels = [
        "$N_{train}$" if t == x_ntrain else size_labels.get(int(t), str(int(t)))
        for t in all_ticks
    ]
    for axis in (ax, ax2):
        axis.set_xticks(all_ticks)
        axis.set_xticklabels(tick_labels)

    reduce_y_ticks(ax)

    if "UCI" in csv_path.name:
        ax2.set_yticks([-1, 0, 1, 2, 3])
    elif "IMDb" in csv_path.name:
        ax2.set_yticks([-0.8, -0.4, 0, 0.4])
    else:
        reduce_y_ticks(ax2)

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    sns.set_theme(
        style="ticks",
        context="paper",
        rc={
            "font.size": 15,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.grid": False,
        },
    )
    for csv_path, title in DATASETS:
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        output_path = plot_csv(csv_path, title)
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
