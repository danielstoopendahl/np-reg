from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    (
        REPO_ROOT / "image/results/Exjobb models - CIFAR test.csv",
        "CIFAR-10 Test Accuracy vs Model Size",
    ),
    (
        REPO_ROOT / "language/results/Exjobb models - IMDb test.csv",
        "IMDb Test Accuracy vs Model Size",
    ),
    (
        REPO_ROOT / "tabular/results/Exjobb models - UCI test.csv",
        "UCI HAR Test Accuracy vs Model Size",
    ),
]

METHODS = ["Vanilla", "Layer Norm", "Batch Norm", "Weight Decay", "Dropout", "Np-reg"]


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

    output_path = csv_path.with_name(
        f"{re.sub(r'[^a-z0-9]+', '_', csv_path.stem.lower()).strip('_')}_accuracy_by_model_size.png"
    )

    palette = {
        "Vanilla": "#E07070",
        "Layer Norm": "#5B8BC5",
        "Batch Norm": "#70B070",
        "Weight Decay": "#C58B5B",
        "Dropout": "#8E70C5",
        "Np-reg": "#5BC5B0",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    sns.lineplot(
        data=plot_df,
        x="model_size",
        y="accuracy",
        hue="method",
        style="method",
        markers=True,
        dashes=False,
        linewidth=2.5,
        markersize=7,
        palette=palette,
        ax=ax,
    )

    for method, method_data in plot_df.groupby("method", sort=False):
        method_data = method_data.sort_values("model_size")
        x = method_data["model_size"].to_numpy()
        y = method_data["accuracy"].to_numpy()
        s = method_data["std"].to_numpy()
        ax.fill_between(x, y - s, y + s, color=palette[method], alpha=0.18)

    # Add vertical line for #parameters = #datapoints
    if "CIFAR" in csv_path.name:
        x_line = 16.223
        ax.axvline(x=x_line, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax.text(x_line, -0.025, "$N_{train}$", ha='center', va='top',  color="black", alpha=0.8,
                transform=ax.get_xaxis_transform())
    elif "IMDb" in csv_path.name:
        x_line = 11.314
        ax.axvline(x=x_line, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax.text(x_line, -0.025, "$N_{train}$", ha='center', va='top',  color="black", alpha=0.8,
                transform=ax.get_xaxis_transform())
    else:  # UCI
        x_line = 12.966
        ax.axvline(x=x_line, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        ax.text(x_line, -0.025, "$N_{train}$", ha='center', va='top', color="black", alpha=0.8,
                transform=ax.get_xaxis_transform())

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend(title="Method", frameon=True, loc='lower right')

    # Set custom x-axis labels based on dataset
    if "CIFAR" in csv_path.name:
        size_labels = {8192: "25M", 2048: "6.3M", 512: "1.6M", 128: "390k", 32: "99k", 8: "25k"}
    elif "IMDb" in csv_path.name:
        size_labels = {256: "13M", 128: "3.2M", 64: "800k", 32: "200k", 16: "50k", 8: "13k"}
    else:  # UCI
        size_labels = {8192: "4.6M", 2048: "120k", 512: "290k", 128: "73k", 32: "18k", 8: "4.5k"}

    unique_sizes = sorted(plot_df["model_size"].unique())
    ax.set_xticks(unique_sizes)
    ax.set_xticklabels([size_labels.get(int(size), str(int(size))) for size in unique_sizes])

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    for csv_path, title in DATASETS:
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        output_path = plot_csv(csv_path, title)
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
