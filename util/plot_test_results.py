from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

CSV_PATH = Path(__file__).with_name("Exjobb models - UCI test.csv")
OUTPUT_PATH = Path(__file__).with_name("uci_test_accuracy_by_model_size.png")

MODEL_SIZES = [8192, 2048, 512, 128, 32, 8]
METHODS = ["Vanilla", "Layer Norm", "Batch Norm", "Weight Decay", "Dropout", "Np-reg"]


@dataclass
class Block:
    size: int
    method_values: dict[str, float]
    method_std: dict[str, float]


def parse_percent(value: str) -> float:
    value = value.strip().replace("%", "")
    return float(value)


def parse_csv(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with path.open(newline="") as f:
        rows = list(csv.reader(f))

    header_sizes = [cell.strip() for cell in rows[0]]
    header_methods = [cell.strip() for cell in rows[1]]
    data_rows = rows[2:]

    blocks: list[Block] = []
    for block_idx, size in enumerate(MODEL_SIZES):
        offset = block_idx * 8
        method_cols = [offset + i + 1 for i in range(len(METHODS))]

        mean_row = next(row for row in data_rows if row and row[0] == "Mean")
        std_row = next(row for row in data_rows if row and row[0] == "Std")

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


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    means, stds = parse_csv(CSV_PATH)

    palette = {
        "Vanilla": "#E07070",
        "Layer Norm": "#5B8BC5",
        "Batch Norm": "#70B070",
        "Weight Decay": "#C58B5B",
        "Dropout": "#8E70C5",
        "Np-reg": "#5BC5B0",
    }

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in METHODS:
        method_means = means[means["method"] == method].sort_values("model_size")
        method_stds = stds[stds["method"] == method].sort_values("model_size")

        x = method_means["model_size"].to_numpy()
        y = method_means["accuracy"].to_numpy()
        s = method_stds["std"].to_numpy()
        color = palette[method]

        sns.lineplot(x=x, y=y, ax=ax, color=color, linewidth=2.0, marker="o", label=method)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Model size")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("UCI Test Accuracy vs Model Size")
    ax.legend(title="Method", frameon=True)

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
