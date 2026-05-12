from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_finetuning(csv_path: Path, output_path: Path) -> None:
    raw = pd.read_csv(csv_path, header=None)

    fraction_labels = [1.0, 0.3, 0.1, 0.03, 0.01]
    model_names = raw.iloc[1].tolist()

    mean_row = raw[raw.iloc[:, 0] == "Mean"].iloc[0]
    std_row = raw[raw.iloc[:, 0] == "Std"].iloc[0]

    valid_cols = [
        c for c in raw.columns
        if isinstance(mean_row[c], str) and "%" in mean_row[c]
    ]

    expected_cols = len(fraction_labels) * 4
    if len(valid_cols) != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} percentage columns, found {len(valid_cols)}. "
            "Check the CSV format or update the parser."
        )

    records = []
    for idx, col in enumerate(valid_cols):
        frac = fraction_labels[idx // 4]
        model = model_names[col]
        mean_str = str(mean_row[col]).replace("%", "")
        std_str = str(std_row[col]).replace("%", "")
        mean_val = float(mean_str) / 100.0
        std_val = float(std_str) / 100.0
        records.append({"fraction": frac, "model": model, "mean": mean_val, "std": std_val})

    df = pd.DataFrame(records)

    model_labels = {
        "Vanilla": "Vanilla",
        "opt": "normalizing + D + WD",
        "np": "NP",
        "np + opt": "NP + D + WD",
    }
    df["model"] = df["model"].map(model_labels).fillna(df["model"])
    marker_styles = {
        "Vanilla": "o",
        "normalizing + D + WD": "s",
        "NP": "D",
        "NP + D + WD": "^",
    }

    sns.set_theme(style="whitegrid")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)

    sns.lineplot(
        data=df,
        x="fraction",
        y="mean",
        hue="model",
        style="model",
        markers=marker_styles,
        dashes=False,
        linewidth=2.2,
        markersize=7,
        ax=ax,
    )

    for model, g in df.groupby("model"):
        g = g.sort_values("fraction")
        ax.fill_between(
            g["fraction"],
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            alpha=0.2,
        )

    ax.set_xscale("log")
    ax.set_xticks(fraction_labels)
    ax.set_xticklabels(["100%", "30%", "10%", "3%", "1%"])
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Accuracy")
    ax.legend(title="Model")

    if "Vanilla" not in df["model"].unique():
        raise ValueError("Could not find 'Vanilla' in model names.")

    vanilla_means = df[df["model"] == "Vanilla"][["fraction", "mean"]]
    vanilla_means = vanilla_means.rename(columns={"mean": "vanilla_mean"})
    vanilla_stds = df[df["model"] == "Vanilla"][["fraction", "std"]]
    vanilla_stds = vanilla_stds.rename(columns={"std": "vanilla_std"})

    diff_df = df.merge(vanilla_means, on="fraction", how="left")
    diff_df = diff_df.merge(vanilla_stds, on="fraction", how="left")
    diff_df["pct_increase"] = (
        (diff_df["mean"] - diff_df["vanilla_mean"]) / diff_df["vanilla_mean"]
    ) * 100.0
    diff_df["pct_increase_std"] = 100.0 * np.sqrt(
        (diff_df["std"] / diff_df["vanilla_mean"]) ** 2
        + (((diff_df["mean"] - diff_df["vanilla_mean"]) * diff_df["vanilla_std"]) ** 2)
        / (diff_df["vanilla_mean"] ** 4)
    )

    sns.lineplot(
        data=diff_df,
        x="fraction",
        y="pct_increase",
        hue="model",
        style="model",
        markers=marker_styles,
        dashes=False,
        linewidth=2.2,
        markersize=7,
        ax=ax2,
    )

    for model, g in diff_df.groupby("model"):
        g = g.sort_values("fraction")
        ax2.fill_between(
            g["fraction"],
            g["pct_increase"] - g["pct_increase_std"],
            g["pct_increase"] + g["pct_increase_std"],
            alpha=0.2,
        )

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xticks(fraction_labels)
    ax2.set_xticklabels(["100%", "30%", "10%", "3%", "1%"])
    ax2.set_xlabel("Training data fraction")
    ax2.set_ylabel("% increase over Vanilla")
    ax2.legend(title="Model")

    fig.savefig(output_path, format="pdf")


base_dir = Path(__file__).resolve().parents[1]
plot_finetuning(
    base_dir / "image_finetuning" / "results" / "Exjobb models - Food101 test.csv",
    base_dir / "image_finetuning" / "results" / "food101_finetuning_plots.pdf",
)
plot_finetuning(
    base_dir / "language_finetuning" / "results" / "Exjobb models - Yelp test.csv",
    base_dir / "language_finetuning" / "results" / "yelp_finetuning_plots.pdf",
)