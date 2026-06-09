from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

PALETTE = {
    "Vanilla": "#C37238",
    "BN/LN/None + D + WD": "#926942",
    "NP-reg": "#789EB8",
    "NP-reg + D + WD": "#386463",
}

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
        "opt": "BN/LN/None + D + WD",
        "np": "NP-reg",
        "np + opt": "NP-reg + D + WD",
    }
    df["model"] = df["model"].map(model_labels).fillna(df["model"])
    # We'll enable markers for all lines, then explicitly set NP-reg to triangle.

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
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 4), constrained_layout=True)

    sns.lineplot(
        data=df,
        x="fraction",
        y="mean",
        hue="model",
        style="model",
        markers=True,
        palette=PALETTE, 
        dashes=False,
        linewidth=2.2,
        markersize=10,
        ax=ax,
    )

    # Force NP-reg to use triangle marker on the actual plotted lines.
    try:
        target_rgba = mcolors.to_rgba(PALETTE["NP-reg"])
    except Exception:
        target_rgba = None
    for line in ax.get_lines():
        lab = (line.get_label() or "")
        color = line.get_color()
        match = False
        # Match exact label 'NP-reg' to avoid matching 'NP-reg + D + WD'
        if lab == "NP-reg":
            match = True
        elif target_rgba is not None:
            try:
                # compare RGB (ignore alpha)
                rgba = mcolors.to_rgba(color)
                if np.allclose(rgba[:3], target_rgba[:3], atol=1e-2):
                    match = True
            except Exception:
                pass
        if match:
            line.set_marker("^")
            line.set_markersize(10)
            line.set_markeredgewidth(0.8)
            line.set_markerfacecolor(line.get_color())

    for model, g in df.groupby("model"):
        g = g.sort_values("fraction")
        ax.fill_between(
            g["fraction"],
            g["mean"] - g["std"],
            g["mean"] + g["std"],
            alpha=0.2,
        )

    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(False)
    ax.tick_params(axis="both", which="both", direction="out", length=4)
    ax.set_xticks(fraction_labels)
    ax.set_xticklabels(["100%", "30%", "10%", "3%", "1%"])
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(title="Model")
    # Reduce number of y-ticks for cleaner readability
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

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
        markers=True,
        palette=PALETTE, 
        dashes=False,
        linewidth=2.2,
        markersize=10,
        ax=ax2,
    )

    for line in ax2.get_lines():
        lab = (line.get_label() or "")
        color = line.get_color()
        match = False
        if lab == "NP-reg":
            match = True
        elif target_rgba is not None:
            try:
                rgba = mcolors.to_rgba(color)
                if np.allclose(rgba[:3], target_rgba[:3], atol=1e-2):
                    match = True
            except Exception:
                pass
        if match:
            line.set_marker("^")
            line.set_markersize(10)
            line.set_markeredgewidth(0.8)
            line.set_markerfacecolor(line.get_color())

    for model, g in diff_df.groupby("model"):
        g = g.sort_values("fraction")
        ax2.fill_between(
            g["fraction"],
            g["pct_increase"] - g["pct_increase_std"],
            g["pct_increase"] + g["pct_increase_std"],
            alpha=0.2,
            color=PALETTE.get(model, None),
        )

    ax2.set_xscale("log")
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.grid(False)
    ax2.tick_params(axis="both", which="both", direction="out", length=4)
    ax2.set_xticks(fraction_labels)
    ax2.set_xticklabels(["100%", "30%", "10%", "3%", "1%"])
    ax2.set_xlabel("Training Data Fraction")
    ax2.set_ylabel("% Increase Over Vanilla")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    ax2.legend(title="Model", loc="upper right")
    # Reduce number of y-ticks for cleaner readability
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

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