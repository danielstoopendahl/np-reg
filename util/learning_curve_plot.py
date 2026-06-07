import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ── Colour palette ────────────────────────────────────────────────────────────
# Each model needs two colours: [train, val].
# Swap these for any hex codes, named colours, or seaborn palette names you like.
PALETTE = {
    "model1": {
        "train": "#C37238",   # muted blue  – train loss (dashed)
        "val":   "#C37238",   # same hue    – val loss   (solid)
        "label": "Vanilla",
    },
    "model2": {
        "train": "#789EB8",   # muted orange – train loss (dashed)
        "val":   "#789EB8",   # same hue     – val loss   (solid)
        "label": "NP-reg",
    },
}


def plot_learning_curves(
    file1_path,
    file2_path,
    palette: dict = PALETTE,
    save_path=None,
    style: str = "ticks",
    context: str = "paper",
    ax=None,
    title: str | None = None,
    show: bool = True,
    show_legend: bool = True,
):
    """
    Plot learning curves from two CSV files using seaborn.

    Parameters
    ----------
    file1_path : str | Path
        Path to the first CSV (expects columns: epoch, train_loss, val_loss,
        val_accuracy).
    file2_path : str | Path
        Path to the second CSV (same schema).
    palette : dict
        Colour palette dict (see PALETTE at the top of this file).
        Keys: "model1", "model2".  Each sub-dict needs "train", "val", "label".
    save_path : str | Path | None
        If given, the figure is written to this path at 300 dpi.
    style : str
        Any seaborn style: "darkgrid", "whitegrid", "dark", "white", "ticks".
    context : str
        Any seaborn context: "paper", "notebook", "talk", "poster".
    ax : matplotlib.axes.Axes | None
        Optional axes to draw on. If None, a new figure is created.
    title : str | None
        Optional plot title override.
    show : bool
        If True, calls plt.show(). Set False when drawing subplots.
    show_legend : bool
        If True, draw a legend on the axes.
    """
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    lw = 2.3
    dash_pattern = (5, 2)          # (ink, gap) in points

    for df, key in [(df1, "model1"), (df2, "model2")]:
        p = palette[key]
        name = p["label"]

        # Training loss – dashed
        sns.lineplot(
            data=df, x="epoch", y="train_loss",
            ax=ax, color=p["train"], linewidth=lw,
            linestyle="--", dashes=dash_pattern,
            label=f"{name} - Train",
            legend=show_legend,
        )
        # Validation loss – solid
        sns.lineplot(
            data=df, x="epoch", y="val_loss",
            ax=ax, color=p["val"], linewidth=lw,
            linestyle="-",
            label=f"{name} - Val",
            legend=show_legend,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.tick_params(axis="both", which="both")
    ticks = ax.get_yticks()
    if len(ticks) > 1:
        ax.set_yticks(ticks[::2])
    ax.set_title(title or "Training and Validation Loss")
    if show_legend:
        ax.legend(loc="upper right", frameon=True)
    else:
        existing_legend = ax.get_legend()
        if existing_legend is not None:
            existing_legend.remove()

    plt.tight_layout()



# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]

    configs = [
       
        {
            "name": "CIFAR-10",
            "file1": root / "image/results/learning_curve_vanilla.csv",
            "file2": root / "image/results/learning_curve_np.csv",
        },
        {
            "name": "IMDb",
            "file1": root / "language/results/learning_curve_vanilla.csv",
            "file2": root / "language/results/learning_curve_np.csv",
        },
        {
            "name": "HAR",
            "file1": root / "tabular/results/learning_curve_har_vanilla.csv",
            "file2": root / "tabular/results/learning_curve_har_np.csv",
        },
    ]


    missing = [c for c in configs if not (c["file1"].exists() and c["file2"].exists())]

    if not missing:
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
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, cfg in zip(axes, configs):
            plot_learning_curves(
                cfg["file1"],
                cfg["file2"],
                palette=PALETTE,
                save_path=None,
                style="ticks",
                context="paper",
                ax=ax,
                title=cfg["name"],
                show=False,
                show_legend=True,
            )
        plt.tight_layout()
        fig.savefig("results/learning_curves_all_domains.pdf", bbox_inches="tight")
        print("Figure saved to learning_curves_all_domains.pdf")
    else:
        print("Please update the file paths to your CSV files.")
        for cfg in missing:
            for f in (cfg["file1"], cfg["file2"]):
                status = "Exists" if Path(f).exists() else "Not found"
                print(f"  {f} - {status}")