from __future__ import annotations

import argparse
import json
from math import fsum
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "tabular/results"
DATASET_SPECTRA_DIRS = {
    "CIFAR-10": REPO_ROOT / "image/results",
    "IMDb": REPO_ROOT / "language/results",
    "UCI HAR": REPO_ROOT / "tabular/results",
}

STAGE_ORDER = ["before_training", "epoch_100", "best_val"]
FINAL_STAGES = {"best_val"}
STAGE_LABELS = {
    "before_training": "Before training",
    "epoch_100": "Epoch 100",
    "best_val": "Best val",
}
MODEL_LABELS = {
    "np": "NP-reg",
    "vanilla": "Vanilla",
    "bn": "Batch Norm",
}
PALETTE = {
    "Vanilla": "#C37238",
    "Batch Norm": "#829750",
    "NP-reg": "#789EB8",
}


def parse_model_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_hidden_singular_values" in stem:
        suffix = stem.split("_hidden_singular_values", 1)[1].lstrip("_")
        if not suffix:
            return "Model"
        return format_model_label(suffix)
    if "_spectrum_" in stem:
        prefix = stem.split("_spectrum_")[0]
    else:
        prefix = stem.split("_")[0]
    return format_model_label(prefix)


def extract_stage(path: Path, payload: dict) -> str:
    stage = payload.get("stage")
    if stage:
        return stage
    stem = path.stem
    if "_spectrum_" in stem:
        return stem.split("_spectrum_")[1]
    if "_hidden_singular_values" in stem:
        return "best_val"
    return "unknown"


def format_model_label(raw: str) -> str:
    tokens = raw.replace("-", "_").split("_")
    parts = []
    for token in tokens:
        if not token:
            continue
        parts.append(MODEL_LABELS.get(token, token.title()))
    if not parts:
        return "Model"
    return " ".join(parts)


def load_singular_values(path: Path) -> tuple[list[float], dict | None]:
    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        return payload.get("singular_values", []), payload
    return [], None



import math
def compute_effective_rank(values: list[float]) -> float:
    # Information-theoretic effective rank: exp(-sum p_i log p_i)
    if not values:
        return 0.0
    total = sum(values)
    if total <= 0:
        return 0.0
    p = [v / total for v in values if v > 0]
    entropy = -sum(pi * math.log(pi) for pi in p)
    return math.exp(entropy)


def load_spectra(input_dir: Path, prefer_hidden: bool = True) -> dict[str, dict[str, list[float]]]:
    spectra: dict[str, dict[str, list[float]]] = {}
    hidden_paths = list(input_dir.glob("*_hidden_singular_values*.json"))
    spectrum_paths = list(input_dir.glob("*_spectrum_*.json"))
    if prefer_hidden and hidden_paths:
        paths = hidden_paths
    else:
        paths = spectrum_paths + hidden_paths
    for path in sorted(paths):
        values, payload = load_singular_values(path)
        if not values:
            continue
        if any(values[i] < values[i + 1] for i in range(len(values) - 1)):
            values = sorted(values, reverse=True)

        stage = extract_stage(path, payload or {})
        model = parse_model_from_filename(path.name)
        spectra.setdefault(stage, {})[model] = values

    return spectra


def reduce_y_ticks(ax: plt.Axes, log_y: bool) -> None:
    y_ticks = ax.get_yticks()
    if len(y_ticks) <= 1:
        return
    target = max(2, len(y_ticks) // 2)
    if log_y:
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=target))
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=target))


def plot_spectra(
    spectra: dict[str, dict[str, list[float]]],
    output_path: Path,
    log_y: bool,
    log_x: bool,
    show: bool,
) -> None:
    stages = [stage for stage in STAGE_ORDER if stage in spectra and stage in FINAL_STAGES]
    stages += sorted(
        stage for stage in spectra if stage not in stages and stage in FINAL_STAGES
    )
    if not stages:
        print("No final spectra found to plot.")
        return

    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4), squeeze=False)
    axes = axes[0]

    for idx, stage in enumerate(stages):
        ax = axes[idx]
        for model, values in spectra[stage].items():
            eff_rank = compute_effective_rank(values)
            print(f"Effective rank | stage={stage} model={model}: {eff_rank:.6f}")
            x = range(1, len(values) + 1)
            sns.lineplot(
                x=x,
                y=values,
                label=model,
                linewidth=2.4,
                color=PALETTE.get(model),
                alpha=0.9,
                ax=ax,
            )

        ax.set_title(STAGE_LABELS.get(stage, stage.replace("_", " ").title()))
        ax.set_xlabel("Index")
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.grid(False)
        ax.tick_params(axis="both", which="both", direction="out", length=4)
        reduce_y_ticks(ax, log_y=log_y)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_spectra_by_dataset(
    spectra_by_dataset: dict[str, dict[str, dict[str, list[float]]]],
    output_path: Path,
    log_y: bool,
    log_x: bool,
    show: bool,
) -> None:
    processed_spectra_by_dataset = split_small_spectra_by_dataset(spectra_by_dataset)
    base_labels = [label for label in DATASET_SPECTRA_DIRS if label in processed_spectra_by_dataset]
    extra_labels = [label for label in processed_spectra_by_dataset if label not in base_labels]
    base_labels += [label for label in extra_labels if not label.endswith(" (small)")]
    small_labels = [f"{label} (small)" for label in base_labels]
    ordered_datasets = base_labels + small_labels
    if not base_labels:
        print("No final spectra found to plot.")
        return

    ncols = max(1, len(base_labels))
    nrows = 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )
    for idx, dataset_label in enumerate(ordered_datasets):
        row = 0 if idx < len(base_labels) else 1
        col = idx if idx < len(base_labels) else idx - len(base_labels)
        ax = axes[row][col]
        spectra = processed_spectra_by_dataset.get(dataset_label)
        if not spectra:
            ax.set_visible(False)
            continue
        stages = [stage for stage in STAGE_ORDER if stage in spectra and stage in FINAL_STAGES]
        if not stages:
            continue
        stage = stages[0]
        for model, values in spectra[stage].items():
            eff_rank = compute_effective_rank(values)
            print(
                f"Effective rank | dataset={dataset_label} stage={stage} model={model}: {eff_rank:.6f}"
            )
            x = range(1, len(values) + 1)
            sns.lineplot(
                x=x,
                y=values,
                label=model,
                linewidth=2.4,
                color=PALETTE.get(model),
                alpha=0.9,
                ax=ax,
            )

        ax.set_title(dataset_label)
        ax.set_xlabel("Index")
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.grid(False)
        ax.tick_params(axis="both", which="both", direction="out", length=4)
        reduce_y_ticks(ax, log_y=log_y)
        # Remove microticks (minor ticks) for small plots
        if dataset_label.endswith("(small)"):
            ax.minorticks_off()

    for row_axes in axes:
        for ax in row_axes:
            if not ax.has_data() and ax.get_visible():
                ax.set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def split_small_spectra_by_dataset(
    spectra_by_dataset: dict[str, dict[str, dict[str, list[float]]]]
) -> dict[str, dict[str, dict[str, list[float]]]]:
    updated = dict(spectra_by_dataset)
    for dataset_label, spectra in spectra_by_dataset.items():
        regular: dict[str, dict[str, list[float]]] = {}
        small: dict[str, dict[str, list[float]]] = {}
        for stage, models in spectra.items():
            regular_models: dict[str, list[float]] = {}
            small_models: dict[str, list[float]] = {}
            for model, values in models.items():
                if "Small" in model.split():
                    label = model[:-6] if model.endswith(" Small") else model
                    small_models[label] = values
                else:
                    regular_models[model] = values
            if regular_models:
                regular[stage] = regular_models
            if small_models:
                small[stage] = small_models

        if small:
            updated[dataset_label] = regular
            updated[f"{dataset_label} (small)"] = small

    return updated


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
    parser = argparse.ArgumentParser(description="Plot singular value spectra from JSON files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing *_spectrum_*.json or *_hidden_singular_values*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: input dir / singular_value_spectra.png).",
    )
    parser.add_argument(
        "--linear-y",
        action="store_true",
        help="Use a linear y-axis instead of log scale.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a log scale on the x-axis.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        input_dir = args.input_dir
        if not input_dir.exists():
            print(f"Input directory does not exist: {input_dir}")
            return
        output_path = args.output or (input_dir / "singular_value_spectra.pdf")
        spectra = load_spectra(input_dir)
        plot_spectra(
            spectra,
            output_path,
            log_y=not args.linear_y,
            log_x=args.log_x,
            show=args.show,
        )
        return

    spectra_by_dataset = {}
    for dataset_label, input_dir in DATASET_SPECTRA_DIRS.items():
        if not input_dir.exists():
            print(f"Input directory does not exist: {input_dir}")
            continue
        spectra_by_dataset[dataset_label] = load_spectra(input_dir, prefer_hidden=True)

    default_output = REPO_ROOT / "util/results/singular_value_spectra_all.pdf"
    output_path = args.output or default_output
    plot_spectra_by_dataset(
        spectra_by_dataset,
        output_path,
        log_y=not args.linear_y,
        log_x=args.log_x,
        show=args.show,
    )


if __name__ == "__main__":
    main()
