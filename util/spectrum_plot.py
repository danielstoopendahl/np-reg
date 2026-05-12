from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    "np": "NP",
    "vanilla": "Vanilla",
}
PALETTE = {
    "Vanilla": "#E07070",
    "NP": "#70B070",
}


def parse_model_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if "_spectrum_" in stem:
        prefix = stem.split("_spectrum_")[0]
    else:
        prefix = stem.split("_")[0]
    return MODEL_LABELS.get(prefix, prefix.replace("-", " ").title())


def extract_stage(path: Path, payload: dict) -> str:
    stage = payload.get("stage")
    if stage:
        return stage
    stem = path.stem
    if "_spectrum_" in stem:
        return stem.split("_spectrum_")[1]
    return "unknown"


def load_spectra(input_dir: Path) -> dict[str, dict[str, list[float]]]:
    spectra: dict[str, dict[str, list[float]]] = {}
    for path in sorted(input_dir.glob("*_spectrum_*.json")):
        with path.open() as f:
            payload = json.load(f)
        values = payload.get("singular_values", [])
        if not values:
            continue
        if any(values[i] < values[i + 1] for i in range(len(values) - 1)):
            values = sorted(values, reverse=True)

        stage = extract_stage(path, payload)
        prefix = path.stem.split("_spectrum_")[0] if "_spectrum_" in path.stem else path.stem.split("_")[0]

        model = parse_model_from_filename(path.name)
        spectra.setdefault(stage, {})[model] = values

    return spectra


def plot_spectra(
    spectra: dict[str, dict[str, list[float]]],
    output_path: Path,
    log_y: bool,
    log_x: bool,
    show: bool,
) -> None:
    sns.set_theme(style="whitegrid")
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
            x = range(1, len(values) + 1)
            sns.lineplot(
                x=x,
                y=values,
                label=model,
                linewidth=1.8,
                color=PALETTE.get(model),
                alpha=0.7,
                ax=ax,
            )

        ax.set_title(STAGE_LABELS.get(stage, stage.replace("_", " ").title()))
        ax.set_xlabel("Singular value")
        ax.set_ylabel("Rank")
        ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

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
    sns.set_theme(style="whitegrid")
    ordered_datasets = [label for label in DATASET_SPECTRA_DIRS if label in spectra_by_dataset]
    ordered_datasets += [
        label for label in spectra_by_dataset if label not in ordered_datasets
    ]
    if not ordered_datasets:
        print("No final spectra found to plot.")
        return

    fig, axes = plt.subplots(
        1,
        len(ordered_datasets),
        figsize=(5 * len(ordered_datasets), 4),
        squeeze=False,
    )
    axes = axes[0]

    for idx, dataset_label in enumerate(ordered_datasets):
        ax = axes[idx]
        spectra = spectra_by_dataset[dataset_label]
        stages = [stage for stage in STAGE_ORDER if stage in spectra and stage in FINAL_STAGES]
        if not stages:
            continue
        stage = stages[0]
        for model, values in spectra[stage].items():
            x = range(1, len(values) + 1)
            sns.lineplot(
                x=x,
                y=values,
                label=model,
                linewidth=1.8,
                color=PALETTE.get(model),
                alpha=0.7,
                ax=ax,
            )

        ax.set_title(dataset_label)
        ax.set_xlabel("Singular value")
        ax.set_ylabel("Rank")
        if log_y:
            ax.set_yscale("log")
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot singular value spectra from JSON files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing *_spectrum_*.json files.",
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
        spectra_by_dataset[dataset_label] = load_spectra(input_dir)

    default_output = REPO_ROOT / "image/results/singular_value_spectra_all.pdf"
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
