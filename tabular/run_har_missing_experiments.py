#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RESULT_RE = re.compile(
    r"RESULT\s+mean_val_acc=([0-9eE+\-.]+)\s+mean_val_loss=([0-9eE+\-.]+)"
)

SECTION_CANONICAL = {
    "vanilla": "vanilla",
    "layer norm": "layer_norm",
    "batch norm": "batch_norm",
    "np-reg": "np_reg",
    "weight decay": "weight_decay",
    "dropout": "dropout",
}


@dataclass(frozen=True)
class PendingExperiment:
    row_idx: int
    block_idx: int
    start_col: int
    hidden_dim: int
    section: str
    batch_size: int
    learning_rate: float
    param_value: float | None
    acc_col_idx: int
    loss_col_idx: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run pending UCI HAR grid experiments from the final CSV and fill "
            "accuracy/validation loss with fixed seed runs."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=script_dir / "Exjobb models - UCI HAR fixed epochs.csv",
        help="Path to the HAR results CSV.",
    )
    parser.add_argument(
        "--target-script",
        type=Path,
        default=script_dir / "har.py",
        help="Training script to execute.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run the target script.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=script_dir / "grid_results" / "har_missing" / "logs",
        help="Directory for per-run logs.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Re-run rows that already have both accuracy and validation loss.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on number of runs to execute (for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list pending runs without executing them.",
    )
    return parser.parse_args()


def normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def parse_hidden_dim(cell: str) -> int | None:
    raw = cell.strip()
    if not raw:
        return None
    if not re.fullmatch(r"\d+", raw):
        return None
    return int(raw)


def is_missing(cell: str) -> bool:
    return cell.strip() == ""


def parse_number(cell: str) -> float | None:
    text = cell.strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def discover_blocks(rows: list[list[str]]) -> list[tuple[int, int]]:
    if not rows:
        raise RuntimeError("CSV appears to be empty.")

    block_info: list[tuple[int, int]] = []
    for idx, cell in enumerate(rows[0]):
        hidden_dim = parse_hidden_dim(cell)
        if hidden_dim is not None:
            block_info.append((idx, hidden_dim))

    if not block_info:
        raise RuntimeError("Could not detect hidden-dimension blocks in row 1.")

    return block_info


def discover_experiments(rows: list[list[str]], rerun_completed: bool) -> list[PendingExperiment]:
    blocks = discover_blocks(rows)
    current_section: str | None = None
    pending: list[PendingExperiment] = []

    for row_idx in range(1, len(rows)):
        row = rows[row_idx]

        first_block_col, _ = blocks[0]
        first_cell = row[first_block_col] if first_block_col < len(row) else ""
        section_candidate = SECTION_CANONICAL.get(normalize_label(first_cell))
        if section_candidate is not None:
            current_section = section_candidate
            continue

        if current_section is None:
            continue

        for block_idx, (start_col, hidden_dim) in enumerate(blocks):
            cells = [row[start_col + i] if start_col + i < len(row) else "" for i in range(5)]
            c0, c1, c2, c3, c4 = cells

            if normalize_label(c3) == "accuracy" and normalize_label(c4) == "val loss":
                continue

            if current_section in {"vanilla", "layer_norm", "batch_norm"}:
                batch_size = parse_number(c1)
                learning_rate = parse_number(c2)
                param_value = None
            else:
                param_value = parse_number(c0)
                batch_size = parse_number(c1)
                learning_rate = parse_number(c2)

            if batch_size is None or learning_rate is None:
                continue
            if current_section in {"np_reg", "weight_decay", "dropout"} and param_value is None:
                continue

            if not rerun_completed and (not is_missing(c3)) and (not is_missing(c4)):
                continue

            pending.append(
                PendingExperiment(
                    row_idx=row_idx,
                    block_idx=block_idx,
                    start_col=start_col,
                    hidden_dim=hidden_dim,
                    section=current_section,
                    batch_size=int(batch_size),
                    learning_rate=learning_rate,
                    param_value=param_value,
                    acc_col_idx=start_col + 3,
                    loss_col_idx=start_col + 4,
                )
            )

    return pending


def build_command(args: argparse.Namespace, exp: PendingExperiment, target_script: Path) -> list[str]:
    cmd = [
        args.python_bin,
        str(target_script),
        "--hidden-dim",
        str(exp.hidden_dim),
        "--batch-size",
        str(exp.batch_size),
        "--lr",
        str(exp.learning_rate),
        "--seed",
        str(args.seed),
    ]

    if exp.section == "layer_norm":
        cmd.append("--layer-norm")
    elif exp.section == "batch_norm":
        cmd.append("--batch-norm")
    elif exp.section == "np_reg":
        cmd.extend(["--np-reg-lambda", str(exp.param_value)])
    elif exp.section == "weight_decay":
        cmd.extend(["--weight-decay", str(exp.param_value)])
    elif exp.section == "dropout":
        cmd.extend(["--dropout", str(exp.param_value)])

    return cmd


def parse_result_metrics(stdout: str) -> tuple[float, float]:
    match = RESULT_RE.search(stdout)
    if not match:
        raise RuntimeError("Could not parse RESULT line with mean validation metrics.")
    mean_val_acc = float(match.group(1))
    mean_val_loss = float(match.group(2))
    return mean_val_acc, mean_val_loss


def write_csv(csv_path: Path, rows: list[list[str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def run_experiment(
    args: argparse.Namespace,
    exp: PendingExperiment,
    target_script: Path,
    work_dir: Path,
    log_file: Path,
) -> tuple[float, float]:
    # Only run the experiment once, as har.py now does 5-fold CV internally
    cmd = build_command(args, exp, target_script)
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    log_file.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Experiment failed at row={exp.row_idx + 1}, block={exp.block_idx + 1}, "
            f"exit={completed.returncode}."
        )
    acc, loss = parse_result_metrics(completed.stdout)
    return acc, loss


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.resolve()
    target_script = args.target_script.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not target_script.exists():
        raise FileNotFoundError(f"Target script not found: {target_script}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    pending = discover_experiments(rows, rerun_completed=args.rerun_completed)
    print(f"Found {len(pending)} experiments to run in {csv_path}")

    if args.max_runs is not None:
        pending = pending[: args.max_runs]
        print(f"Limiting execution to first {len(pending)} runs due to --max-runs")

    if not pending:
        print("Nothing to run. CSV already complete.")
        return

    if args.dry_run:
        for i, exp in enumerate(pending, start=1):
            print(
                f"[{i}/{len(pending)}] row={exp.row_idx + 1} block={exp.block_idx + 1} "
                f"section={exp.section} hidden_dim={exp.hidden_dim} "
                f"bs={exp.batch_size} lr={exp.learning_rate} param={exp.param_value}"
            )
        return

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    work_dir = target_script.parent

    for i, exp in enumerate(pending, start=1):
        print(
            f"[{i}/{len(pending)}] row={exp.row_idx + 1} block={exp.block_idx + 1} "
            f"section={exp.section} hidden_dim={exp.hidden_dim} "
            f"bs={exp.batch_size} lr={exp.learning_rate} param={exp.param_value}"
        )

        log_name = (
            f"row{exp.row_idx + 1}_block{exp.block_idx + 1}_{exp.section}_"
            f"hd{exp.hidden_dim}_bs{exp.batch_size}_lr{exp.learning_rate}_"
            f"p{exp.param_value}_seed{args.seed}.log"
        )
        safe_log_name = log_name.replace("/", "_").replace(" ", "_")
        log_path = args.logs_dir / safe_log_name

        best_val_acc, best_val_loss = run_experiment(
            args=args,
            exp=exp,
            target_script=target_script,
            work_dir=work_dir,
            log_file=log_path,
        )

        row = rows[exp.row_idx]
        while len(row) <= max(exp.acc_col_idx, exp.loss_col_idx):
            row.append("")


        row[exp.acc_col_idx] = f"{best_val_acc * 100:.2f}% (mean of 5 folds)"
        row[exp.loss_col_idx] = f"{best_val_loss:.6f} (mean of 5 folds)"
        write_csv(csv_path, rows)

        print(
            f"  saved mean accuracy={row[exp.acc_col_idx]} mean val loss={row[exp.loss_col_idx]}"
        )

    print(f"Completed. Updated CSV: {csv_path}")


if __name__ == "__main__":
    main()
