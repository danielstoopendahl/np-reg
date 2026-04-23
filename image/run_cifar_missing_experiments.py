#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


BEST_LOSS_RE = re.compile(r"Best val loss:\s*([0-9eE+\-.]+)")
BEST_ACC_RE = re.compile(r"Best val accuracy:\s*([0-9eE+\-.]+)%")


@dataclass(frozen=True)
class PendingExperiment:
    row_idx: int
    block_idx: int
    hidden_dim: int
    np_reg_lambda: float
    batch_size: int
    learning_rate: float
    acc_col_idx: int
    loss_col_idx: int


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run only missing CIFAR NP-reg experiments from the final CSV and "
            "fill validation accuracy/loss cells."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=script_dir / "Exjobb models - Final Experiment CIFAR2.csv",
        help="Path to the CIFAR results CSV.",
    )
    parser.add_argument(
        "--target-script",
        type=Path,
        default=script_dir / "reg_cifar.py",
        help="Training script to execute.",
    )
    parser.add_argument(
        "--hidden-dims",
        default="8192,2048,512",
        help=(
            "Comma-separated hidden dims, one per NP-reg block in the CSV (left to right). "
            "Example: 8192,2048,512"
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run the target script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print missing runs without executing them.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=script_dir / "grid_results" / "cifar_missing" / "logs",
        help="Directory for per-run logs.",
    )
    return parser.parse_args()


def parse_hidden_dims(raw: str) -> list[int]:
    dims = [part.strip() for part in raw.split(",") if part.strip()]
    if not dims:
        raise ValueError("--hidden-dims must contain at least one value")
    return [int(v) for v in dims]


def _is_missing(value: str) -> bool:
    return value.strip() == ""


def _parse_float(cell: str) -> float:
    return float(cell.strip())


def discover_missing_experiments(rows: list[list[str]], hidden_dims: list[int]) -> list[PendingExperiment]:
    if len(rows) < 3:
        raise RuntimeError("CSV does not contain expected header + data rows.")

    header_row = rows[1]
    block_starts: list[int] = []
    idx = 0
    while idx < len(header_row):
        if idx + 4 < len(header_row):
            block = [cell.strip().lower() for cell in header_row[idx : idx + 5]]
            if block == ["lambda", "batch_size", "learning_rate", "accuracy", "val loss"]:
                block_starts.append(idx)
                idx += 5
                if idx < len(header_row) and header_row[idx].strip() == "":
                    idx += 1
                continue
        idx += 1

    if not block_starts:
        raise RuntimeError("Could not find NP-reg blocks in CSV header row.")

    if len(hidden_dims) != len(block_starts):
        raise RuntimeError(
            f"Hidden dims count ({len(hidden_dims)}) must match CSV blocks ({len(block_starts)})."
        )

    pending: list[PendingExperiment] = []
    for row_idx in range(2, len(rows)):
        row = rows[row_idx]
        for block_idx, start in enumerate(block_starts):
            while len(row) <= start + 4:
                row.append("")

            lam_cell = row[start]
            bs_cell = row[start + 1]
            lr_cell = row[start + 2]
            acc_cell = row[start + 3]
            loss_cell = row[start + 4]

            if any(_is_missing(v) for v in (lam_cell, bs_cell, lr_cell)):
                continue

            if not (_is_missing(acc_cell) or _is_missing(loss_cell)):
                continue

            pending.append(
                PendingExperiment(
                    row_idx=row_idx,
                    block_idx=block_idx,
                    hidden_dim=hidden_dims[block_idx],
                    np_reg_lambda=_parse_float(lam_cell),
                    batch_size=int(_parse_float(bs_cell)),
                    learning_rate=_parse_float(lr_cell),
                    acc_col_idx=start + 3,
                    loss_col_idx=start + 4,
                )
            )

    return pending


def run_experiment(args: argparse.Namespace, exp: PendingExperiment, work_dir: Path, log_file: Path) -> tuple[float, float]:
    cmd = [
        args.python_bin,
        str(args.target_script),
        "--hidden-dim",
        str(exp.hidden_dim),
        "--batch-size",
        str(exp.batch_size),
        "--learning-rate",
        str(exp.learning_rate),
        "--np-reg-lambda",
        str(exp.np_reg_lambda),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    log_file.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")

    if completed.returncode != 0:
        raise RuntimeError(
            f"Experiment failed (row={exp.row_idx + 1}, block={exp.block_idx + 1}, exit={completed.returncode})."
        )

    loss_match = BEST_LOSS_RE.search(completed.stdout)
    acc_match = BEST_ACC_RE.search(completed.stdout)
    if not loss_match or not acc_match:
        raise RuntimeError(
            f"Could not parse best validation metrics (row={exp.row_idx + 1}, block={exp.block_idx + 1})."
        )

    best_loss = float(loss_match.group(1))
    best_acc = float(acc_match.group(1))
    return best_acc, best_loss


def write_rows(csv_path: Path, rows: list[list[str]]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    hidden_dims = parse_hidden_dims(args.hidden_dims)
    csv_path = args.csv_path.resolve()
    target_script = args.target_script.resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not target_script.exists():
        raise FileNotFoundError(f"Target script not found: {target_script}")

    work_dir = target_script.parent
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    pending = discover_missing_experiments(rows, hidden_dims)
    print(f"Found {len(pending)} missing experiments in {csv_path}")

    if not pending:
        print("Nothing to run. CSV already complete.")
        return

    for i, exp in enumerate(pending, start=1):
        print(
            f"[{i}/{len(pending)}] row={exp.row_idx + 1} block={exp.block_idx + 1} "
            f"hidden_dim={exp.hidden_dim} lambda={exp.np_reg_lambda} "
            f"bs={exp.batch_size} lr={exp.learning_rate}"
        )

        if args.dry_run:
            continue

        run_id = (
            f"row{exp.row_idx + 1}_block{exp.block_idx + 1}_hd{exp.hidden_dim}"
            f"_lam{exp.np_reg_lambda}_bs{exp.batch_size}_lr{exp.learning_rate}.log"
        )
        safe_run_id = run_id.replace("/", "_").replace(" ", "_")
        log_file = args.logs_dir / safe_run_id

        best_acc, best_loss = run_experiment(args, exp, work_dir, log_file)

        row = rows[exp.row_idx]
        while len(row) <= max(exp.acc_col_idx, exp.loss_col_idx):
            row.append("")

        row[exp.acc_col_idx] = f"{best_acc:.2f}%"
        row[exp.loss_col_idx] = f"{best_loss:.6f}"
        write_rows(csv_path, rows)

        print(
            f"  saved accuracy={row[exp.acc_col_idx]} val loss={row[exp.loss_col_idx]}"
        )

    print(f"Completed. Updated CSV: {csv_path}")


if __name__ == "__main__":
    main()