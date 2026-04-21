#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple


BATCH_SIZES = [64, 128, 256]
LEARNING_RATES = ["1e-5", "3e-5", "1e-4"]
WEIGHT_DECAYS = ["1e-5", "1e-4", "1e-3"]
DROPOUTS = ["0.1", "0.3", "0.5"]
NP_REGS = ["0.01", "0.1", "1"]


@dataclass(frozen=True)
class Experiment:
    group_name: str
    param_name: str
    param_value: str
    batch_size: int
    learning_rate: str


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Run all CIFAR experiments for one hidden_dim using a simple built-in grid."
    )
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument(
        "--target-script",
        default=os.path.join(script_dir, "reg_cifar.py"),
        help="Training script to execute for each experiment.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used to run target script.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(script_dir, "grid_results", "cifar_simple"),
        help="Directory for logs and summary CSV.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip runs that already have a completed log (default: enabled).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for quick checks.",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra raw args appended to every command.",
    )
    return parser.parse_args()


def build_grid() -> List[Experiment]:
    runs: List[Experiment] = []

    # Vanilla
    for bs in BATCH_SIZES:
        for lr in LEARNING_RATES:
            runs.append(Experiment("vanilla", "none", "", bs, lr))

    # Weight decay
    for wd in WEIGHT_DECAYS:
        for bs in BATCH_SIZES:
            for lr in LEARNING_RATES:
                runs.append(Experiment("weight_decay", "weight_decay", wd, bs, lr))

    # Dropout
    for p in DROPOUTS:
        for bs in BATCH_SIZES:
            for lr in LEARNING_RATES:
                runs.append(Experiment("dropout", "dropout", p, bs, lr))

    # Layer norm
    for bs in BATCH_SIZES:
        for lr in LEARNING_RATES:
            runs.append(Experiment("layer_norm", "layer_norm", "true", bs, lr))

    # Batch norm
    for bs in BATCH_SIZES:
        for lr in LEARNING_RATES:
            runs.append(Experiment("batch_norm", "batch_norm", "true", bs, lr))

    # NP-reg
    for lam in NP_REGS:
        for bs in BATCH_SIZES:
            for lr in LEARNING_RATES:
                runs.append(Experiment("np_reg", "np_reg_lambda", lam, bs, lr))

    return runs


def build_command(args: argparse.Namespace, exp: Experiment) -> List[str]:
    cmd = [
        args.python_bin,
        args.target_script,
        "--hidden-dim",
        str(args.hidden_dim),
        "--batch-size",
        str(exp.batch_size),
        "--learning-rate",
        exp.learning_rate,
    ]

    if exp.group_name == "weight_decay":
        cmd += ["--weight-decay", exp.param_value]
    elif exp.group_name == "dropout":
        cmd += ["--dropout", exp.param_value]
    elif exp.group_name == "layer_norm":
        cmd += ["--layer-norm"]
    elif exp.group_name == "batch_norm":
        cmd += ["--batch-norm"]
    elif exp.group_name == "np_reg":
        cmd += ["--np-reg-lambda", exp.param_value]

    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    if args.extra_args:
        cmd += shlex.split(args.extra_args)

    return cmd


def sanitize_tag(value: str) -> str:
    return (
        value.replace("+", "")
        .replace(".", "p")
        .replace("-", "m")
        .replace("/", "_")
        .replace(" ", "_")
    )


def log_is_completed(log_path: str) -> bool:
    if not os.path.exists(log_path):
        return False

    markers = [
        "Run finished with arguments:",
        "Best val loss:",
        "Best val accuracy:",
    ]

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return False

    return all(m in content for m in markers)


def run_and_tee(command: List[str], log_path: str) -> int:
    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return process.wait()


def extract_best_metrics_from_log(log_path: str) -> Tuple[str, str]:
    """Return (best_val_loss, best_val_accuracy) extracted from a run log.

    Empty strings are returned when the log does not exist or no final metrics were found.
    """
    if not os.path.exists(log_path):
        return "", ""

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except OSError:
        return "", ""

    loss_match = re.search(r"Best val loss:\s*([0-9]+(?:\.[0-9]+)?)", content)
    acc_match = re.search(r"Best val accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", content)

    best_val_loss = loss_match.group(1) if loss_match else ""
    best_val_accuracy = acc_match.group(1) if acc_match else ""
    return best_val_loss, best_val_accuracy


def write_summary_header(summary_path: str) -> None:
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "hidden_dim",
                "group_name",
                "param_name",
                "param_value",
                "batch_size",
                "learning_rate",
                "status",
                "best_val_loss",
                "best_val_accuracy",
                "log_file",
                "command",
            ]
        )


def append_summary_row(
    summary_path: str,
    args: argparse.Namespace,
    exp: Experiment,
    status: str,
    best_val_loss: str,
    best_val_accuracy: str,
    log_file: str,
    command: List[str],
) -> None:
    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                args.hidden_dim,
                exp.group_name,
                exp.param_name,
                exp.param_value,
                exp.batch_size,
                exp.learning_rate,
                status,
                best_val_loss,
                best_val_accuracy,
                log_file,
                shlex.join(command),
            ]
        )


def main() -> None:
    args = parse_args()

    experiments = build_grid()
    if args.max_runs is not None:
        experiments = experiments[: args.max_runs]

    if not experiments:
        raise RuntimeError("No experiments selected.")

    output_dir = os.path.abspath(args.output_dir)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, f"summary_hidden_{args.hidden_dim}.csv")
    write_summary_header(summary_path)

    print("Simple CIFAR grid runner")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Runs selected: {len(experiments)}")
    print(f"Logs: {log_dir}")
    print(f"Summary CSV: {summary_path}")
    print(f"Resume mode: {'on' if args.resume else 'off'}")
    print(f"Dry run: {'yes' if args.dry_run else 'no'}")

    for idx, exp in enumerate(experiments, start=1):
        param_tag = exp.param_value if exp.param_value else "none"
        run_tag = (
            f"{idx:03d}_{exp.group_name}_{exp.param_name}_{sanitize_tag(param_tag)}"
            f"_bs_{exp.batch_size}_lr_{sanitize_tag(exp.learning_rate)}"
        )
        log_path = os.path.join(log_dir, f"{run_tag}.log")
        command = build_command(args, exp)

        print(
            f"[{idx}/{len(experiments)}] {exp.group_name} "
            f"{exp.param_name}={param_tag} bs={exp.batch_size} lr={exp.learning_rate}"
        )

        if args.resume and log_is_completed(log_path):
            print("  skipping completed run")
            best_val_loss, best_val_accuracy = extract_best_metrics_from_log(log_path)
            append_summary_row(
                summary_path,
                args,
                exp,
                "skipped_existing",
                best_val_loss,
                best_val_accuracy,
                log_path,
                command,
            )
            continue

        if args.dry_run:
            print(f"  DRY RUN: {shlex.join(command)}")
            append_summary_row(summary_path, args, exp, "dry_run", "", "", log_path, command)
            continue

        exit_code = run_and_tee(command, log_path)
        status = "ok" if exit_code == 0 else f"failed_exit_{exit_code}"
        best_val_loss, best_val_accuracy = extract_best_metrics_from_log(log_path)
        append_summary_row(
            summary_path,
            args,
            exp,
            status,
            best_val_loss,
            best_val_accuracy,
            log_path,
            command,
        )

        if exit_code != 0:
            print(f"Stopping after failure in run {idx}: exit code {exit_code}")
            sys.exit(exit_code)

    print("All requested runs processed.")


if __name__ == "__main__":
    main()
