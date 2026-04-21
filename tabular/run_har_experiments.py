import csv
import re
import subprocess
import sys
from pathlib import Path


SECTION_NAMES = {
    "vanilla",
    "layer norm",
    "batch norm",
    "np-reg",
    "weight decay",
    "dropout",
}

RESULT_RE = re.compile(
    r"RESULT\s+best_val_acc=([0-9eE+\-.]+)\s+best_val_loss=([0-9eE+\-.]+)"
)


def normalize_header(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def parse_float(value: str) -> float:
    return float(value.strip())


def build_command(section: str, params: dict[str, float], har_script: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(har_script),
        "--hidden-dim",
        "512",
        "--seed",
        "42",
        "--batch-size",
        str(int(params["batch size"])),
        "--lr",
        str(params["learning rate"]),
    ]

    if section == "layer norm":
        cmd.extend(["--np-reg-lambda", "0"])
        cmd.append("--layer-norm")
    elif section == "batch norm":
        cmd.extend(["--np-reg-lambda", "0"])
        cmd.append("--batch-norm")
    elif section == "np-reg":
        cmd.extend(["--np-reg-lambda", str(params["lambda"])])
    elif section == "weight decay":
        cmd.extend(["--np-reg-lambda", "0"])
        cmd.extend(["--weight-decay", str(params["lambda"])])
    elif section == "dropout":
        cmd.extend(["--np-reg-lambda", "0"])
        cmd.extend(["--dropout", str(params["p"])])
    else:
        cmd.extend(["--np-reg-lambda", "0"])

    return cmd


def run_experiment(section: str, params: dict[str, float], har_script: Path, root_dir: Path) -> tuple[float, float]:
    cmd = build_command(section, params, har_script)
    print(f"Running {section} with params={params}")
    result = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Experiment failed for section={section}, params={params}")

    match = RESULT_RE.search(result.stdout)
    if not match:
        print(result.stdout)
        raise RuntimeError(f"Could not parse RESULT line for section={section}, params={params}")

    best_val_acc = float(match.group(1))
    best_val_loss = float(match.group(2))
    return best_val_acc, best_val_loss


def main() -> None:
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    csv_path = script_path.parent / "Exjobb models - Final experiment UCI HAR512.csv"
    har_script = root_dir / "tabular" / "har.py"

    with csv_path.open("r", newline="") as f:
        rows = list(csv.reader(f))

    if rows and rows[0]:
        rows[0][0] = "8"

    current_section = None
    header_map = {}

    for i, row in enumerate(rows):
        row0 = row[0].strip().lower() if len(row) > 0 else ""

        if row0 in SECTION_NAMES:
            current_section = row0
            header_map = {}
            continue

        if current_section is None:
            continue

        if all(cell.strip() == "" for cell in row):
            continue

        normalized_row = [normalize_header(cell) for cell in row]
        if "accuracy" in normalized_row and "val loss" in normalized_row:
            header_map = {
                name: idx for idx, name in enumerate(normalized_row) if name != ""
            }
            continue

        if not header_map:
            continue

        required = ["batch size", "learning rate", "accuracy", "val loss"]
        if current_section in {"np-reg", "weight decay"}:
            required.append("lambda")
        if current_section == "dropout":
            required.append("p")

        if any(key not in header_map for key in required):
            continue

        try:
            params = {
                "batch size": parse_float(row[header_map["batch size"]]),
                "learning rate": parse_float(row[header_map["learning rate"]]),
            }

            if current_section in {"np-reg", "weight decay"}:
                params["lambda"] = parse_float(row[header_map["lambda"]])
            if current_section == "dropout":
                params["p"] = parse_float(row[header_map["p"]])
        except (ValueError, IndexError):
            continue

        acc, val_loss = run_experiment(current_section, params, har_script, root_dir)

        acc_idx = header_map["accuracy"]
        val_idx = header_map["val loss"]

        while len(rows[i]) <= max(acc_idx, val_idx):
            rows[i].append("")

        rows[i][acc_idx] = f"{acc:.6f}"
        rows[i][val_idx] = f"{val_loss:.6f}"

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    print(f"Completed all experiments. Updated: {csv_path}")


if __name__ == "__main__":
    main()
