import argparse
from pathlib import Path

import torch


DEFAULT_INPUT = Path("data/dataset_higgs.pt")
DEFAULT_OUTPUT = Path("data/dataset_higgs_small2.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a smaller HIGGS dataset .pt file by taking the first N rows per split.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=1_000_000)
    parser.add_argument("--val-size", type=int, default=50_000)
    parser.add_argument("--test-size", type=int, default=50_000)
    return parser.parse_args()


def require_keys(data, keys):
    missing = [key for key in keys if key not in data]
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(f"Missing required key(s) in dataset: {missing_text}")


def first_n_rows(tensor, n, name):
    if tensor.size(0) < n:
        raise ValueError(
            f"Not enough rows for {name}: requested {n}, found {tensor.size(0)}"
        )
    # Clone to detach from original backing storage so torch.save writes only sliced data.
    return tensor[:n].clone()


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Loading dataset from {args.input}...")
    data = torch.load(args.input)

    required = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    require_keys(data, required)

    small_data = {
        "X_train": first_n_rows(data["X_train"], args.train_size, "X_train"),
        "y_train": first_n_rows(data["y_train"], args.train_size, "y_train"),
        "X_val": first_n_rows(data["X_val"], args.val_size, "X_val"),
        "y_val": first_n_rows(data["y_val"], args.val_size, "y_val"),
        "X_test": first_n_rows(data["X_test"], args.test_size, "X_test"),
        "y_test": first_n_rows(data["y_test"], args.test_size, "y_test"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(small_data, args.output)
    print(f"Saved smaller dataset to {args.output}")


if __name__ == "__main__":
    main()
