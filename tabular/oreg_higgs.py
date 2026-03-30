from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F



# Hardcoded config
DATA_PATH = None
OPENML_FALLBACK_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
MAX_ROWS = 200_000
READ_CHUNK_SIZE = 25_000
REMOTE_TIMEOUT_SECONDS = 20
TEST_SIZE = 0.2
HIDDEN_UNITS = 64
EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SEED = 42
reg_lambda = 1e-3

def orthogonal_regularization(weight):

    sym = torch.mm(weight.t(), weight)

    identity = torch.eye(sym.size(0), device=weight.device)

    loss_ortho = torch.norm(sym - identity, p='fro')**2

    return reg_lambda * loss_ortho

def normperserving_regularization(data, features):
    """
    Computes the norm-preserving regularization penalty.
    Penalizes differences between the norm of input data and the norm of output features.
    """
    
    # Calculate norm for each datapoint in the data batch
    data_norm = torch.norm(data.view(data.size(0), -1), p='fro', dim=1)
    
    # Calculate norm for each datapoint in the features batch
    features_norm = torch.norm(features.view(features.size(0), -1), p='fro', dim=1)
    
    # Create loss that penalizes when norms differ
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    
    return reg_lambda * norm_diff_loss

def log(message: str) -> None:
    print(message, flush=True)


class HiggsNN(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_units)
        self.nonlinear = nn.ReLU()
        self.linear2 = nn.Linear(hidden_units, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.nonlinear(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.linear2(x)
        return x.squeeze(1)


def load_higgs(data_path: Path | None, max_rows: int):
    def read_frame_in_chunks(source, compression: str = "infer") -> pd.DataFrame:
        chunks = []
        rows_loaded = 0
        for chunk in pd.read_csv(
            source,
            header=None,
            chunksize=READ_CHUNK_SIZE,
            compression=compression,
        ):
            remaining = max_rows - rows_loaded
            if remaining <= 0:
                break

            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]

            chunks.append(chunk)
            rows_loaded += len(chunk)
            log(f"Loaded {rows_loaded:,}/{max_rows:,} rows...")

            if rows_loaded >= max_rows:
                break

        if not chunks:
            raise RuntimeError("No rows were loaded from the HIGGS dataset source.")

        return pd.concat(chunks, ignore_index=True)

    if data_path is not None:
        if not data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")

        frame = read_frame_in_chunks(data_path, compression="infer")
        y = frame.iloc[:, 0].to_numpy(dtype=np.int64)
        X = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
        return X, y

    try:
        with urlopen(OPENML_FALLBACK_URL, timeout=REMOTE_TIMEOUT_SECONDS) as response:
            frame = read_frame_in_chunks(response, compression="gzip")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load HIGGS from remote URL. Set DATA_PATH to a local HIGGS.csv.gz file "
            f"or check network access. Original error: {exc}"
        ) from exc

    y = frame.iloc[:, 0].to_numpy(dtype=np.int64)
    X = frame.iloc[:, 1:].to_numpy(dtype=np.float32)

    return X, y


def stratified_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        cls_test_count = int(round(len(cls_indices) * test_size))
        cls_test_count = min(max(cls_test_count, 1), len(cls_indices) - 1)

        test_indices.append(cls_indices[:cls_test_count])
        train_indices.append(cls_indices[cls_test_count:])

    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


def binary_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC is undefined when only one class is present.")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    pos_rank_sum = ranks[pos].sum()

    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def binary_classification_report(y_true: np.ndarray, y_pred: np.ndarray, digits: int = 4) -> str:
    lines = []
    header = f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines.append(header)
    lines.append("")

    total_support = len(y_true)
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for cls in (0, 1):
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        support = int(np.sum(y_true == cls))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

        lines.append(
            f"{str(cls):>12} {precision:>10.{digits}f} {recall:>10.{digits}f} {f1:>10.{digits}f} {support:>10d}"
        )

    accuracy = float(np.mean(y_true == y_pred))
    weighted_precision /= total_support
    weighted_recall /= total_support
    weighted_f1 /= total_support

    lines.append("")
    lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.{digits}f} {total_support:>10d}")
    lines.append(
        f"{'weighted avg':>12} {weighted_precision:>10.{digits}f} {weighted_recall:>10.{digits}f} {weighted_f1:>10.{digits}f} {total_support:>10d}"
    )

    return "\n".join(lines)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_source = str(DATA_PATH) if DATA_PATH is not None else OPENML_FALLBACK_URL
    log(f"Loading data (max_rows={MAX_ROWS:,}) from: {data_source}")
    X, y = load_higgs(DATA_PATH, MAX_ROWS)
    log(f"Loaded dataset with {len(X):,} rows and {X.shape[1]} features")

    log("Creating train/test split...")
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        seed=SEED,
    )
    log(f"Split done: train={len(X_train):,}, test={len(X_test):,}")

    log("Standardizing features...")
    X_train, X_test = standardize_train_test(X_train, X_test)

    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Training on device: {device}")
    model = HiggsNN(input_dim=X_train.shape[1], hidden_units=HIDDEN_UNITS).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=16,
        min_lr=1e-8,
    )

    log("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            npreg = normperserving_regularization(batch_X, model.forward_features(batch_X))
            loss = criterion(logits, batch_y) + npreg
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_test_tensor.to(device))
            val_loss = criterion(val_logits, y_test_tensor.to(device)).item()
            val_prob = torch.sigmoid(val_logits)
            val_pred = (val_prob >= 0.5).float()
            val_accuracy = (val_pred == y_test_tensor.to(device)).float().mean().item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        log(
            f"Epoch {epoch + 1:02d}/{EPOCHS} - "
            f"loss: {epoch_loss:.4f} - "
            f"val_loss: {val_loss:.4f} - "
            f"val_acc: {val_accuracy:.4f} - "
            f"lr: {current_lr:.2e}"
        )

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor.to(device))
        y_prob = torch.sigmoid(logits).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(np.int64)

    accuracy = float(np.mean(y_test == y_pred))
    roc_auc = binary_roc_auc_score(y_test, y_prob)

    log(f"Samples: train={len(X_train):,}, test={len(X_test):,}")
    log(f"Accuracy: {accuracy:.4f}")
    log(f"ROC-AUC:  {roc_auc:.4f}")
    log("\nClassification report:")
    log(binary_classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    main()
