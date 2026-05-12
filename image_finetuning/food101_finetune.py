import argparse
import copy
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


HIDDEN_DIM = 2048
BATCH_SIZE = 128
TRAIN_FRACTION = 1.0
NP_REG_LAMBDA = 0
WEIGHT_DECAY = 0
DROPOUT = 0
BATCH_NORM = False
LAYER_NORM = False
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
EPOCHS = 500
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
NUM_CLASSES = 101


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Food-101 transfer learning with frozen ResNet-50 embeddings",
    )
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--np-reg-lambda", type=float, default=NP_REG_LAMBDA)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-norm", action="store_true", default=BATCH_NORM)
    parser.add_argument("--layer-norm", action="store_true", default=LAYER_NORM)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def normpreserving_regularization(inputs, features, reg_lambda):
    input_norm = torch.norm(inputs.view(inputs.size(0), -1), p=2, dim=1)
    feature_norm = torch.norm(features.view(features.size(0), -1), p=2, dim=1)
    norm_diff_loss = F.mse_loss(input_norm, feature_norm)
    return reg_lambda * norm_diff_loss


def build_class_indices(labels: torch.Tensor, num_classes: int):
    class_indices = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels.tolist()):
        class_indices[int(label)].append(idx)
    return class_indices


def stratified_sample(class_indices: list[list[int]], fraction: float, rng: random.Random):
    subset = []
    for indices in class_indices:
        total = len(indices)
        if fraction >= 1.0:
            count = total
        else:
            count = int(round(fraction * total))
            count = max(1, min(count, total))
        indices_copy = indices.copy()
        rng.shuffle(indices_copy)
        subset.extend(indices_copy[:count])
    rng.shuffle(subset)
    return subset


def stratified_split(class_indices: list[list[int]], rng: random.Random):
    train_class_indices = []
    val_indices = []
    for indices in class_indices:
        indices_copy = indices.copy()
        rng.shuffle(indices_copy)
        total = len(indices_copy)
        if total < 2:
            val_count = 0
        else:
            val_count = int(round(0.1 * total))
            val_count = max(1, min(val_count, total - 1))
        val_indices.extend(indices_copy[:val_count])
        train_class_indices.append(indices_copy[val_count:])
    rng.shuffle(val_indices)
    return train_class_indices, val_indices


class SNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim, dropout, use_batch_norm, use_layer_norm):
        super().__init__()
        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.second_linear = nn.Linear(hidden_dim, NUM_CLASSES)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.first_linear(inputs)
        if self.use_batch_norm:
            features = self.batch_norm(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)
        features = self.activation(features)
        return features

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(inputs)
        features = self.dropout(features)
        logits = self.second_linear(features)
        return logits, features


def train_one_epoch(head, dataloader, optimizer, loss_fn, device, np_reg_lambda):
    head.train()
    total_loss = 0.0
    total_samples = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, hidden_features = head(features)
        loss = loss_fn(logits, labels)
        if np_reg_lambda > 0:
            loss = loss + normpreserving_regularization(
                features,
                hidden_features,
                np_reg_lambda,
            )
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(head, dataloader, loss_fn, device):
    head.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        logits, _ = head(features)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()

    total = max(total_samples, 1)
    accuracy = correct / total
    return total_loss / total, accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    combined_embeddings_path = os.path.join(DATA_DIR, "food101_embeddings.pt")
    embeddings = torch.load(combined_embeddings_path, map_location="cpu")
    train_features = embeddings["train_features"]
    train_labels = embeddings["train_labels"]
    test_features = embeddings["test_features"]
    test_labels = embeddings["test_labels"]

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    input_dim = int(train_features.shape[1])

    class_indices = build_class_indices(train_labels, NUM_CLASSES)
    rng = random.Random(args.seed)
    train_class_indices, val_indices = stratified_split(class_indices, rng)

    train_indices = stratified_sample(train_class_indices, args.train_fraction, rng)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    val_loader = DataLoader(
        val_subset,
        batch_size=512,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    head = SNN(input_dim, HIDDEN_DIM, args.dropout, args.batch_norm, args.layer_norm,).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    best_val_loss = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_val_accuracy = 0.0
    epochs_since_improve = 0
    early_stop_patience = 3

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            head=head,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            np_reg_lambda=args.np_reg_lambda,
        )
        val_loss, val_accuracy = evaluate(
            head=head,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )
        if epochs_since_improve >= early_stop_patience:
            print(
                "Early stopping: val_loss did not improve for "
                f"{early_stop_patience} epochs."
            )
            break

    head.load_state_dict(best_state)
    _, accuracy = evaluate(
        head=head,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print(
        f"fraction={args.train_fraction:.3f} seed={args.seed} "
        f"train={len(train_indices):>5} val={len(val_indices):>5} "
        f"test_acc={accuracy:.4f} best_val_acc={best_val_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()
