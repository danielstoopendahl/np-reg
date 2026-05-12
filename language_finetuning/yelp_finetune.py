import argparse
import copy
import os
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
HIDDEN_DIM = 768
BATCH_SIZE = 256
TRAIN_FRACTION = 1.0
NP_REG_LAMBDA = 0
WEIGHT_DECAY = 0
DROPOUT = 0
BATCH_NORM = False
LAYER_NORM = False
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
EPOCHS = 500
EMBEDDINGS_PT = os.path.join(SCRIPT_DIR, "data", "yelp_embeddings.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune shallow head on Yelp restaurant embeddings",
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


def build_class_indices(labels: torch.Tensor, num_classes: int) -> list[list[int]]:
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
    def __init__(self, input_dim, hidden_dim, num_classes, dropout, use_batch_norm, use_layer_norm):
        super().__init__()
        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.second_linear = nn.Linear(hidden_dim, num_classes)

    def forward_features(self, inputs):
        features = self.first_linear(inputs)
        if self.use_batch_norm:
            features = self.batch_norm(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)
        features = self.activation(features)
        return features

    def forward(self, inputs):
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
            loss = loss + normpreserving_regularization(features, hidden_features, np_reg_lambda)

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


def load_embedding_tensors(embeddings_pt):

    payload = torch.load(embeddings_pt, map_location="cpu")
    train_embeddings = payload["train_embeddings"].float()
    train_labels = payload["train_labels"].long()
    test_embeddings = payload["test_embeddings"].float()
    test_labels = payload["test_labels"].long()

    num_classes = int(torch.max(train_labels).item()) + 1
    train_dataset = TensorDataset(train_embeddings, train_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)
    return train_dataset, test_dataset, num_classes


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, num_classes = load_embedding_tensors(EMBEDDINGS_PT)
    input_dim = int(train_dataset.tensors[0].shape[1])
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    train_labels = train_dataset.tensors[1]
    class_indices = build_class_indices(train_labels, num_classes)
    rng = random.Random(args.seed)
    train_class_indices, val_indices = stratified_split(class_indices,rng)
    
    train_indices = stratified_sample(train_class_indices, args.train_fraction, rng)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    val_loader = DataLoader(
        val_subset,
        batch_size=512,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    head = SNN(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        dropout=args.dropout,
        use_batch_norm=args.batch_norm,
        use_layer_norm=args.layer_norm,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
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
    _, test_accuracy = evaluate(
        head=head,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Best val accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()
