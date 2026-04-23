import argparse
from collections import Counter
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from datasets import load_from_disk
import random


DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "imdb_hf")


def parser():
    parser = argparse.ArgumentParser(description="BoW embedder + MLP for IMDB sentiment")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--o-reg-lambda", type=float, default=0)
    parser.add_argument("--np-reg-lambda", type=float, default=0)
    parser.add_argument("--batch-norm", action="store_true", default=False)
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH)

    return parser.parse_args()

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normperserving_regularization(data, features, reg_lambda):
    data_norm = torch.norm(data.view(data.size(0), -1), p=2, dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p=2, dim=1)
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    return reg_lambda * norm_diff_loss


def orthogonal_regularization(weight, reg_lambda):
    sym = torch.mm(weight.t(), weight)
    identity = torch.eye(sym.size(0), device=weight.device)
    loss_ortho = torch.norm(sym - identity, p="fro") ** 2
    return reg_lambda * loss_ortho


class SLFN_IMDB(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, mlp_dropout: float, use_batch_norm: bool, use_layer_norm: bool):
        super().__init__()
        num_classes = 2

        self.first_linear = nn.Linear(embedding_dim, hidden_dim)
        self.non_linear = nn.ReLU()
        self.second_linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(mlp_dropout)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward_features(self, bow_embedding: torch.Tensor):
        features = self.first_linear(bow_embedding)
        if self.use_batch_norm:
            features = self.batch_norm(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)
        features = self.non_linear(features)
        features = self.dropout(features)
        return features

    def forward(self, bow_embedding: torch.Tensor):
        features = self.forward_features(bow_embedding)
        logits = self.second_linear(features)
        return logits, features, bow_embedding


def tokenize_text(text: str):
    return text.lower().split()


def build_vocab(texts, max_vocab_size: int=80000, min_freq: int=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    vocab_tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab_tokens = sorted(vocab_tokens, key=lambda token: counter[token], reverse=True)
    vocab_tokens = vocab_tokens[:max_vocab_size]

    return {token: idx for idx, token in enumerate(vocab_tokens)}


def vectorize_text(text: str, vocab: dict):
    tokens = tokenize_text(text)
    indices = [vocab[token] for token in tokens if token in vocab]

    vec = torch.zeros(len(vocab), dtype=torch.float32)
    if not indices:
        return vec

    index_tensor = torch.tensor(indices, dtype=torch.long)
    counts = torch.bincount(index_tensor, minlength=len(vocab)).float()

    return counts


def encode_split(dataset_split, vocab: dict):
    embeddings = [vectorize_text(text, vocab) for text in dataset_split["text"]]
    labels = torch.tensor(dataset_split["label"], dtype=torch.long)
    features = torch.stack(embeddings, dim=0)
    return features, labels


def get_imdb_dataset(local_dataset_path: str):
    if os.path.isdir(local_dataset_path):
        print(f"Loading IMDb dataset from local disk: {local_dataset_path}")
        return load_from_disk(local_dataset_path)

    print("Local IMDb dataset not found. Downloading once from HF Hub...")
    dataset = load_dataset("imdb")
    os.makedirs(os.path.dirname(local_dataset_path), exist_ok=True)
    dataset.save_to_disk(local_dataset_path)
    print(f"Saved IMDb dataset locally to: {local_dataset_path}")
    return dataset


def build_tensor_splits_from_bow(max_vocab_size: int, dataset_path: str, device: torch.device):
    dataset = get_imdb_dataset(local_dataset_path=dataset_path)
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_split = split["train"]
    val_split = split["test"]
    test_split = dataset["test"]

    vocab = build_vocab(
        texts=train_split["text"],
        max_vocab_size=max_vocab_size
    )

    train_features, train_labels = encode_split(train_split, vocab)
    val_features, val_labels = encode_split(val_split, vocab)
    test_features, test_labels = encode_split(test_split, vocab)

    return (
        train_features.to(device),
        train_labels.to(device),
        val_features.to(device),
        val_labels.to(device),
        test_features.to(device),
        test_labels.to(device),
        len(vocab),
    )


def train(model, x_train, y_train, criterion, optimizer, batch_size, o_reg_lambda, np_reg_lambda):
    model.train()

    n_samples = x_train.size(0)
    permutation = torch.randperm(n_samples, device=x_train.device)
    total_loss = 0.0
    correct = 0
    total = n_samples

    for batch_start in range(0, n_samples, batch_size):
        idx = permutation[batch_start : batch_start + batch_size]
        bow_embedding = x_train[idx]
        labels = y_train[idx]

        optimizer.zero_grad(set_to_none=True)

        logits, features, bow_embedding = model(bow_embedding=bow_embedding)
        loss = criterion(logits, labels)

        if np_reg_lambda > 0:
            loss = loss + normperserving_regularization(bow_embedding, features, np_reg_lambda)
        if o_reg_lambda > 0:
            loss = loss + orthogonal_regularization(model.first_linear.weight, o_reg_lambda)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def test(model, x, y, criterion):
    model.eval()

    logits, _, _ = model(bow_embedding=x)
    loss = criterion(logits, y)
    predictions = logits.argmax(dim=1)
    accuracy = (predictions == y).float().mean().item()
    return loss.item(), accuracy


def main():
    torch.set_float32_matmul_precision('high')
    args = parser()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y, val_x, val_y, test_x, test_y, vocab_size = build_tensor_splits_from_bow(
        max_vocab_size=args.vocab_size,
        dataset_path=args.dataset_path,
        device=device,
    )

    model = SLFN_IMDB(
        embedding_dim=vocab_size,
        hidden_dim=args.hidden_dim,
        mlp_dropout=args.dropout,
        use_batch_norm=args.batch_norm,
        use_layer_norm=args.layer_norm
    ).to(device)

    model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_loss_improvement = 0
    early_stop_patience = 16
    saved_checkpoint = False
    fd, save_path = tempfile.mkstemp(prefix="bow_mlp_imdb_", suffix=".pt")
    os.close(fd)
    print(f"Using device: {device}")
    print(f"BoW vocab size: {vocab_size}")

    for epoch in range(1, 401):
        train_loss, train_acc = train(
            model,
            train_x,
            train_y,
            criterion,
            optimizer,
            args.batch_size,
            args.o_reg_lambda,
            args.np_reg_lambda,
        )
        val_loss, val_acc = test(
            model,
            val_x,
            val_y,
            criterion,
        )

        print(
            f"Epoch {epoch}/{400} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1

        if epochs_without_loss_improvement >= early_stop_patience:
            print("Validation loss has not improved for 8 epochs. Stopping training.")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            saved_checkpoint = True
            print(f"Saved new best model to {save_path}")

    try:
        if saved_checkpoint:
            checkpoint = torch.load(save_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint)

        test_loss, test_acc = test(
            model,
            test_x,
            test_y,
            criterion,
        )
        print(f"Test loss={test_loss:.4f} | Test accuracy={test_acc * 100:.2f}%")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


if __name__ == "__main__":
    main()
