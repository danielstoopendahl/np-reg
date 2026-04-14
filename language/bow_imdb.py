import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

def parser():
    parser = argparse.ArgumentParser(description="BoW embedder + MLP for IMDB sentiment")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--o-reg-lambda", type=float, default=0)
    parser.add_argument("--np-reg-lambda", type=float, default=0)
    parser.add_argument("--batch-norm", action="store_true", default=False)
    return parser.parse_args()


def normperserving_regularization(data, features, reg_lambda):
    data_norm = torch.norm(data.view(data.size(0), -1), p="fro", dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p="fro", dim=1)
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    return reg_lambda * norm_diff_loss


def orthogonal_regularization(weight, reg_lambda):
    sym = torch.mm(weight.t(), weight)
    identity = torch.eye(sym.size(0), device=weight.device)
    loss_ortho = torch.norm(sym - identity, p="fro") ** 2
    return reg_lambda * loss_ortho


class SLFN_IMDB(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, mlp_dropout: float, use_batch_norm: bool):
        super().__init__()
        num_classes = 2

        self.first_linear = nn.Linear(embedding_dim, hidden_dim)
        self.non_linear = nn.ReLU()
        self.second_linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(mlp_dropout)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward_features(self, bow_embedding: torch.Tensor):
        features = self.first_linear(bow_embedding)
        if self.use_batch_norm:
            features = self.batch_norm(features)
        features = self.non_linear(features)
        features = self.dropout(features)
        return features

    def forward(self, bow_embedding: torch.Tensor):
        features = self.forward_features(bow_embedding)
        logits = self.second_linear(features)
        return logits, features, bow_embedding


def tokenize_text(text: str):
    return text.lower().split()


def build_vocab(texts, max_vocab_size: int=20000, min_freq: int=2):
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


def build_tensor_splits_from_bow(device: torch.device):
    dataset = load_dataset("imdb")
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_split = split["train"]
    val_split = split["test"]
    test_split = dataset["test"]

    vocab = build_vocab(
        texts=train_split["text"]
    )

    train_embeddings, train_labels = encode_split(train_split, vocab)
    val_embeddings, val_labels = encode_split(val_split, vocab)
    test_embeddings, test_labels = encode_split(test_split, vocab)

    train_embeddings = train_embeddings.to(device, non_blocking=True)
    train_labels = train_labels.to(device, non_blocking=True)
    val_embeddings = val_embeddings.to(device, non_blocking=True)
    val_labels = val_labels.to(device, non_blocking=True)
    test_embeddings = test_embeddings.to(device, non_blocking=True)
    test_labels = test_labels.to(device, non_blocking=True)

    return (
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        test_embeddings,
        test_labels,
        len(vocab),
    )


def train(model, embeddings, labels, batch_size, criterion, optimizer, o_reg_lambda, np_reg_lambda):
    model.train()

    total_loss = 0.0
    correct = 0
    total = labels.size(0)
    permutation = torch.randperm(total, device=embeddings.device)

    for start_idx in range(0, total, batch_size):
        batch_indices = permutation[start_idx:start_idx + batch_size]
        bow_embedding = embeddings[batch_indices]
        batch_labels = labels[batch_indices]

        optimizer.zero_grad(set_to_none=True)

        logits, features, bow_embedding = model(bow_embedding=bow_embedding)
        npreg = normperserving_regularization(bow_embedding, features, np_reg_lambda)
        oreg = orthogonal_regularization(model.first_linear.weight, o_reg_lambda)
        loss = criterion(logits, batch_labels) + npreg + oreg

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == batch_labels).sum().item()

    return total_loss / total, correct / total


def test(model, embeddings, labels, batch_size, criterion, o_reg_lambda, np_reg_lambda):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = labels.size(0)

    for start_idx in range(0, total, batch_size):
        end_idx = start_idx + batch_size
        bow_embedding = embeddings[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        with torch.no_grad():
            logits, features, bow_embedding = model(bow_embedding=bow_embedding)
            npreg = normperserving_regularization(bow_embedding, features, np_reg_lambda)
            oreg = orthogonal_regularization(model.first_linear.weight, o_reg_lambda)
            loss = criterion(logits, batch_labels) # + npreg + oreg

        total_loss += loss.item() * batch_labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == batch_labels).sum().item()

    return total_loss / total, correct / total


def main():
    args = parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        test_embeddings,
        test_labels,
        vocab_size,
    ) = build_tensor_splits_from_bow(
        device=device,
    )

    model = SLFN_IMDB(
        embedding_dim=vocab_size,
        hidden_dim=args.hidden_dim,
        mlp_dropout=args.dropout,
        use_batch_norm=args.batch_norm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_loss_improvement = 0
    early_stop_patience = 8
    print(f"Using device: {device}")
    print(f"BoW vocab size: {vocab_size}")
    print(
        f"Loaded BoW tensors on {device}: "
        f"train={tuple(train_embeddings.shape)}, "
        f"val={tuple(val_embeddings.shape)}, "
        f"test={tuple(test_embeddings.shape)}"
    )
    save_path = "models/bow_mlp_imdb.pt"

    for epoch in range(1, 301):
        train_loss, train_acc = train(
            model,
            train_embeddings,
            train_labels,
            args.batch_size,
            criterion,
            optimizer,
            args.o_reg_lambda,
            args.np_reg_lambda,
        )
        val_loss, val_acc = test(
            model,
            val_embeddings,
            val_labels,
            args.batch_size,
            criterion,
            args.o_reg_lambda,
            args.np_reg_lambda,
        )

        print(
            f"Epoch {epoch}/{301} | "
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
            print(f"Saved new best model to {save_path}")

    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    test_loss, test_acc = test(
        model,
        test_embeddings,
        test_labels,
        args.batch_size,
        criterion,
        args.o_reg_lambda,
        args.np_reg_lambda,
    )
    print(f"Test loss={test_loss:.4f} | Test accuracy={test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
