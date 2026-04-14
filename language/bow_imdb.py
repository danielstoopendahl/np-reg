import argparse
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset

# 0.2753 lr=1e-4, 1024 batch size


def parser():
    parser = argparse.ArgumentParser(description="BoW embedder + MLP for IMDB sentiment")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=16)
    parser.add_argument("--min-lr", type=float, default=1e-8)
    parser.add_argument("--o-reg-lambda", type=float, default=0)
    parser.add_argument("--np-reg-lambda", type=float, default=0)
    parser.add_argument("--batch-norm", action="store_true", default=False)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--binary-bow", action="store_true", default=False)
    parser.add_argument("--save-path", type=str, default="models/bow_mlp_imdb.pt")
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


def build_vocab(texts, max_vocab_size: int, min_freq: int):
    counter = Counter()
    for text in texts:
        counter.update(tokenize_text(text))

    vocab_tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab_tokens = sorted(vocab_tokens, key=lambda token: counter[token], reverse=True)
    vocab_tokens = vocab_tokens[:max_vocab_size]

    return {token: idx for idx, token in enumerate(vocab_tokens)}


def vectorize_text(text: str, vocab: dict, binary_bow: bool):
    tokens = tokenize_text(text)
    indices = [vocab[token] for token in tokens if token in vocab]

    vec = torch.zeros(len(vocab), dtype=torch.float32)
    if not indices:
        return vec

    index_tensor = torch.tensor(indices, dtype=torch.long)
    counts = torch.bincount(index_tensor, minlength=len(vocab)).float()
    if binary_bow:
        counts = counts.clamp(max=1.0)
    return counts


def encode_split(dataset_split, vocab: dict, binary_bow: bool):
    embeddings = [vectorize_text(text, vocab, binary_bow) for text in dataset_split["text"]]
    labels = torch.tensor(dataset_split["label"], dtype=torch.long)
    features = torch.stack(embeddings, dim=0)
    return TensorDataset(features, labels)


def build_dataloaders_from_bow(batch_size: int, max_vocab_size: int, min_freq: int, binary_bow: bool):
    dataset = load_dataset("imdb")
    split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_split = split["train"]
    val_split = split["test"]
    test_split = dataset["test"]

    vocab = build_vocab(
        texts=train_split["text"],
        max_vocab_size=max_vocab_size,
        min_freq=min_freq,
    )

    train_dataset = encode_split(train_split, vocab, binary_bow)
    val_dataset = encode_split(val_split, vocab, binary_bow)
    test_dataset = encode_split(test_split, vocab, binary_bow)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, len(vocab)


def train(model, dataloader, criterion, optimizer, device, o_reg_lambda, np_reg_lambda):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        bow_embedding, labels = batch
        bow_embedding = bow_embedding.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, features, bow_embedding = model(bow_embedding=bow_embedding)
        npreg = normperserving_regularization(bow_embedding, features, np_reg_lambda)
        oreg = orthogonal_regularization(model.first_linear.weight, o_reg_lambda)
        loss = criterion(logits, labels) + npreg + oreg

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total
# 40.73 bn
# 39.14

def test(model, dataloader, criterion, device, o_reg_lambda, np_reg_lambda):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        bow_embedding, labels = batch
        bow_embedding = bow_embedding.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits, features, bow_embedding = model(bow_embedding=bow_embedding)
            npreg = normperserving_regularization(bow_embedding, features, np_reg_lambda)
            oreg = orthogonal_regularization(model.first_linear.weight, o_reg_lambda)
            loss = criterion(logits, labels) # + npreg + oreg

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, vocab_size = build_dataloaders_from_bow(
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab_size,
        min_freq=args.min_freq,
        binary_bow=args.binary_bow,
    )

    model = SLFN_IMDB(
        embedding_dim=vocab_size,
        hidden_dim=args.hidden_dim,
        mlp_dropout=args.dropout,
        use_batch_norm=args.batch_norm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    best_val_acc = -1.0
    print(f"Using device: {device}")
    print(f"BoW vocab size: {vocab_size}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.o_reg_lambda,
            args.np_reg_lambda,
        )
        val_loss, val_acc = test(
            model,
            val_loader,
            criterion,
            device,
            args.o_reg_lambda,
            args.np_reg_lambda,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% | "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")
        if current_lr <= 2 * args.min_lr:
            print("Minimum learning rate reached. Stopping training.")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved new best model to {args.save_path}")

    checkpoint = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    test_loss, test_acc = test(
        model,
        test_loader,
        criterion,
        device,
        args.o_reg_lambda,
        args.np_reg_lambda,
    )
    print(f"Test loss={test_loss:.4f} | Test accuracy={test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
