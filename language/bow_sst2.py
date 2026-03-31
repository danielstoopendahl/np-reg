import random
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


TOKEN_PATTERN = re.compile(r"[A-Za-z']+")

SEED = 42
BATCH_SIZE = 64
HIDDEN_DIM = 1024
LR = 1e-3
EPOCHS = 500
MAX_VOCAB_SIZE = 20000
MIN_FREQ = 2
SAVE_PATH = "language/models/bow_sst2.pt"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(texts: list[str], max_vocab_size: int, min_freq: int) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab_items = [
        token
        for token, freq in counter.most_common(max_vocab_size)
        if freq >= min_freq
    ]
    return {token: i for i, token in enumerate(vocab_items)}


def vectorize_bow(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in tokenize(text):
            idx = vocab.get(token)
            if idx is not None:
                x[row, idx] += 1.0
    return x


class BoWMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


def main() -> None:
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading SST-2...")
    ds = load_dataset("glue", "sst2")

    train_texts = ds["train"]["sentence"]
    train_labels = np.array(ds["train"]["label"], dtype=np.int64)

    val_texts = ds["validation"]["sentence"]
    val_labels = np.array(ds["validation"]["label"], dtype=np.int64)

    print("Building vocabulary...")
    vocab = build_vocab(train_texts, MAX_VOCAB_SIZE, MIN_FREQ)
    print(f"Vocab size: {len(vocab)}")

    print("Vectorizing BoW features...")
    x_train = vectorize_bow(train_texts, vocab)
    x_val = vectorize_bow(val_texts, vocab)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(train_labels)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(val_labels)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = BoWMLP(input_dim=len(vocab), hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=500,
        eta_min=0,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )
        scheduler.step()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "hidden_dim": HIDDEN_DIM,
        },
        SAVE_PATH,
    )
    print(f"Saved model checkpoint to {SAVE_PATH}")


if __name__ == "__main__":
    main()
