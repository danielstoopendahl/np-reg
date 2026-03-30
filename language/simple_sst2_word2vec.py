import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datasets import load_dataset
import gensim.downloader as api

# Hardcoded values
SEED = 42
BATCH_SIZE = 64
HIDDEN_SIZE = 128
LR = 1e-4
EPOCHS = 20
EMB_DIM = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Loading SST-2...")
ds = load_dataset("glue", "sst2")

print("Loading Word2Vec (word2vec-google-news-300)...")
w2v = api.load("word2vec-google-news-300")  # static pretrained vectors

token_pattern = re.compile(r"[A-Za-z']+")

def sentence_to_sum_vector(text: str) -> np.ndarray:
    tokens = token_pattern.findall(text)
    v = np.zeros(EMB_DIM, dtype=np.float32)
    for t in tokens:
        if t in w2v:
            v += w2v[t]
        elif t.lower() in w2v:
            v += w2v[t.lower()]
    return v

def build_arrays(split_name: str):
    texts = ds[split_name]["sentence"]
    labels = ds[split_name]["label"]
    X = np.zeros((len(texts), EMB_DIM), dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    for i, text in enumerate(texts):
        X[i] = sentence_to_sum_vector(text)
    return X, y

print("Vectorizing train...")
X_train, y_train = build_arrays("train")
print("Vectorizing validation...")
X_val, y_val = build_arrays("validation")

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = nn.Sequential(
    nn.Linear(EMB_DIM, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, 2),
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

torch.save(model.state_dict(), "sst2_word2vec_ffn.pt")
print("Saved model to sst2_word2vec_ffn.pt")
