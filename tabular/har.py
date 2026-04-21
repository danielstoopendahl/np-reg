import argparse
import os
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



def parse_args():
    parser = argparse.ArgumentParser(description="Train an SLFN on UCI HAR")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--np-reg-lambda", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normperserving_regularization(data, features, reg_lambda):
    data_norm = torch.norm(data.view(data.size(0), -1), p='fro', dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p='fro', dim=1)
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    
    return reg_lambda * norm_diff_loss

def orthogonal_regularization(weight, o_reg_lambda):
    sym = torch.mm(weight.t(), weight)
    identity = torch.eye(sym.size(0), device=weight.device)
    loss_ortho = torch.norm(sym - identity, p='fro')**2
    
    return o_reg_lambda * loss_ortho

class SLFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size, device, np_reg_lambda):
    model.train()
    n_samples = x_train.size(0)
    permutation = torch.randperm(n_samples, device=device)
    epoch_loss = 0.0
    epoch_correct = 0

    for i in range(0, n_samples, batch_size):
        idx = permutation[i : i + batch_size]
        xb = x_train[idx]
        yb = y_train[idx]

        optimizer.zero_grad()
        first_layer_features = model.fc1(xb)
        logits = model.fc2(model.dropout(model.relu(first_layer_features)))
        loss = loss_fn(logits, yb) + normperserving_regularization(xb, first_layer_features, np_reg_lambda)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        epoch_correct += (logits.argmax(dim=1) == yb).sum().item()

    return epoch_loss / n_samples, epoch_correct / n_samples


@torch.no_grad()
def evaluate(model, loss_fn, x, y):
    model.eval()
    logits = model(x)
    loss = loss_fn(logits, y).item()
    preds = logits.argmax(dim=1)
    acc = (preds == y).float().mean().item()
    return loss, acc


def main():
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load("tabular/data/dataset_har.pt")

    x_train = dataset["X_train"].to(device)
    y_train = dataset["y_train"].to(device)
    x_val = dataset["X_val"].to(device)
    y_val = dataset["y_val"].to(device)
    x_test = dataset["X_test"].to(device)
    y_test = dataset["y_test"].to(device)
    
    input_dim = x_train.size(1)
    num_classes = int(torch.max(y_train).item() + 1)

    model = SLFN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state_dict = None

    print("Starting SLFN training on UCI HAR...")
    for epoch in range(120):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            x_train=x_train,
            y_train=y_train,
            batch_size=args.batch_size,
            device=device,
            np_reg_lambda=args.np_reg_lambda,
        )
        val_loss, val_acc = evaluate(model, loss_fn, x_val, y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:03d}/120 | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(model, loss_fn, x_test, y_test)
    print(f"Finished training. Best val accuracy: {best_val_acc:.4f} | test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
