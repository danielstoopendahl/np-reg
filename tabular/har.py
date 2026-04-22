import argparse
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SLFN on UCI HAR")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--np-reg-lambda", type=float, default=0.1)
    parser.add_argument("--o-reg-lambda", type=float, default=0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--batch-norm", action="store_true", default=False)
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
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
    data_norm = torch.norm(data.view(data.size(0), -1), p=2, dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p=2, dim=1)
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    
    return reg_lambda * norm_diff_loss

def orthogonal_regularization(weight, o_reg_lambda):
    sym = torch.mm(weight.t(), weight)
    identity = torch.eye(sym.size(0), device=weight.device)
    loss_ortho = torch.norm(sym - identity, p='fro')**2
    
    return o_reg_lambda * loss_ortho

class SLFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout, use_batch_norm, use_layer_norm):
        super().__init__()
        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.non_linear = nn.ReLU()
        self.second_linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward_features(self, x):
        features = self.first_linear(x)
        if self.use_batch_norm:
            features = self.batch_norm(features)
        if self.use_layer_norm:
            features = self.layer_norm(features)
        features = self.non_linear(features)
        features = self.dropout(features)
        return features

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.second_linear(features)
        return logits, features, x


def train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size, device, np_reg_lambda, o_reg_lambda):
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
        logits, features, inputs = model(xb)
        loss = loss_fn(logits, yb)
        if np_reg_lambda > 0:
            loss = loss + normperserving_regularization(inputs, features, np_reg_lambda)
        if o_reg_lambda > 0:
            loss = loss + orthogonal_regularization(model.first_linear.weight, o_reg_lambda)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        epoch_correct += (logits.argmax(dim=1) == yb).sum().item()

    return epoch_loss / n_samples, epoch_correct / n_samples


@torch.no_grad()
def evaluate(model, loss_fn, x, y):
    model.eval()
    logits, _, _ = model(x)
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
        use_batch_norm=args.batch_norm,
        use_layer_norm=args.layer_norm,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_state_dict = None

    print("Starting SLFN training on UCI HAR...")
    for epoch in range(100):
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            x_train=x_train,
            y_train=y_train,
            batch_size=args.batch_size,
            device=device,
            np_reg_lambda=args.np_reg_lambda,
            o_reg_lambda=args.o_reg_lambda,
        )
        val_loss, val_acc = evaluate(model, loss_fn, x_val, y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"Epoch {epoch + 1:03d}/100 | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc = evaluate(model, loss_fn, x_test, y_test)
    print(
        f"Finished training. Best val accuracy: {best_val_acc:.4f} | "
        f"best_val_loss={best_val_loss:.4f} | test_acc={test_acc:.4f}"
    )
    print(
        f"RESULT best_val_acc={best_val_acc:.6f} "
        f"best_val_loss={best_val_loss:.6f} test_acc={test_acc:.6f} test_loss={test_loss:.6f}"
    )


if __name__ == "__main__":
    main()
