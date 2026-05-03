import argparse
import random
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SLFN on UCI HAR")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--np-reg-lambda", type=float, default=0)
    parser.add_argument("--o-reg-lambda", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
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

loss_landscapes = []
gradient_stabilities = []
gradient_relative_stabilities = []
effective_betas = []

def calculate_ll_n_gp(model, model_input, target, base_lr):
    model.eval()

    # Expect gradients to be present for all trainable parameters
    assert any(p.grad is not None for p in model.parameters()), "Expected gradients to be present"

    # Save original gradients (will be needed by optimizer.step() after this function)
    orig_grads = {name: p.grad.clone() if p.grad is not None else None for name, p in model.named_parameters()}

    # Precompute L2 norm of original gradients
    orig_grad_norm = 0.0
    for g in orig_grads.values():
        if g is not None:
            orig_grad_norm += (g ** 2).sum().item()
    orig_grad_norm = orig_grad_norm ** 0.5

    # Save original state dict (cloned tensors)
    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    losses_for_steps = []
    grad_norms_for_steps = []
    rel_grad_norms_for_steps = []
    beta_values = []
    eps = 1e-12
    
    # Apply initial 0.5*lr step
    for p in model.parameters():
        if p.grad is not None:
            p.data = p.data - 0.5 * base_lr * p.grad

    # Evaluate loss and gradient norm at each of 36 steps
    for i in range(36):
        # Compute loss at this step

        logits, _, _ = model(model_input)
        loss_val = F.cross_entropy(logits, target)
        losses_for_steps.append(loss_val.item())
        
        # Clear old gradients
        for p in model.parameters():
            p.grad = None
        loss_val.backward()
        
        # Compute Euclidean distance between original grad and current grad.
        grad_distance = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None and name in orig_grads and orig_grads[name] is not None:
                grad_diff = p.grad - orig_grads[name]
                grad_distance += (grad_diff ** 2).sum().item()
        grad_distance = grad_distance ** 0.5
        grad_norms_for_steps.append(grad_distance)

        # Relative gradient change (scale-invariant)
        rel = grad_distance / (orig_grad_norm + eps)
        rel_grad_norms_for_steps.append(rel)

        # Effective beta smoothness estimate: ||grad(theta+u)-grad(theta)|| / ||u||
        # Approximate ||u|| as cumulative_step * ||orig_grad|| where cumulative_step = (0.5 + i*0.1) * base_lr
        step_multiplier = 0.5 + i * 0.1
        disp_norm = step_multiplier * base_lr * orig_grad_norm

        if disp_norm == 0.0:
            beta = float('nan')
        else:
            beta = grad_distance / disp_norm
        beta_values.append(beta)
        
        # Advance by 0.1*lr for next step (skip after last)
        if i != 35:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.data = p.data - 0.1 * base_lr * p.grad

    # Restore original parameters
    model.load_state_dict(orig_state)

    # Restore original gradients for optimizer.step()
    for name, p in model.named_parameters():
        if name in orig_grads:
            p.grad = orig_grads[name]

    loss_landscapes.append(losses_for_steps)
    gradient_stabilities.append(grad_norms_for_steps)
    gradient_relative_stabilities.append(rel_grad_norms_for_steps)

    effective_betas.append(max(beta_values))

    model.train()

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

        current_lr = optimizer.param_groups[0]["lr"]

        calculate_ll_n_gp(model, inputs, yb, current_lr,)
        optimizer.step()

        epoch_loss += loss.item() * xb.size(0)
        epoch_correct += (logits.argmax(dim=1) == yb).sum().item()

    return epoch_loss / n_samples, epoch_correct / n_samples


@torch.no_grad()
def evaluate(model, x, y, loss_fn):
    model.eval()
    logits, _, _ = model(x)
    preds = logits.argmax(dim=1)
    loss = loss_fn(logits, y)
    acc = (preds == y).float().mean().item()
    return acc, loss


def main():
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset = torch.load("data/dataset_har.pt")
    x_all = dataset["X_train"].to(device)
    y_all = dataset["y_train"].to(device)
    subject_all = dataset["subject_train"].cpu().numpy()

    # Load subject-disjoint folds
    import numpy as np
    folds_data = np.load("data/har_cv_folds.npz")
    folds = [folds_data[f"fold{i}"] for i in range(5)]
    # folds = [[1,5,10,15]]

    val_accs = []
    val_losses = []

    for fold_idx in range(1):
        val_subjects = folds[fold_idx]
        train_subjects = np.concatenate([folds[j] for j in range(5) if j != fold_idx])
        # train_subjects = [2,6,7,8,9,3,11,12,13,14,4,16,17,18,19,1,5,10,15]
        val_mask = np.isin(subject_all, val_subjects)
        train_mask = np.isin(subject_all, train_subjects)

        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        x_val = x_all[val_mask]
        y_val = y_all[val_mask]

        # Normalize using training split statistics only
        train_mean = x_train.mean(dim=0, keepdim=True)
        train_std = x_train.std(dim=0, keepdim=True)
        train_std[train_std < 1e-6] = 1.0
        x_train = (x_train - train_mean) / train_std
        x_val = (x_val - train_mean) / train_std

        input_dim = x_train.size(1)
        num_classes = int(torch.max(y_train).item() + 1)
        loss_fn = nn.CrossEntropyLoss()
        model = SLFN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            use_batch_norm=args.batch_norm,
            use_layer_norm=args.layer_norm,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        val_acc = 0
        val_loss = 0

        for epoch in range(args.epochs):
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
            val_acc, val_loss = evaluate(model, x_val, y_val, loss_fn)
            print(
                f"Epoch {epoch + 1:03d}/50 | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            

        val_accs.append(float(val_acc))
        val_losses.append(float(val_loss))
        print(f"Fold {fold_idx+1}/5: val_acc={val_acc:.6f} val_loss={val_loss:.6f}")

    mean_val_acc = float(np.mean(val_accs))
    mean_val_loss = float(np.mean(val_losses))
    
    print(f"RESULT mean_val_acc={mean_val_acc:.6f} mean_val_loss={mean_val_loss:.6f}")

    # Save only min/max ranges for the requested metrics.
    loss_landscapes_ranges = [
        {"min": min(values), "max": max(values)}
        for values in loss_landscapes
    ]
    gradient_relative_stabilities_ranges = [
        {"min": min(values), "max": max(values)}
        for values in gradient_relative_stabilities
    ]
    out_metrics = {
        "loss_landscapes": loss_landscapes_ranges,
        "gradient_relative_stabilities": gradient_relative_stabilities_ranges,
        "effective_betas": effective_betas,
    }
    save_path = "har_stability_metrics.json"
    with open(save_path, "w") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"Saved metrics to {save_path}")

if __name__ == "__main__":
    main()
    