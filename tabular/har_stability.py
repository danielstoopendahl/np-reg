import argparse
import json
import math
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Train an SNN on UCI HAR")
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


def get_adamw_effective_update(optimizer, model):
    """
    Reconstruct the per-parameter AdamW update vector using the current
    optimizer state (moments accumulated up to but not including this step).

    Returns a dict mapping parameter -> effective update tensor of the same
    shape as the parameter, i.e. the vector AdamW is about to subtract from
    each parameter:

        update_p = lr * m_hat / (sqrt(v_hat) + eps)

    Weight-decay is intentionally excluded: it acts as a simple parameter
    shrinkage and does not affect the *direction* of the landscape probe.

    Falls back to the raw gradient scaled by lr when the optimizer state has
    not yet been initialised (first step).
    """
    effective_update = {}
    for group in optimizer.param_groups:
        lr = group['lr']
        eps = group['eps']
        beta1, beta2 = group['betas']

        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state[p]

            if len(state) == 0:
                # First step: moments not yet initialised, fall back to
                # scaled raw gradient (equivalent to what Adam would do).
                effective_update[p] = lr * p.grad.clone()
                continue

            step = state['step']
            # PyTorch stores step as a scalar tensor since ~1.13
            step_val = step.item() if isinstance(step, torch.Tensor) else step

            exp_avg = state['exp_avg']       # m_t  (biased first moment)
            exp_avg_sq = state['exp_avg_sq'] # v_t  (biased second moment)

            bias_corr1 = 1.0 - beta1 ** step_val
            bias_corr2 = 1.0 - beta2 ** step_val

            m_hat = exp_avg / bias_corr1
            v_hat = exp_avg_sq / bias_corr2

            update = lr * m_hat / (v_hat.sqrt() + eps)
            effective_update[p] = update

    return effective_update


def calculate_ll_n_gp(model, optimizer, model_input, target, np_reg_lambda):
    """
    Probe the loss landscape and gradient behaviour along the AdamW effective
    update direction.

    The probe walks 40 steps of size 0.1 * ||effective_update|| in the
    direction of the AdamW update, starting 0.1 steps before the current
    parameters, so the actual optimizer step falls roughly in the middle of
    the probed region.

    Metrics recorded
    ----------------
    loss_landscapes  : list of 40 loss values along the probe path.
    gradient_stabilities : list of 40 Euclidean distances between the
                           gradient at each probe point and the gradient at
                           the starting parameters.
    effective_betas  : max over the probe of ||Δg|| / ||Δθ||, i.e. an
                       empirical estimate of the β-smoothness constant along
                       the update direction.
    """
    was_training = model.training

    # ------------------------------------------------------------------ #
    # 1. Snapshot original parameters, gradients, and compute the AdamW   #
    #    effective update direction.                                       #
    # ------------------------------------------------------------------ #
    orig_grads = {
        name: p.grad.clone() if p.grad is not None else None
        for name, p in model.named_parameters()
    }

    # Build a name -> parameter map so we can index effective_update by name
    name_to_param = dict(model.named_parameters())

    # effective_update is keyed by parameter *object*
    effective_update_by_param = get_adamw_effective_update(optimizer, model)

    # Re-key by parameter name for convenience
    effective_update = {
        name: effective_update_by_param[p]
        for name, p in name_to_param.items()
        if p in effective_update_by_param
    }

    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------ #
    # 2. Compute the scalar norm of one probe step so that the β          #
    #    denominator is the true parameter-space displacement.            #
    # ------------------------------------------------------------------ #
    probe_scale = 0.1  # fraction of the full AdamW step per probe step
    step_norm = math.sqrt(
        sum((probe_scale * u).pow(2).sum().item() for u in effective_update.values())
    )
    # Guard against degenerate case (should not happen after first step)
    if step_norm < 1e-12:
        step_norm = 1e-12

    # ------------------------------------------------------------------ #
    # 3. Apply the initial backward offset so the probe straddles the     #
    #    current parameter location.                                      #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        for name, p in model.named_parameters():
            u = effective_update.get(name)
            if u is not None:
                p.data.sub_(probe_scale * u)

    # ------------------------------------------------------------------ #
    # 4. Walk 40 probe steps, recording loss and gradient information.    #
    # ------------------------------------------------------------------ #
    losses_for_steps = []
    grad_norms_for_steps = []
    beta_values = []
    prev_grads = None

    for i in range(40):
        features = model.forward_features(model_input)
        logits = model.second_linear(features)
        loss_val = F.cross_entropy(logits, target)
        if np_reg_lambda > 0:
            loss_val = loss_val + normperserving_regularization(
                model_input, features, np_reg_lambda
            )
        losses_for_steps.append(loss_val.item())

        # Recompute gradients at this probe point
        for p in model.parameters():
            p.grad = None
        loss_val.backward()

        # Distance between current gradient and the gradient at θ_0
        grad_distance = math.sqrt(sum(
            ((p.grad - orig_grads[name]) ** 2).sum().item()
            for name, p in model.named_parameters()
            if p.grad is not None and orig_grads.get(name) is not None
        ))
        grad_norms_for_steps.append(grad_distance)

        # β-smoothness: ||∇L(θ_i) - ∇L(θ_{i-1})|| / ||θ_i - θ_{i-1}||
        # The denominator is the actual parameter displacement per step.
        if prev_grads is None:
            beta_values.append(float("nan"))
        else:
            adj_grad_distance = math.sqrt(sum(
                ((p.grad - prev_grads[name]) ** 2).sum().item()
                for name, p in model.named_parameters()
                if p.grad is not None and prev_grads.get(name) is not None
            ))
            beta_values.append(adj_grad_distance / step_norm)

        prev_grads = {
            name: p.grad.clone() if p.grad is not None else None
            for name, p in model.named_parameters()
        }

        # Advance one probe step
        with torch.no_grad():
            for name, p in model.named_parameters():
                u = effective_update.get(name)
                if u is not None:
                    p.data.sub_(probe_scale * u)

    # ------------------------------------------------------------------ #
    # 5. Restore original parameters and gradients for optimizer.step()  #
    # ------------------------------------------------------------------ #
    model.load_state_dict(orig_state)
    for name, p in model.named_parameters():
        p.grad = orig_grads.get(name)

    # ------------------------------------------------------------------ #
    # 6. Record metrics                                                   #
    # ------------------------------------------------------------------ #
    loss_landscapes.append(losses_for_steps)
    gradient_stabilities.append(grad_norms_for_steps)
    finite_betas = [b for b in beta_values if not math.isnan(b)]
    effective_betas.append(max(finite_betas) if finite_betas else float("nan"))

    if was_training:
        model.train()
    else:
        model.eval()


class SNN(nn.Module):
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

        batch_idx = i // batch_size
        if batch_idx % 1 == 0:
            # Pass optimizer so the probe can read AdamW moment estimates
            calculate_ll_n_gp(model, optimizer, xb, yb, np_reg_lambda)

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

    folds_data = np.load("data/har_cv_folds.npz")
    folds = [folds_data[f"fold{i}"] for i in range(5)]

    val_accs = []
    val_losses = []

    for fold_idx in range(1):
        val_subjects = folds[fold_idx]
        train_subjects = np.concatenate([folds[j] for j in range(5) if j != fold_idx])
        val_mask = np.isin(subject_all, val_subjects)
        train_mask = np.isin(subject_all, train_subjects)

        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        x_val = x_all[val_mask]
        y_val = y_all[val_mask]

        train_mean = x_train.mean(dim=0, keepdim=True)
        train_std = x_train.std(dim=0, keepdim=True)
        train_std[train_std < 1e-6] = 1.0
        x_train = (x_train - train_mean) / train_std
        x_val = (x_val - train_mean) / train_std

        input_dim = x_train.size(1)
        num_classes = int(torch.max(y_train).item() + 1)
        loss_fn = nn.CrossEntropyLoss()
        model = SNN(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            use_batch_norm=args.batch_norm,
            use_layer_norm=args.layer_norm,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
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

    loss_landscapes_ranges = [
        {"min": min(values), "max": max(values)}
        for values in loss_landscapes
    ]
    out_metrics = {
        "loss_landscapes": loss_landscapes_ranges,
        "gradient_stabilities": gradient_stabilities,
        "effective_betas": effective_betas,
    }
    save_metrics_path = "stability_metrics_np.json"
    with open(save_metrics_path, "w") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"Saved metrics to {save_metrics_path}")


if __name__ == "__main__":
    main()