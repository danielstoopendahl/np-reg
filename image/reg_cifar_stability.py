import argparse
import copy
import json
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

HIDDEN_DIM = 8192 # [8, 32, 128, 512, 2048, 8192]
BATCH_SIZE = 128 # [64, 128, 256]
NP_REG_LAMBDA = 0 # [0.01, 0.1, 1]
WEIGHT_DECAY=0
DROPOUT=0
BATCH_NORM=False
LEARNING_RATE=3e-4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--np-reg-lambda", type=float, default=NP_REG_LAMBDA)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-norm", action="store_true", default=BATCH_NORM)
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)

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

def normperserving_regularization(data, features, np_reg_lambda):

    data_norm = torch.norm(data.view(data.size(0), -1), p=2, dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p=2, dim=1)
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    
    return np_reg_lambda * norm_diff_loss

loss_landscapes = []
gradient_stabilities = []
gradient_relative_stabilities = []
effective_betas = []


def get_adamw_effective_update(optimizer, model):
    """
    Reconstruct the per-parameter AdamW update vector using the current
    optimizer state (moments accumulated up to but not including this step).
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
                # First step fallback to scaled raw gradient
                effective_update[p] = lr * p.grad.clone()
                continue

            step = state['step']
            step_val = step.item() if isinstance(step, torch.Tensor) else step

            exp_avg = state['exp_avg']       
            exp_avg_sq = state['exp_avg_sq'] 

            bias_corr1 = 1.0 - beta1 ** step_val
            bias_corr2 = 1.0 - beta2 ** step_val

            m_hat = exp_avg / bias_corr1
            v_hat = exp_avg_sq / bias_corr2

            update = lr * m_hat / (v_hat.sqrt() + eps)
            effective_update[p] = update

    return effective_update


def calculate_ll_n_gp(model, optimizer, model_input, target, np_reg_lambda):
    was_training = model.training

    # Expect gradients to be present for all trainable parameters
    assert any(p.grad is not None for p in model.parameters()), "Expected gradients to be present"

    # Save original gradients (needed by optimizer.step() after this function)
    orig_grads = {
        name: p.grad.clone() if p.grad is not None else None
        for name, p in model.named_parameters()
    }

    # Map parameter objects to names for effective update extraction
    name_to_param = dict(model.named_parameters())
    effective_update_by_param = get_adamw_effective_update(optimizer, model)

    effective_update = {
        name: effective_update_by_param[p]
        for name, p in name_to_param.items()
        if p in effective_update_by_param
    }

    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Compute step norm for accurate beta-smoothness denominator
    probe_scale = 0.1  
    step_norm = math.sqrt(
        sum((probe_scale * u).pow(2).sum().item() for u in effective_update.values())
    )
    if step_norm < 1e-12:
        step_norm = 1e-12

    # Offset parameters backward initially to straddle the current point
    with torch.no_grad():
        for name, p in model.named_parameters():
            u = effective_update.get(name)
            if u is not None:
                p.data.sub_(probe_scale * u)

    losses_for_steps = []
    grad_norms_for_steps = []
    beta_values = []
    prev_grads = None

    # Walk 40 steps along the true effective update direction
    for i in range(40):
        features = model.forward_features(model_input)
        logits = model.second_linear(features)
        loss_val = F.cross_entropy(logits, target)

        if np_reg_lambda > 0:
            loss_val = loss_val + normperserving_regularization(model_input, features, np_reg_lambda)

        losses_for_steps.append(loss_val.item())

        # Clear old gradients and perform a local backward pass
        for p in model.parameters():
            p.grad = None
        loss_val.backward()

        # Compute Euclidean distance between original grad and current grad
        grad_distance = math.sqrt(sum(
            ((p.grad - orig_grads[name]) ** 2).sum().item()
            for name, p in model.named_parameters()
            if p.grad is not None and orig_grads.get(name) is not None
        ))
        grad_norms_for_steps.append(grad_distance)

        # Calculate effective beta smoothness using the real displacement norm
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

        # Advance along the update vector
        with torch.no_grad():
            for name, p in model.named_parameters():
                u = effective_update.get(name)
                if u is not None:
                    p.data.sub_(probe_scale * u)

    # Restore original parameters and gradients
    model.load_state_dict(orig_state)
    for name, p in model.named_parameters():
        if name in orig_grads:
            p.grad = orig_grads[name]

    loss_landscapes.append(losses_for_steps)
    gradient_stabilities.append(grad_norms_for_steps)
    finite_betas = [b for b in beta_values if not math.isnan(b)]
    effective_betas.append(max(finite_betas) if finite_betas else float("nan"))

    if was_training:
        model.train()
    else:
        model.eval()


def augment_batch_on_gpu(data):
    padded = F.pad(data, (4, 4, 4, 4), mode="constant", value=0.0)
    n, _, h, w = data.shape
    offsets_y = torch.randint(0, 9, (n,), device=data.device)
    offsets_x = torch.randint(0, 9, (n,), device=data.device)

    # Extract random 32x32 crops per sample from the padded tensor.
    patches = padded.unfold(2, h, 1).unfold(3, w, 1)
    sample_idx = torch.arange(n, device=data.device)
    cropped = patches[sample_idx, :, offsets_y, offsets_x, :, :]

    flip_mask = torch.rand(n, device=data.device) < 0.5
    flipped = torch.flip(cropped, dims=[3])
    return torch.where(flip_mask.view(-1, 1, 1, 1), flipped, cropped)


class SNN_CIFAR(nn.Module):
    def __init__(self, hidden_dim, dropout, use_batch_norm, use_layer_norm):
        super(SNN_CIFAR, self).__init__()

        input_dim = 3 * 32 * 32
        output_dim = 10

        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.non_linear = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)
        self.second_linear = nn.Linear(hidden_dim, output_dim)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward_features(self, x):
        x = torch.flatten(x, 1)
        x = self.first_linear(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.non_linear(x)
        x = self.drop(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.second_linear(x)
        return x


def dataset_to_device_tensors(dataset, device, indices=None):
    if indices is None:
        indices = range(len(dataset))

    data_batches = []
    target_batches = []
    for idx in indices:
        sample, target = dataset[idx]
        data_batches.append(sample)
        target_batches.append(target)

    x = torch.stack(data_batches, dim=0).to(device)
    y = torch.tensor(target_batches, dtype=torch.long, device=device)
    return x, y


def train_one_epoch(model, optimizer, epoch, x_train, y_train, batch_size, np_reg_lambda):
    model.train()
    n_samples = x_train.size(0)
    permutation = torch.randperm(n_samples, device=x_train.device)
    epoch_loss = 0.0
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=x_train.device).view(1, 3, 1, 1)
    std = torch.tensor((0.2023, 0.1994, 0.2010), device=x_train.device).view(1, 3, 1, 1)

    for batch_start in range(0, n_samples, batch_size):
        idx = permutation[batch_start : batch_start + batch_size]
        data = x_train[idx]
        target = y_train[idx]
        model_input = (augment_batch_on_gpu(data) - mean) / std

        optimizer.zero_grad()
        features = model.forward_features(model_input)
        logits = model.second_linear(features)
        loss = F.cross_entropy(logits, target)

        if np_reg_lambda > 0:
            loss = loss + normperserving_regularization(model_input, features, np_reg_lambda)

        loss.backward()
        batch_idx = batch_start // batch_size
        if batch_idx % 100 == 0:
            calculate_ll_n_gp(model, optimizer, model_input, target, np_reg_lambda)
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_start}/{n_samples}] "
                f"Loss: {loss.item():.6f}"
            )

    return epoch_loss / n_samples


@torch.no_grad()
def evaluate_tensor_split(model, x, y, split_name="Validation"):
    model.eval()
    logits = model(x)
    test_loss = F.cross_entropy(logits, y).item()
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    accuracy = 100.0 * correct / y.size(0)
    print(
        f"\n{split_name} set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{y.size(0)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy



def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_transform = transforms.ToTensor()
    full_train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=base_transform)
    val_base_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=base_transform)
    test_base_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=base_transform)

    split_generator = torch.Generator().manual_seed(42)
    all_indices = torch.randperm(len(full_train_dataset), generator=split_generator).tolist()
    val_indices = all_indices[:5000]
    train_indices = all_indices[5000:]

    print("Loading CIFAR tensors to device memory...")
    x_train, y_train = dataset_to_device_tensors(full_train_dataset, device, train_indices)
    x_val_raw, y_val = dataset_to_device_tensors(val_base_dataset, device, val_indices)
    x_test_raw, y_test = dataset_to_device_tensors(test_base_dataset, device)
    mean = torch.tensor((0.4914, 0.4822, 0.4465), device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.2023, 0.1994, 0.2010), device=device).view(1, 3, 1, 1)
    x_val = (x_val_raw - mean) / std
    x_test = (x_test_raw - mean) / std
    del x_val_raw
    del x_test_raw

    model = SNN_CIFAR(args.hidden_dim, args.dropout, args.batch_norm, args.layer_norm).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    min_lr = 1e-8


    best_val_loss = float("inf")
    best_accuracy = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    
    for epoch in range(1,201):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            x_train=x_train,
            y_train=y_train,
            batch_size=args.batch_size,
            np_reg_lambda=args.np_reg_lambda,
        )

        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        val_loss, accuracy = evaluate_tensor_split(model, x_val, y_val, split_name="Validation")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")
        if current_lr <= 2*min_lr:
            print("Minimum learning rate reached. Stopping training.")
            break

    # Evaluate test accuracy from the checkpoint selected by best validation accuracy.
    model.load_state_dict(best_model_state)
    test_loss, test_accuracy = evaluate_tensor_split(model, x_test, y_test, split_name="Test")

    print(
        f"Run finished with arguments: \nbatch_size={args.batch_size}\n"
        f"hidden_dim={args.hidden_dim}\n"
        f"np_reg_lambda={args.np_reg_lambda}\n"
        f"weight_degay={args.weight_decay}\n"
        f"dropout={args.dropout}\n"
        f"batchnorm={args.batch_norm}\n"
        f"layernorm={args.layer_norm}\n"
        f"learning_rate={args.learning_rate}\n"
        f"seed={args.seed}\n"
    )
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val accuracy: {best_accuracy:.2f}%")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Final test accuracy: {test_accuracy:.2f}%")

    # Save min/max ranges for loss, and full curves for gradient-change metrics.
    loss_landscapes_ranges = [
        {"min": min(values), "max": max(values)}
        for values in loss_landscapes
    ]
    out_metrics = {
        "loss_landscapes": loss_landscapes_ranges,
        "gradient_stabilities": gradient_stabilities,
        "effective_betas": effective_betas,
    }
    save_metrics_path = "results/stability_metrics.json"
    with open(save_metrics_path, "w") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"Saved metrics to {save_metrics_path}")


if __name__ == "__main__":
    main()