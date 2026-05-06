import argparse
import copy
import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

HIDDEN_DIM = 8192 # [8, 32, 128, 512, 2048, 8192]
BATCH_SIZE = 128 # [64, 128, 256]
NP_REG_LAMBDA = 0 # [0.01, 0.1, 1]
O_REG_LAMBDA = 0 # []
WEIGHT_DECAY=0
DROPOUT=0
BATCH_NORM=False
LEARNING_RATE=3e-4
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DEFAULT_SPECTRUM_JSON_DIR = os.path.join(DEFAULT_RESULTS_DIR, "spectra")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--np-reg-lambda", type=float, default=NP_REG_LAMBDA)
    parser.add_argument("--o-reg-lambda", type=float, default=O_REG_LAMBDA)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-norm", action="store_true", default=BATCH_NORM)
    parser.add_argument("--layer-norm", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--spectrum-json-dir", type=str, default=DEFAULT_SPECTRUM_JSON_DIR)

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
    """
    Computes the norm-preserving regularization penalty.
    Penalizes differences between the norm of input data and the norm of output features.
    """

    data_norm = torch.norm(data.view(data.size(0), -1), p=2, dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p=2, dim=1)
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    
    return np_reg_lambda * norm_diff_loss

def orthogonal_regularization(weight, o_reg_lambda):
    """
    Computes the orthogonal regularization penalty: 
    L = lambda * ||W^T W - I||_F^2
    """

    sym = torch.mm(weight.t(), weight)
    identity = torch.eye(sym.size(0), device=weight.device)
    loss_ortho = torch.norm(sym - identity, p='fro')**2
    
    return o_reg_lambda * loss_ortho


def get_cifar10_normalization_tensors(device):
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return mean, std


def normalize_batch(data, mean, std):
    return (data - mean) / std


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


class SLFN_CIFAR(nn.Module):
    def __init__(self, hidden_dim, dropout, use_batch_norm, use_layer_norm):
        super(SLFN_CIFAR, self).__init__()

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


def train_one_epoch(model, optimizer, epoch, x_train, y_train, batch_size, np_reg_lambda, o_reg_lambda, mean, std):
    model.train()
    n_samples = x_train.size(0)
    permutation = torch.randperm(n_samples, device=x_train.device)
    epoch_loss = 0.0

    for batch_start in range(0, n_samples, batch_size):
        idx = permutation[batch_start : batch_start + batch_size]
        data = x_train[idx]
        target = y_train[idx]
        model_input = normalize_batch(augment_batch_on_gpu(data), mean, std)

        optimizer.zero_grad()
        features = model.forward_features(model_input)
        logits = model.second_linear(features)
        loss = F.cross_entropy(logits, target)

        if np_reg_lambda > 0:
            loss = loss + normperserving_regularization(model_input, features, np_reg_lambda)
        if o_reg_lambda > 0:
            loss = loss + orthogonal_regularization(model.first_linear.weight, o_reg_lambda)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * data.size(0)
        batch_idx = batch_start // batch_size
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


def compute_singular_values(weight):
    with torch.no_grad():
        return torch.linalg.svdvals(weight).float().cpu()


def save_singular_values_json(singular_values, output_path, stage, epoch, weight_shape):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "stage": stage,
        "epoch": epoch,
        "weight_shape": list(weight_shape),
        "singular_values": [float(value) for value in singular_values.tolist()],
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)



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
    mean, std = get_cifar10_normalization_tensors(device)
    x_val = normalize_batch(x_val_raw, mean, std)
    x_test = normalize_batch(x_test_raw, mean, std)
    del x_val_raw
    del x_test_raw

    model = SLFN_CIFAR(args.hidden_dim, args.dropout, args.batch_norm, args.layer_norm).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    min_lr = 1e-8
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=16,
        min_lr=min_lr,
    )

    best_val_loss = float("inf")
    best_accuracy = 0.0
    best_epoch = None
    best_model_state = copy.deepcopy(model.state_dict())

    spectrum_dir = args.spectrum_json_dir
    before_train_path = os.path.join(spectrum_dir, "spectrum_before_training.json")
    singular_values = compute_singular_values(model.first_linear.weight)
    save_singular_values_json(
        singular_values,
        before_train_path,
        stage="before_training",
        epoch=0,
        weight_shape=model.first_linear.weight.shape,
    )
    print(f"Saved singular value spectrum to {before_train_path}")
    
    for epoch in range(1,1001):
        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            x_train=x_train,
            y_train=y_train,
            batch_size=args.batch_size,
            np_reg_lambda=args.np_reg_lambda,
            o_reg_lambda=args.o_reg_lambda,
            mean=mean,
            std=std,
        )

        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        val_loss, accuracy = evaluate_tensor_split(model, x_val, y_val, split_name="Validation")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")
        if epoch == 100:
            epoch_100_path = os.path.join(spectrum_dir, "spectrum_epoch_100.json")
            singular_values = compute_singular_values(model.first_linear.weight)
            save_singular_values_json(
                singular_values,
                epoch_100_path,
                stage="epoch_100",
                epoch=epoch,
                weight_shape=model.first_linear.weight.shape,
            )
            print(f"Saved singular value spectrum to {epoch_100_path}")
        if current_lr <= 2*min_lr:
            print("Minimum learning rate reached. Stopping training.")
            break

    # Evaluate test accuracy from the checkpoint selected by best validation accuracy.
    model.load_state_dict(best_model_state)
    test_loss, test_accuracy = evaluate_tensor_split(model, x_test, y_test, split_name="Test")
    best_path = os.path.join(spectrum_dir, "spectrum_best_val.json")
    singular_values = compute_singular_values(model.first_linear.weight)
    save_singular_values_json(
        singular_values,
        best_path,
        stage="best_val",
        epoch=best_epoch,
        weight_shape=model.first_linear.weight.shape,
    )
    print(f"Saved singular value spectrum to {best_path}")

    print(
        f"Run finished with arguments: \nbatch_size={args.batch_size}\n"
        f"hidden_dim={args.hidden_dim}\n"
        f"np_reg_lambda={args.np_reg_lambda}\n"
        f"o_reg_lambda={args.o_reg_lambda}\n"
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


if __name__ == "__main__":
    main()
