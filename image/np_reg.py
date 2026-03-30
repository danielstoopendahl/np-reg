import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import csv

from ResNet18 import ResNet18

def normperserving_regularization(data, features, reg_lambda=1):
    """
    Computes the norm-preserving regularization penalty.
    Penalizes differences between the norm of input data and the norm of output features.
    """
    
    # Calculate norm for each datapoint in the data batch
    data_norm = torch.norm(data.view(data.size(0), -1), p='fro', dim=1)
    
    # Calculate norm for each datapoint in the features batch
    features_norm = torch.norm(features.view(features.size(0), -1), p='fro', dim=1)
    
    # Create loss that penalizes when norms differ
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    
    return reg_lambda * norm_diff_loss

class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()

        input_dim = 3 * 32 * 32
        rank = 1000
        hidden_dim = 16384
        self.bottleneck = nn.Linear(input_dim, rank)
        self.first_linear = nn.Linear(rank, hidden_dim)
        self.non_linear = nn.ReLU()
        self.second_linear = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        nn.init.orthogonal_(self.bottleneck.weight)
        nn.init.orthogonal_(self.first_linear.weight)

    def forward_features(self, x):
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = self.first_linear(x)
        x = self.non_linear(x)

        return x

    def forward_logits(self, x):
        x = self.forward_features(x)
        x = self.second_linear(x)
        return x

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits)


def ResNet1(num_classes=10):
    return ResNetCIFAR(num_classes=num_classes)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        features = model.forward_features(data)
        logits = model.forward_logits(data)
        normperserving_loss = normperserving_regularization(data, features)
        loss = F.cross_entropy(logits, target) + normperserving_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    return running_loss / len(train_loader)


def train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher):
    model.train()
    running_loss = 0.0
    teacher.eval()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        features = model.forward_features(data)
        logits = model.forward_logits(data)
         
        with torch.no_grad():
            target = teacher.forward_logits(data)

        normperserving_loss = normperserving_regularization(data, features)
        loss = criterion(logits, target) + normperserving_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 100 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                f"Loss: {loss.item():.6f}"
            )

    return running_loss / len(train_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model.forward_logits(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy


def analyze_active_hidden_neurons(model, device, test_loader):
    model.eval()
    ever_positive = None

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            hidden = model.forward_features(data)
            batch_positive = (hidden > 0).any(dim=0)
            ever_positive = batch_positive if ever_positive is None else (ever_positive | batch_positive)

    if ever_positive is None:
        print("No test samples found; hidden neuron analysis skipped.")
        return 0, 0, 0.0

    active_count = int(ever_positive.sum().item())
    total_count = int(ever_positive.numel())
    active_ratio = 100.0 * active_count / total_count

    print(
        f"Hidden neurons active at least once (>0): "
        f"{active_count}/{total_count} ({active_ratio:.2f}%)"
    )
    return active_count, total_count, active_ratio


def histogram_hidden_activations(model, device, test_loader, output_path="rankgraphs/hidden_activation_histogram.txt"):
    """
    Histogram over hidden neurons by number of test samples where activation is > 0.
    Example bucket meaning: hist[k] = number of hidden neurons active in exactly k samples.
    """
    model.eval()
    activation_counts = None
    total_samples = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            hidden = model.forward_features(data)
            batch_counts = (hidden > 0).sum(dim=0).to(torch.int64)
            activation_counts = batch_counts if activation_counts is None else (activation_counts + batch_counts)
            total_samples += data.size(0)

    if activation_counts is None or total_samples == 0:
        print("No test samples found; hidden activation histogram skipped.")
        return None

    hist = torch.bincount(activation_counts, minlength=total_samples + 1)

    nonzero_bins = torch.nonzero(hist, as_tuple=False).flatten().tolist()
    print("\nHidden-neuron activation histogram (k active samples -> number of neurons):")
    for k in nonzero_bins:
        print(f"{k} -> {int(hist[k].item())}")

    with open(output_path, "w") as f:
        f.write("# k_active_samples\tneuron_count\n")
        for k in nonzero_bins:
            f.write(f"{k}\t{int(hist[k].item())}\n")

    print(f"Saved hidden activation histogram to {output_path}\n")
    return hist


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_log_path = "rankgraphs/gsvd_normperserving_losses.csv"
    os.makedirs(os.path.dirname(loss_log_path), exist_ok=True)
    with open(loss_log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss"])

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    real_train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)


    train_loader = DataLoader(real_train_dataset, batch_size=128, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=12)

    filename = 'models/resnet1_norm_regularization.pth'

    model = ResNet1(num_classes=10).to(device)
    # print("continuing from previous save")
    # model.load_state_dict(torch.load(filename))

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=12,
        min_lr=1e-8,
    )
    criterion = nn.MSELoss()

    # print('loading teacher...')
    # teacher = ResNet18().to(device)
    # teacher.load_state_dict(torch.load('models/resnet18_cifar10_vanilla.pth'))

    for epoch in range(1,1001):
        #train_loss = train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher)
        train_loss = train(model, device, train_loader, optimizer, epoch)

        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        val_loss, accuracy = test(model, device, test_loader)

        with open(loss_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss])

        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")

    analyze_active_hidden_neurons(model, device, test_loader)
    histogram_hidden_activations(model, device, test_loader)

    torch.save(model.state_dict(), filename)
    print("Model saved to", filename)
    print("Loss history saved to", loss_log_path)


if __name__ == "__main__":
    main()