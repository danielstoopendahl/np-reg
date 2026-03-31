import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

HIDDEN_DIM = 16384 # [256, 512, 1024, 2048, 4096, 8192, 16384]
BATCH_SIZE = 128 # [64, 128, 256, 512]
REG_LAMBDA = 1 # [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]

# python np_reg.py --reg-lambda=3e-4 --batch-size=64

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--reg-lambda", type=float, default=REG_LAMBDA)
    return parser.parse_args()

def normperserving_regularization(data, features, reg_lambda):
    """
    Computes the norm-preserving regularization penalty.
    Penalizes differences between the norm of input data and the norm of output features.
    """

    data_norm = torch.norm(data.view(data.size(0), -1), p='fro', dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p='fro', dim=1)
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
    
    return reg_lambda * norm_diff_loss


class SLFN_CIFAR(nn.Module):
    def __init__(self, hidden_dim):
        super(SLFN_CIFAR, self).__init__()

        input_dim = 3 * 32 * 32
        output_dim = 10

        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.non_linear = nn.ReLU()
        self.second_linear = nn.Linear(hidden_dim, output_dim)

    def forward_features(self, x):
        x = torch.flatten(x, 1)
        x = self.first_linear(x)
        x = self.non_linear(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.second_linear(x)
        return x


def train(model, device, train_loader, optimizer, epoch, reg_lambda):
    model.train()
    running_loss = 0.0


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        features = model.forward_features(data)
        normperserving_loss = normperserving_regularization(data, features, reg_lambda)
        
        logits = model.second_linear(features)
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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nValidation set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    full_train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    val_base_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=test_transform)

    train_subset, val_subset = random_split(
        full_train_dataset,
        [len(full_train_dataset) - 10000, 10000],
        generator=torch.Generator().manual_seed(42),
    )
    val_subset = Subset(val_base_dataset, val_subset.indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False, num_workers=12)

    model = SLFN_CIFAR(HIDDEN_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    min_lr = 3e-9
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=16,
        min_lr=min_lr,
    )

    best_val_loss = float("inf")
    best_accuracy = 0.0
    
    for epoch in range(1,1001):
        train_loss = train(model, device, train_loader, optimizer, epoch, args.reg_lambda)

        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        val_loss, accuracy = test(model, device, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")
        if current_lr <= min_lr:
            print("Minimum learning rate reached. Stopping training.")
            break

    print(
        f"Run finished with arguments: batch_size={args.batch_size}, "
        f"reg_lambda={args.reg_lambda}"
    )
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Best val accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()