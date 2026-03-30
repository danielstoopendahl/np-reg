import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ResNet18 import ResNet18

class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()

        input_dim = 3 * 32 * 32
        rank = 1000
        hidden_dim = 16384
        self.bottleneck = nn.Linear(input_dim, rank)
        self.first_linear = nn.Linear(rank, hidden_dim)
        self.non_linear = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.second_linear = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

        nn.init.orthogonal_(self.bottleneck.weight)
        nn.init.orthogonal_(self.first_linear.weight)

    def forward_features(self, x):
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = self.first_linear(x)
        x = self.non_linear(x)
        x = self.batch_norm(x)

        return x

    def forward_logits(self, x):
        x = self.forward_features(x)
        x = self.second_linear(x)
        return x

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits)

    def first_two_linear_combined_weight(self):
        return torch.mm(self.first_linear.weight, self.bottleneck.weight)


def ResNet1(num_classes=10):
    return ResNetCIFAR(num_classes=num_classes)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model.forward_logits(data)
        loss = F.cross_entropy(logits, target)
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
        logits = model.forward_logits(data)
         
        with torch.no_grad():
            target = teacher.forward_logits(data)

        loss = criterion(logits, target)
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


def main():
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

    real_train_dataset = datasets.CIFAR10("./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)


    train_loader = DataLoader(real_train_dataset, batch_size=128, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=12)

    filename = 'models/resnet1_batchnorm.pth'

    model = ResNet1(num_classes=10).to(device)
    # print("continuing from previous save")
    # model.load_state_dict(torch.load(filename))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=12,
        min_lr=1e-8,
    )
    criterion = nn.MSELoss()

    print('loading teacher...')
    teacher = ResNet18().to(device)
    teacher.load_state_dict(torch.load('models/resnet18_cifar10_vanilla.pth'))

    for epoch in range(1,1001):
        train_loss = train_with_teacher(model, device, train_loader, optimizer, criterion, epoch, teacher)
        # train_loss = train(model, device, train_loader, optimizer, epoch)

        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        val_loss, accuracy = test(model, device, test_loader)
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")

    torch.save(model.state_dict(), filename)
    print("Model saved to", filename)


if __name__ == "__main__":
    main()