import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 93% accuracy

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10):

        super(ResNetCIFAR, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        self.softmax = nn.Softmax(dim=1)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride_value in strides:
            layers.append(block(self.in_channels, out_channels, stride_value))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward_logits(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits)


def ResNet18(num_classes=10):
    return ResNetCIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model.forward_logits(data)
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


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model.forward_logits(data)
            test_loss += criterion(logits, target).item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return test_loss, accuracy


def get_dataloaders(data_dir="./data", train_batch_size=128, test_batch_size=256, num_workers=4):
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

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders()

    model = ResNet18(num_classes=10).to(device)
    checkpoint_path = "models/resnet18_cifar10_vanilla.pth"

    # print("continuing from previous save")
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), eps=1e-10)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=15,
        min_lr=1e-8,
    )

    for epoch in range(1, 81):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        test_loss, acc = test(model, device, test_loader, criterion)
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()