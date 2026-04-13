import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

def npreg(data, features, reg_lambda=1e-2):
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


def oreg(weight, reg_lambda=1e-4):
    """
    Computes the orthogonal regularization penalty: 
    L = lambda * ||W^T W - I||_F^2
    """
    
    # Calculate W^T * W
    sym = torch.mm(weight.t(), weight)
    
    # Create the Identity matrix of the same size on the same device
    identity = torch.eye(sym.size(0), device=weight.device)
    
    # Compute the squared Frobenius norm of the difference
    loss_ortho = torch.norm(sym - identity, p='fro')**2
    
    return reg_lambda * loss_ortho

class DNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DNN, self).__init__()

        input_dim = 3 * 32 * 32
        hidden_dim = 2048
        self.first_linear = nn.Linear(input_dim, hidden_dim)
        self.second_linear = nn.Linear(hidden_dim, hidden_dim)
        self.third_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fourth_linear = nn.Linear(hidden_dim, hidden_dim)
        self.fifth_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, num_classes)
        self.non_linear = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward_hidden_layers(self, x):
        x = torch.flatten(x, 1)
        h1 = self.non_linear(self.first_linear(x))
        h2 = self.non_linear(self.second_linear(h1))
        h3 = self.non_linear(self.third_linear(h2))
        h4 = self.non_linear(self.fourth_linear(h3))
        h5 = self.non_linear(self.fifth_linear(h4))
        return [h1, h2, h3, h4, h5]

    def forward_half(self, x):
        return self.forward_hidden_layers(x)[0]

    def forward_features(self, x):
        return self.forward_hidden_layers(x)[-1]

    def forward_logits(self, x):
        x = self.forward_features(x)
        x = self.output_linear(x)
        return x

    def forward(self, x):
        logits = self.forward_logits(x)
        return self.softmax(logits)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0


    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        hidden = model.forward_hidden_layers(data)
        logits = model.output_linear(hidden[-1])
        # first_ortho_loss = oreg(model.first_linear.weight)
        # second_ortho_loss = oreg(model.second_linear.weight)
        npreg_terms = [npreg(data, hidden[0])]
        npreg_terms += [npreg(hidden[i], hidden[i + 1]) for i in range(len(hidden) - 1)]
        loss = F.cross_entropy(logits, target) + sum(npreg_terms)
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

    filename = 'models/resnet1_oreg_dnn.pth'

    model = DNN(num_classes=10).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=12,
        min_lr=1e-8,
    )

    for epoch in range(1,1001):
        train_loss = train(model, device, train_loader, optimizer, epoch)

        print(f"Epoch {epoch}: Train loss {train_loss:.6f}")
        val_loss, accuracy = test(model, device, test_loader)
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Learning rate {current_lr:.2e}")

    torch.save(model.state_dict(), filename)
    print("Model saved to", filename)


if __name__ == "__main__":
    main()