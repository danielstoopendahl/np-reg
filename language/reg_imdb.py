import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# pip uninstall torch torchvision -y
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

SEED = 42
EPOCHS = 40

def parser():
    parser = argparse.ArgumentParser(description="Frozen pretrained BERT + MLP for IMDB sentiment")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--o-reg-lambda", type=float, default=0)
    parser.add_argument("--np-reg-lambda", type=float, default=0)
    return parser.parse_args()

def normperserving_regularization(data, features, reg_lambda):
                        
    data_norm = torch.norm(data.view(data.size(0), -1), p='fro', dim=1)
    features_norm = torch.norm(features.view(features.size(0), -1), p='fro', dim=1)                                                
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
                                                            
    return reg_lambda * norm_diff_loss

def orthogonal_regularization(weight, reg_lambda):

    sym = torch.mm(weight.t(), weight)
    identity = torch.eye(sym.size(0), device=weight.device)
    loss_ortho = torch.norm(sym - identity, p='fro')**2
    
    return reg_lambda * loss_ortho


class SLFN_IMDB(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dropout: float):
        super().__init__()
        embedding_dim = 768
        num_classes = 2

        self.first_linear = nn.Linear(embedding_dim, hidden_dim)
        self.non_linear = nn.ReLU()
        self.second_linear = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(mlp_dropout)

    def forward_features(self, cls_embedding: torch.Tensor):
        features = self.first_linear(cls_embedding)
        features = self.non_linear(features)
        features = self.dropout(features)
        return features

    def forward(self, cls_embedding: torch.Tensor):
        features = self.forward_features(cls_embedding)
        logits = self.second_linear(features)
        return logits, features, cls_embedding


def build_dataloaders_from_cache(embedding_path, batch_size: int):
    
    cache = torch.load(embedding_path, map_location="cpu")
    
    train_dataset = TensorDataset(cache["train_embeddings"], cache["train_labels"])
    val_dataset = TensorDataset(cache["val_embeddings"], cache["val_labels"])
    test_dataset = TensorDataset(cache["test_embeddings"], cache["test_labels"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def train(model, dataloader, criterion, optimizer, device, o_reg_lambda, np_reg_lambda):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        cls_embedding, labels = batch
        cls_embedding = cls_embedding.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, features, cls_embedding = model(cls_embedding=cls_embedding)
        npreg = normperserving_regularization(cls_embedding, features, np_reg_lambda)
        oreg = orthogonal_regularization(model.first_linear.weight, o_reg_lambda)
        loss = criterion(logits, labels) + npreg + oreg
            
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def test(model, dataloader, criterion, device, o_reg_lambda, np_reg_lambda):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        cls_embedding, labels = batch
        cls_embedding = cls_embedding.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            logits, features, cls_embedding = model(cls_embedding=cls_embedding)
            npreg = normperserving_regularization(cls_embedding, features, np_reg_lambda)
            oreg = orthogonal_regularization(model.first_linear.weight, o_reg_lambda)
            loss = criterion(logits, labels) + npreg + oreg
            
        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_path = "embeddings/imdb_bert_embeddings.pt"
    train_loader, val_loader, test_loader = build_dataloaders_from_cache(embedding_path=embedding_path, batch_size=args.batch_size)

    model = SLFN_IMDB(hidden_dim=args.hidden_dim, mlp_dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=16,
        min_lr=1e-8,
    )

    best_val_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, args.o_reg_lambda, args.np_reg_lambda)
        val_loss, val_acc = test(model, val_loader, criterion, device, args.o_reg_lambda, args.np_reg_lambda)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "frozen_bert_mlp_imdb.pt")
            print(f"Saved new best model to {"frozen_bert_mlp_imdb.pt"}")

    checkpoint = torch.load("frozen_bert_mlp_imdb.pt", map_location=device)
    model.load_state_dict(checkpoint)

    test_loss, test_acc = test(model, test_loader, criterion, device, args.o_reg_lambda, args.np_reg_lambda)
    print(f"Test loss={test_loss:.4f} | Test accuracy={test_acc:.4f}")


if __name__ == "__main__":
    main()


