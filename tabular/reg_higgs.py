import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
import argparse
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMENSION = 1024
BATCH_SIZE = 65536 # [32768, 65536, 131072]
NP_REG_LAMBDA = 0 # [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]
O_REG_LAMBDA = 0
WEIGHT_DECAY=0
DROPOUT=0
BATCH_NORM=False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--np-reg-lambda", type=float, default=NP_REG_LAMBDA)
    parser.add_argument("--o-reg-lambda", type=float, default=O_REG_LAMBDA)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--batch-norm", action="store_true", default=BATCH_NORM)
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

def orthogonal_regularization(weight, o_reg_lambda):
    """
    Computes the orthogonal regularization penalty: 
    L = lambda * ||W^T W - I||_F^2
    """

    sym = torch.mm(weight.t(), weight)
    identity = torch.eye(sym.size(0), device=weight.device)
    loss_ortho = torch.norm(sym - identity, p='fro')**2
    
    return o_reg_lambda * loss_ortho

class SLFN(nn.Module):
    def __init__(self, dropout, use_batch_norm):
        super().__init__()
        self.fc1 = nn.Linear(21, HIDDEN_DIMENSION)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_DIMENSION, 1)
        self.drop = nn.Dropout(p=dropout)
        self.use_batch_norm = use_batch_norm
        self.batch_norm = nn.BatchNorm1d(HIDDEN_DIMENSION)
          
    def forward_features(self, x):
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.relu(x)
        x = self.drop(x)
        return x
      
    def forward(self, x):
        x = self.forward_features(x)
        return self.fc2(x)
    

def train_one_epoch(model, optimizer, loss_fn, train_loader, np_reg_lambda, o_reg_lambda):
    model.train()
    train_size = len(train_loader.dataset)
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad()
        features = model.forward_features(batch_X)
        logits = model.fc2(features)
        loss = loss_fn(logits, batch_y) + normperserving_regularization(batch_X, features, np_reg_lambda) + orthogonal_regularization(model.fc1.weight, o_reg_lambda)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    return running_loss / train_size

def validate(model, loss_fn, val_loader):
    with torch.no_grad():
        model.eval()
        val_loss_sum = 0.0
        all_probs = []
        all_labels = []
        total = 0

        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            val_logits = model(batch_X)
            batch_loss = loss_fn(val_logits, batch_y)

            batch_size = batch_X.size(0)
            val_loss_sum += batch_loss.item() * batch_size
            total += batch_size

            all_probs.append(torch.sigmoid(val_logits).view(-1).cpu())
            all_labels.append(batch_y.view(-1).cpu())

        val_probs = torch.cat(all_probs).numpy()
        val_labels = torch.cat(all_labels).numpy()
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        val_roc_auc = auc(fpr, tpr)
        val_loss = val_loss_sum / total

    return val_loss, val_roc_auc


def main():
    args = parse_args()
    processed_file = "data/dataset_higgs.pt"

    print("Loading dataset...")
    data = torch.load(processed_file)
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    y_train = y_train.float().view(-1, 1)
    y_val = y_val.float().view(-1, 1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    model = SLFN(args.dropout, args.batch_norm).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-2, weight_decay=args.weight_decay)
    min_lr = 1e-7
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=16, min_lr=min_lr,)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Training")
    best_val_loss = float("inf")
    best_val_roc_auc = float("-inf")
    for epoch in range(1000):
        avg_train_loss = train_one_epoch(model, opt, loss_fn, train_loader, args.np_reg_lambda, args.o_reg_lambda)
        val_loss, val_roc_auc = validate(model, loss_fn, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
        scheduler.step(avg_train_loss)
        current_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{1000} - Loss: {avg_train_loss:.4f} - Val ROC-AUC: {val_roc_auc:.4f} - Learning rate: {current_lr}")
        if current_lr < 2 * min_lr:
            print("Minimum learning rate reached. Stopping training.")
            break

    print("Training finished")
    print(f"Best Val Loss: {best_val_loss:.4f} - Best Val ROC-AUC: {best_val_roc_auc:.4f}")
    print(
        f"Run finished with arguments: \nbatch_size={args.batch_size}\n"
        f"np_reg_lambda={args.np_reg_lambda}\n"
        f"o_reg_lambda={args.o_reg_lambda}\n"
        f"weight_degay={args.weight_decay}\n"
        f"dropout={args.dropout}\n"
        f"batchnorm={args.batch_norm}\n"
    )


main()