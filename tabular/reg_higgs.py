import torch
from torch import nn
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
        x = self.relu(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.drop(x)
        return x
      
    def forward(self, x):
        x = self.forward_features(x)
        return self.fc2(x)
    

def train_one_epoch(model, optimizer, loss_fn, X_train, y_train, batch_size, np_reg_lambda, o_reg_lambda):
    model.train()
    train_size = X_train.size(0)
    running_loss = 0.0
    perm = torch.randperm(train_size, device=X_train.device)
    for start in range(0, train_size, batch_size):
        idx = perm[start:start + batch_size]
        batch_X = X_train[idx]
        batch_y = y_train[idx]
        optimizer.zero_grad()
        features = model.forward_features(batch_X)
        logits = model.fc2(features)
        loss = loss_fn(logits, batch_y) + normperserving_regularization(batch_X, features, np_reg_lambda) + orthogonal_regularization(model.fc1.weight, o_reg_lambda)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)

    return running_loss / train_size

def validate(model, loss_fn, X_val, y_val):
    with torch.no_grad():
        model.eval()
        val_logits = model(X_val)
        val_loss = loss_fn(val_logits, y_val)
        val_probs = torch.sigmoid(val_logits).view(-1).cpu().numpy()
        val_labels = y_val.view(-1).cpu().numpy()
        fpr, tpr, _ = roc_curve(val_labels, val_probs)
        val_roc_auc = auc(fpr, tpr)

    return val_loss.item(), val_roc_auc


def main():
    args = parse_args()
    processed_file = "data/dataset_higgs.pt"

    print("Loading dataset...")
    data = torch.load(processed_file)
    X_train, y_train = data['X_train'].to(device), data['y_train'].to(device)
    X_val, y_val = data['X_val'].to(device), data['y_val'].to(device)
    y_train = y_train.float().view(-1, 1)
    y_val = y_val.float().view(-1, 1)

    model = SLFN(args.dropout, args.batch_norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
    min_lr = 1e-7
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=16, min_lr=min_lr,)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Training")
    best_val_loss = float("inf")
    best_val_roc_auc = float("-inf")
    for epoch in range(1000):
        avg_train_loss = train_one_epoch(model, opt, loss_fn, X_train, y_train, args.batch_size, args.np_reg_lambda, args.o_reg_lambda)
        val_loss, val_roc_auc = validate(model, loss_fn, X_val, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
        scheduler.step(val_loss)
        current_lr = opt.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{1000} - Loss: {avg_train_loss:.4f} - Val ROC-AUC: {val_roc_auc:.4f}")
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