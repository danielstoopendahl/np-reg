import torch
from torch import nn
from sklearn.metrics import roc_curve, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMENSION = 1024
BATCH_SIZE = 65536
EPOCHS = 200

class SLFN(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(28, HIDDEN_DIMENSION)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(HIDDEN_DIMENSION, 1)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		return self.fc2(x)

processed_file = "dataset_higgs.pt"

print("Loading pre-processed dataset...")
data = torch.load(processed_file)
X_train, y_train = data['X_train'].to(device), data['y_train'].to(device)
X_val, y_val = data['X_val'].to(device), data['y_val'].to(device)

train_size = X_train.size(0)

model = SLFN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

print("Training")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    perm = torch.randperm(train_size, device=device)
    for start in range(0, train_size, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        batch_X = X_train[idx]
        batch_y = y_train[idx]
        opt.zero_grad()
        loss = loss_fn(model(batch_X), batch_y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * batch_X.size(0)
    
    avg_train_loss = running_loss / train_size

    with torch.no_grad():
        model.eval()
        val_logits = model(X_val)
        val_loss = loss_fn(val_logits, y_val)
        preds = (torch.sigmoid(val_logits) > 0.5).float()
        val_acc = (preds == y_val).float().mean().item()
    
    print(f"Epoch {epoch+1:02d}/{EPOCHS} - Loss: {avg_train_loss:.4f} - Val Accuracy: {val_acc:.4f}")

print("Training finished")

# Calculate ROC
with torch.no_grad():
    model.eval()
    test_logits = model(X_val)
    test_probs = torch.sigmoid(test_logits).cpu().numpy()
    test_labels = y_val.cpu().numpy()
    
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nROC-AUC Score: {roc_auc:.4f}")


