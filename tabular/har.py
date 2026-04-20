import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_URL = (
	"https://archive.ics.uci.edu/ml/machine-learning-databases/"
	"00240/UCI%20HAR%20Dataset.zip"
)

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

def parse_args():
	parser = argparse.ArgumentParser(description="Train an SLFN on UCI HAR")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=120)
	parser.add_argument("--hidden-dim", type=int, default=1024)
	parser.add_argument("--np-reg-lambda", type=float, default=0)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--dropout", type=float, default=0.2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--data-url", type=str, default=DEFAULT_URL)
	parser.add_argument("--force-redownload", action="store_true")
	parser.add_argument("--processed-name", type=str, default="dataset_har.pt")
	parser.add_argument(
		"--save-model",
		action="store_true",
		help="Save best checkpoint to tabular/models/slfn_har.pt",
	)
	return parser.parse_args()


class SLFN(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_classes, dropout):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)
		self.fc2 = nn.Linear(hidden_dim, num_classes)

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		return self.fc2(x)


def _load_har_split(split_dir):
	x_path = split_dir / f"X_{split_dir.name}.txt"
	y_path = split_dir / f"y_{split_dir.name}.txt"

	x = np.loadtxt(x_path, dtype=np.float32)
	y = np.loadtxt(y_path, dtype=np.int64) - 1  # labels are 1..6 in source files

	return x, y


def _download_and_extract(raw_data_dir, data_url, force_redownload):
	dataset_root = raw_data_dir / "UCI HAR Dataset"
	if dataset_root.exists() and not force_redownload:
		return dataset_root

	if raw_data_dir.exists() and force_redownload:
		shutil.rmtree(raw_data_dir)
	raw_data_dir.mkdir(parents=True, exist_ok=True)

	archive_path = raw_data_dir / "uci_har.zip"
	print(f"Downloading UCI HAR from {data_url}")
	urllib.request.urlretrieve(data_url, archive_path)

	print("Extracting archive...")
	with zipfile.ZipFile(archive_path, "r") as zip_ref:
		zip_ref.extractall(raw_data_dir)

	if not dataset_root.exists():
		raise FileNotFoundError("Extracted archive does not contain expected 'UCI HAR Dataset' folder")

	return dataset_root


def prepare_dataset(processed_path, data_url, force_redownload=False):
	if processed_path.exists() and not force_redownload:
		print(f"Loading preprocessed dataset from {processed_path}")
		return torch.load(processed_path)

	data_dir = processed_path.parent
	raw_data_dir = data_dir / "raw_har"
	dataset_root = _download_and_extract(raw_data_dir, data_url, force_redownload)

	x_train_np, y_train_np = _load_har_split(dataset_root / "train")
	x_test_np, y_test_np = _load_har_split(dataset_root / "test")

	# Normalize using train statistics to avoid leakage from test split.
	train_mean = x_train_np.mean(axis=0, keepdims=True)
	train_std = x_train_np.std(axis=0, keepdims=True)
	train_std[train_std < 1e-6] = 1.0

	x_train_np = (x_train_np - train_mean) / train_std
	x_test_np = (x_test_np - train_mean) / train_std

	dataset = {
		"X_train": torch.from_numpy(x_train_np),
		"y_train": torch.from_numpy(y_train_np),
		"X_test": torch.from_numpy(x_test_np),
		"y_test": torch.from_numpy(y_test_np),
	}

	torch.save(dataset, processed_path)
	print(f"Saved preprocessed dataset to {processed_path}")
	return dataset


def accuracy_from_logits(logits, targets):
	preds = logits.argmax(dim=1)
	return (preds == targets).float().mean().item()


def train_one_epoch(model, optimizer, loss_fn, x_train, y_train, batch_size, device, np_reg_lambda):
	model.train()
	n_samples = x_train.size(0)
	permutation = torch.randperm(n_samples, device=device)
	epoch_loss = 0.0
	epoch_correct = 0

	for i in range(0, n_samples, batch_size):
		idx = permutation[i : i + batch_size]
		xb = x_train[idx]
		yb = y_train[idx]

		optimizer.zero_grad()
		first_layer_features = model.fc1(xb)
		logits = model.fc2(model.dropout(model.relu(first_layer_features)))
		loss = loss_fn(logits, yb) + normperserving_regularization(xb, first_layer_features, np_reg_lambda)
		loss.backward()
		optimizer.step()

		epoch_loss += loss.item() * xb.size(0)
		epoch_correct += (logits.argmax(dim=1) == yb).sum().item()

	return epoch_loss / n_samples, epoch_correct / n_samples


@torch.no_grad()
def evaluate(model, loss_fn, x, y):
	model.eval()
	logits = model(x)
	loss = loss_fn(logits, y).item()
	acc = accuracy_from_logits(logits, y)
	return loss, acc


def main():
	args = parse_args()

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	script_dir = Path(__file__).resolve().parent
	data_dir = script_dir / "data"
	models_dir = script_dir / "models"
	data_dir.mkdir(parents=True, exist_ok=True)
	models_dir.mkdir(parents=True, exist_ok=True)

	processed_path = data_dir / args.processed_name
	dataset = prepare_dataset(
		processed_path=processed_path,
		data_url=args.data_url,
		force_redownload=args.force_redownload,
	)

	x_train = dataset["X_train"].to(device)
	y_train = dataset["y_train"].to(device)
	x_test = dataset["X_test"].to(device)
	y_test = dataset["y_test"].to(device)

	input_dim = x_train.size(1)
	num_classes = int(torch.max(y_train).item() + 1)

	model = SLFN(
		input_dim=input_dim,
		hidden_dim=args.hidden_dim,
		num_classes=num_classes,
		dropout=args.dropout,
	).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	loss_fn = nn.CrossEntropyLoss()

	best_test_acc = -1.0
	best_state_dict = None

	print("Starting SLFN training on UCI HAR...")
	for epoch in range(args.epochs):
		train_loss, train_acc = train_one_epoch(
			model=model,
			optimizer=optimizer,
			loss_fn=loss_fn,
			x_train=x_train,
			y_train=y_train,
			batch_size=args.batch_size,
			device=device,
			np_reg_lambda=args.np_reg_lambda,
		)
		test_loss, test_acc = evaluate(model, loss_fn, x_test, y_test)
		scheduler.step()

		if test_acc > best_test_acc:
			best_test_acc = test_acc
			best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

		lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch + 1:03d}/{args.epochs} | "
			f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
			f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} | "
			f"lr={lr:.2e}"
		)

	print(f"Finished training. Best test accuracy: {best_test_acc:.4f}")

	if args.save_model and best_state_dict is not None:
		save_path = models_dir / "slfn_har.pt"
		torch.save(
			{
				"state_dict": best_state_dict,
				"input_dim": input_dim,
				"hidden_dim": args.hidden_dim,
				"num_classes": num_classes,
				"dropout": args.dropout,
			},
			save_path,
		)
		print(f"Saved best model to {save_path}")


if __name__ == "__main__":
	main()
