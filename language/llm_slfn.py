import argparse
import math
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
	parser = argparse.ArgumentParser(
		description="Simple NPLM (fixed-window feed-forward language model)."
	)
	parser.add_argument("--text-path", type=str, required=True)
	parser.add_argument("--context-size", type=int, default=15)
	parser.add_argument("--embed-dim", type=int, default=128)
	parser.add_argument("--hidden-dim", type=int, default=512)
	parser.add_argument("--num-layers", type=int, default=4)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
	parser.add_argument("--global-context", choices=["none", "mean", "conv"], default="mean")
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-5)
	parser.add_argument("--max-vocab", type=int, default=30000)
	parser.add_argument("--min-freq", type=int, default=2)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def tokenize(text: str):
	return text.lower().split()


def split_tokens(tokens, train_ratio=0.8, val_ratio=0.1):
	n = len(tokens)
	n_train = int(n * train_ratio)
	n_val = int(n * val_ratio)
	train_tokens = tokens[:n_train]
	val_tokens = tokens[n_train:n_train + n_val]
	test_tokens = tokens[n_train + n_val:]
	return train_tokens, val_tokens, test_tokens


def build_vocab(tokens, max_vocab=30000, min_freq=2):
	counter = Counter(tokens)
	kept = [tok for tok, freq in counter.items() if freq >= min_freq]
	kept.sort(key=lambda t: counter[t], reverse=True)
	kept = kept[:max_vocab - 2]

	idx2tok = ["<unk>", "<pad>"] + kept
	tok2idx = {tok: idx for idx, tok in enumerate(idx2tok)}
	return tok2idx, idx2tok


def numericalize(tokens, tok2idx):
	unk = tok2idx["<unk>"]
	return [tok2idx.get(tok, unk) for tok in tokens]


def make_ngram_dataset(token_ids, context_size):
	if len(token_ids) <= context_size:
		raise ValueError("Not enough tokens for the selected context size.")

	xs = []
	past = []
	ys = []
	for i in range(context_size, len(token_ids)):
		xs.append(token_ids[i - context_size:i])
		past.append(token_ids[:i - context_size])
		ys.append(token_ids[i])

	x_tensor = torch.tensor(xs, dtype=torch.long)
	# We keep variable-length distant prefixes in a Python list to avoid huge padding tensors.
	past_tensor = past
	y_tensor = torch.tensor(ys, dtype=torch.long)
	return x_tensor, past_tensor, y_tensor


class NPLMDataset(torch.utils.data.Dataset):
	def __init__(self, contexts, distant_prefixes, targets):
		self.contexts = contexts
		self.distant_prefixes = distant_prefixes
		self.targets = targets

	def __len__(self):
		return self.targets.size(0)

	def __getitem__(self, idx):
		return self.contexts[idx], self.distant_prefixes[idx], self.targets[idx]


def collate_nplm(batch):
	contexts = torch.stack([b[0] for b in batch], dim=0)
	distant = [b[1] for b in batch]
	targets = torch.stack([b[2] for b in batch], dim=0)
	return contexts, distant, targets


class ResidualBlock(nn.Module):
	def __init__(self, hidden_dim, dropout, activation):
		super().__init__()
		self.norm = nn.LayerNorm(hidden_dim)
		self.ff = nn.Linear(hidden_dim, hidden_dim)
		self.dropout = nn.Dropout(dropout)
		self.activation = nn.ReLU() if activation == "relu" else nn.Tanh()

	def forward(self, x):
		residual = x
		x = self.norm(x)
		x = self.ff(x)
		x = self.activation(x)
		x = self.dropout(x)
		return x + residual


class NPLM(nn.Module):
	"""
	Bengio-style NPLM core:
	- Look up embeddings for a fixed context window
	- Concatenate embeddings
	- Feed through MLP
	- Predict next-token distribution
	"""

	def __init__(
		self,
		vocab_size,
		context_size,
		embed_dim,
		hidden_dim,
		num_layers,
		dropout,
		activation,
		global_context,
	):
		super().__init__()
		self.global_context = global_context
		self.context_size = context_size
		self.embedding = nn.Embedding(vocab_size, embed_dim)
		input_tokens = context_size + (1 if global_context != "none" else 0)
		self.input_proj = nn.Linear(input_tokens * embed_dim, hidden_dim)
		self.in_norm = nn.LayerNorm(hidden_dim)
		self.in_act = nn.ReLU() if activation == "relu" else nn.Tanh()
		self.in_dropout = nn.Dropout(dropout)
		self.blocks = nn.ModuleList(
			[ResidualBlock(hidden_dim, dropout, activation) for _ in range(max(num_layers - 1, 0))]
		)
		if global_context == "conv":
			# Lightweight learned kernel over distant context as in the paper's global context variant.
			self.global_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
		else:
			self.global_conv = None
		self.output = nn.Linear(hidden_dim, vocab_size)

	def _global_embedding(self, distant_prefixes, device):
		if self.global_context == "none":
			return None

		global_vecs = []
		for prefix in distant_prefixes:
			if len(prefix) == 0:
				global_vecs.append(torch.zeros(self.embedding.embedding_dim, device=device))
				continue

			prefix_ids = torch.tensor(prefix, dtype=torch.long, device=device)
			prefix_emb = self.embedding(prefix_ids)

			if self.global_context == "mean":
				g = prefix_emb.mean(dim=0)
			else:
				conv_in = prefix_emb.transpose(0, 1).unsqueeze(0)
				conv_out = self.global_conv(conv_in).squeeze(0).transpose(0, 1)
				g = conv_out.mean(dim=0)

			global_vecs.append(g)

		return torch.stack(global_vecs, dim=0)

	def forward(self, x, distant_prefixes):
		local_emb = self.embedding(x)
		if self.global_context == "none":
			emb = local_emb.reshape(local_emb.size(0), -1)
		else:
			global_emb = self._global_embedding(distant_prefixes, x.device)
			merged = torch.cat([local_emb, global_emb.unsqueeze(1)], dim=1)
			emb = merged.reshape(merged.size(0), -1)

		h = self.input_proj(emb)
		h = self.in_norm(h)
		h = self.in_act(h)
		h = self.in_dropout(h)
		for block in self.blocks:
			h = block(h)
		return self.output(h)


def run_epoch(model, loader, criterion, optimizer, device):
	is_train = optimizer is not None
	model.train(is_train)

	total_loss = 0.0
	total_tokens = 0

	for x, distant, y in loader:
		x = x.to(device)
		y = y.to(device)

		if is_train:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(is_train):
			logits = model(x, distant)
			loss = criterion(logits, y)
			if is_train:
				loss.backward()
				optimizer.step()

		batch_size = y.size(0)
		total_loss += loss.item() * batch_size
		total_tokens += batch_size

	avg_loss = total_loss / max(total_tokens, 1)
	ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
	return avg_loss, ppl


def build_loaders(args):
	with open(args.text_path, "r", encoding="utf-8") as f:
		text = f.read()

	tokens = tokenize(text)
	if len(tokens) < 100:
		raise ValueError("Input text is too small. Provide a larger corpus.")

	train_tokens, val_tokens, test_tokens = split_tokens(tokens)
	tok2idx, idx2tok = build_vocab(
		train_tokens,
		max_vocab=args.max_vocab,
		min_freq=args.min_freq,
	)

	train_ids = numericalize(train_tokens, tok2idx)
	val_ids = numericalize(val_tokens, tok2idx)
	test_ids = numericalize(test_tokens, tok2idx)

	train_ctx, train_distant, train_y = make_ngram_dataset(train_ids, args.context_size)
	val_ctx, val_distant, val_y = make_ngram_dataset(val_ids, args.context_size)
	test_ctx, test_distant, test_y = make_ngram_dataset(test_ids, args.context_size)

	train_ds = NPLMDataset(train_ctx, train_distant, train_y)
	val_ds = NPLMDataset(val_ctx, val_distant, val_y)
	test_ds = NPLMDataset(test_ctx, test_distant, test_y)

	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=0,
		collate_fn=collate_nplm,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=collate_nplm,
	)
	test_loader = DataLoader(
		test_ds,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,
		collate_fn=collate_nplm,
	)

	return train_loader, val_loader, test_loader, len(idx2tok)


def main():
	args = parse_args()
	set_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, val_loader, test_loader, vocab_size = build_loaders(args)
	model = NPLM(
		vocab_size=vocab_size,
		context_size=args.context_size,
		embed_dim=args.embed_dim,
		hidden_dim=args.hidden_dim,
		num_layers=args.num_layers,
		dropout=args.dropout,
		activation=args.activation,
		global_context=args.global_context,
	).to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr=args.lr,
		weight_decay=args.weight_decay,
	)

	best_val_loss = float("inf")
	best_state = None

	print(f"Device: {device}")
	print(f"Vocab size: {vocab_size}")
	print(f"Context size: {args.context_size}")
	print(f"Layers: {args.num_layers}")
	print(f"Global context: {args.global_context}")

	for epoch in range(1, args.epochs + 1):
		train_loss, train_ppl = run_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_ppl = run_epoch(model, val_loader, criterion, None, device)

		print(
			f"Epoch {epoch:02d}/{args.epochs} | "
			f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} | "
			f"val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
		)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

	if best_state is not None:
		model.load_state_dict(best_state)

	test_loss, test_ppl = run_epoch(model, test_loader, criterion, None, device)
	print(f"Test loss={test_loss:.4f} | Test ppl={test_ppl:.2f}")


if __name__ == "__main__":
	main()
