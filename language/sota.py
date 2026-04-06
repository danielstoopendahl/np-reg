import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Metrics:
    loss: float
    accuracy: float


def normperserving_regularization(data, features, reg_lambda=1):
                        
    data_norm = torch.norm(data.view(data.size(0), -1), p='fro', dim=1)
                                    
    features_norm = torch.norm(features.view(features.size(0), -1), p='fro', dim=1)
                                                
    norm_diff_loss = F.mse_loss(data_norm, features_norm)
                                                            
    return reg_lambda * norm_diff_loss

def orthogonal_regularization(weight, reg_lambda=1e-4):
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

class FrozenBertMLP(nn.Module):
    def __init__(
        self,
        model_name: str,
        mlp_hidden_dim: int,
        mlp_dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.eval()

        self.first_linear = nn.Linear(hidden_size, mlp_hidden_dim)
        self.non_linear = nn.ReLU()
        self.second_linear = nn.Linear(mlp_hidden_dim, num_classes)

        # not used:
        self.dropout = nn.Dropout(mlp_dropout)

    def train(self, mode: bool = True):
        super().train(mode)
        self.bert.eval()
        return self

    def embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    def forward_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cls_embedding = self.embedding(input_ids, attention_mask)
        features = self.first_linear(cls_embedding)
        features = self.non_linear(features)
        return features, cls_embedding

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features, cls_embedding = self.forward_features(input_ids, attention_mask)
        logits = self.second_linear(features)
        return logits, features, cls_embedding


def build_dataloaders(tokenizer_name: str, max_length: int, batch_size: int, seed: int):
    dataset = load_dataset("imdb")
    split = dataset["train"].train_test_split(test_size=0.2, seed=seed)
    train_dataset = split["train"]
    val_dataset = split["test"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=12)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=12)

    return train_loader, val_loader, test_loader


def run_epoch(model, dataloader, criterion, optimizer, device, train: bool) -> Metrics:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels")
        if labels is None:
            labels = batch.get("label")
        if labels is None:
            raise KeyError(f"Batch missing label keys. Available keys: {list(batch.keys())}")
        labels = labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits, features, cls_embedding = model(input_ids=input_ids, attention_mask=attention_mask)
            npreg = normperserving_regularization(cls_embedding, features)
            oreg = orthogonal_regularization(model.first_linear.weight)
            # loss = criterion(logits, labels) + npreg 
            loss = criterion(logits, labels) + oreg
            # loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return Metrics(loss=total_loss / total, accuracy=correct / total)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = FrozenBertMLP(
        model_name=args.model_name,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        num_classes=2,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
        )
        val_metrics = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.4f}"
        )

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "mlp_hidden_dim": args.mlp_hidden_dim,
                    "mlp_dropout": args.mlp_dropout,
                    "max_length": args.max_length,
                },
                args.output_path,
            )
            print(f"Saved new best model to {args.output_path}")

    checkpoint = torch.load(args.output_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        train=False,
    )
    print(f"Test loss={test_metrics.loss:.4f} | Test accuracy={test_metrics.accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frozen pretrained BERT + MLP for IMDB sentiment")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=8200)
    parser.add_argument("--mlp-dropout", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", type=str, default="frozen_bert_mlp_imdb.pt")
    main(parser.parse_args())
