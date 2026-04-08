import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding

SEED = 42

def build_tokenized_datasets(tokenizer_name: str):
    dataset = load_dataset("imdb")
    split = dataset["train"].train_test_split(test_size=0.2, seed=SEED)
    train_dataset = split["train"]
    val_dataset = split["test"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    val_dataset = val_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    return train_dataset, val_dataset, test_dataset, tokenizer


def precompute_split_embeddings(bert_model, dataloader, device):
    all_embeddings = []
    all_labels = []

    bert_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            if labels is None:
                labels = batch.get("label")
            if labels is None:
                raise KeyError(f"Batch missing label keys. Available keys: {list(batch.keys())}")

            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embedding.cpu())
            all_labels.append(labels.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embeddings, labels


def build_embedding_cache(device):
    model_name = "bert-base-uncased"
    embedding_path = "embeddings/imdb_bert_embeddings.pt"
        
    print("Building tokenized IMDB dataset for embedding precompute...")
    train_dataset, val_dataset, test_dataset, tokenizer = build_tokenized_datasets(tokenizer_name=model_name)

    bert_model = AutoModel.from_pretrained(model_name).to(device)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False,collate_fn=collator)
    val_loader = train_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=collator)
    test_loader = train_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collator)

    print("Precomputing train embeddings...")
    train_embeddings, train_labels = precompute_split_embeddings(bert_model, train_loader, device)
    print("Precomputing validation embeddings...")
    val_embeddings, val_labels = precompute_split_embeddings(bert_model, val_loader, device)
    print("Precomputing test embeddings...")
    test_embeddings, test_labels = precompute_split_embeddings(bert_model, test_loader, device)

    cache = {
        "train_embeddings": train_embeddings,
        "train_labels": train_labels,
        "val_embeddings": val_embeddings,
        "val_labels": val_labels,
        "test_embeddings": test_embeddings,
        "test_labels": test_labels,
    }
    torch.save(cache, embedding_path)
    print(f"Saved embedding cache to {embedding_path}")
    return cache

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_embedding_cache(device)