import argparse
import os
import random
import time
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


DATASET_DIR = os.path.join(os.path.dirname(__file__), "Yelp JSON", "yelp_dataset")
BUSINESS_FILE = "yelp_academic_dataset_business.json"
REVIEW_FILE = "yelp_academic_dataset_review.json"
BACKBONE = "bert-base-uncased"
MAX_LENGTH = 512
BATCH_SIZE = 128
NUM_WORKERS = 8
SEED = 42
TEST_RATIO = 0.1
LOG_EVERY_BATCHES = 1
OUTPUT_DIR = os.path.abspath(
	os.path.join(os.path.dirname(__file__), "data")
)
EMBEDDINGS_PT = os.path.join(OUTPUT_DIR, "yelp_embeddings.pt")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Precompute BERT embeddings for Yelp restaurant reviews",
	)
	parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
	parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR)
	parser.add_argument("--backbone", type=str, default=BACKBONE)
	parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
	parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
	parser.add_argument("--seed", type=int, default=SEED)
	parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
	parser.add_argument("--max-examples", type=int, default=200000)
	parser.add_argument("--embeddings-pt", type=str, default=EMBEDDINGS_PT)
	return parser.parse_args()


def set_seed(seed: Optional[int]) -> None:
	if seed is None:
		return
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
	worker_seed = torch.initial_seed() % 2**32
	random.seed(worker_seed)
	np.random.seed(worker_seed)


def get_primary_split(dataset: object) -> Dataset:
	if isinstance(dataset, DatasetDict):
		if "train" in dataset:
			return dataset["train"]
		return dataset[next(iter(dataset.keys()))]
	if isinstance(dataset, Dataset):
		return dataset
	raise TypeError("Unsupported dataset type.")


def normalize_categories(categories: Optional[object]) -> List[str]:
	if categories is None:
		return []
	if isinstance(categories, list):
		return [str(cat).strip() for cat in categories if cat]
	if isinstance(categories, str):
		return [cat.strip() for cat in categories.split(",") if cat.strip()]
	return []


def is_restaurant(categories: Iterable[str]) -> bool:
	for category in categories:
		if "restaurant" in category.lower():
			return True
	return False


def build_restaurant_id_set(business_split: Dataset) -> Set[str]:
	restaurant_ids: Set[str] = set()
	for record in business_split:
		categories = normalize_categories(record.get("categories"))
		if not categories:
			continue
		if is_restaurant(categories):
			business_id = record.get("business_id")
			if business_id:
				restaurant_ids.add(str(business_id))
	if not restaurant_ids:
		raise ValueError("No restaurant business IDs found in Yelp JSON dataset.")
	return restaurant_ids


def resolve_review_text(record: dict) -> str:
	text = record.get("text")
	if text:
		return str(text)
	review_text = record.get("review_text")
	if review_text:
		return str(review_text)
	return ""


def resolve_star_label(stars: object) -> Optional[int]:
	if stars is None:
		return None
	try:
		star_value = int(float(stars))
	except (TypeError, ValueError):
		return None
	if not 1 <= star_value <= 5:
		return None
	return star_value - 1


def filter_reviews_to_restaurants(
	review_split: Dataset,
	restaurant_ids: Set[str],
	max_examples: Optional[int],
) -> Dataset:
	def is_restaurant_review(record: dict) -> bool:
		business_id = record.get("business_id")
		return business_id is not None and str(business_id) in restaurant_ids

	filtered = review_split.filter(is_restaurant_review)
	if max_examples is not None:
		max_examples = max(0, int(max_examples))
		if max_examples > 0:
			filtered = filtered.select(range(min(max_examples, len(filtered))))
	return filtered


def build_review_dataset(
	review_split: Dataset,
	restaurant_ids: Set[str],
	max_examples: Optional[int],
) -> Dataset:
	filtered = filter_reviews_to_restaurants(review_split, restaurant_ids, max_examples)

	def to_text_and_label(record: dict) -> dict:
		text = resolve_review_text(record)
		label = resolve_star_label(record.get("stars"))
		return {"text": text, "labels": label}

	processed = filtered.map(
		to_text_and_label,
		remove_columns=filtered.column_names,
	)
	processed = processed.filter(
		lambda record: bool(record.get("text")) and record.get("labels") is not None
	)
	return processed


def tokenize_reviews(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
	def tokenize_batch(batch: dict) -> dict:
		return tokenizer(
			batch["text"],
			truncation=True,
			padding="max_length",
			max_length=max_length,
		)

	tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
	tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
	return tokenized


def build_embedding_loader(dataset: Dataset, seed: int, batch_size: int) -> DataLoader:
	generator = torch.Generator().manual_seed(seed)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=NUM_WORKERS,
		worker_init_fn=seed_worker,
		generator=generator,
	)


def compute_embeddings(
	backbone: torch.nn.Module,
	dataloader: DataLoader,
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
	backbone.eval()
	all_embeddings = []
	all_labels = []
	start = time.perf_counter()
	last_log = start

	with torch.no_grad():
		for step, batch in enumerate(dataloader, start=1):
			input_ids = batch["input_ids"].to(device)
			attention_mask = batch["attention_mask"].to(device)
			labels = batch["labels"]

			outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
			token_embeddings = outputs.last_hidden_state
			mask = attention_mask.unsqueeze(-1).float()
			summed = (token_embeddings * mask).sum(dim=1)
			denom = mask.sum(dim=1).clamp(min=1e-9)
			pooled_embeddings = summed / denom

			all_embeddings.append(pooled_embeddings.cpu().numpy())
			all_labels.append(labels.cpu().numpy())

			if step % LOG_EVERY_BATCHES == 0:
				now = time.perf_counter()
				elapsed = now - last_log
				total = now - start
				print(
					f"Embedding batches: {step}, +{elapsed:.1f}s, total {total:.1f}s"
				)
				last_log = now

	embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
	labels = np.concatenate(all_labels, axis=0)
	return embeddings, labels


def build_embedding_dataset(
	split: Dataset,
	tokenizer,
	backbone: torch.nn.Module,
	device: torch.device,
	seed: int,
	batch_size: int,
	max_length: int,
) -> Dataset:
	tokenized = tokenize_reviews(split, tokenizer, max_length)
	
	loader = build_embedding_loader(tokenized, seed, batch_size)
	embeddings, labels = compute_embeddings(backbone, loader, device)
	return Dataset.from_dict({"embeddings": embeddings, "labels": labels})


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	start_total = time.perf_counter()

	business_path = os.path.join(args.dataset_dir, BUSINESS_FILE)
	review_path = os.path.join(args.dataset_dir, REVIEW_FILE)
	load_start = time.perf_counter()
	business_dataset = load_dataset("json", data_files={"train": business_path})
	review_dataset = load_dataset("json", data_files={"train": review_path})
	print(f"Loaded JSON in {time.perf_counter() - load_start:.1f}s")
	business_split = get_primary_split(business_dataset)
	review_split = get_primary_split(review_dataset)

	filter_start = time.perf_counter()
	restaurant_ids = build_restaurant_id_set(business_split)
	reviews = build_review_dataset(review_split, restaurant_ids, args.max_examples)
	print(f"Filtered restaurant reviews: {len(reviews)}")
	print(f"Filtering+processing took {time.perf_counter() - filter_start:.1f}s")

	split_start = time.perf_counter()
	split_dataset = reviews.train_test_split(test_size=args.test_ratio, seed=args.seed)
	print(f"Train/test split took {time.perf_counter() - split_start:.1f}s")
	train_split = split_dataset["train"]
	test_split = split_dataset["test"]

	model_start = time.perf_counter()
	tokenizer = AutoTokenizer.from_pretrained(args.backbone)
	backbone = AutoModel.from_pretrained(args.backbone).to(device)
	for param in backbone.parameters():
		param.requires_grad = False
	print(f"Loaded backbone in {time.perf_counter() - model_start:.1f}s")

	train_start = time.perf_counter()
	train_dataset = build_embedding_dataset(
		train_split,
		tokenizer,
		backbone,
		device,
		seed=args.seed,
		batch_size=args.batch_size,
		max_length=args.max_length,
	)
	print(f"Train embeddings took {time.perf_counter() - train_start:.1f}s")

	test_start = time.perf_counter()
	test_dataset = build_embedding_dataset(
		test_split,
		tokenizer,
		backbone,
		device,
		seed=args.seed,
		batch_size=args.batch_size,
		max_length=args.max_length,
	)
	print(f"Test embeddings took {time.perf_counter() - test_start:.1f}s")

	os.makedirs(args.output_dir, exist_ok=True)
	DatasetDict(
		{
			"train": train_dataset,
			"test": test_dataset,
		}
	).save_to_disk(args.output_dir)

	pt_payload = {
		"train_embeddings": torch.from_numpy(train_dataset["embeddings"]),
		"train_labels": torch.from_numpy(train_dataset["labels"]),
		"test_embeddings": torch.from_numpy(test_dataset["embeddings"]),
		"test_labels": torch.from_numpy(test_dataset["labels"]),
	}
	torch.save(pt_payload, args.embeddings_pt)

	print(f"Saved embeddings dataset to: {args.output_dir}")
	print(f"Saved embeddings tensors to: {args.embeddings_pt}")
	print(f"Total runtime: {time.perf_counter() - start_total:.1f}s")


if __name__ == "__main__":
	main()
