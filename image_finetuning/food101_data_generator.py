import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Food101
from torchvision.models import ResNet50_Weights, resnet50


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "data"))
BATCH_SIZE = 128
NUM_WORKERS = 4


def build_feature_extractor(device: torch.device) -> Tuple[nn.Module, torch.nn.Module]:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    return model, weights.transforms()


@torch.no_grad()
def extract_embeddings(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        batch_features = model(images).cpu()
        features.append(batch_features)
        labels.append(batch_labels.cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def create_embeddings_split(
    root: str,
    split: str,
    model: nn.Module,
    transform: torch.nn.Module,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = Food101(
        root=root,
        split=split,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_init_fn,
    )
    return extract_embeddings(loader, model, device)


def create_food101_embeddings() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_feature_extractor(device)

    train_features, train_labels = create_embeddings_split(
        root=DATA_ROOT,
        split="train",
        model=model,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )
    test_features, test_labels = create_embeddings_split(
        root=DATA_ROOT,
        split="test",
        model=model,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        device=device,
    )

    os.makedirs(DATA_ROOT, exist_ok=True)
    embeddings_path = os.path.join(DATA_ROOT, "food101_embeddings.pt")
    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "test_features": test_features,
            "test_labels": test_labels,
        },
        embeddings_path,
    )


if __name__ == "__main__":
    create_food101_embeddings()
