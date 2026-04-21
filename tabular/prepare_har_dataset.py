import os
import shutil
import urllib.request
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split
import torch


DEFAULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)


def _load_har_split(split_name):
    x_path = f"tabular/data/raw_har/UCI HAR Dataset/{split_name}/X_{split_name}.txt"
    y_path = f"tabular/data/raw_har/UCI HAR Dataset/{split_name}/y_{split_name}.txt"

    x = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int64) - 1  # labels are 1..6 in source files

    return x, y


def _download_and_extract(data_url):
    raw_data_dir = "tabular/data/raw_har"
    dataset_root = "tabular/data/raw_har/UCI HAR Dataset"
    if os.path.exists(dataset_root):
        return dataset_root

    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    os.makedirs(raw_data_dir, exist_ok=True)

    archive_path = "tabular/data/raw_har/uci_har.zip"
    print(f"Downloading UCI HAR from {data_url}")
    urllib.request.urlretrieve(data_url, archive_path)

    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(raw_data_dir)

    if not os.path.exists(dataset_root):
        raise FileNotFoundError("Extracted archive does not contain expected 'UCI HAR Dataset' folder")

    return dataset_root


def prepare_dataset(processed_path, data_url):

    _download_and_extract(data_url)

    x_train_np, y_train_np = _load_har_split("train")
    x_test_np, y_test_np = _load_har_split("test")

    # Normalize using train statistics to avoid leakage from test split.
    train_mean = x_train_np.mean(axis=0, keepdims=True)
    train_std = x_train_np.std(axis=0, keepdims=True)
    train_std[train_std < 1e-6] = 1.0

    x_train_np = (x_train_np - train_mean) / train_std
    x_test_np = (x_test_np - train_mean) / train_std

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_np,
        y_train_np,
        test_size=0.2,
        stratify=y_train_np,
        random_state=42,
    )

    dataset = {
        "X_train": torch.from_numpy(x_train),
        "y_train": torch.from_numpy(y_train),
        "X_val": torch.from_numpy(x_val),
        "y_val": torch.from_numpy(y_val),
        "X_test": torch.from_numpy(x_test_np),
        "y_test": torch.from_numpy(y_test_np),
    }



    torch.save(dataset, processed_path)
    print(f"Saved preprocessed dataset to {processed_path}")
    return dataset

prepare_dataset("tabular/data/dataset_har.pt", DEFAULT_URL)