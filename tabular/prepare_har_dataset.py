import os
import shutil
import urllib.request
import zipfile

import numpy as np
import torch


DEFAULT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)


def _load_har_split(split_name):
    x_path = f"data/raw_har/UCI HAR Dataset/{split_name}/X_{split_name}.txt"
    y_path = f"data/raw_har/UCI HAR Dataset/{split_name}/y_{split_name}.txt"

    x = np.loadtxt(x_path, dtype=np.float32)
    y = np.loadtxt(y_path, dtype=np.int64) - 1  # labels are 1..6 in source files

    return x, y


def _load_subject_ids(split_name):
    subject_path = f"data/raw_har/UCI HAR Dataset/{split_name}/subject_{split_name}.txt"
    return np.loadtxt(subject_path, dtype=np.int64)


def _select_validation_subjects(subject_ids, target_fraction=0.2):
    total = int(subject_ids.shape[0])
    if total < 2:
        raise ValueError("Need at least 2 samples to build train/validation split")

    unique_subjects, counts = np.unique(subject_ids, return_counts=True)
    subject_counts = {int(s): int(c) for s, c in zip(unique_subjects, counts)}

    target = int(round(total * target_fraction))
    # Subset-sum DP over subject sample counts to get as close as possible to target.
    chosen_by_sum = {0: []}
    for subject in sorted(subject_counts):
        count = subject_counts[subject]
        updated = dict(chosen_by_sum)
        for running_sum, selected in chosen_by_sum.items():
            new_sum = running_sum + count
            if new_sum not in updated:
                updated[new_sum] = selected + [subject]
        chosen_by_sum = updated

    candidate_sums = [s for s in chosen_by_sum if 0 < s < total]
    if not candidate_sums:
        raise ValueError("Could not build a non-empty subject-disjoint validation split")

    best_sum = min(candidate_sums, key=lambda s: (abs(s - target), -s))
    return np.array(chosen_by_sum[best_sum], dtype=np.int64)


def _download_and_extract(data_url):
    raw_data_dir = "data/raw_har"
    dataset_root = "data/raw_har/UCI HAR Dataset"
    if os.path.exists(dataset_root):
        return dataset_root

    if os.path.exists(raw_data_dir):
        shutil.rmtree(raw_data_dir)
    os.makedirs(raw_data_dir, exist_ok=True)

    archive_path = "data/raw_har/uci_har.zip"
    print(f"Downloading UCI HAR from {data_url}")
    urllib.request.urlretrieve(data_url, archive_path)

    print("Extracting archive...")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(raw_data_dir)

    if not os.path.exists(dataset_root):
        raise FileNotFoundError("Extracted archive does not contain expected 'UCI HAR Dataset' folder")

    return dataset_root



def prepare_dataset(processed_path, data_url, folds_path="data/har_cv_folds.npz", n_folds=5):
    """
    Prepares the HAR dataset and generates subject-disjoint cross-validation folds.
    Saves the folds (subject IDs for each fold) to disk for reuse.
    """
    _download_and_extract(data_url)

    x_train_np, y_train_np = _load_har_split("train")
    subject_train_np = _load_subject_ids("train")
    x_test_np, y_test_np = _load_har_split("test")

    # Generate subject-disjoint folds
    unique_subjects = np.unique(subject_train_np)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    shuffled_subjects = rng.permutation(unique_subjects)
    folds = np.array_split(shuffled_subjects, n_folds)

    # Save folds for reuse
    np.savez(folds_path, **{f"fold{i}": fold for i, fold in enumerate(folds)})
    print(f"Saved {n_folds} subject-disjoint folds to {folds_path}")

    # Save the normalized test set for convenience
    # (Normalization will be done per fold during training)
    dataset = {
        "X_train": torch.from_numpy(x_train_np),
        "y_train": torch.from_numpy(y_train_np),
        "subject_train": torch.from_numpy(subject_train_np),
        "X_test": torch.from_numpy(x_test_np),
        "y_test": torch.from_numpy(y_test_np),
    }
    torch.save(dataset, processed_path)
    print(f"Saved preprocessed dataset to {processed_path}")
    return dataset

prepare_dataset("data/dataset_har.pt", DEFAULT_URL)
