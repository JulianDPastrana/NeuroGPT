#!/usr/bin/env python3
"""
Stratified Group K-Fold for ADHD dataset to preserve class balance across folds.
Handles label detection from NPZ files and creates balanced train/val/test splits.

Usage:
    python scripts/stratified_fold_generator.py --data /path/to/data --output manifest.txt
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_labels_from_mat(label_path: str, subject_name: str) -> int:
    """Load label from .mat file. Returns 0 or 1."""
    try:
        mat = loadmat(label_path)
        # Assuming mat file has 'classlabel' key with integer values (1 or 2)
        label = mat.get("classlabel", None)
        if label is not None:
            label = int(label.squeeze()) - 1  # Convert 1,2 to 0,1
            return label
    except Exception as e:
        print(f"Warning: Could not load label for {subject_name}: {e}")
    return None


def load_labels_from_npz(npz_path: str) -> int:
    """Try to load label from NPZ file if it contains label data."""
    try:
        data = np.load(npz_path, allow_pickle=True)
        # Check for common label keys
        for key in ["label", "class", "classlabel", "y"]:
            if key in data:
                return int(data[key].squeeze())
    except Exception:
        pass
    return None


def detect_labels(data_path: str, label_dir: str = "true_labels") -> Dict[str, int]:
    """
    Detect labels for all subjects.
    Looks for .mat files in label_dir or checks NPZ files for embedded labels.

    Returns:
        Dict mapping filename to label (0 or 1)
    """
    labels = {}
    label_path = os.path.join(data_path, label_dir)

    # Get all .npz files
    npz_files = sorted(
        [
            f
            for f in os.listdir(data_path)
            if f.endswith(".npz") and os.path.isfile(os.path.join(data_path, f))
        ]
    )

    for npz_file in npz_files:
        subject_name = os.path.splitext(npz_file)[0]

        # Try .mat file first
        mat_file = os.path.join(label_path, subject_name + ".mat")
        if os.path.isfile(mat_file):
            label = load_labels_from_mat(mat_file, subject_name)
            if label is not None:
                labels[npz_file] = label
                continue

        # Try embedded label in NPZ
        npz_full_path = os.path.join(data_path, npz_file)
        label = load_labels_from_npz(npz_full_path)
        if label is not None:
            labels[npz_file] = label

    return labels


def check_imbalance(labels: Dict[str, int]) -> Tuple[bool, Dict[int, int]]:
    """
    Check if labels are imbalanced.

    Returns:
        (is_imbalanced, class_counts)
    """
    class_counts = {}
    for label in labels.values():
        class_counts[label] = class_counts.get(label, 0) + 1

    total = sum(class_counts.values())
    percentages = {cls: count / total * 100 for cls, count in class_counts.items()}

    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)
    for cls in sorted(class_counts.keys()):
        count = class_counts[cls]
        pct = percentages[cls]
        print(f"  Class {cls}: {count} samples ({pct:.1f}%)")

    # Imbalanced if any class is <40% or >60%
    is_imbalanced = any(pct < 40 or pct > 60 for pct in percentages.values())

    if is_imbalanced:
        print("\n⚠️  DATA IS IMBALANCED (using Stratified GroupKFold)")
    else:
        print("\n✓ Data is relatively balanced")

    return is_imbalanced, class_counts


def create_stratified_folds(
    files: List[str],
    labels: Dict[str, int],
    n_splits: int = 10,
    val_ratio: float = 0.2,
) -> List[Tuple[List[str], List[str], List[str]]]:
    """
    Create stratified group k-fold splits with balanced class distribution.

    Args:
        files: List of NPZ filenames
        labels: Dict mapping filename to label
        n_splits: Number of folds
        val_ratio: Validation ratio from train+val set

    Returns:
        List of (train_files, val_files, test_files) tuples for each fold
    """
    # Prepare arrays for sklearn
    y = np.array([labels.get(f, 0) for f in files])
    groups = np.arange(len(files))

    # Create stratified group k-fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(
        sgkf.split(np.arange(len(files)), y, groups)
    ):
        # Further split train_val into train and val
        n_train_val = len(train_val_idx)
        n_val = max(1, int(n_train_val * val_ratio))

        # Use stratified split for train/val too
        train_val_y = y[train_val_idx]
        train_val_groups = np.arange(n_train_val)

        # Simple stratified split without sklearn
        class_indices = {}
        for idx, label in enumerate(train_val_y):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(train_val_idx[idx])

        val_files_idx = []
        train_files_idx = []
        for class_label, indices in class_indices.items():
            n_val_class = max(1, int(len(indices) * val_ratio))
            val_files_idx.extend(indices[:n_val_class])
            train_files_idx.extend(indices[n_val_class:])

        train_files = [files[i] for i in train_files_idx]
        val_files = [files[i] for i in val_files_idx]
        test_files = [files[i] for i in test_idx]

        folds.append((train_files, val_files, test_files))

        # Print fold statistics
        train_y = y[train_files_idx]
        val_y = y[val_files_idx]
        test_y = y[test_idx]

        print(f"\n{'─' * 80}")
        print(f"Fold {fold_idx + 1}/{n_splits}")
        print(f"{'─' * 80}")
        print(f"Train: {len(train_files)} samples", end="")
        for cls in sorted(np.unique(train_y)):
            count = (train_y == cls).sum()
            pct = count / len(train_y) * 100
            print(f" | Class {cls}: {count} ({pct:.1f}%)", end="")
        print()

        print(f"Val:   {len(val_files)} samples", end="")
        for cls in sorted(np.unique(val_y)):
            count = (val_y == cls).sum()
            pct = count / len(val_y) * 100
            print(f" | Class {cls}: {count} ({pct:.1f}%)", end="")
        print()

        print(f"Test:  {len(test_files)} samples", end="")
        for cls in sorted(np.unique(test_y)):
            count = (test_y == cls).sum()
            pct = count / len(test_y) * 100
            print(f" | Class {cls}: {count} ({pct:.1f}%)", end="")
        print()

    return folds


def write_manifests(folds: List[Tuple[List[str], List[str], List[str]]]) -> None:
    """Write fold manifests in format expected by train_gpt.py"""
    for fold_idx, (train_files, val_files, test_files) in enumerate(folds):
        manifest_file = f".adhd_fold_{fold_idx + 1}_files.txt"

        with open(manifest_file, "w") as f:
            f.write("[TRAIN]\n")
            for fname in train_files:
                f.write(f"{fname}\n")

            f.write("\n[VAL]\n")
            for fname in val_files:
                f.write(f"{fname}\n")

            f.write("\n[TEST]\n")
            for fname in test_files:
                f.write(f"{fname}\n")

        print(f"✓ Wrote {manifest_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified group k-fold splits for ADHD dataset"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data directory containing .npz files",
    )
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    data_path = args.data
    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found: {data_path}")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print("STRATIFIED GROUP K-FOLD GENERATOR")
    print(f"{'=' * 80}")
    print(f"Data path: {data_path}")
    print(f"N-splits: {args.n_splits}")
    print(f"Val ratio: {args.val_ratio}")

    # Get all files
    all_files = sorted(
        [
            f
            for f in os.listdir(data_path)
            if f.endswith(".npz") and os.path.isfile(os.path.join(data_path, f))
        ]
    )
    print(f"Total files found: {len(all_files)}")

    # Detect labels
    print("\nDetecting labels...")
    labels = detect_labels(data_path)
    print(f"Labels loaded for {len(labels)} files")

    if len(labels) == 0:
        print("\n❌ No labels found! Cannot create stratified folds.")
        print(
            "Please ensure .mat files exist in 'true_labels' or labels "
            "are embedded in NPZ files."
        )
        sys.exit(1)

    # Check imbalance
    is_imbalanced, class_counts = check_imbalance(labels)

    # Create stratified folds
    print(f"\n{'=' * 80}")
    print("CREATING STRATIFIED FOLDS")
    print(f"{'=' * 80}")

    folds = create_stratified_folds(
        all_files,
        labels,
        n_splits=args.n_splits,
        val_ratio=args.val_ratio,
    )

    # Write manifests
    print(f"\n{'=' * 80}")
    print("WRITING MANIFEST FILES")
    print(f"{'=' * 80}")
    write_manifests(folds)

    print(f"\n{'=' * 80}")
    print("✅ STRATIFIED FOLD GENERATION COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
