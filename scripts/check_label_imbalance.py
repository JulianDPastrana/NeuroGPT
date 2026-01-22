#!/usr/bin/env python3
"""
Check label distribution and imbalance in ADHD dataset.
Analyzes what's causing low eval accuracy: label polarity mismatch or extreme imbalance.
"""

import argparse
import os
import sys
from typing import Tuple

from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def load_label_from_mat(label_dir: str, subject_name: str) -> int:
    """Load label from .mat file in true_labels directory."""
    mat_file = os.path.join(label_dir, subject_name + ".mat")
    try:
        mat = loadmat(mat_file)
        label = mat.get("classlabel", None)
        if label is not None:
            return int(label.squeeze()) - 1  # Convert 1,2 to 0,1
    except Exception as e:
        print(f"  Warning: Failed to load {mat_file}: {e}")
    return None


def analyze_split_labels(
    data_path: str, label_dir: str, split_files: list
) -> Tuple[int, int, float]:
    """Analyze labels in a split. Returns (count_class0, count_class1, imbalance_ratio)"""
    if not split_files:
        return 0, 0, 0.0

    class_counts = {0: 0, 1: 0}
    missing = 0

    for fname in split_files:
        subject_name = os.path.splitext(fname)[0]
        label = load_label_from_mat(label_dir, subject_name)
        if label is not None:
            class_counts[label] += 1
        else:
            missing += 1

    total = class_counts[0] + class_counts[1]
    if total == 0:
        return 0, 0, 0.0

    count0 = class_counts[0]
    count1 = class_counts[1]
    imbalance = max(count0, count1) / (count0 + count1) if total > 0 else 0

    return count0, count1, imbalance


def parse_manifest(manifest_file: str) -> Tuple[list, list, list]:
    """Parse manifest file with [TRAIN]/[VAL]/[TEST] sections."""
    train_files = []
    val_files = []
    test_files = []
    current = "TRAIN"

    try:
        with open(manifest_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.upper() in ["[TRAIN]", "TRAIN:"]:
                    current = "TRAIN"
                    continue
                if line.upper() in ["[VAL]", "[VALID]", "VAL:", "VALID:"]:
                    current = "VAL"
                    continue
                if line.upper() in ["[TEST]", "TEST:"]:
                    current = "TEST"
                    continue

                if current == "TRAIN":
                    train_files.append(line)
                elif current == "VAL":
                    val_files.append(line)
                elif current == "TEST":
                    test_files.append(line)
    except FileNotFoundError:
        print(f"Error: Manifest file not found: {manifest_file}")
        sys.exit(1)

    return train_files, val_files, test_files


def main():
    parser = argparse.ArgumentParser(
        description="Analyze label distribution across fold splits"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to data directory containing .npz files",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to fold manifest file",
    )
    args = parser.parse_args()

    data_path = args.data
    manifest_file = args.manifest
    label_dir = os.path.join(data_path, "true_labels")

    if not os.path.isdir(data_path):
        print(f"Error: Data directory not found: {data_path}")
        sys.exit(1)

    if not os.path.isfile(manifest_file):
        print(f"Error: Manifest file not found: {manifest_file}")
        sys.exit(1)

    if not os.path.isdir(label_dir):
        print(f"Error: Label directory not found: {label_dir}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)

    train_files, val_files, test_files = parse_manifest(manifest_file)

    print(f"\nManifest: {manifest_file}")
    print(f"Data path: {data_path}")
    print(f"Label dir: {label_dir}\n")

    # Analyze each split
    print("TRAINING SET:")
    train_c0, train_c1, train_imb = analyze_split_labels(
        data_path, label_dir, train_files
    )
    print(f"  Total: {train_c0 + train_c1} samples")
    print(f"  Class 0: {train_c0} ({100 * train_c0 / (train_c0 + train_c1):.1f}%)")
    print(f"  Class 1: {train_c1} ({100 * train_c1 / (train_c0 + train_c1):.1f}%)")
    print(f"  Imbalance ratio: {train_imb:.3f}")

    print("\nVALIDATION SET:")
    val_c0, val_c1, val_imb = analyze_split_labels(data_path, label_dir, val_files)
    print(f"  Total: {val_c0 + val_c1} samples")
    print(f"  Class 0: {val_c0} ({100 * val_c0 / (val_c0 + val_c1):.1f}%)")
    print(f"  Class 1: {val_c1} ({100 * val_c1 / (val_c0 + val_c1):.1f}%)")
    print(f"  Imbalance ratio: {val_imb:.3f}")

    print("\nTEST SET:")
    test_c0, test_c1, test_imb = analyze_split_labels(data_path, label_dir, test_files)
    print(f"  Total: {test_c0 + test_c1} samples")
    print(f"  Class 0: {test_c0} ({100 * test_c0 / (test_c0 + test_c1):.1f}%)")
    print(f"  Class 1: {test_c1} ({100 * test_c1 / (test_c0 + test_c1):.1f}%)")
    print(f"  Imbalance ratio: {test_imb:.3f}")

    # Overall analysis
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)

    overall_c0 = train_c0 + val_c0 + test_c0
    overall_c1 = train_c1 + val_c1 + test_c1
    overall_imb = max(overall_c0, overall_c1) / (overall_c0 + overall_c1)

    print(
        f"\nOverall: {overall_c0} class-0, {overall_c1} class-1 (ratio: {overall_imb:.3f})"
    )

    if overall_imb > 0.65:
        print("\n⚠️  SEVERE IMBALANCE DETECTED!")
        print("  The dataset has extreme class imbalance. This causes:")
        print(
            "    1. Model may predict all majority class (baseline ~{:.1f}%)".format(
                100 * max(overall_c0, overall_c1) / (overall_c0 + overall_c1)
            )
        )
        print("    2. BCEWithLogitsLoss may need class weighting")
        print("    3. Consider using pos_weight parameter or class weights")
        print("\n  Solutions:")
        print("    - Use StratifiedGroupKFold to balance splits")
        print("    - Apply pos_weight in BCEWithLogitsLoss(pos_weight=...)")
        print("    - Use class weights in loss function")
    elif overall_imb > 0.60:
        print("\n⚠️  MODERATE IMBALANCE - Consider using Stratified K-Fold")
    else:
        print("\n✓ Data is reasonably balanced")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
