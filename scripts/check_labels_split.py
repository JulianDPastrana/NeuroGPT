#!/usr/bin/env python3
"""Quick label histogram per split to detect imbalance/inversion.
Usage: python scripts/check_labels_split.py --data /path/to/data --manifest /path/to/manifest.txt
Manifest may contain [TRAIN]/[VAL]/[TEST] sections or plain file list (legacy).
"""

import argparse
import os

# Allow import of project modules
import sys
from typing import List, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from batcher.downstream_dataset import MotorImageryDataset  # type: ignore


def parse_manifest(path: str) -> Tuple[List[str], List[str], List[str]]:
    train, val, test = [], [], []
    current = "TRAIN"
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper in {"[TRAIN]", "TRAIN:"}:
                current = "TRAIN"
                continue
            if upper in {"[VAL]", "[VALID]", "VAL:", "VALID:"}:
                current = "VAL"
                continue
            if upper in {"[TEST]", "TEST:"}:
                current = "TEST"
                continue
            if current == "TRAIN":
                train.append(line)
            elif current == "VAL":
                val.append(line)
            else:
                test.append(line)
    if not (train or val or test):
        with open(path, "r") as f:
            files = [ln.strip() for ln in f if ln.strip()]
        n_val = max(2, len(files) // 5)
        train, val = files[:-n_val], files[-n_val:]
        test = val
    return train, val, test


def count_labels(split_name: str, files: List[str], data_path: str) -> None:
    if not files:
        print(f"{split_name}: no files")
        return
    ds = MotorImageryDataset(
        files,
        sample_keys=["inputs", "attention_mask"],
        chunk_len=500,
        num_chunks=2,
        ovlp=0,
        root_path=data_path,
        gpt_only=True,
    )
    labels = ds.labels
    uniq, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    frac = {int(u): float(c) / total for u, c in zip(uniq, counts)}
    print(
        f"{split_name}: N={total}, counts={dict(zip(uniq.tolist(), counts.tolist()))}, frac={frac}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        required=True,
        help="Path to data directory containing .npz and true_labels",
    )
    ap.add_argument("--manifest", required=True, help="Path to fold manifest file")
    args = ap.parse_args()

    train, val, test = parse_manifest(args.manifest)
    print(f"Loaded manifest: train={len(train)}, val={len(val)}, test={len(test)}")
    count_labels("TRAIN", train, args.data)
    count_labels("VAL", val, args.data)
    count_labels("TEST", test, args.data)


if __name__ == "__main__":
    main()
