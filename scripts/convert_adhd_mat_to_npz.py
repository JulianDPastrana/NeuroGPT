#!/usr/bin/env python3
"""
Convert ADHD-Control .mat files to .npz format for NeuroGPT training.
"""

import os
from pathlib import Path

import numpy as np
import scipy.io as sio


def convert_mat_to_npz(mat_file_path, output_dir, label):
    """
    Convert a single .mat file to .npz format.

    Args:
        mat_file_path: Path to the .mat file
        output_dir: Directory to save the .npz file
        label: Class label (0 for Control, 1 for ADHD)
    """
    # Load .mat file
    mat_data = sio.loadmat(mat_file_path)

    # Get the filename without extension
    filename = Path(mat_file_path).stem

    # Extract the data (key matches filename)
    if filename in mat_data:
        data = mat_data[filename]
    else:
        # Try to find the data key (skip metadata keys)
        data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
        if data_keys:
            data = mat_data[data_keys[0]]
        else:
            raise ValueError(f"Could not find data in {mat_file_path}")

    # Data is already in shape (n_samples, n_channels)
    # Keep it in this format to match BCI2A

    # Create output filename
    output_path = os.path.join(output_dir, f"{filename}.npz")

    # Save as .npz with the required format matching BCI2A
    # The entire recording is treated as one trial
    n_samples = data.shape[0]

    np.savez(
        output_path,
        s=data.astype(np.float64),  # EEG data: (n_samples, n_channels)
        etyp=np.array([[label]], dtype=np.uint16),  # Event type (label)
        epos=np.array([[0]], dtype=np.int32),  # Event position (start)
        edur=np.array([[n_samples]], dtype=np.uint16),  # Event duration
        artifacts=np.array([[0]], dtype=np.uint8),  # No artifacts marked
    )

    print(f"Converted {filename}: shape {data.shape}, label {label}")
    return output_path


def main():
    # Define paths
    base_dir = "/home/usuarioutp/Documents/data/IEE-EEG-ADHD-CONTROL"
    adhd_dir = os.path.join(base_dir, "ADHD")
    control_dir = os.path.join(base_dir, "Control")
    output_dir = "/home/usuarioutp/Documents/data/adhd_control_npz"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Convert ADHD files (label = 1)
    print("Converting ADHD files...")
    adhd_files = sorted([f for f in os.listdir(adhd_dir) if f.endswith(".mat")])
    for mat_file in adhd_files:
        mat_path = os.path.join(adhd_dir, mat_file)
        convert_mat_to_npz(mat_path, output_dir, label=1)

    # Convert Control files (label = 0)
    print("\nConverting Control files...")
    control_files = sorted([f for f in os.listdir(control_dir) if f.endswith(".mat")])
    for mat_file in control_files:
        mat_path = os.path.join(control_dir, mat_file)
        convert_mat_to_npz(mat_path, output_dir, label=0)

    print("\n✓ Conversion complete!")
    print(f"  Total ADHD files: {len(adhd_files)}")
    print(f"  Total Control files: {len(control_files)}")
    print(f"  Output directory: {output_dir}")

    # List all converted files
    npz_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".npz")])
    print(f"\nConverted files ({len(npz_files)}):")
    print(npz_files)


if __name__ == "__main__":
    main()
