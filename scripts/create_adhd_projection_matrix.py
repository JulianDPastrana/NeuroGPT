#!/usr/bin/env python3
"""
Create a projection matrix to interpolate 19 EEG channels to 22 channels.

The ADHD dataset has 19 channels while the pretrained model expects 22 channels.
This script creates an interpolation matrix that maps 19 channels to 22 channels
using linear interpolation based on spatial relationships.

ADHD dataset (19 channels): Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2,
                             F7, F8, T3, T4, T5, T6, Fz, Cz, Pz

BCI2A dataset (22 channels): Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz,
                              C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz

Strategy:
- Map similar channels directly (1:1 mapping)
- For missing channels, interpolate from nearby channels
- Preserve as much spatial information as possible



Based on the projection matrix creation script, here are the shared channels between the ADHD and BCI2A datasets:

Exact Matches (5 channels):
Fz - ADHD[16] → BCI[0]
Cz - ADHD[17] → BCI[9]
Pz - ADHD[18] → BCI[19]
C3 - ADHD[4] → BCI[7]
C4 - ADHD[5] → BCI[11]
These 5 channels have direct 1:1 mappings in the projection matrix (weight = 1.0).

Channel Lists:
ADHD dataset (19 channels):
Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3(T7), T4(T8), T5(P7), T6(P8), Fz, Cz, Pz

BCI2A dataset (22 channels):
Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz

The remaining 17 BCI2A channels (FC3, FC1, FCz, FC2, FC4, C5, C1, C2, C6, CP3, CP1, CPz, CP2, CP4, P1, P2, POz) are interpolated from weighted combinations of nearby ADHD channels since they don't exist in the ADHD montage.
"""

import os

import numpy as np


def create_interpolation_matrix():
    """
    Create a 22x19 interpolation matrix to map 19 ADHD channels to 22 BCI channels.

    ADHD channel indices (0-18):
    0:Fp1, 1:Fp2, 2:F3, 3:F4, 4:C3, 5:C4, 6:P3, 7:P4, 8:O1, 9:O2,
    10:F7, 11:F8, 12:T3(T7), 13:T4(T8), 14:T5(P7), 15:T6(P8), 16:Fz, 17:Cz, 18:Pz

    BCI target indices (0-21):
    0:Fz, 1:FC3, 2:FC1, 3:FCz, 4:FC2, 5:FC4, 6:C5, 7:C3, 8:C1, 9:Cz,
    10:C2, 11:C4, 12:C6, 13:CP3, 14:CP1, 15:CPz, 16:CP2, 17:CP4, 18:P1, 19:Pz, 20:P2, 21:POz
    """

    # Initialize 22x19 matrix with zeros
    P = np.zeros((22, 19), dtype=np.float64)

    # Direct mappings where channels match or are very close
    # BCI[0] = Fz -> ADHD[16] = Fz
    P[0, 16] = 1.0

    # BCI[9] = Cz -> ADHD[17] = Cz
    P[9, 17] = 1.0

    # BCI[19] = Pz -> ADHD[18] = Pz
    P[19, 18] = 1.0

    # BCI[7] = C3 -> ADHD[4] = C3
    P[7, 4] = 1.0

    # BCI[11] = C4 -> ADHD[5] = C4
    P[11, 5] = 1.0

    # Interpolations for frontocentral channels
    # BCI[1] = FC3: interpolate from F3, C3, T3
    P[1, 2] = 0.4  # F3
    P[1, 4] = 0.4  # C3
    P[1, 12] = 0.2  # T3

    # BCI[2] = FC1: interpolate from F3, Fz, C3, Cz
    P[2, 2] = 0.3  # F3
    P[2, 16] = 0.2  # Fz
    P[2, 4] = 0.3  # C3
    P[2, 17] = 0.2  # Cz

    # BCI[3] = FCz: interpolate from Fz, Cz
    P[3, 16] = 0.5  # Fz
    P[3, 17] = 0.5  # Cz

    # BCI[4] = FC2: interpolate from F4, Fz, C4, Cz
    P[4, 3] = 0.3  # F4
    P[4, 16] = 0.2  # Fz
    P[4, 5] = 0.3  # C4
    P[4, 17] = 0.2  # Cz

    # BCI[5] = FC4: interpolate from F4, C4, T4
    P[5, 3] = 0.4  # F4
    P[5, 5] = 0.4  # C4
    P[5, 13] = 0.2  # T4

    # BCI[6] = C5: interpolate from C3, T3
    P[6, 4] = 0.6  # C3
    P[6, 12] = 0.4  # T3

    # BCI[8] = C1: interpolate from C3, Cz
    P[8, 4] = 0.6  # C3
    P[8, 17] = 0.4  # Cz

    # BCI[10] = C2: interpolate from C4, Cz
    P[10, 5] = 0.6  # C4
    P[10, 17] = 0.4  # Cz

    # BCI[12] = C6: interpolate from C4, T4
    P[12, 5] = 0.6  # C4
    P[12, 13] = 0.4  # T4

    # Interpolations for centroparietal channels
    # BCI[13] = CP3: interpolate from C3, P3, T5
    P[13, 4] = 0.3  # C3
    P[13, 6] = 0.5  # P3
    P[13, 14] = 0.2  # T5

    # BCI[14] = CP1: interpolate from C3, Cz, P3, Pz
    P[14, 4] = 0.2  # C3
    P[14, 17] = 0.2  # Cz
    P[14, 6] = 0.3  # P3
    P[14, 18] = 0.3  # Pz

    # BCI[15] = CPz: interpolate from Cz, Pz
    P[15, 17] = 0.5  # Cz
    P[15, 18] = 0.5  # Pz

    # BCI[16] = CP2: interpolate from C4, Cz, P4, Pz
    P[16, 5] = 0.2  # C4
    P[16, 17] = 0.2  # Cz
    P[16, 7] = 0.3  # P4
    P[16, 18] = 0.3  # Pz

    # BCI[17] = CP4: interpolate from C4, P4, T6
    P[17, 5] = 0.3  # C4
    P[17, 7] = 0.5  # P4
    P[17, 15] = 0.2  # T6

    # Interpolations for parietal channels
    # BCI[18] = P1: interpolate from P3, Pz
    P[18, 6] = 0.6  # P3
    P[18, 18] = 0.4  # Pz

    # BCI[20] = P2: interpolate from P4, Pz
    P[20, 7] = 0.6  # P4
    P[20, 18] = 0.4  # Pz

    # BCI[21] = POz: interpolate from Pz, O1, O2
    P[21, 18] = 0.5  # Pz
    P[21, 8] = 0.25  # O1
    P[21, 9] = 0.25  # O2

    return P


def main():
    # Create the interpolation matrix
    P = create_interpolation_matrix()

    # Verify the matrix shape
    print(f"Interpolation matrix shape: {P.shape}")
    print("Expected: (22, 19)")

    # Verify that each row sums to approximately 1.0 (conservation of signal)
    row_sums = P.sum(axis=1)
    print("\nRow sums (should be close to 1.0):")
    print(
        f"Min: {row_sums.min():.3f}, Max: {row_sums.max():.3f}, Mean: {row_sums.mean():.3f}"
    )

    # Check for any zero rows (channels not mapped)
    zero_rows = np.where(row_sums == 0)[0]
    if len(zero_rows) > 0:
        print(f"\nWarning: Zero rows found at indices: {zero_rows}")
    else:
        print("\nAll channels properly mapped!")

    # Save the matrix
    output_dir = "../inputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tMatrix_19to22_adhd.npy")
    np.save(output_path, P)

    print(f"\nSaved interpolation matrix to: {output_path}")
    print("Matrix statistics:")
    print(f"  Non-zero elements: {np.count_nonzero(P)}")
    print(f"  Sparsity: {(1 - np.count_nonzero(P) / P.size) * 100:.1f}%")

    # Display the matrix structure
    print("\nMatrix structure (showing non-zero elements):")
    for i in range(P.shape[0]):
        non_zero_idx = np.where(P[i] > 0)[0]
        if len(non_zero_idx) > 0:
            contributions = ", ".join(
                [f"{idx}:{P[i, idx]:.2f}" for idx in non_zero_idx]
            )
            print(f"  BCI channel {i:2d}: <- ADHD [{contributions}]")


if __name__ == "__main__":
    main()
