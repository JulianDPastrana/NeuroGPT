#!/usr/bin/env python3
"""
Script to summarize 10-fold cross-validation results for ADHD classification.
Displays accuracy statistics (mean, std) for each fold and overall performance.
"""

import os
import sys

import numpy as np
import pandas as pd


def load_fold_results(base_dir, num_folds=10):
    """
    Load test results from all folds.

    Args:
        base_dir: Base directory containing fold output folders
        num_folds: Number of folds (default: 10)

    Returns:
        Dictionary with fold results
    """
    results = {}

    for fold_idx in range(1, num_folds + 1):
        fold_name = f"fold{fold_idx}"

        # Try different possible paths for test_metrics.csv
        possible_paths = [
            os.path.join(
                base_dir,
                f"outputs_adhd_10fold_fold{fold_idx}",
                f"adhd_fold{fold_idx}-{fold_idx}",
                "test_metrics.csv",
            ),
            os.path.join(
                base_dir,
                f"outputs_adhd_10fold_fold{fold_idx}",
                f"adhd_fold{fold_idx}-1",
                "test_metrics.csv",
            ),
            os.path.join(
                base_dir,
                f"outputs_adhd_fold{fold_idx}",
                f"adhd_fold{fold_idx}-1",
                "test_metrics.csv",
            ),
            os.path.join(base_dir, f"fold{fold_idx}", "test_metrics.csv"),
        ]

        test_csv = None
        for path in possible_paths:
            if os.path.exists(path):
                test_csv = path
                break

        if test_csv is None:
            print(f"⚠️  Warning: Could not find test_metrics.csv for fold {fold_idx}")
            continue

        try:
            df = pd.read_csv(test_csv)

            if "test_accuracy" not in df.columns:
                print(f"⚠️  Warning: No 'test_accuracy' column in fold {fold_idx}")
                continue

            # Get test accuracy (should be only one row)
            test_accuracy = df["test_accuracy"].iloc[0]
            test_loss = df["test_loss"].iloc[0] if "test_loss" in df.columns else None

            # Store results
            results[fold_name] = {
                "fold_idx": fold_idx,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            }

        except Exception as e:
            print(f"⚠️  Error loading fold {fold_idx}: {e}")
            continue

    return results


def print_summary(results):
    """
    Print a formatted summary of the 10-fold cross-validation test results.

    Args:
        results: Dictionary with fold results
    """
    if not results:
        print("❌ No results found!")
        return

    print("\n" + "=" * 100)
    print("10-FOLD CROSS-VALIDATION TEST RESULTS SUMMARY")
    print("=" * 100)
    print("Binary Classification: ADHD vs Control")
    print(f"Number of folds analyzed: {len(results)}/10")
    print("=" * 100)

    # Table header
    print(f"\n{'Fold':<10} {'Test Accuracy':<20} {'Test Loss':<20}")
    print("-" * 100)

    # Individual fold results
    test_accs = []
    test_losses = []

    for fold_name in sorted(results.keys(), key=lambda x: results[x]["fold_idx"]):
        r = results[fold_name]
        test_accs.append(r["test_accuracy"])
        if r["test_loss"] is not None:
            test_losses.append(r["test_loss"])

        loss_str = f"{r['test_loss']:.6f}" if r["test_loss"] is not None else "N/A"
        print(f"{fold_name:<10} {r['test_accuracy']:<20.4f} {loss_str:<20}")

    print("-" * 100)

    # Overall statistics
    test_acc_mean = np.mean(test_accs)
    test_acc_std = np.std(test_accs)
    test_loss_mean = np.mean(test_losses) if test_losses else None
    test_loss_std = np.std(test_losses) if test_losses else None

    print(f"\n{'OVERALL TEST STATISTICS (across folds)':^100}")
    print("=" * 100)
    print("Test Accuracy:")
    print(f"  Mean ± Std: {test_acc_mean:.4f} ± {test_acc_std:.4f}")
    print(f"  Range: [{min(test_accs):.4f}, {max(test_accs):.4f}]")

    if test_loss_mean is not None:
        print("\nTest Loss:")
        print(f"  Mean ± Std: {test_loss_mean:.6f} ± {test_loss_std:.6f}")

    print("=" * 100)

    # Performance interpretation
    print(f"\n{'PERFORMANCE INTERPRETATION':^100}")
    print("=" * 100)

    if test_acc_mean > 0.80:
        performance = "Excellent"
    elif test_acc_mean > 0.70:
        performance = "Good"
    elif test_acc_mean > 0.60:
        performance = "Moderate"
    else:
        performance = "Needs Improvement"

    print(f"Overall Performance: {performance}")
    print(f"Test Accuracy: {test_acc_mean * 100:.2f}% ± {test_acc_std * 100:.2f}%")

    if test_acc_std < 0.05:
        consistency = "Highly consistent across folds"
    elif test_acc_std < 0.10:
        consistency = "Reasonably consistent across folds"
    else:
        consistency = "High variability across folds"

    print(f"Model Stability: {consistency}")
    print("=" * 100 + "\n")


def save_summary_csv(results, output_file):
    """
    Save summary results to a CSV file.

    Args:
        results: Dictionary with fold results
        output_file: Path to output CSV file
    """
    if not results:
        print("⚠️  No results to save!")
        return

    # Create DataFrame
    rows = []
    for fold_name in sorted(results.keys(), key=lambda x: results[x]["fold_idx"]):
        rows.append(results[fold_name])

    df = pd.DataFrame(rows)

    # Add overall statistics
    summary_row = {
        "fold_idx": "MEAN",
        "test_accuracy": df["test_accuracy"].mean(),
        "test_loss": df["test_loss"].mean() if "test_loss" in df.columns else None,
    }

    std_row = {
        "fold_idx": "STD",
        "test_accuracy": df["test_accuracy"].std(),
        "test_loss": df["test_loss"].std() if "test_loss" in df.columns else None,
    }

    df = pd.concat([df, pd.DataFrame([summary_row, std_row])], ignore_index=True)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Summary saved to: {output_file}")


def main():
    """Main function."""
    # Default base directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = script_dir

    # Allow custom directory from command line
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]

    print(f"\n📂 Searching for fold results in: {base_dir}")

    # Load results
    results = load_fold_results(base_dir)

    # Print summary
    print_summary(results)

    # Save summary CSV
    output_csv = os.path.join(base_dir, "10fold_summary.csv")
    save_summary_csv(results, output_csv)


if __name__ == "__main__":
    main()
