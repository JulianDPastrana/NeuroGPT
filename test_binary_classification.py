#!/usr/bin/env python3
"""
Test binary classification setup: loss function, classification head, and accuracy computation.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from trainer.make import decoding_accuracy_metrics


def test_bce_with_logits_loss():
    """Test that BCEWithLogitsLoss works as expected for binary classification."""
    print("\n" + "=" * 80)
    print("TEST 1: BCEWithLogitsLoss Behavior")
    print("=" * 80)

    loss_fn = nn.BCEWithLogitsLoss()

    # Test case 1: Perfect predictions
    logits_perfect = torch.tensor(
        [10.0, -10.0, 10.0, -10.0]
    )  # Strong positives and negatives
    labels_perfect = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_perfect = loss_fn(logits_perfect, labels_perfect)
    print(
        f"✓ Perfect predictions: logits={logits_perfect.tolist()}, labels={labels_perfect.tolist()}"
    )
    print(f"  Loss: {loss_perfect.item():.6f} (should be ~0)")

    # Test case 2: Wrong predictions
    logits_wrong = torch.tensor([-10.0, 10.0, -10.0, 10.0])  # Opposite of labels
    labels_wrong = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_wrong = loss_fn(logits_wrong, labels_wrong)
    print(
        f"✓ Wrong predictions: logits={logits_wrong.tolist()}, labels={labels_wrong.tolist()}"
    )
    print(f"  Loss: {loss_wrong.item():.6f} (should be ~large)")

    # Test case 3: Neutral predictions
    logits_neutral = torch.tensor([0.0, 0.0, 0.0, 0.0])
    labels_neutral = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss_neutral = loss_fn(logits_neutral, labels_neutral)
    print(
        f"✓ Neutral predictions: logits={logits_neutral.tolist()}, labels={labels_neutral.tolist()}"
    )
    print(f"  Loss: {loss_neutral.item():.6f} (should be ~ln(2)={np.log(2):.6f})")

    assert loss_perfect < loss_neutral < loss_wrong, "Loss ordering is wrong!"
    print("\n✅ BCEWithLogitsLoss test PASSED")
    return True


def test_binary_classification_head():
    """Test that binary classification head produces correct output shape."""
    print("\n" + "=" * 80)
    print("TEST 2: Binary Classification Head Output")
    print("=" * 80)

    embed_dim = 768
    batch_size = 4

    # Create a simple classification head like in decoder/gpt.py
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ELU(),
        nn.Dropout(0.5),
        nn.Linear(256, 32),
        nn.ELU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),  # Binary: output_dim = 1
    )
    head.eval()  # No dropout in eval mode

    # Create fake embeddings
    embeddings = torch.randn(batch_size, embed_dim)

    with torch.no_grad():
        logits = head(embeddings)

    print(f"✓ Input embeddings shape: {embeddings.shape}")
    print(f"  Expected: ({batch_size}, {embed_dim})")
    print(f"✓ Output logits shape: {logits.shape}")
    print(f"  Expected: ({batch_size}, 1)")

    assert logits.shape == (batch_size, 1), (
        f"Output shape {logits.shape} != ({batch_size}, 1)"
    )
    print(f"✓ Logits values (raw): {logits.squeeze().tolist()}")
    print("  (No sigmoid applied yet - these are raw logits for BCEWithLogitsLoss)")

    print("\n✅ Binary classification head test PASSED")
    return True


def test_loss_backward():
    """Test that loss backward works correctly."""
    print("\n" + "=" * 80)
    print("TEST 3: Loss Backward Pass")
    print("=" * 80)

    embed_dim = 768
    batch_size = 4

    # Create head with gradient tracking
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ELU(),
        nn.Dropout(0.5),
        nn.Linear(256, 32),
        nn.ELU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    )
    head.train()

    loss_fn = nn.BCEWithLogitsLoss()

    # Forward pass
    embeddings = torch.randn(batch_size, embed_dim, requires_grad=True)
    logits = head(embeddings)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = loss_fn(logits.squeeze(-1), labels)
    print(f"✓ Batch size: {batch_size}")
    print(f"  Loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients exist
    for name, param in head.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"✓ {name}: grad_norm = {grad_norm:.6f}")
            assert grad_norm > 0, f"Gradient for {name} is zero!"
        else:
            print(f"⚠ {name}: No gradient!")

    print("\n✅ Loss backward pass test PASSED")
    return True


def test_accuracy_metrics():
    """Test decoding_accuracy_metrics function with binary classification."""
    print("\n" + "=" * 80)
    print("TEST 4: Accuracy Metrics Computation")
    print("=" * 80)

    # Test case 1: Perfect predictions (1D logits)
    preds_perfect_1d = np.array([10.0, -10.0, 5.0, -5.0, 0.1, -0.1])
    labels_perfect = np.array([1, 0, 1, 0, 1, 0])

    metrics = decoding_accuracy_metrics(
        (preds_perfect_1d, labels_perfect), num_decoding_classes=2
    )
    print(f"✓ Perfect 1D logits: preds={preds_perfect_1d.tolist()}")
    print(f"  labels={labels_perfect.tolist()}")
    print(f"  Accuracy: {metrics['accuracy']} (expected: 1.0)")
    assert metrics["accuracy"] == 1.0, (
        f"Expected accuracy 1.0, got {metrics['accuracy']}"
    )

    # Test case 2: Predictions with some errors
    preds_mixed = np.array([5.0, 3.0, -5.0, -3.0, 0.5, -0.5])
    labels_mixed = np.array([1, 0, 0, 1, 1, 0])  # 4 correct, 2 wrong

    metrics = decoding_accuracy_metrics(
        (preds_mixed, labels_mixed), num_decoding_classes=2
    )
    expected_acc = 4.0 / 6.0  # 4 correct out of 6
    print(f"\n✓ Mixed 1D logits: preds={preds_mixed.tolist()}")
    print(f"  labels={labels_mixed.tolist()}")
    print(f"  Accuracy: {metrics['accuracy']} (expected: {expected_acc:.3f})")
    assert abs(metrics["accuracy"] - expected_acc) < 0.01, "Accuracy mismatch"

    # Test case 3: 2D array with shape (batch, 1) - should be squeezed
    preds_2d = np.array([[10.0], [-10.0], [5.0], [-5.0]])
    labels_2d = np.array([1, 0, 1, 0])

    metrics = decoding_accuracy_metrics((preds_2d, labels_2d), num_decoding_classes=2)
    print(f"\n✓ 2D logits shape (batch, 1): preds.shape={preds_2d.shape}")
    print(f"  Accuracy: {metrics['accuracy']} (expected: 1.0)")
    assert metrics["accuracy"] == 1.0, (
        f"Expected accuracy 1.0, got {metrics['accuracy']}"
    )

    # Test case 4: Imbalanced data (like the ADHD case)
    preds_imbalanced = np.array([5.0] * 100 + [-5.0] * 30)  # More class 1
    labels_imbalanced = np.array(
        [1] * 95 + [0] * 5 + [0] * 28 + [1] * 2
    )  # Imbalanced labels

    metrics = decoding_accuracy_metrics(
        (preds_imbalanced, labels_imbalanced), num_decoding_classes=2
    )
    correct = ((preds_imbalanced > 0).astype(int) == labels_imbalanced).sum()
    expected_acc = correct / len(labels_imbalanced)
    print(f"\n✓ Imbalanced data: {len(preds_imbalanced)} samples")
    print(
        f"  Pred class distribution: {(preds_imbalanced > 0).sum()} class-1, {(preds_imbalanced <= 0).sum()} class-0"
    )
    print(
        f"  True class distribution: {labels_imbalanced.sum()} class-1, {(1 - labels_imbalanced).sum()} class-0"
    )
    print(f"  Accuracy: {metrics['accuracy']} (expected: {expected_acc:.3f})")
    assert abs(metrics["accuracy"] - expected_acc) < 0.01, "Accuracy mismatch"

    print("\n✅ Accuracy metrics test PASSED")
    return True


def test_end_to_end():
    """End-to-end test: embedding → head → loss → backward."""
    print("\n" + "=" * 80)
    print("TEST 5: End-to-End Forward-Backward Pass")
    print("=" * 80)

    embed_dim = 768
    batch_size = 8

    # Create model components
    head = nn.Sequential(
        nn.Linear(embed_dim, 256),
        nn.ELU(),
        nn.Dropout(0.5),
        nn.Linear(256, 32),
        nn.ELU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1),
    )
    head.train()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)

    # Create fake data: 80% class 1, 20% class 0 (like ADHD imbalance)
    embeddings = torch.randn(batch_size, embed_dim, requires_grad=True)
    labels_np = np.array([1, 1, 1, 1, 1, 1, 0, 0])  # 75% class 1
    labels = torch.tensor(labels_np, dtype=torch.float32)

    print(f"✓ Batch: {batch_size} samples")
    print(
        f"  Label distribution: {labels_np.sum()} class-1, {len(labels_np) - labels_np.sum()} class-0"
    )

    # Training loop
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        logits = head(embeddings)
        loss = loss_fn(logits.squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Compute accuracy
        with torch.no_grad():
            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()
            acc = (preds == labels.long()).float().mean().item()

        print(f"  Step {step}: loss={loss.item():.6f}, accuracy={acc:.3f}")

    # Check that loss decreased
    print(f"\n✓ Loss trend: {[f'{l:.6f}' for l in losses]}")
    assert losses[-1] < losses[0], "Loss should decrease during training"
    print(f"  Loss decreased from {losses[0]:.6f} to {losses[-1]:.6f} ✓")

    print("\n✅ End-to-end test PASSED")
    return True


def main():
    print("\n" + "=" * 80)
    print("BINARY CLASSIFICATION SETUP TEST SUITE")
    print("=" * 80)

    tests = [
        ("BCEWithLogitsLoss", test_bce_with_logits_loss),
        ("Classification Head", test_binary_classification_head),
        ("Loss Backward", test_loss_backward),
        ("Accuracy Metrics", test_accuracy_metrics),
        ("End-to-End", test_end_to_end),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"\n❌ {name} FAILED with error:")
            print(f"   {str(e)}")
            results.append((name, "FAILED"))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for name, status in results:
        symbol = "✅" if status == "PASSED" else "❌"
        print(f"{symbol} {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASSED")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests PASSED! Binary classification setup is correct.\n")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
