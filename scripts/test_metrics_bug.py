#!/usr/bin/env python3
"""
Test to demonstrate the metrics bug with binary classification
"""

import numpy as np
from sklearn.metrics import accuracy_score

# Simulate what happens with binary classification
print("=" * 80)
print("BINARY CLASSIFICATION METRICS BUG DEMONSTRATION")
print("=" * 80)

# Scenario 1: Binary classification predictions from BCEWithLogitsLoss
# Model outputs raw logits [batch, 1]
batch_size = 10
np.random.seed(42)

# Real logits (varying values)
logits_binary = np.random.randn(batch_size, 1)  # Shape [10, 1]
labels = np.random.randint(0, 2, batch_size)  # True labels: [0, 1, 0, 1, ...]

print("\nBinary Classification Setup:")
print(f"  Logits shape: {logits_binary.shape}")
print(f"  Logits values:\n{logits_binary.flatten()}")
print(f"  True labels: {labels}")

# WRONG way (current code): argmax on [batch, 1]
preds_wrong = logits_binary.argmax(axis=-1)
print("\n❌ WRONG: Using argmax(axis=-1) on [batch, 1]:")
print(f"  Predictions: {preds_wrong}")
print(f"  All predictions are 0: {np.all(preds_wrong == 0)}")
acc_wrong = accuracy_score(labels, preds_wrong)
print(f"  Accuracy: {acc_wrong:.3f}")

# CORRECT way: Apply sigmoid + threshold
preds_correct = (1 / (1 + np.exp(-logits_binary.flatten())) > 0.5).astype(int)
print("\n✅ CORRECT: Using sigmoid + threshold:")
print(f"  Predictions: {preds_correct}")
print(f"  Accuracy: {accuracy_score(labels, preds_correct):.3f}")

# Alternative CORRECT way: Compare to 0
preds_correct_alt = (logits_binary.flatten() > 0).astype(int)
print("\n✅ CORRECT (alternative): Using logits > 0:")
print(f"  Predictions: {preds_correct_alt}")
print(f"  Accuracy: {accuracy_score(labels, preds_correct_alt):.3f}")

print("\n" + "=" * 80)
print("MULTI-CLASS CLASSIFICATION")
print("=" * 80)

# Scenario 2: Multi-class classification
num_classes = 4
logits_multiclass = np.random.randn(batch_size, num_classes)
labels_mc = np.random.randint(0, num_classes, batch_size)

print("\nMulti-Class Setup:")
print(f"  Logits shape: {logits_multiclass.shape}")
print(f"  True labels: {labels_mc}")

# CORRECT: argmax on [batch, num_classes]
preds_mc = logits_multiclass.argmax(axis=-1)
print("\n✅ CORRECT: Using argmax(axis=-1) on [batch, 4]:")
print(f"  Predictions: {preds_mc}")
print(f"  Accuracy: {accuracy_score(labels_mc, preds_mc):.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The bug:
- For binary classification, model outputs shape [batch, 1]
- argmax(axis=-1) on [batch, 1] ALWAYS returns 0
- This causes all predictions to be class 0
- Accuracy is constant only if chance level happens to match!

The fix:
- For binary classification (num_classes=2):
  * Apply sigmoid + threshold to [batch, 1]
  * OR compare logits to 0 threshold
- For multi-class (num_classes>2):
  * Use argmax(axis=-1) on [batch, num_classes]
""")
