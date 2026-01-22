#!/usr/bin/env python3
"""
Debug script to check what shapes and values the metrics function receives
"""

import sys

sys.path.insert(0, "/home/usuarioutp/Documents/NeuroGPT/src")

import numpy as np

from trainer.make import decoding_accuracy_metrics

print("=" * 80)
print("DEBUG: METRICS COMPUTATION")
print("=" * 80)

# Test 1: What we expect for binary classification
print("\n1. Expected binary classification inputs:")
print("   Predictions from model: [batch, 1] raw logits")
print("   Labels: [batch] binary labels")

logits = np.array([[0.5], [-0.3], [1.2], [-2.1], [0.1], [-0.5]])
labels = np.array([1, 0, 1, 0, 1, 0])

print(f"\n   Logits shape: {logits.shape}")
print(f"   Logits: {logits.flatten()}")
print(f"   Labels shape: {labels.shape}")
print(f"   Labels: {labels}")

# What should predictions be?
sigmoid = 1 / (1 + np.exp(-logits.squeeze(-1)))
expected_preds = (sigmoid > 0.5).astype(int)
print(f"\n   Expected predictions (sigmoid > 0.5): {expected_preds}")
print(f"   Sigmoid values: {sigmoid}")

# What does our function return?
metrics = decoding_accuracy_metrics((logits, labels), num_decoding_classes=2)
print(f"\n   Accuracy from function: {metrics['accuracy']}")

# Test 2: What if predictions are probabilities instead of logits?
print("\n2. What if predictions are already probabilities (0-1)?")
probs = np.array([[0.7], [0.3], [0.8], [0.2], [0.6], [0.1]])
print(f"   Probs shape: {probs.shape}")
print(f"   Probs values: {probs.flatten()}")

# Our function would try sigmoid on these
sigmoid_on_probs = 1 / (1 + np.exp(-probs.squeeze(-1)))
preds_from_probs = (sigmoid_on_probs > 0.5).astype(int)
print(f"   After sigmoid: {sigmoid_on_probs}")
print(f"   Predictions: {preds_from_probs}")

metrics_probs = decoding_accuracy_metrics((probs, labels), num_decoding_classes=2)
print(f"   Accuracy: {metrics_probs['accuracy']}")

print("\n" + "=" * 80)
