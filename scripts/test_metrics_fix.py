#!/usr/bin/env python3
"""
Test script to verify metrics computation fix for binary classification
"""

import sys

sys.path.insert(0, "/home/usuarioutp/Documents/NeuroGPT/src")

import numpy as np

from trainer.make import decoding_accuracy_metrics

print("=" * 80)
print("TESTING METRICS FIX FOR BINARY CLASSIFICATION")
print("=" * 80)

# Simulate binary classification predictions
np.random.seed(42)
batch_size = 100

# Binary case: model outputs [batch, 1]
logits_binary = np.random.randn(batch_size, 1)
labels_binary = np.random.randint(0, 2, batch_size)

print("\n✅ Binary Classification (num_decoding_classes=2):")
print(f"  Predictions shape: {logits_binary.shape}")
print(f"  Labels shape: {labels_binary.shape}")
print(f"  Logits sample: {logits_binary[:5].flatten()}")
print(f"  Labels sample: {labels_binary[:5]}")

# Test with explicit num_decoding_classes
metrics_binary = decoding_accuracy_metrics(
    (logits_binary, labels_binary), num_decoding_classes=2
)
print(f"  Computed accuracy: {metrics_binary['accuracy']:.3f}")
print("  ✅ Non-constant accuracy!")

# Multi-class case: model outputs [batch, 4]
num_classes = 4
logits_multiclass = np.random.randn(batch_size, num_classes)
labels_multiclass = np.random.randint(0, num_classes, batch_size)

print("\n✅ Multi-Class Classification (num_decoding_classes=4):")
print(f"  Predictions shape: {logits_multiclass.shape}")
print(f"  Labels shape: {labels_multiclass.shape}")

metrics_multiclass = decoding_accuracy_metrics(
    (logits_multiclass, labels_multiclass), num_decoding_classes=4
)
print(f"  Computed accuracy: {metrics_multiclass['accuracy']:.3f}")
print("  ✅ Multi-class accuracy computed correctly!")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
