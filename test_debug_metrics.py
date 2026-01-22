#!/usr/bin/env python3
"""Test script to directly test the metrics function with various input formats."""

import sys

sys.path.insert(0, "/home/usuarioutp/Documents/NeuroGPT/src")

import numpy as np

from trainer.make import decoding_accuracy_metrics

print("\n" + "=" * 80)
print("TEST 1: Raw logits (expected format for BCEWithLogitsLoss)")
print("=" * 80)
# Simulate [batch=6, 1] shaped logits from model
logits = np.array([[0.5], [-0.3], [1.2], [-2.1], [0.1], [-0.5]])
labels = np.array([1, 0, 1, 0, 1, 0])
result = decoding_accuracy_metrics((logits, labels), num_decoding_classes=2)
print(f"Input logits shape: {logits.shape}")
print(f"Input logits: {logits.flatten()}")
print(f"Input labels: {labels}")
print(f"Result: {result}")
print()

print("\n" + "=" * 80)
print("TEST 2: Probabilities (0-1 range, wrong input format)")
print("=" * 80)
# If model was outputting probabilities instead
probs = np.array([[0.7], [0.3], [0.8], [0.2], [0.6], [0.1]])
labels = np.array([1, 0, 1, 0, 1, 0])
result = decoding_accuracy_metrics((probs, labels), num_decoding_classes=2)
print(f"Input probs shape: {probs.shape}")
print(f"Input probs: {probs.flatten()}")
print(f"Input labels: {labels}")
print(f"Result: {result}")
print()

print("\n" + "=" * 80)
print("TEST 3: All-zero predictions (would indicate model predicting nothing)")
print("=" * 80)
zeros = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
labels = np.array([1, 0, 1, 0, 1, 0])
result = decoding_accuracy_metrics((zeros, labels), num_decoding_classes=2)
print(f"Input all zeros shape: {zeros.shape}")
print(f"Input all zeros: {zeros.flatten()}")
print(f"Input labels: {labels}")
print(f"Result: {result}")
print()

print("\n" + "=" * 80)
print("TEST 4: Inverted labels (to see if labels are flipped)")
print("=" * 80)
logits = np.array([[0.5], [-0.3], [1.2], [-2.1], [0.1], [-0.5]])
labels_inverted = np.array([0, 1, 0, 1, 0, 1])  # Opposite
result = decoding_accuracy_metrics((logits, labels_inverted), num_decoding_classes=2)
print(f"Input logits shape: {logits.shape}")
print(f"Input logits: {logits.flatten()}")
print(f"Input labels (inverted): {labels_inverted}")
print(f"Result: {result}")
print()

print("\n" + "=" * 80)
print("TEST 5: High-magnitude logits (testing sigmoid stability)")
print("=" * 80)
logits_high = np.array([[10.0], [-10.0], [5.0], [-5.0], [100.0], [-100.0]])
labels = np.array([1, 0, 1, 0, 1, 0])
result = decoding_accuracy_metrics((logits_high, labels), num_decoding_classes=2)
print(f"Input high logits shape: {logits_high.shape}")
print(f"Input high logits: {logits_high.flatten()}")
print(f"Input labels: {labels}")
print(f"Result: {result}")
print()

print("\n" + "=" * 80)
print("TEST 6: All labels are 0 (testing when all gt is single class)")
print("=" * 80)
logits = np.array([[0.5], [-0.3], [1.2], [-2.1], [0.1], [-0.5]])
labels_all_zero = np.array([0, 0, 0, 0, 0, 0])
result = decoding_accuracy_metrics((logits, labels_all_zero), num_decoding_classes=2)
print(f"Input logits shape: {logits.shape}")
print(f"Input logits: {logits.flatten()}")
print(f"Input labels (all 0): {labels_all_zero}")
print(f"Result: {result}")
print()

print("\n" + "=" * 80)
print("TEST 7: All labels are 1 (testing when all gt is single class)")
print("=" * 80)
logits = np.array([[0.5], [-0.3], [1.2], [-2.1], [0.1], [-0.5]])
labels_all_one = np.array([1, 1, 1, 1, 1, 1])
result = decoding_accuracy_metrics((logits, labels_all_one), num_decoding_classes=2)
print(f"Input logits shape: {logits.shape}")
print(f"Input logits: {logits.flatten()}")
print(f"Input labels (all 1): {labels_all_one}")
print(f"Result: {result}")

print("\n\nDone!")
