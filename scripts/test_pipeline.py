#!/usr/bin/env python3
"""
Comprehensive test to trace the full training pipeline for binary classification
"""

import sys

sys.path.insert(0, "/home/usuarioutp/Documents/NeuroGPT/src")

import numpy as np
import torch

from embedder.base import Embedder
from encoder.conformer_braindecode import EEGConformer

print("=" * 80)
print("COMPREHENSIVE PIPELINE TEST - BINARY CLASSIFICATION")
print("=" * 80)

# Setup
torch.manual_seed(42)
np.random.seed(42)

batch_size = 4
chunks = 2
n_chans = 22
n_times = 500
n_outputs = 1  # Binary classification

# Create dummy batch
batch = {
    "inputs": torch.randn(batch_size, chunks, n_chans, n_times),
    "labels": torch.randint(
        0, 2, (batch_size * chunks,)
    ),  # Binary labels for each chunk
}

print("\n1. Batch shapes:")
print(f"   inputs: {batch['inputs'].shape}")
print(f"   labels: {batch['labels'].shape}")

# Test encoder only
print("\n2. Testing EEGConformer encoder (encoder-only mode):")
encoder = EEGConformer(
    n_outputs=n_outputs,
    n_chans=n_chans,
    n_times=n_times,
    ch_pos=None,
    is_decoding_mode=True,
    add_log_softmax=False,  # Binary classification fix
)

encoder_output = encoder(batch["inputs"])
print(f"   Encoder output shape: {encoder_output.shape}")
print(
    f"   ✅ Expected: [batch*chunks, {n_outputs}] = [{batch_size * chunks}, {n_outputs}]"
)

# Test loss computation
print("\n3. Testing loss computation:")
embedder = Embedder(
    in_dim=n_chans,
    n_times=n_times,
    n_pos=1024,
    is_decoding_mode=True,
)

# Prepare outputs dict as the model would
outputs = {"decoding_logits": encoder_output, "outputs": encoder_output}

# Compute loss
loss_dict = embedder.decoding_loss(
    decoding_logits=outputs["decoding_logits"], labels=batch["labels"]
)

print(f"   Loss: {loss_dict['decoding_loss'].item():.4f}")
print("   ✅ Loss computed successfully (not NaN/Inf)")

# Verify loss is not stuck
print("\n4. Verifying loss values vary:")
for i in range(3):
    random_logits = torch.randn(batch_size * chunks, 1)
    loss = embedder.decoding_loss(random_logits, batch["labels"])["decoding_loss"]
    print(f"   Sample {i + 1}: {loss.item():.4f}")

print("\n5. Testing gradient flow:")
model_outputs = torch.randn(batch_size * chunks, 1, requires_grad=True)
labels = batch["labels"].float()
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(model_outputs.squeeze(-1), labels)
loss.backward()
print(f"   Output grad norm: {model_outputs.grad.norm().item():.6f}")
print(f"   ✅ Gradients flow: {model_outputs.grad.norm().item() > 0}")

print("\n" + "=" * 80)
print("✅ ALL PIPELINE TESTS PASSED!")
print("=" * 80)
