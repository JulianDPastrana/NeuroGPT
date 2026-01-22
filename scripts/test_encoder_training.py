#!/usr/bin/env python3
"""
Quick test to check if encoder final layer is trainable
"""

import sys

sys.path.insert(0, "/home/usuarioutp/Documents/NeuroGPT/src")

import torch

from encoder.conformer_braindecode import EEGConformer

# Create encoder with decoding mode
encoder = EEGConformer(
    n_outputs=1,  # Binary classification
    n_chans=22,
    n_times=500,
    ch_pos=None,
    is_decoding_mode=True,
    add_log_softmax=False,  # Disable LogSoftmax for binary classification
)

print("=" * 80)
print("ENCODER CONFIGURATION")
print("=" * 80)
print(f"add_log_softmax: {encoder.add_log_softmax}")
print(f"return_features: {encoder.final_layer.return_features}")
print(f"is_decoding_mode: {encoder.is_decoding_mode}")
print("\nFinal layer modules:")
for name, module in encoder.final_layer.final_layer.named_children():
    print(f"  {name}: {module}")

print("=" * 80)
print("PARAMETERS THAT REQUIRE GRADIENTS")
print("=" * 80)
trainable_params = 0
frozen_params = 0
for name, param in encoder.named_parameters():
    if param.requires_grad:
        trainable_params += param.numel()
        if "final" in name or "fc" in name:
            print(f"✓ {name}: {param.shape} - requires_grad={param.requires_grad}")
    else:
        frozen_params += param.numel()
        if "final" in name or "fc" in name:
            print(f"✗ {name}: {param.shape} - requires_grad={param.requires_grad}")

print(f"\nTotal trainable parameters: {trainable_params:,}")
print(f"Total frozen parameters: {frozen_params:,}")

# Test forward pass
print("\n" + "=" * 80)
print("FORWARD PASS TEST")
print("=" * 80)
batch_size = 4
x = torch.randn(batch_size, 2, 22, 500)  # (batch, chunks, channels, time)

# Check intermediate outputs
print(f"Input shape: {x.shape}")
print(f"Input mean/std: {x.mean():.4f} / {x.std():.4f}")

# Hook to capture intermediate values
intermediate_outputs = {}


def make_hook(name):
    def hook(module, input, output):
        print(
            f"  {name} input shape: {input[0].shape if isinstance(input, tuple) else input.shape}"
        )
        print(f"  {name} output shape: {output.shape}")
        print(f"  {name} output mean/std: {output.mean():.6f} / {output.std():.6f}")
        intermediate_outputs[name] = output.detach()

    return hook


encoder.patch_embedding.register_forward_hook(make_hook("patch_embedding"))
encoder.transformer.register_forward_hook(make_hook("transformer"))
encoder.fc.register_forward_hook(make_hook("fc"))
encoder.final_layer.register_forward_hook(make_hook("final_layer"))

output = encoder(x)
print(f"\nOutput shape: {output.shape}")
print(f"Output values (all samples):\n{output.detach().numpy()}")
print(f"Output mean/std: {output.mean():.6f} / {output.std():.6f}")

if "fc" in intermediate_outputs:
    fc_out = intermediate_outputs["fc"]
    print(f"\nFC layer output shape: {fc_out.shape}")
    print(f"FC layer output mean/std: {fc_out.mean():.6f} / {fc_out.std():.6f}")
    print(f"FC layer output sample: {fc_out[0, :5].numpy()}")

# Check final layer weights
print("\n" + "=" * 80)
print("FINAL LAYER WEIGHTS")
print("=" * 80)
final_weight = encoder.final_layer.final_layer[0].weight.data
final_bias = encoder.final_layer.final_layer[0].bias.data
print(f"Weight shape: {final_weight.shape}")
print(f"Weight mean/std: {final_weight.mean():.6f} / {final_weight.std():.6f}")
print(f"Weight values: {final_weight.numpy()}")
print(f"\nBias shape: {final_bias.shape}")
print(f"Bias value: {final_bias.numpy()}")

# Manual computation
print("\n" + "=" * 80)
print("MANUAL COMPUTATION CHECK")
print("=" * 80)
if "fc" in intermediate_outputs:
    fc_out = intermediate_outputs["fc"]
    print(f"FC output (first sample): {fc_out[0].numpy()}")
    print(f"Final weight: {final_weight.numpy()}")
    manual_output = torch.matmul(fc_out[0], final_weight.T) + final_bias
    print(f"Manual computation (should match final output): {manual_output.numpy()}")
    print(f"Actual final output (first sample): {output[0].detach().numpy()}")
    print(f"Match: {torch.allclose(manual_output, output[0])}")

# Test backward pass
print("\n" + "=" * 80)
print("BACKWARD PASS TEST")
print("=" * 80)
labels = torch.randint(0, 2, (batch_size,)).float()
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(output.squeeze(), labels)
print(f"Loss: {loss.item():.6f}")

loss.backward()

# Check gradients
print("\nGradients for final layer:")
for name, param in encoder.named_parameters():
    if "final" in name or "fc" in name:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"  {name}: grad=None (NO GRADIENT!)")
