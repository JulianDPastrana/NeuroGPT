#!/bin/bash

# Finetune NeuroGPT for ADHD vs Control classification
# This script trains on the IEE-EEG-ADHD-CONTROL dataset
# Usage: ./finetune_adhd.sh [DATA_PATH] [OUTPUT_DIR] [FOLD_NUM] [FOLD_MANIFEST]

# Default parameters
DATA_PATH="${1:-../../data/adhd_control_npz/}"
OUTPUT_DIR="${2:-results/models/upstream/}"
FOLD_NUM="${3:-0}"
FOLD_MANIFEST="${4:-}"  # Optional: file listing fold-specific train+val files

# Activate virtual environment and run training
cd "$(dirname "$0")"
source ../venv/bin/activate

python3 ../src/train_gpt.py \
    --training-style='decoding' \
    --num-decoding-classes=2 \
    --training-steps=10000 \
    --eval_every_n_steps=500 \
    --log-every-n-steps=1000 \
    --num_chunks=2 \
    --per-device-training-batch-size=32 \
    --per-device-validation-batch-size=32 \
    --chunk_len=500 \
    --chunk_ovlp=0 \
    --run-name="adhd_fold${FOLD_NUM}" \
    --ft-only-encoder='True' \
    --fold_i="${FOLD_NUM}" \
    --num-encoder-layers=6 \
    --num-hidden-layers=6 \
    --learning-rate=1e-4 \
    --use-encoder='True' \
    --embedding-dim=1024 \
    --optim='adamw_torch' \
    --pretrained-model='../../pretrained_model/pytorch_model.bin' \
    --log-dir="${OUTPUT_DIR}" \
    --dst-data-path="${DATA_PATH}" \
    --fold-manifest-file="${FOLD_MANIFEST}"
