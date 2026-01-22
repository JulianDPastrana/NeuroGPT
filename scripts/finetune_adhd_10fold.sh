#!/bin/bash

# ADHD dataset finetuning with 10-fold cross-validation
# Binary classification: ADHD vs Control
# This script creates fold-specific data directories and calls finetune_adhd.sh for each fold

cd "$(dirname "$0")" || exit

# Source dataset path
SOURCE_DATA="../../data/adhd_control_npz/"

# Activate virtual environment
source ../venv/bin/activate

# Python script to generate fold splits and display progress
python3 << 'PYEOF'
import os
import sys
import numpy as np
import shutil
import time
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedGroupKFold
import scipy.io as sio

def format_time(seconds):
    """Format seconds to human-readable time"""
    return str(timedelta(seconds=int(seconds)))

def print_progress_bar(current, total, bar_length=50, prefix='Progress'):
    """Print a progress bar"""
    percent = float(current) / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    print(f"\r{prefix}: |{bar}| {current}/{total} ({percent*100:.1f}%)", end='', flush=True)
    if current == total:
        print()

# Dataset path (full dataset with 121 subjects: 61 ADHD + 60 Control)
data_path = "../../data/adhd_control_npz/"

# Get list of subject files (excluding true_labels directory)
all_files = sorted([f for f in os.listdir(data_path) 
                    if f.endswith('.npz') and os.path.isfile(os.path.join(data_path, f))])

print(f"\n{'='*100}")
print(f"ADHD Dataset: 10-Fold Group K-Fold Cross-Validation")
print(f"{'='*100}")
print(f"Total subjects: {len(all_files)}")
print(f"Total folds: 10")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Extract subject groups and labels for stratification
subjects = np.array(range(len(all_files)))
groups = np.array(range(len(all_files)))

# Load labels for stratification
label_dir = "../../data/adhd_control_npz/true_labels/"
labels = []
for fname in all_files:
    stem = fname.replace('.npz', '')
    mat_file = os.path.join(label_dir, f"{stem}.mat")
    try:
        mat = sio.loadmat(mat_file)
        label = int(mat['classlabel'].squeeze()) - 1  # Convert 1,2 to 0,1
        labels.append(label)
    except:
        labels.append(0)  # Default to 0 if load fails

labels = np.array(labels)

# Initialize stratified 10-fold CV
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)

# Create fold information file for bash to read
with open('.adhd_folds_info.txt', 'w') as f:
    fold_idx = 0
    for train_val_idx, test_idx in sgkf.split(subjects, labels, groups=groups):
        fold_idx += 1
        
        # Further split train_val into train and val (80/20 split) with stratification
        train_val_labels = labels[train_val_idx]
        n_train_val = len(train_val_idx)
        n_val = max(1, n_train_val // 5)
        
        # Use stratified split for train/val to maintain class balance
        train_val_subjects = np.array(range(len(train_val_idx)))
        sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Get the 5th fold (20%) as validation
        inner_fold_count = 0
        for inner_train, inner_val in sgkf_inner.split(train_val_subjects, train_val_labels, groups=train_val_subjects):
            if inner_fold_count == 4:  # Last fold (20%)
                train_idx = train_val_idx[inner_train]
                val_idx = train_val_idx[inner_val]
                break
            inner_fold_count += 1
        
        # Get file names
        train_files = [all_files[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        test_files = [all_files[i] for i in test_idx]
        
        # Write fold info
        f.write(f"FOLD_{fold_idx}_TRAIN:{','.join(train_files)}\n")
        f.write(f"FOLD_{fold_idx}_VAL:{','.join(val_files)}\n")
        f.write(f"FOLD_{fold_idx}_TEST:{','.join(test_files)}\n")
        
        # Display fold information
        print(f"\n{'='*100}")
        print(f"FOLD {fold_idx}/10")
        print(f"{'='*100}")
        
        print(f"\n📊 TRAINING SET ({len(train_files)} subjects):")
        print(f"{'─'*100}")
        for i, fname in enumerate(train_files, 1):
            print(f"  {i:2d}. {fname}")
        
        print(f"\n✓ VALIDATION SET ({len(val_files)} subjects):")
        print(f"{'─'*100}")
        for i, fname in enumerate(val_files, 1):
            print(f"  {i:2d}. {fname}")
        
        print(f"\n🧪 TEST SET ({len(test_files)} subjects):")
        print(f"{'─'*100}")
        for i, fname in enumerate(test_files, 1):
            print(f"  {i:2d}. {fname}")
        
        print(f"\n{'─'*100}")
        print(f"Fold {fold_idx} Summary:")
        print(f"  Training:   {len(train_files):2d} subjects ({len(train_files)/len(all_files)*100:5.1f}%)")
        print(f"  Validation: {len(val_files):2d} subjects ({len(val_files)/len(all_files)*100:5.1f}%)")
        print(f"  Test:       {len(test_files):2d} subjects ({len(test_files)/len(all_files)*100:5.1f}%)")
        print(f"{'─'*100}\n")

print(f"\n{'='*100}")
print(f"Fold splits generated. Starting training...")
print(f"{'='*100}\n")
PYEOF

# Record start time
TOTAL_START=$(date +%s)

# Read fold information and train each fold
for FOLD in {1..10}; do
    FOLD_START=$(date +%s)
    
    echo ""
    echo "======================================================================================================"
    echo "Starting training for FOLD ${FOLD}/10"
    echo "======================================================================================================"
    
    # Progress bar
    FILLED=$((FOLD - 1))
    EMPTY=$((10 - FILLED))
    printf "Overall Progress: |"
    printf '█%.0s' $(seq 1 $((FILLED * 5)))
    printf '░%.0s' $(seq 1 $((EMPTY * 5)))
    printf "| %d/10 (%.1f%%)\n" $((FOLD - 1)) $(awk "BEGIN {printf \"%.1f\", ($((FOLD - 1)) / 10) * 100}")
    echo ""
    
    # Extract fold files from info file
    TRAIN_FILES=$(grep "^FOLD_${FOLD}_TRAIN:" .adhd_folds_info.txt | cut -d: -f2)
    VAL_FILES=$(grep "^FOLD_${FOLD}_VAL:" .adhd_folds_info.txt | cut -d: -f2)
    TEST_FILES=$(grep "^FOLD_${FOLD}_TEST:" .adhd_folds_info.txt | cut -d: -f2)
    
    
    # Create temporary fold manifest file with structured sections to avoid leakage
    FOLD_MANIFEST=".adhd_fold_${FOLD}_files.txt"

    echo "[TRAIN]" > "$FOLD_MANIFEST"
    IFS=',' read -ra FILES <<< "$TRAIN_FILES"
    for FILE in "${FILES[@]}"; do
        echo "$FILE" >> "$FOLD_MANIFEST"
    done

    echo "[VAL]" >> "$FOLD_MANIFEST"
    IFS=',' read -ra FILES <<< "$VAL_FILES"
    for FILE in "${FILES[@]}"; do
        echo "$FILE" >> "$FOLD_MANIFEST"
    done

    echo "[TEST]" >> "$FOLD_MANIFEST"
    IFS=',' read -ra FILES <<< "$TEST_FILES"
    for FILE in "${FILES[@]}"; do
        echo "$FILE" >> "$FOLD_MANIFEST"
    done
    
    echo "🚀 Preparing data for Fold ${FOLD}..."
    echo "📊 Training Fold ${FOLD}..."
    echo ""
    
    # Train using source dataset with fold manifest
    OUTPUT_DIR="./outputs_adhd_10fold_fold${FOLD}"
    
    # Call finetune_adhd.sh with source data path and fold manifest
    ./finetune_adhd.sh "${SOURCE_DATA}" "${OUTPUT_DIR}" "${FOLD}" "${FOLD_MANIFEST}"
    
    EXIT_CODE=$?
    
    # Cleanup fold manifest
    rm -f "$FOLD_MANIFEST"
    
    # Calculate timing
    FOLD_END=$(date +%s)
    FOLD_ELAPSED=$((FOLD_END - FOLD_START))
    
    # Calculate ETA
    CURRENT_ELAPSED=$((FOLD_END - TOTAL_START))
    AVG_FOLD_TIME=$((CURRENT_ELAPSED / FOLD))
    REMAINING_FOLDS=$((10 - FOLD))
    ETA_SECONDS=$((AVG_FOLD_TIME * REMAINING_FOLDS))
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✅ Fold ${FOLD} completed successfully!"
        printf "⏱️  Fold time: %02d:%02d:%02d\n" $((FOLD_ELAPSED/3600)) $(((FOLD_ELAPSED%3600)/60)) $((FOLD_ELAPSED%60))
        if [ $REMAINING_FOLDS -gt 0 ]; then
            printf "⏳ Estimated time remaining: %02d:%02d:%02d\n" $((ETA_SECONDS/3600)) $(((ETA_SECONDS%3600)/60)) $((ETA_SECONDS%60))
            COMPLETION_TIME=$(date -d "+${ETA_SECONDS} seconds" "+%H:%M:%S" 2>/dev/null || date -v "+${ETA_SECONDS}S" "+%H:%M:%S" 2>/dev/null || echo "N/A")
            echo "🕐 Expected completion: ${COMPLETION_TIME}"
        fi
        echo ""
    else
        echo ""
        echo "⚠️  Fold ${FOLD} completed with errors (exit code: ${EXIT_CODE})"
        printf "⏱️  Fold time: %02d:%02d:%02d\n" $((FOLD_ELAPSED/3600)) $(((FOLD_ELAPSED%3600)/60)) $((FOLD_ELAPSED%60))
        echo ""
    fi
done

# Cleanup info file
rm -f .adhd_folds_info.txt

# Final summary
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "======================================================================================================"
echo "All 10 folds completed!"
echo "======================================================================================================"
printf "Overall Progress: |"
printf '█%.0s' $(seq 1 50)
printf "| 10/10 (100.0%%)\n"
echo ""
printf "⏱️  Total time: %02d:%02d:%02d\n" $((TOTAL_ELAPSED/3600)) $(((TOTAL_ELAPSED%3600)/60)) $((TOTAL_ELAPSED%60))
printf "⏱️  Average time per fold: %02d:%02d:%02d\n" $((TOTAL_ELAPSED/36000)) $(((TOTAL_ELAPSED/10%3600)/60)) $((TOTAL_ELAPSED/10%60))
echo "📁 Results saved in outputs_adhd_10fold_fold1/ through outputs_adhd_10fold_fold10/"
echo "🏁 Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================================================"
echo ""
