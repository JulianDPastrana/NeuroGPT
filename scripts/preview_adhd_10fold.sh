#!/bin/bash

# Preview 10-fold cross-validation splits for ADHD dataset
# Shows train/val/test distributions without running training

cd "$(dirname "$0")" || exit

# Activate virtual environment
source ../venv/bin/activate

# Show the CV splits
python3 << 'EOF'
import os
import sys
import numpy as np
from sklearn.model_selection import GroupKFold

# Dataset path
data_path = "../../data/adhd_control_npz/"

# Get list of subject files (excluding true_labels directory)
all_files = sorted([f for f in os.listdir(data_path) 
                    if f.endswith('.npz') and os.path.isfile(os.path.join(data_path, f))])
print(f"\n{'='*100}")
print(f"ADHD Dataset: 10-Fold Group K-Fold Cross-Validation - PREVIEW")
print(f"{'='*100}")
print(f"Total subjects found: {len(all_files)}\n")

# Extract subject groups
subjects = np.array(range(len(all_files)))
groups = np.array(range(len(all_files)))

# Initialize 10-fold CV
gkf = GroupKFold(n_splits=10)
fold_idx = 0

for train_val_idx, test_idx in gkf.split(subjects, groups=groups):
    fold_idx += 1
    
    # Further split train_val into train and val (80/20 split)
    n_train_val = len(train_val_idx)
    n_val = max(1, n_train_val // 5)  # 20% for validation
    
    train_idx = train_val_idx[:-n_val]
    val_idx = train_val_idx[-n_val:]
    
    # Get file names
    train_files = [all_files[i] for i in train_idx]
    val_files = [all_files[i] for i in val_idx]
    test_files = [all_files[i] for i in test_idx]
    
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
print(f"Summary Statistics:")
print(f"{'='*100}")
print(f"Each fold uses different subjects for testing (leave-one-out style)")
print(f"Validation is done on 20% of training data")
print(f"Training uses the remaining 80% of non-test data")
print(f"{'='*100}\n")

EOF
