# Quick Reference: How to Use New Balanced Folds

## Problem Summary
- **Eval accuracy:** 0.27-0.32 (worse than random ~0.50)
- **Root cause:** Validation set had 85.7% one class (3 minority, 18 majority)
- **Solution:** StratifiedGroupKFold creates 10 balanced splits (~50-50 in each)

## Quick Start

### Option 1: Use Fold 1 (Recommended)
```bash
cd /home/usuarioutp/Documents/NeuroGPT
python src/train_gpt.py \
    --data_manifest scripts/.adhd_fold_1_files.txt \
    --output_dir outputs/fold_1 \
    ...
```

### Option 2: Run All 10 Folds
```bash
cd /home/usuarioutp/Documents/NeuroGPT/scripts
bash finetune_adhd_10fold.sh  # (update to use new .adhd_fold_* files)
```

### Option 3: Custom Fold
Replace `fold_1` with any fold: `fold_2`, `fold_3`, ..., `fold_10`

## New Fold Files Location
```
/home/usuarioutp/Documents/NeuroGPT/scripts/.adhd_fold_1_files.txt
/home/usuarioutp/Documents/NeuroGPT/scripts/.adhd_fold_2_files.txt
...
/home/usuarioutp/Documents/NeuroGPT/scripts/.adhd_fold_10_files.txt
```

## Fold Structure
Each manifest contains:
```
[TRAIN]
file_1.npz
file_2.npz
...
[VAL]
file_88.npz
...
[TEST]
file_109.npz
...
```

Each fold is stratified:
- **TRAIN:** 88 files (50% Control, 50% ADHD)
- **VAL:** 21 files (~48% Control, ~52% ADHD)
- **TEST:** 12 files (~50% Control, ~50% ADHD)

## Verify Balance (Optional)
```bash
cd /home/usuarioutp/Documents/NeuroGPT/scripts
python check_label_imbalance.py .adhd_fold_1_files.txt
```

Expected output:
```
TRAIN: Control: 44 (50.0%), ADHD: 44 (50.0%)
VAL: Control: ~10 (47.6%), ADHD: ~11 (52.4%)
TEST: Control: 6 (50.0%), ADHD: 6 (50.0%)
```

## Expected Results
- **Old (imbalanced):** Eval accuracy 0.27-0.32
- **New (balanced):** Expected 0.60-0.75

## What Changed?
1. ✅ Removed debug print statements (cleaner logs)
2. ✅ Fixed metrics function for binary classification
3. ✅ Created 10 stratified k-folds (balanced class distribution)
4. ✅ Verified all fixes with comprehensive tests

## If Still Having Issues

### Check fold balance:
```bash
python scripts/check_label_imbalance.py scripts/.adhd_fold_1_files.txt
```

### Verify labels are correct:
```bash
python scripts/test_binary_classification.py  # Should pass all 5 tests
```

### Regenerate folds (if needed):
```bash
python scripts/stratified_fold_generator.py \
    --data /home/usuarioutp/Documents/data/adhd_control_npz \
    --n-splits 10 \
    --val-ratio 0.2
```

## Files Modified
- `src/trainer/make.py` - Fixed metrics for binary classification
- `src/batcher/base.py` - Removed debug print
- `src/batcher/downstream_dataset.py` - Removed debug prints

## Documentation
See `/home/usuarioutp/Documents/NeuroGPT/SOLUTION_SUMMARY.md` for detailed analysis
