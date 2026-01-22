# 10-Fold Cross-Validation Results Summary

This directory contains a script to summarize the results from 10-fold cross-validation training.

## Usage

### Basic Usage
Run the script from the scripts directory:

```bash
cd /home/usuarioutp/Documents/NeuroGPT/scripts
python3 summarize_10fold_results.py
```

### Custom Directory
You can also specify a custom directory containing the fold results:

```bash
python3 summarize_10fold_results.py /path/to/results/directory
```

## Output

The script produces two outputs:

1. **Console Output**: A formatted table showing:
   - Individual fold accuracies (final, mean, std, min, max)
   - Number of evaluation steps per fold
   - Overall statistics across all folds (mean ± std)
   - Performance interpretation

2. **CSV File** (`10fold_summary.csv`): Contains all metrics in CSV format for further analysis

## Example Output

```
====================================================================================================
10-FOLD CROSS-VALIDATION RESULTS SUMMARY
====================================================================================================
Binary Classification: ADHD vs Control
Number of folds analyzed: 10/10
====================================================================================================

Fold     Final Acc    Mean Acc     Std Acc      Min Acc      Max Acc      Eval Steps  
----------------------------------------------------------------------------------------------------
fold1    0.7760       0.7760       0.0000       0.7760       0.7760       20          
fold2    0.6540       0.6540       0.0000       0.6540       0.6540       20          
fold3    0.8160       0.8160       0.0000       0.8160       0.8160       20          
...
----------------------------------------------------------------------------------------------------

                                 OVERALL STATISTICS (across folds)                                  
====================================================================================================
Final Accuracy (last checkpoint):
  Mean ± Std: 0.7942 ± 0.0494
  Range: [0.6540, 0.8210]

Average Accuracy (across all eval steps):
  Mean ± Std: 0.7942 ± 0.0494
====================================================================================================

                                     PERFORMANCE INTERPRETATION                                     
====================================================================================================
Overall Performance: Good
Classification Accuracy: 79.42%
Consistency (lower is better): 4.94% std deviation
Model Stability: Highly consistent across folds
====================================================================================================
```

## Current Results

Based on the latest 10-fold cross-validation:

- **Mean Accuracy**: 79.42% ± 4.94%
- **Best Fold**: Fold 5, 6, 8 (82.1%)
- **Worst Fold**: Fold 2 (65.4%)
- **Overall Performance**: Good for binary ADHD vs Control classification
- **Model Stability**: Highly consistent across folds

## Notes

- The script automatically searches for `eval_history.csv` files in the standard fold output directories
- Fold directories should follow the naming pattern: `outputs_adhd_10fold_fold{N}/adhd_fold{N}-{N}/`
- Missing folds will generate a warning but won't prevent the script from running
