import pandas as pd
import argparse
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, 
    f1_score, recall_score, precision_score, confusion_matrix
)

def compute_metrics_from_predictions(df):
    """Compute classification metrics from a predictions CSV."""
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    
    # Get probability columns (for AUC)
    prob_cols = [c for c in df.columns if c.startswith('probability_class_')]
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # Compute AUC if we have probability columns
    if len(prob_cols) >= 2:
        y_prob = df[prob_cols].values
        try:
            if len(prob_cols) == 2:
                # Binary classification - use probability of positive class
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"Warning: Could not compute AUC: {e}")
            metrics['auc_roc'] = np.nan
    
    # Compute sensitivity and specificity for binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics

def main(args):
    all_metrics = []

    print(f"Aggregating results from: {args.output_dir}")

    for fold in range(1, args.n_splits + 1):
        # Find the CSV file in the fold directory whose name contains the prefix
        fold_dir = os.path.join(args.output_dir, f"fold_{fold}")
        results_path = None
        if os.path.isdir(fold_dir):
            matches = [f for f in os.listdir(fold_dir) if args.csv_prefix in f and f.endswith('.csv')]
            if matches:
                results_path = os.path.join(fold_dir, matches[0])
                if len(matches) > 1:
                    print(f"Warning: Multiple matches for prefix '{args.csv_prefix}' in fold {fold}: {matches}. Using '{matches[0]}'.")

        if results_path and os.path.exists(results_path):
            print(f"Reading results from: {results_path}")
            predictions_df = pd.read_csv(results_path)
            metrics = compute_metrics_from_predictions(predictions_df)
            metrics['fold'] = fold
            all_metrics.append(metrics)
        else:
            print(f"Warning: No CSV file matching prefix '{args.csv_prefix}' found for fold {fold} in {fold_dir}")

    if not all_metrics:
        print("No metrics found to aggregate. Exiting.")
        return

    # Aggregate the metrics across folds
    aggregated_df = pd.DataFrame(all_metrics)
    
    # Reorder columns to put fold first
    cols = ['fold'] + [c for c in aggregated_df.columns if c != 'fold']
    aggregated_df = aggregated_df[cols]

    # Calculate mean and std for numeric columns (excluding 'fold')
    numeric_cols = [c for c in aggregated_df.columns if c != 'fold']
    mean_metrics = aggregated_df[numeric_cols].mean().to_frame('mean').T
    std_metrics = aggregated_df[numeric_cols].std().to_frame('std').T

    final_summary = pd.concat([mean_metrics, std_metrics])

    print("\n--- Per-Fold Results ---")
    print(aggregated_df.to_string(index=False))
    print("\n--- Aggregated Cross-Validation Results ---")
    print(final_summary.to_string())

    # Save aggregated results
    aggregated_df.to_csv(os.path.join(args.output_dir, "all_folds_results.csv"), index=False)
    final_summary.to_csv(os.path.join(args.output_dir, "final_summary_results.csv"))
    print(f"\nSaved final results to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Aggregate results from k-fold CV')
    parser.add_argument('--output_dir', required=True, help='Path to the main output directory containing fold subdirectories')
    parser.add_argument('--n_splits', required=True, type=int, help='Number of folds that were run')
    parser.add_argument('--csv_prefix', required=True, type=str, help='Substring to match against CSV filenames in each fold directory')
    args = parser.parse_args()
    main(args)