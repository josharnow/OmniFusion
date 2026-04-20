import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve
from tqdm import tqdm

# BASE_PATH = "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3"
BASE_PATH = "/Users/josh/Downloads/untitled folder 13/Model 2-DS"

# Set to True to save a new CSV with predicted_label re-applied at the best threshold
SAVE_RETHRESHOLDED_CSV = True


def process_fold(fold_number):
    csv_path = os.path.join(BASE_PATH, f"fold_{fold_number}", "test.csv")

    if not os.path.exists(csv_path):
        print(f"  [SKIP] File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    y_true = df['true_label']
    y_prob = df['probability_class_1']

    print(f"\n{'='*60}")
    print(f"Fold {fold_number}  —  {csv_path}")
    print(f"{'='*60}")
    print(f"{'Threshold':<10} {'Sens (Cancer)':<15} {'Spec (Benign)':<15} {'Balanced Acc':<15}")
    print("-" * 60)

    for t in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        b_acc = (sens + spec) / 2

        print(f"{t:<10.2f} {sens:<15.4f} {spec:<15.4f} {b_acc:<15.4f}")

    # --- Calculus-Based Global Maximum ---
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ba_scores = (tpr + (1 - fpr)) / 2
    best_idx = np.argmax(ba_scores)
    best_t = thresholds[best_idx]
    best_ba = ba_scores[best_idx]
    best_sens = tpr[best_idx]
    best_spec = 1 - fpr[best_idx]

    print("-" * 60)
    print(f"GLOBAL MAX (Calculated):")
    print(f"{'Threshold':<10} {'Sens (Cancer)':<15} {'Spec (Benign)':<15} {'Balanced Acc':<15}")
    print(f"{best_t:<10.4f} {best_sens:<15.4f} {best_spec:<15.4f} {best_ba:<15.4f}")

    if SAVE_RETHRESHOLDED_CSV:
        out_df = df.copy()
        out_df.loc[:, 'predicted_label'] = (out_df['probability_class_1'] >= best_t).astype(int)
        original_pos = df['predicted_label'].sum()
        new_pos = out_df['predicted_label'].sum()
        out_path = os.path.join(os.path.dirname(csv_path), f"test_threshold_{best_t:.4f}.csv")
        out_df.to_csv(out_path, index=False)
        print(f"Saved re-thresholded CSV to {out_path}")
        print(f"  predicted_label: {int(original_pos)} → {int(new_pos)} positives (threshold {best_t:.4f})")

    return best_t, y_true.values, y_prob.values


def find_cross_fold_optimal_threshold(fold_data):
    """
    For a dense grid of thresholds, compute balanced accuracy on each fold
    and return the threshold that maximises the average across folds.
    """
    # Build a common threshold grid from all folds' probability values
    all_probs = np.concatenate([y_prob for _, y_prob in fold_data])
    thresholds = np.unique(all_probs)

    best_t = None
    best_avg_ba = -1.0

    n = len(thresholds)
    print(f"Scanning {n} thresholds across {len(fold_data)} fold(s)...")

    for t in tqdm(thresholds, unit="threshold"):
        bas = []
        for y_true, y_prob in fold_data:
            y_pred = (y_prob >= t).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                bas.append((sens + spec) / 2)
        if bas:
            avg_ba = np.mean(bas)
            if avg_ba > best_avg_ba:
                best_avg_ba = avg_ba
                best_t = t

    return best_t, best_avg_ba


# --- Prompt ---
fold_input = input("Enter fold number (or 'all' to process all 10 folds): ").strip().lower()

if fold_input == 'all':
    best_thresholds = []
    fold_data = []
    for i in range(1, 11):
        result = process_fold(i)
        if result is not None:
            best_t, y_true, y_prob = result
            best_thresholds.append(best_t)
            fold_data.append((y_true, y_prob))
    if best_thresholds:
        print(f"\n{'='*60}")
        print(f"Average best threshold across {len(best_thresholds)} fold(s): {np.mean(best_thresholds):.4f}")
        print(f"  (per fold: {', '.join(f'{t:.4f}' for t in best_thresholds)})")

        cross_t, cross_ba = find_cross_fold_optimal_threshold(fold_data) # NOTE - This can take a long time (>1 hour)
        print(f"\nCross-fold optimal threshold (maximises average BAcc): {cross_t:.4f}")
        print(f"  Average BAcc at this threshold: {cross_ba:.4f}")
else:
    process_fold(fold_input)
