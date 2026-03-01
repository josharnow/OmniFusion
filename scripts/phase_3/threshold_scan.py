import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

# 1. Point this to your actual file path
csv_path = "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3/fold_1/test.csv" 

df = pd.read_csv(csv_path)
y_true = df['true_label']
y_prob = df['probability_class_1']

print(f"{'Threshold':<10} {'Sens (Cancer)':<15} {'Spec (Benign)':<15} {'Balanced Acc':<15}")
print("-" * 60)

# --- Your Original Iterative Scanning ---
for t in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    b_acc = (sens + spec) / 2
    
    print(f"{t:<10.2f} {sens:<15.4f} {spec:<15.4f} {b_acc:<15.4f}")

# --- NEW: Calculus-Based Global Maximum Calculation ---
# roc_curve calculates the TPR and FPR for all mathematically significant thresholds
fpr, tpr, thresholds = roc_curve(y_true, y_prob)

# Balanced Accuracy = (Sensitivity + Specificity) / 2
# Specificity is (1 - FPR)
ba_scores = (tpr + (1 - fpr)) / 2

# Find the index of the maximum score
best_idx = np.argmax(ba_scores)
best_t = thresholds[best_idx]
best_ba = ba_scores[best_idx]
best_sens = tpr[best_idx]
best_spec = 1 - fpr[best_idx]

print("-" * 60)
print(f"GLOBAL MAX (Calculated):")
print(f"{'Threshold':<10} {'Sens (Cancer)':<15} {'Spec (Benign)':<15} {'Balanced Acc':<15}")
print(f"{best_t:<10.4f} {best_sens:<15.4f} {best_spec:<15.4f} {best_ba:<15.4f}")