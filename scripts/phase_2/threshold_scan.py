import pandas as pd
from sklearn.metrics import confusion_matrix

# 1. Point this to your actual file path
# If on server: "output/phase_2/fold_1/test.csv"
# If local: "test_threshold_5.csv"
csv_path = "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_2/fold_1/test.csv" 

df = pd.read_csv(csv_path)
y_true = df['true_label']
y_prob = df['probability_class_1']

print(f"{'Threshold':<10} {'Sens (Cancer)':<15} {'Spec (Benign)':<15} {'Balanced Acc':<15}")
print("-" * 60)

# Scanning low thresholds because 0.4 was still too high
for t in [0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    b_acc = (sens + spec) / 2
    
    print(f"{t:<10.2f} {sens:<15.4f} {spec:<15.4f} {b_acc:<15.4f}")