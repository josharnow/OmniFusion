import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# --- Configuration ---
# Assumes output files are in a folder named "output/phase_1" relative to the project root.
# OUTPUT_DIR = "output/phase_1" 
# TODO - prompt for fold number
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def generate_confusion_matrix(csv_file_path, base_filename):
    """
    Reads a CSV file, generates, plots, and interprets a confusion matrix.
    """
    try:
        df = pd.read_csv(csv_file_path)

        # Ensure required columns exist
        if 'true_label' not in df.columns or 'predicted_label' not in df.columns:
            print(f"Error: 'true_label' or 'predicted_label' columns not found in {csv_file_path}")
            return

        # Generate the confusion matrix
        cm = confusion_matrix(df['true_label'], df['predicted_label'])

        # Calculate percentages for annotation
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        with np.errstate(divide='ignore', invalid='ignore'):
             cm_percent = cm.astype('float') / cm_sum
             cm_percent[np.isnan(cm_percent)] = 0.0


        # Create labels for the heatmap
        group_names = ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)']
        group_counts = [f"{value}" for value in cm.flatten()]
        group_percentages = [f"{value:.2%}" for value in cm_percent.flatten()]
        labels = [f"{name}\n{count}\n{percent}" for name, count, percent in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                    xticklabels=['Benign (0)', 'Malignant (1)'], 
                    yticklabels=['Benign (0)', 'Malignant (1)'])
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix ({base_filename})', fontsize=14)
        
        # Save the figure to the same directory as this script
        output_dir = os.path.dirname(__file__)
        plot_path = os.path.join(output_dir, f"{base_filename}_confusion_matrix.png")
        
        plt.savefig(plot_path)
        print(f"Confusion matrix plot saved to {plot_path}")

        # Print the matrix values for interpretation
        tn, fp, fn, tp = cm.flatten()
        print("\n--- Metrics ---")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")

        # Calculate metrics
        benign_recall_specificity = 0.0
        if (tn + fp) > 0:
            benign_recall_specificity = tn / (tn + fp)
            
        malignant_recall_sensitivity = 0.0
        if (tp + fn) > 0:
            malignant_recall_sensitivity = tp / (tp + fn)
            
        bacc = (benign_recall_specificity + malignant_recall_sensitivity) / 2
        
        print(f"\nBenign Recall (Specificity): {benign_recall_specificity:.4f}")
        print(f"Malignant Recall (Sensitivity): {malignant_recall_sensitivity:.4f}")
        print(f"Balanced Accuracy (BAcc): {bacc:.4f}")

    except Exception as e:
        print(f"An error occurred during CSV processing or plotting: {e}")

def main():
    fold_num = input(f"Please enter the fold number: ")

    FOLD_DIR = f"fold_{fold_num}"
    OUTPUT_DIR = os.path.join("output", "phase_1", FOLD_DIR)

    # Build absolute output directory under the project root
    abs_output_dir = os.path.join(PROJECT_ROOT, OUTPUT_DIR)

    if not os.path.isdir(abs_output_dir):
      print(f"Error: The output directory '{abs_output_dir}' was not found.")
      print("Please check the OUTPUT_DIR variable.")
      return

    # Prompt the user for the filename
    csv_filename = input(f"Please enter the name of the CSV file (must be in '{abs_output_dir}'): ")

    # Build the full file path
    csv_file_path = os.path.join(abs_output_dir, csv_filename)

    if not os.path.exists(csv_file_path):
        print(f"Error: File not found at {csv_file_path}")
        return
    
    # Get base name for output file (e.g., "test")
    base_output_name = f"{FOLD_DIR}_{os.path.splitext(csv_filename)[0]}"
    
    print(base_output_name)

    # Generate the matrix
    generate_confusion_matrix(csv_file_path, base_output_name)

if __name__ == "__main__":
    main()