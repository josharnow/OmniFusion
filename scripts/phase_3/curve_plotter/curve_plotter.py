# This file is used to plot training, validation, and loss curves after training a model. It reads the log files generated during training and generates plots for analysis.
# (This Python script should prompt the user to specify the log file name; it will look in the 'logs' directory to find the file to parse.)
import pandas as pd
import matplotlib.pyplot as plt
import re
import ast
import numpy as np
import os

# --- Configuration ---
# Assumes log files are in a folder named "logs" in the same directory as this script.
LOG_DIR = "logs/phase_3" 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def parse_log_file(log_file_path):
    """
    Parses the training log file to extract epoch-level metrics.
    """
    epochs = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_baccs = []
    val_accs = []
    val_rocs = []

    # Regex to find the averaged training stats (some logs don't include class_acc)
    # Capture the averaged loss in parentheses after `loss: X (AVG)`
    train_stats_pattern = re.compile(r"Averaged stats:.*?loss: \d+\.\d+ \((\d+\.\d+)\)")
    
    # Regex to find the validation stats dictionary
    val_stats_pattern = re.compile(r"-------------------------- (\{.*?\})")

    temp_train_loss = None
    temp_train_acc = None

    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        print(f"Successfully read file. Parsing {len(lines)} lines...")

        for line in lines:
            # 1. Look for the training summary line
            train_match = train_stats_pattern.search(line)
            if train_match:
                try:
                    temp_train_loss = float(train_match.group(1))
                except (IndexError, ValueError):
                    temp_train_loss = None
                # Many logs do not include a training class_acc in the averaged line.
                # Use NaN for training accuracy so validation rows are still preserved.
                temp_train_acc = np.nan

            # 2. Look for the validation summary line
            val_match = val_stats_pattern.search(line)
            if val_match:
                # 3. If we find a val line, check if we have pending train stats
                try:
                    stats_dict = ast.literal_eval(val_match.group(1))
                    epoch = stats_dict.get('Epoch')
                    
                    # 4. If we have all data for this epoch, save it
                    if epoch is not None and temp_train_loss is not None:
                        epochs.append(epoch)
                        train_losses.append(temp_train_loss)
                        train_accs.append(temp_train_acc)
                        
                        val_losses.append(stats_dict.get('Val Loss'))
                        val_accs.append(stats_dict.get('Val Acc'))
                        val_baccs.append(stats_dict.get('Val BAcc'))
                        val_rocs.append(stats_dict.get('Val ROC'))
                    
                    # 5. Reset temp stats for the next epoch
                    temp_train_loss = None
                    temp_train_acc = None
                except Exception as e:
                    print(f"Warning: Failed to parse stats line. Error: {e}")
                    if temp_train_loss is not None: # Discard if val fails
                        temp_train_loss = None
                        temp_train_acc = None

        # Create DataFrame from the lists
        data = {
            'Epoch': epochs,
            'Training Loss': train_losses,
            'Training Accuracy': train_accs,
            'Validation Loss': val_losses,
            'Validation Accuracy': val_accs,
            'Validation BAcc': val_baccs,
            'Validation ROC': val_rocs
        }
        
        # Create DataFrame; do not drop NaNs so rows with validation-only metrics are kept
        df = pd.DataFrame(data)
        df['Epoch'] = df['Epoch'].astype(int)
        
        return df

    except FileNotFoundError:
        print(f"Error: The file '{log_file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during file reading: {e}")
        return None

def plot_metrics(df, base_output_name, output_dir):
    """
    Generates and saves plots from the metrics DataFrame.
    """
    try:
        print("\n--- Successfully Parsed Data ---")
        print(df.to_markdown(index=False, floatfmt=".4f"))
        
        # --- Save the plots ---
        combined_plot_path = os.path.join(output_dir, f"{base_output_name}_curves.png")
        
        # Get the list of epochs that actually have data
        epoch_ticks = df['Epoch'].unique().astype(int)
        # Create ticks every 2 epochs, or just 1 if the range is small
        tick_step = 2 if (max(epoch_ticks) - min(epoch_ticks)) > 10 else 1

        # --- MODIFICATION: Changed figure size to accommodate 3 plots ---
        plt.figure(figsize=(18, 5)) # Was (12, 5)

        # Plot 1: Loss Curves
        # --- MODIFICATION: Changed subplot from (1, 2, 1) to (1, 3, 1) ---
        plt.subplot(1, 3, 1)
        plt.plot(df['Epoch'], df['Training Loss'], label='Training Loss', marker='o')
        plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss', marker='x')
        plt.title(f'{base_output_name}: Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(min(df['Epoch']), max(df['Epoch']) + 1, tick_step))

        # Plot 2: Accuracy Curves
        # Using Validation BAcc as it's more informative for imbalanced datasets
        # --- MODIFICATION: Changed subplot from (1, 2, 2) to (1, 3, 2) ---
        plt.subplot(1, 3, 2)
        plt.plot(df['Epoch'], df['Training Accuracy'], label='Training Accuracy', marker='o')
        plt.plot(df['Epoch'], df['Validation BAcc'], label='Validation Balanced Acc (BAcc)', marker='x')
        plt.title(f'{base_output_name}: Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy / BAcc')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(min(df['Epoch']), max(df['Epoch']) + 1, tick_step))
        
        # --- NEW PLOT ---
        # Plot 3: AUC-ROC Curve
        plt.subplot(1, 3, 3)
        plt.plot(df['Epoch'], df['Validation ROC'], label='Validation AUC-ROC', marker='s', color='g')
        plt.title(f'{base_output_name}: AUC-ROC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC-ROC')
        plt.legend()
        plt.grid(True)
        plt.xticks(np.arange(min(df['Epoch']), max(df['Epoch']) + 1, tick_step))
        # --- END NEW PLOT ---

        plt.tight_layout()
        
        # Save the combined plot
        plt.savefig(combined_plot_path)
        print(f"\nSuccessfully generated plots and saved to {combined_plot_path}")

    except Exception as e:
        print(f"An error occurred during plotting: {e}")

def main():
    # --- 1. Check for 'logs' directory ---
    # Build absolute logs directory under the project root
    global LOG_DIR
    LOG_DIR = os.path.join(PROJECT_ROOT, LOG_DIR)

    if not os.path.isdir(LOG_DIR):
        print(f"Error: A folder named '{LOG_DIR}' was not found.")
        print("Please create it and place your log files inside.")
        return

    # --- 2. Find all files in the logs directory ---
    log_files = [f for f in os.listdir(LOG_DIR) if os.path.isfile(os.path.join(LOG_DIR, f))]

    if not log_files:
        print(f"No files found in '{LOG_DIR}'. Exiting.")
        return

    print(f"Found {len(log_files)} file(s) in '{LOG_DIR}'. Processing each...\n")
    output_dir = os.path.dirname(__file__)

    # --- 3. Iteratively process each file ---
    for log_filename in sorted(log_files):
        log_file_path = os.path.join(LOG_DIR, log_filename)
        print(f"\n{'='*60}")
        print(f"Processing: {log_filename}")
        print(f"{'='*60}")

        df = parse_log_file(log_file_path)

        if df is not None and not df.empty:
            base_output_name = os.path.splitext(log_filename)[0]
            plot_metrics(df, base_output_name, output_dir)
        else:
            print(f"No data extracted from '{log_filename}'. Skipping.")

if __name__ == "__main__":
    main()