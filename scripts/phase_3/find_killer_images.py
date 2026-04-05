import pandas as pd
import os
import sys
import argparse

# ================= CONFIGURATION =================
CSV_PATH_TEMPLATE = '/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3/fold_{fold}/{file_name}'
# =================================================

def get_args():
    parser = argparse.ArgumentParser(description='Scan dataset for toxic images')
    parser.add_argument('--file_name', type=str, required=True, help="CSV file name (e.g. fold_data_target.csv)")
    parser.add_argument('--fold', type=str, default=None, help="Fold Number")
    parser.add_argument('--indices', type=str, default=None, help="Comma separated indices (e.g. '10,20,30')")
    return parser.parse_args()

def main():
    # Parse arguments first
    args = get_args()
    file_name = args.file_name
    
    print("--- PanDerm Crash Investigator (With Shuffle Replication) ---")
    
    # 1. Get Inputs (Use Args or Fallback to Interactive)
    fold_num = args.fold
    indices_str = args.indices

    try:
        if fold_num is None:
            fold_num = input("Enter Fold Number (e.g., 3): ").strip()
        
        if indices_str is None:
            indices_str = input("Enter Crash Indices (comma separated): ").strip()
        
        target_indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
        
    except ValueError:
        print("Error: Invalid input.")
        return

    # 2. Load CSV
    csv_path = CSV_PATH_TEMPLATE.format(fold=fold_num, file_name=file_name)
    if not os.path.exists(csv_path):
        # Fallback check
        alt_path = csv_path.replace("fold_data_external.csv", f"fold_data_external_{fold_num}.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
    
    print(f"\n>>> Loading CSV: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # ==============================================================================
    # CRITICAL FIX: Replicate the Shuffle from run_class_finetuning.py (Line 253)
    # ==============================================================================
    print(">>> Replicating 'df.sample(frac=1, random_state=42)' shuffle...")
    df = df.sample(frac=1, random_state=42)
    # ==============================================================================

    # 3. Filter for Validation Set
    if 'split' in df.columns:
        val_df = df[df['split'] == 'val'] # Do NOT reset index yet, preserve structure
    elif 'fold' in df.columns:
        val_df = df[df['fold'] == int(fold_num)]
    else:
        print("ERROR: CSV missing 'split' or 'fold' column.")
        return

    # 4. Find the Files
    print(f"Validation Set Size: {len(val_df)}")
    print("\n" + "="*40)
    print(f"KILLER IMAGES FOR FOLD {fold_num}")
    print("="*40)
    
    for idx in target_indices:
        if idx < 0 or idx >= len(val_df):
            print(f"[INDEX {idx}] -> OUT OF BOUNDS")
            continue
            
        # Use iloc to grab the row by its INTEGER position in the validation set
        row = val_df.iloc[idx]
        
        img_name = row.get('image', row.get('image_id', 'UNKNOWN'))
        label = row.get('binary_label', row.get('label', 'N/A'))
        
        print(f"Index:    {idx}")
        print(f"Filename: {img_name}")
        print(f"Label:    {label}")
        print("-" * 20)

if __name__ == "__main__":
    main()