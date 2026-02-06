import pandas as pd
import os
import sys

# ================= CONFIGURATION =================
CSV_PATH_TEMPLATE = '/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3/fold_{fold}/fold_data_external.csv'
# =================================================

def main():
    print("--- PanDerm Crash Investigator (With Shuffle Replication) ---")
    
    # 1. Get Inputs
    try:
        if len(sys.argv) > 2:
            fold_num = sys.argv[1]
            indices_str = sys.argv[2]
        else:
            fold_num = input("Enter Fold Number (e.g., 3): ").strip()
            indices_str = input("Enter Crash Indices (comma separated): ").strip()
        
        target_indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
        
    except ValueError:
        print("Error: Invalid input.")
        return

    # 2. Load CSV
    csv_path = CSV_PATH_TEMPLATE.format(fold=fold_num)
    if not os.path.exists(csv_path):
        # Fallback check
        alt_path = csv_path.replace("fold_data_external.csv", f"fold_data_external_{fold_num}.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
    
    print(f"\n>>> Loading CSV: {csv_path}")
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
    # The DataLoader accesses the validation set using iloc (positional indexing)
    # on the filtered dataframe.
    print(f"Validation Set Size: {len(val_df)}")
    print("\n" + "="*40)
    print(f"KILLER IMAGES FOR FOLD {fold_num}")
    print("="*40)
    
    for idx in target_indices:
        if idx < 0 or idx >= len(val_df):
            print(f"[INDEX {idx}] -> OUT OF BOUNDS")
            continue
            
        # Use iloc to grab the row by its INTEGER position in the validation set
        # This matches how __getitem__ works in your Uni_Dataset
        row = val_df.iloc[idx]
        
        img_name = row.get('image', row.get('image_id', 'UNKNOWN'))
        label = row.get('binary_label', row.get('label', 'N/A'))
        
        print(f"Index:    {idx}")
        print(f"Filename: {img_name}")
        print(f"Label:    {label}")
        print("-" * 20)

if __name__ == "__main__":
    main()