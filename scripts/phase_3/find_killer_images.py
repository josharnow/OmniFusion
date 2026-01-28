import pandas as pd
import os
import sys

# ================= CONFIGURATION =================
# We assume the file is named 'fold_data.csv' inside the fold folder
# based on your previous logs.
CSV_PATH_TEMPLATE = '/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3/fold_{fold}/fold_data.csv'

# Path to images (for verification)
IMAGE_ROOT = '/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/'
# =================================================

def main():
    print("--- PanDerm Crash Investigator (Split Column Fix) ---")
    
    # 1. Get Inputs
    try:
        # If running from bash script with args
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
    print(f"\n>>> Loading CSV: {csv_path}")

    if not os.path.exists(csv_path):
        # Fallback check: maybe it's named fold_data_X.csv?
        alt_path = csv_path.replace("fold_data.csv", f"fold_data_{fold_num}.csv")
        if os.path.exists(alt_path):
            csv_path = alt_path
            print(f">>> Found alternate file: {csv_path}")
        else:
            print(f"CRITICAL ERROR: Could not find CSV file at {csv_path}")
            return

    df = pd.read_csv(csv_path)
    
    # 3. Filter for Validation Set
    # The crash happens in VALIDATION, so we must look at the 'val' split.
    if 'split' in df.columns:
        # Filter for validation rows
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        print(f"Validation Set Size: {len(val_df)} images (Filtered by split='val')")
    elif 'fold' in df.columns:
        # Fallback for old style just in case
        val_df = df[df['fold'] == int(fold_num)].reset_index(drop=True)
    else:
        print("ERROR: CSV must have 'split' or 'fold' column.")
        print(f"Found columns: {list(df.columns)}")
        return

    # 4. Find the Files
    print("\n" + "="*40)
    print(f"KILLER IMAGES FOR FOLD {fold_num}")
    print("="*40)
    
    for idx in target_indices:
        if idx < 0 or idx >= len(val_df):
            print(f"[INDEX {idx}] -> OUT OF BOUNDS")
            continue
            
        row = val_df.iloc[idx]
        img_name = row.get('image', row.get('image_id', 'UNKNOWN'))
        
        print(f"Index:    {idx}")
        print(f"Filename: {img_name}")
        print("-" * 20)

if __name__ == "__main__":
    main()