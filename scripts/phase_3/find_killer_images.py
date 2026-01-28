# This will find images that produce NaN during evaluation and crash the run. Index values are derived from debug exception statements printed during crashed evaluations.
import pandas as pd
import os
import sys

# ================= CONFIGURATION =================
# Base path pattern. The script will replace {fold} with your input.
# Verify this matches your directory structure exactly.
CSV_PATH_TEMPLATE = '/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3/fold_{fold}/fold_data.csv'

# Path to the actual images (optional, helps verify if they exist)
IMAGE_ROOT = '/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/'
# =================================================

def main():
    print("--- PanDerm Crash Investigator ---")
    
    # 1. Get Inputs
    try:
        fold_num = input("Enter Fold Number (e.g., 3): ").strip()
        indices_str = input("Enter Crash Indices (comma separated, e.g., 28136, 32569): ").strip()
        
        # Parse indices into a list of integers
        target_indices = [int(idx.strip()) for idx in indices_str.split(',') if idx.strip().isdigit()]
        
        if not target_indices:
            print("Error: No valid indices provided.")
            return

    except ValueError:
        print("Error: Invalid input format.")
        return

    # 2. Construct Path
    csv_path = CSV_PATH_TEMPLATE.format(fold=fold_num)
    print(f"\n>>> Loading CSV: {csv_path}")

    if not os.path.exists(csv_path):
        print(f"CRITICAL ERROR: CSV file not found at {csv_path}")
        print("Check your path template in the script.")
        return

    # 3. Load and Filter Data
    df = pd.read_csv(csv_path)
    
    # We must replicate the DataLoarder logic exactly:
    # The indices from the crash log correspond to the VALIDATION set for that fold.
    if 'fold' not in df.columns:
        print("Error: 'fold' column missing in CSV.")
        return
        
    # Filter for validation (where fold == fold_num) and RESET INDEX
    # This aligns the DataFrame index (0, 1, 2...) with the DataLoader index
    val_df = df[df['fold'] == int(fold_num)].reset_index(drop=True)
    
    print(f"Validation Set Size: {len(val_df)} images")
    
    # 4. Find the Files
    print("\n" + "="*40)
    print(f"IDENTIFYING IMAGES FOR FOLD {fold_num}")
    print("="*40)
    
    found_count = 0
    for idx in target_indices:
        if idx < 0 or idx >= len(val_df):
            print(f"[INDEX {idx}] -> OUT OF BOUNDS (Max index is {len(val_df)-1})")
            continue
            
        row = val_df.iloc[idx]
        
        # Try to find the image filename column
        # Adjust 'image' if your CSV uses 'image_id' or 'id'
        img_name = row.get('image', row.get('image_id', row.get('id', 'UNKNOWN_COL')))
        label = row.get('target', row.get('label', 'N/A'))
        
        full_path = os.path.join(IMAGE_ROOT, img_name)
        file_status = "EXISTS" if os.path.exists(full_path) else "MISSING"
        
        print(f"[INDEX {idx}]")
        print(f"  Filename: {img_name}")
        print(f"  Label:    {label}")
        print(f"  Status:   {file_status}")
        print(f"  Path:     {full_path}")
        print("-" * 40)
        found_count += 1

    print(f"\nJob Complete. Found {found_count} files.")
    print("Run the 'sed' command or manual deletion to remove these from your CSV.")

if __name__ == "__main__":
    main()