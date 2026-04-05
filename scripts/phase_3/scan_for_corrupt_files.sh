#!/bin/bash

# =================================================================
#  Data Integrity Scanner for PanDerm (Fold 1)
# =================================================================

cd ../..

# create virtualenv with...
if [ ! -d venv ]; then
    python3 -m virtualenv -p python3 venv
    # install libraries with...
    venv/bin/pip install -r requirements.txt
    venv/bin/pip install -r classification/requirements.txt
    venv/bin/pip install -r segmentation/requirements.txt
fi
source venv/bin/activate

cd scripts/phase_3

# 1. Create the Python scanning script
echo ">>> Creating 'scan_fold1.py'..."

cat << 'EOF' > scan_fold1.py
import pandas as pd
import cv2
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION (Based on your previous logs) ---
CSV_PATH = '/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3/fold_1/fold_data.csv'
ROOT_PATH = '/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/'
# ---------------------------------------------------

print(f"\n>>> Loading CSV from: {CSV_PATH}")
if not os.path.exists(CSV_PATH):
    print(f"ERROR: CSV file not found at {CSV_PATH}")
    exit(1)

df = pd.read_csv(CSV_PATH)

# Filter for Fold 1 Validation only (since that is where the crash happens)
# Assuming 'fold' column exists and validation is where fold == 1
if 'fold' in df.columns:
    val_df = df[df['fold'] == 1]
    print(f">>> Scanning {len(val_df)} images in Fold 1 Validation set...")
else:
    print("WARNING: 'fold' column not found. Scanning entire dataset...")
    val_df = df

bad_files = []

for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
    # Handle possible column names
    img_name = row.get('image', row.get('image_id'))
    
    if not isinstance(img_name, str):
        print(f"SKIPPING Row {idx}: Invalid image name {img_name}")
        continue

    img_path = os.path.join(ROOT_PATH, img_name)
    
    try:
        # 1. Check file existence
        if not os.path.exists(img_path):
            print(f"\n[MISSING] {img_path}")
            bad_files.append((img_path, "Missing File"))
            continue
            
        # 2. Try to decode image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"\n[CORRUPT_DECODE] {img_path}")
            bad_files.append((img_path, "OpenCV Decode Failed"))
            continue
            
        # 3. Check for Math-Breakers (NaNs / Infs)
        if np.isnan(img).any() or np.isinf(img).any():
            print(f"\n[POISONED_PIXEL] {img_path}")
            bad_files.append((img_path, "Contains NaN/Inf pixels"))
            continue
            
        # 4. Check for Dead Images (Zero Variance / Solid Black)
        # This causes Divide-by-Zero in Batch Norm
        if np.std(img) == 0:
            print(f"\n[ZERO_VARIANCE] {img_path}")
            bad_files.append((img_path, "Zero Variance (Solid Color)"))

    except Exception as e:
        print(f"\n[READ_ERROR] {img_path} | Error: {e}")
        bad_files.append((img_path, str(e)))

print("\n" + "="*40)
print(f"SCAN COMPLETE. Found {len(bad_files)} bad files.")
print("="*40)

if len(bad_files) > 0:
    print("List of bad files:")
    for f, reason in bad_files:
        print(f"{reason}: {f}")
else:
    print("SUCCESS: No corrupted files found in Fold 1.")
EOF

# 2. Define your environment python path (Based on your error logs)
PYTHON_EXEC="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/venv/bin/python"

# 3. Execution
echo ">>> Running scanner using: $PYTHON_EXEC"
echo "---------------------------------------------------"

# Run the python script
"$PYTHON_EXEC" scan_fold1.py

echo "---------------------------------------------------"
echo ">>> Done."