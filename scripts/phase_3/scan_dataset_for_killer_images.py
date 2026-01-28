from ...classification.datasets.derm_data import Uni_Dataset
import argparse
import pandas as pd
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
import os

# ================= CONFIGURATION =================
# The template path based on your HPC structure
CSV_PATH_TEMPLATE = "/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3/fold_{}/fold_data.csv"

# The image root (remains constant)
DEFAULT_ROOT_PATH = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/"
INPUT_SIZE = 224
# =================================================

def get_args():
    parser = argparse.ArgumentParser(description='Scan dataset for toxic images')
    # Removed --csv_path, added --fold
    parser.add_argument('--fold', type=str, default=None, help="Fold number (e.g. 1)")
    parser.add_argument('--root_path', type=str, default=DEFAULT_ROOT_PATH)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    return parser.parse_args()

def check_image_tensor(tensor, filename):
    """
    Checks a tensor for 3 types of toxicity:
    1. NaNs
    2. Infs
    3. Zero Variance (Solid Color) -> Crushes LayerNorm
    """
    # 1. NaN Check
    if torch.isnan(tensor).any():
        return False, f"NaN detected in {filename}"
    
    # 2. Inf Check
    if torch.isinf(tensor).any():
        return False, f"Inf detected in {filename}"

    # 3. Zero Variance Check (The LayerNorm Killer)
    # Calculate std dev per channel
    std = torch.std(tensor, dim=(1, 2)) # [C, H, W] -> [C]
    if (std < 1e-6).any():
        return False, f"Flat/Dead Signal (Zero Variance) in {filename} - Will crash LayerNorm"

    return True, "OK"

def main():
    args = get_args()

    # 1. Prompt for Fold if not provided
    if args.fold is None:
        fold_num = input("Enter Fold Number (e.g., 1): ").strip()
    else:
        fold_num = args.fold

    # 2. Construct the CSV path dynamically
    csv_path = CSV_PATH_TEMPLATE.format(fold_num)

    print(f"\n--- Starting Pre-Flight Check for FOLD {fold_num} ({args.split}) ---")
    print(f"Target CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ CRITICAL ERROR: CSV file not found at:\n{csv_path}")
        return

    # 3. Setup Transforms
    mean = [0.485, 0.456, 0.406]
    std = [0.228, 0.224, 0.225]
    
    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 4. Load Dataframe
    print(">>> Loading and shuffling dataframe (seed=42)...")
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=42) # Matches training shuffle exactly
    
    # 5. Create Dataset
    dataset = Uni_Dataset(
        df=df,
        root=args.root_path,
        val=(args.split == 'val'),
        train=(args.split == 'train'),
        test=(args.split == 'test'),
        transforms=val_trans,
        binary=True
    )

    print(f"Dataset size: {len(dataset)} images")
    
    # 6. Iterate and Check
    toxic_files = []
    
    print(f"\n>>> Scanning {len(dataset)} images for NaN/Inf/Zero-Variance...")
    for i in tqdm(range(len(dataset))):
        try:
            # Uni_Dataset returns: x, filename, y
            item = dataset[i]
            img_tensor, filename, _ = item
            
            if img_tensor is None:
                print(f"\n[FAIL] Could not load image: {filename}")
                toxic_files.append(filename)
                continue
                
            is_safe, reason = check_image_tensor(img_tensor, filename)
            
            if not is_safe:
                print(f"\n[TOXIC] {reason}")
                toxic_files.append(filename)

        except Exception as e:
            print(f"\n[CRASH] Error at index {i}: {e}")
            try:
                fname = df.iloc[i].get('image', 'Unknown')
                toxic_files.append(fname)
            except:
                pass

    # 7. Report
    print("\n" + "="*40)
    print("SCAN COMPLETE")
    print("="*40)
    if len(toxic_files) == 0:
        print(f"✅ SUCCESS: Fold {fold_num} ({args.split}) is clean.")
    else:
        print(f"❌ FAILURE: Found {len(toxic_files)} toxic images.")
        print("Culprits:")
        for f in toxic_files:
            print(f" - {f}")

if __name__ == "__main__":
    main()