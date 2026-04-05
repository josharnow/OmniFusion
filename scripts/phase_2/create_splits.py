import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def create_splits(csv_path, output_path, val_ratio=0.2, random_state=42):
    """
    Create train/val/test splits for phase 2.
    - Force all images starting with "ISIC_2024__" into the test split
    - Remaining rows are split into train/val (stratified by binary_label) when not already labeled
    """
    df = pd.read_csv(csv_path)
    
    # Check that binary_label column exists for stratification
    if 'binary_label' not in df.columns:
        raise ValueError("CSV must have a 'binary_label' column for stratified splitting")

    if 'image' not in df.columns:
        raise ValueError("CSV must have an 'image' column to identify ISIC_2024 images")

    # Ensure split column exists
    if 'split' not in df.columns:
        df['split'] = pd.NA

    # Force ISIC_2024 images to test
    test_mask = df['image'].astype(str).str.startswith("ISIC_2024__")
    df.loc[test_mask, 'split'] = 'test'
    
    # Split remaining (non-ISIC_2024) rows into train/val if unlabeled
    remaining_mask = ~test_mask
    unlabeled_mask = remaining_mask & df['split'].isna()

    if unlabeled_mask.any():
        remaining_df = df.loc[unlabeled_mask]
        train_df, val_df = train_test_split(
            remaining_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=remaining_df['binary_label']
        )

        df.loc[val_df.index, 'split'] = 'val'
        df.loc[train_df.index, 'split'] = 'train'
    
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"Split summary:")
    print(f"  Train:{len(df[df['split'] == 'train']):,} samples ({len(df[df['split'] == 'train']) / len(df) * 100:.1f}%)")
    print(f"  Val:  {len(df[df['split'] == 'val']):,} samples ({len(df[df['split'] == 'val']) / len(df) * 100:.1f}%)")
    print(f"  Test: {len(df[df['split'] == 'test']):,} samples ({len(df[df['split'] == 'test']) / len(df) * 100:.1f}%)")
    
    print(f"\nClass distribution (binary_label) in train:")
    print(df[df['split'] == 'train']['binary_label'].value_counts())
    print(f"\nClass distribution (binary_label) in val:")
    print(df[df['split'] == 'val']['binary_label'].value_counts())
    print(f"\nClass distribution (binary_label) in test:")
    print(df[df['split'] == 'test']['binary_label'].value_counts())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    create_splits(args.input, args.output, args.val_ratio, args.seed)