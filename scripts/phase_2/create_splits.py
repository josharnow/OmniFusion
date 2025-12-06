import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def create_splits(csv_path, output_path, val_ratio=0.2, random_state=42):
    """
    Create train/val/test splits for phase 2.
    - Train: comes from Dataset A (should already be labeled 'train')
    - Val/Test: split from Dataset B
    """
    df = pd.read_csv(csv_path)
    
    # Check that binary_label column exists for stratification
    if 'binary_label' not in df.columns:
        raise ValueError("CSV must have a 'binary_label' column for stratified splitting")
    
    # If all data is from Dataset B (no split column yet)
    if 'split' not in df.columns:
        # Split B into val and test, stratified by binary_label
        val_df, test_df = train_test_split(
            df,
            test_size=1 - val_ratio,  # 80% for test by default
            random_state=random_state,
            stratify=df['binary_label']
        )
        
        df.loc[val_df.index, 'split'] = 'val'
        df.loc[test_df.index, 'split'] = 'test'
    
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"Split summary:")
    print(f"  Val:  {len(df[df['split'] == 'val']):,} samples ({len(df[df['split'] == 'val']) / len(df) * 100:.1f}%)")
    print(f"  Test: {len(df[df['split'] == 'test']):,} samples ({len(df[df['split'] == 'test']) / len(df) * 100:.1f}%)")
    
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