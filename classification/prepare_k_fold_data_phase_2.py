# import argparse
# import os
# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, train_test_split
# import sys

# def get_args_parser():
#     """
#     Parses arguments for the k-fold data preparation script.
#     """
#     parser = argparse.ArgumentParser('K-Fold Data Preparation', add_help=False)
#     parser.add_argument('--csv_path', required=True, type=str, 
#                         help='Path to the full, unsplit dataset CSV file.')
#     parser.add_argument('--output_dir', required=True, type=str, 
#                         help='Directory where the fold subdirectories will be saved.')
#     parser.add_argument('--n_splits', default=10, type=int, 
#                         help='Number of folds for cross-validation.')
#     parser.add_argument('--label_column', default='binary_label', type=str, 
#                         help='Name of the column in the CSV containing the labels for stratification.')
#     parser.add_argument('--validation_size', default=0.15, type=float, 
#                         help='Proportion of the training data to hold out for validation.')
#     return parser

# def main(args):
#     """
#     Main function to execute the data splitting.
#     """
#     print(f"Reading full dataset from: {args.csv_path}")
#     df = pd.read_csv(args.csv_path)

#     if args.label_column not in df.columns:
#         raise ValueError(f"Label column '{args.label_column}' not found in the CSV file.")

#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#         print(f"Created output directory: {args.output_dir}")

#     # Initialize the k-fold splitter
#     outer_skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

#     # Loop through each fold
#     for fold, (train_val_index, test_index) in enumerate(outer_skf.split(df, df[args.label_column])):
#         fold_num = fold + 1
#         print(f"--- Preparing Fold {fold_num}/{args.n_splits} ---")

#         # Create explicit copies to prevent pandas warnings
#         train_val_df = df.iloc[train_val_index].copy()
#         test_df = df.iloc[test_index].copy()

#         # Perform the inner split to create a validation set
#         train_df, val_df = train_test_split(
#             train_val_df,
#             test_size=args.validation_size,
#             stratify=train_val_df[args.label_column],
#             random_state=42
#         )
        
#         print(f"  Fold {fold_num} sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

#         # Create the fold-specific output directory
#         fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_num}")
#         if not os.path.exists(fold_output_dir):
#             os.makedirs(fold_output_dir)

#         # Add the 'split' column that the training script expects
#         train_df.loc[:, 'split'] = 'train'
#         val_df.loc[:, 'split'] = 'val'
#         test_df.loc[:, 'split'] = 'test'
        
#         # Combine the splits into a single dataframe for this fold
#         fold_df = pd.concat([train_df, val_df, test_df])
        
#         # Save the dataframe to a CSV file
#         fold_csv_path = os.path.join(fold_output_dir, 'fold_data.csv')
#         fold_df.to_csv(fold_csv_path, index=False)
#         print(f"  Saved fold data to: {fold_csv_path}")

#     print("\nData preparation for all folds is complete.")

# if __name__ == '__main__':
#     parser = get_args_parser()
#     args = parser.parse_args()
#     main(args)