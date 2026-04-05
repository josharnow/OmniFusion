import os
import pandas as pd

def main():
    # Look for test.csv in the current working directory
    csv_filename = "test.csv"
    csv_path = os.path.join(os.getcwd(), csv_filename)
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find '{csv_filename}' in the current directory.")
        print(f"Looked in: {os.getcwd()}")
        return

    # Prompt the user for the desired threshold
    while True:
        try:
            user_input = input("Enter the desired threshold (e.g., 0.2764): ").strip()
            threshold = float(user_input)
            if 0.0 <= threshold <= 1.0:
                break
            else:
                print("Please enter a threshold between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    print(f"\nReading {csv_filename}...")
    df = pd.read_csv(csv_path)

    # Validate that the necessary columns exist
    required_cols = ['probability_class_1', 'predicted_label']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: '{csv_filename}' is missing the required column: '{col}'")
            return

    # Keep track of original positives to show the difference
    original_pos = df['predicted_label'].sum()

    # Apply the logic from threshold_scan.py
    out_df = df.copy()
    out_df['predicted_label'] = (out_df['probability_class_1'] >= threshold).astype(int)
    
    new_pos = out_df['predicted_label'].sum()

    # Generate the output filename and save
    out_filename = f"test_threshold_{threshold:.4f}.csv"
    out_filepath = os.path.join(os.getcwd(), out_filename)
    
    out_df.to_csv(out_filepath, index=False)
    
    # Print the summary stats mimicking threshold_scan.py
    print(f"\nSaved re-thresholded CSV to {out_filename}")
    print(f"  predicted_label: {int(original_pos)} → {int(new_pos)} positives (threshold {threshold:.4f})")

if __name__ == "__main__":
    main()