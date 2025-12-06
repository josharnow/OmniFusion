#!/bin/bash
# Transfers completed folders using rsync

# Prompt user for destination directory
read -p "Enter destination directory: " DEST_DIR

# Ensure destination directory was provided
if [ -z "$DEST_DIR" ]; then
    echo "No destination directory provided. Exiting."
    exit 1
fi

# Prompt user for fold range
read -p "Enter start fold (1-10): " START_FOLD
read -p "Enter end fold (1-10): " END_FOLD

# Validate fold range
if [ -z "$START_FOLD" ] || [ -z "$END_FOLD" ]; then
    echo "Start and end folds must be provided. Exiting."
    exit 1
fi

if [ "$START_FOLD" -lt 1 ] || [ "$START_FOLD" -gt 10 ] || [ "$END_FOLD" -lt 1 ] || [ "$END_FOLD" -gt 10 ]; then
    echo "Fold numbers must be between 1 and 10. Exiting."
    exit 1
fi

if [ "$START_FOLD" -gt "$END_FOLD" ]; then
    echo "Start fold must be less than or equal to end fold. Exiting."
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Build include patterns using a BASH ARRAY (Cleaner and safer than strings)
INCLUDE_ARGS=()
for fold in $(seq $START_FOLD $END_FOLD); do
    INCLUDE_ARGS+=( --include="fold_${fold}/" --include="fold_${fold}/**" )
done

# Define the remote path with DOUBLE QUOTES around the directory path
# This ensures the remote shell handles "MS Thesis" as a single name
REMOTE_PATH='ja50529n@hpcmaster.seidenberg.pace.edu:"/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_1/"'

# Use rsync with the array expansion
echo "Starting transfer..."
rsync -avhP "${INCLUDE_ARGS[@]}" --exclude='checkpoint-last.pth' --exclude='fold_data.csv' --exclude='val.csv' --exclude='*.gpu*' --exclude='*' "$REMOTE_PATH" "$DEST_DIR/"

echo "Transfer of fold_${START_FOLD} to fold_${END_FOLD} to $DEST_DIR completed."