#!/bin/bash

# Prompt the user for the input filename
read -p "Enter the input filename: " INPUT_FILE

# Define the search string
SEARCH_STRING="--- Checking PyTorch and CUDA environment ---"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Find the line number of the second occurrence
# We use '--' to ensure grep treats the string as a pattern, not a flag
SPLIT_LINE=$(grep -nF -- "$SEARCH_STRING" "$INPUT_FILE" | cut -d: -f1 | sed -n '2p')

# Check if a second occurrence was found
if [ -z "$SPLIT_LINE" ]; then
    echo "Error: Second occurrence of delimiter not found in '$INPUT_FILE'."
    exit 1
fi

# Calculate the line immediately before the split for the first file
LINE_BEFORE=$((SPLIT_LINE - 1))

# Extract filename parts for correct naming
# ${VAR##*.} gets the extension (matches last dot onwards)
# ${VAR%.*} gets the basename (removes last dot onwards)
EXTENSION="${INPUT_FILE##*.}"
BASENAME="${INPUT_FILE%.*}"

# Construct output filenames
# Check if the file actually has an extension to avoid naming errors
if [ "$BASENAME" == "$INPUT_FILE" ]; then
    # No extension detected
    OUT_LP="${INPUT_FILE}_lp_only"
    OUT_FT="${INPUT_FILE}_ft_only"
else
    # Extension detected, insert suffix before it
    OUT_LP="${BASENAME}_lp_only.${EXTENSION}"
    OUT_FT="${BASENAME}_ft_only.${EXTENSION}"
fi

# Create the first file (Content prior to the split line)
head -n "$LINE_BEFORE" "$INPUT_FILE" > "$OUT_LP"
echo "Created $OUT_LP"

# Create the second file (Content from the split line to the end)
tail -n "+$SPLIT_LINE" "$INPUT_FILE" > "$OUT_FT"
echo "Created $OUT_FT"