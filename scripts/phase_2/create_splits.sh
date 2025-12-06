# NOTE - Represents last stage of phase 1; aggregates results from each fold into a single CSV file

PYTHON="python3"
OUTPUT="/Users/josh/Downloads/split.csv" # NOTE - Adjust name according to phase of experiment

CSV_PATH="/Users/josh/Downloads/phase_2_slice_3d.csv"


${PYTHON} create_splits.py \
    --input "${CSV_PATH}" \
    --output "${OUTPUT}" \
    --val_ratio 0.10 \
    --seed 42