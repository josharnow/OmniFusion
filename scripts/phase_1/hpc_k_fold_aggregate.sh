# NOTE - Represents last stage of phase 1; aggregates results from each fold into a single CSV file

PYTHON="python3"
OUTPUT_DIR="/Users/josh/Software Development/University/Thesis/PanDerm/output/phase_1" # NOTE - Adjust name according to phase of experiment
N_SPLITS=10
CSV_FILENAME="test_threshold_4.csv"

# --- Change to the project root directory ---
cd ../..

cd classification

${PYTHON} aggregate_slurm_results.py \
    --output_dir "${OUTPUT_DIR}" \
    --n_splits ${N_SPLITS} \
    --csv_filename "${CSV_FILENAME}"