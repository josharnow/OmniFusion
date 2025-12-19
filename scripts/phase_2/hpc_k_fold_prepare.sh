# NOTE - Prepares k-fold CSV files for HPC finetune script

PYTHON="python3"
OUTPUT_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_2" # NOTE - Adjust name according to phase of experiment
N_SPLITS=10

# # --- Change to the project root directory ---
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

mkdir -p ${OUTPUT_DIR}


# Navigate to your classification directory
cd classification

# Run the preparation script
${PYTHON} prepare_k_fold_data_phase_2.py \
    --csv_path "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/phase_2.csv" \
    --output_dir "${OUTPUT_DIR}" \
    --n_splits ${N_SPLITS} \
    --label_column "binary_label"