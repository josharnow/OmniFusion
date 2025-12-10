#!/bin/bash
#SBATCH --job-name=panderm_%a
#SBATCH --output=../../logs/phase_2/finetune_job_%A_%a.out
#SBATCH --error=../../logs/phase_2/finetune_job_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# --- Set the number of splits for your array ---
#SBATCH --array=1-1%3  # Creates 10 jobs, with task IDs from 1 to 10; matches n_splits in Makefile. % limits to 3 concurrent jobs



# --- Your Project Variables ---
PYTHON="python3"
MODEL="PanDerm_Large_FT" # Base model to start fine-tuning from
NB_CLASSES=2
OUTPUT_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_2"
# CSV_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/ISIC 2024 (SLICE-3D)/ISIC_2024_Training_GroundTruth.csv"
CSV_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2" # IMPORTANT: Directory containing your fold_data.csv files
# CSV_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_1" # IMPORTANT: Directory containing your fold_data.csv files
ROOT_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/"
# ROOT_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/ISIC 2024 (SLICE-3D)/ISIC_2024_Training_Input/"
# Below is the pretrained checkpoint for the ENTIRE PanDerm dataset (DO NOT USE FOR THESIS)
# PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"
# Below is the pretrained checkpoint for Imagenet, which the authors used for their pretraining prior to training on the entire dataset from their paper (USE FOR THESIS)
PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/vit_large_patch16_224.orig_in21k.bin"
NUM_WORKERS=0

# --- Define Fold-Specific Paths ---
# This assumes your k-fold CSVs are named like fold_1/fold_data.csv, fold_2/fold_data.csv, etc.
# This path construction is crucial for the job array to work correctly.
# NOTE - ONLY A SINGLE FOLD FOR PHASE 2
FOLD_OUTPUT_DIR="${OUTPUT_BASE_DIR}"
FOLD_CSV_PATH="${CSV_BASE_DIR}/phase_2.csv"

# --- Hyperparameters from README.md ---
# Note: Batch size and LR are adjusted for a single GPU setup
BATCH_SIZE=80             # Smaller batch size for fine-tuning (FOR AMP ENABLED)
# BATCH_SIZE=64             # Smaller batch size for fine-tuning (FOR AMP DISABLED)
LEARNING_RATE=1e-5        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
# LEARNING_RATE=5e-4       
# LEARNING_RATE=1e-5        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
WARMUP_EPOCHS=5           #
EPOCHS=20                 # Start with 10 epochs and increase if needed (README uses 50)
LAYER_DECAY=0.75          #
DROP_PATH=0.3             #
WEIGHT_DECAY=0.1         #
# NOTE - Disabling mixup and cutmix for now
MIXUP=0.0                 #
CUTMIX=0.0                #
# MIXUP=0.8                 #
# CUTMIX=1.0                #
SMOOTHING=0.0
REPROB=0.25               #
MONITOR="recall"     # Metric to monitor for saving best model (ONLY FOR MULTICLASS CLASSIFICATION; otherwise max AUC-ROC is used)

DROPOUT_RATE=0.5          # Adding dropout for fine-tuning to reduce overfitting

# --- Change to the project root directory ---
cd ../..

mkdir -p logs

# --- Environment Setup ---
echo "Starting job on " `date`
echo "Running on node " `hostname`


export LD_LIBRARY_PATH="/opt/nvidia/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/venv/lib/python3.9/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/venv/lib/python3.9/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"



echo "Visible CUDA device environment variables: $CUDA_VISIBLE_DEVICES"
echo "Starting fine-tuning job for Fold ${SLURM_ARRAY_TASK_ID} on $(hostname)"

# create virtualenv with...
if [ ! -d venv ]; then
    python3 -m virtualenv -p python3 venv
    # install libraries with...
    venv/bin/pip install -r requirements.txt
    venv/bin/pip install -r classification/requirements.txt
    venv/bin/pip install -r segmentation/requirements.txt
fi
source venv/bin/activate





# --- Change to the correct directory ---
cd classification


# --- FIX: Set environment variables for torch.distributed ---
# This tells the script to use the local machine for communication.
export MASTER_ADDR=localhost
export MASTER_PORT=12355 # A random, unused port

# --- Linear probing warmup ---
${PYTHON} run_class_finetuning.py \
    --model ${MODEL} \
    --csv_path "${FOLD_CSV_PATH}" \
    --data_path "${FOLD_CSV_PATH}" \
    --batch_size 1024 \
    --root_path "${ROOT_PATH}" \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --output_dir "${FOLD_OUTPUT_DIR}" \
    --log_dir "${FOLD_OUTPUT_DIR}" \
    --nb_classes ${NB_CLASSES} \
    --num_workers ${NUM_WORKERS} \
    --layer_decay ${LAYER_DECAY} \
    --drop_path ${DROP_PATH} \
    --weight_decay ${WEIGHT_DECAY} \
    --mixup ${MIXUP} \
    --cutmix ${CUTMIX} \
    --smoothing ${SMOOTHING} \
    --reprob ${REPROB} \
    --monitor ${MONITOR} \
    --TTA \
    --wandb_name "${SLURM_ARRAY_TASK_ID}_Phase2_LP" \
    --drop ${DROPOUT_RATE} \
    --clip_grad 1.0 \
    --lr 1e-2 \
    --warmup_epochs 2 \
    --epochs 20 \
    --enable_linear_eval \
    --freeze_backbone


# --- Run the Fine-Tuning Script (loads best checkpoint weights produced by linear probing warmup) ---
${PYTHON} run_class_finetuning.py \
    --model ${MODEL} \
    --csv_path "${FOLD_CSV_PATH}" \
    --data_path "${FOLD_CSV_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --root_path "${ROOT_PATH}" \
    --pretrained_checkpoint "${FOLD_OUTPUT_DIR}/checkpoint-last.pth" \
    --output_dir "${FOLD_OUTPUT_DIR}" \
    --log_dir "${FOLD_OUTPUT_DIR}" \
    --nb_classes ${NB_CLASSES} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LEARNING_RATE} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --epochs ${EPOCHS} \
    --layer_decay ${LAYER_DECAY} \
    --drop_path ${DROP_PATH} \
    --weight_decay ${WEIGHT_DECAY} \
    --mixup ${MIXUP} \
    --cutmix ${CUTMIX} \
    --smoothing ${SMOOTHING} \
    --reprob ${REPROB} \
    --monitor ${MONITOR} \
    --TTA \
    --drop ${DROPOUT_RATE} \
    --clip_grad 1.0 \
    --wandb_name "${SLURM_ARRAY_TASK_ID}_Phase2_FT" \
    --aa 'rand-m9-mstd0.5-inc1' \
    --enable_linear_eval

    # NOTE - weights FLAG ENABLES THE WEIGHTED RANDOM SAMPLER

echo "Finished fine-tuning job for ${SLURM_ARRAY_TASK_ID}"