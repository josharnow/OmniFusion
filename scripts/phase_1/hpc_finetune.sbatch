#!/bin/bash
#SBATCH --job-name=panderm_%a
#SBATCH --output=../../logs/phase_1/finetune_job_%A_%a.out
#SBATCH --error=../../logs/phase_1/finetune_job_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# --- Set the number of splits for your array ---
#SBATCH --array=1-10%1  # Creates 10 jobs, with task IDs from 1 to 10; matches n_splits in Makefile. % limits to 2 concurrent jobs



# --- Your Project Variables ---
PYTHON="python3"
MODEL="PanDerm_Large_FT" # Base model to start fine-tuning from
NB_CLASSES=2
OUTPUT_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_1"
# CSV_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/ISIC 2024 (SLICE-3D)/ISIC_2024_Training_GroundTruth.csv"
CSV_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_1" # IMPORTANT: Directory containing your fold_data.csv files
ROOT_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/ISIC 2024 (SLICE-3D)/ISIC_2024_Training_Input/"
PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"
NUM_WORKERS=0

DISABLE_AMP=0  # Set to 1 to disable automatic mixed precision training

# --- Define Fold-Specific Paths ---
# This assumes your k-fold CSVs are named like fold_1/fold_data.csv, fold_2/fold_data.csv, etc.
# This path construction is crucial for the job array to work correctly.
FOLD_OUTPUT_DIR="${OUTPUT_BASE_DIR}/fold_${SLURM_ARRAY_TASK_ID}"
FOLD_CSV_PATH="${CSV_BASE_DIR}/fold_${SLURM_ARRAY_TASK_ID}/fold_data.csv"

# --- Hyperparameters from README.md ---
# Note: Batch size and LR are adjusted for a single GPU setup
BATCH_SIZE=80             # Smaller batch size for fine-tuning
LEARNING_RATE=5e-5        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
# LEARNING_RATE=5e-4       
# LEARNING_RATE=1e-5        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
WARMUP_EPOCHS=5           #
EPOCHS=70                 # Start with 10 epochs and increase if needed (README uses 50)
MIN_LR=1e-6 # Default: 1e-6
LAYER_DECAY=0.75          #
DROP_PATH=0.2             #
WEIGHT_DECAY=0.05         #
# NOTE - Disabling mixup and cutmix for now
MIXUP=0.0                 #
CUTMIX=0.0                #
# MIXUP=0.8                 #
# CUTMIX=1.0                #
SMOOTHING=0.0
REPROB=0.25               #
MONITOR="recall"     # Metric to monitor for saving best model

EARLY_STOPPING_PATIENCE=10  # Number of epochs with no improvement after which training will be stopped. To disable early stopping, set to 0.
EARLY_STOPPING_DELTA=0.001   # Minimum change in the monitored metric to qualify as an improvement

# --- Change to the project root directory ---
cd ../..

mkdir -p logs

# --- Environment Setup ---
echo "Starting job on " `date`
echo "Running on node " `hostname`

# Purge modules to ensure a clean environment
# module purge

# Load the required CUDA and NCCL modules
# module load cuda
# module load nccl
# module load cuda/12.1
# module load nccl/2.11.4 

# export LD_LIBRARY_PATH="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/venv/lib/python3.9/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
# export LD_LIBRARY_PATH="/opt/nvidia/cuda-12.8/lib64:$LD_LIBRARY_PATH"

export LD_LIBRARY_PATH="/opt/nvidia/cuda-12.8/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/venv/lib/python3.9/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/venv/lib/python3.9/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH"

# Purge any modules to ensure a clean environment
# module purge
# module load cuda/12.8

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

# --- Run the Fine-Tuning Script ---
${PYTHON} run_class_finetuning.py \
    --model ${MODEL} \
    --csv_path "${FOLD_CSV_PATH}" \
    --data_path "${FOLD_CSV_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --root_path "${ROOT_PATH}" \
    --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
    --output_dir "${FOLD_OUTPUT_DIR}" \
    --log_dir "${FOLD_OUTPUT_DIR}" \
    --nb_classes ${NB_CLASSES} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LEARNING_RATE} \
    --min_lr ${MIN_LR} \
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
    --early_stopping_patience ${EARLY_STOPPING_PATIENCE} \
    --early_stopping_delta ${EARLY_STOPPING_DELTA} \
    --TTA \
    --disable_amp ${DISABLE_AMP} \
    --weights 
    

    # NOTE - weights FLAG ENABLES THE WEIGHTED RANDOM SAMPLER

echo "Finished fine-tuning job for Fold ${SLURM_ARRAY_TASK_ID}"