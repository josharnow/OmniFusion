#!/bin/bash
#SBATCH --job-name=PDeval
#SBATCH --output=../../logs/phase_3/evaluate_job_%A_%a.out
#SBATCH --error=../../logs/phase_3/evaluate_job_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# --- Set the number of splits for your array ---
#SBATCH --array=1-10%3  # Creates 10 jobs, with task IDs from 1 to 10; matches n_splits in Makefile. % limits to 3 concurrent jobs



# --- Your Project Variables ---
PYTHON="python3"
MODEL="SkinEHDLF_Hybrid" # Base model to start fine-tuning from
NB_CLASSES=2
OUTPUT_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3"
CSV_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3" # IMPORTANT: Directory containing your fold_data.csv files
ROOT_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/preprocessed_images/" # This is the same as phase 2 since images are unchanged
# Below is the pretrained checkpoint for the ENTIRE PanDerm dataset (DO NOT USE FOR THESIS)
# PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"
# Below is the pretrained checkpoint for Imagenet, which the authors used for their pretraining prior to training on the entire dataset from their paper (USE FOR THESIS)
# NOTE - This cannot be used with a hybrid model like SkinEHDLF_Hybrid; need to use the SkinEHDLF-specific pretrained weights. This is handled by timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=pretrained, num_classes=0) and other backbone initializations in skin_ehdlf.py
# PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/vit_large_patch16_224.orig_in21k.bin"
NUM_WORKERS=8

# --- Define Fold-Specific Paths ---
# This assumes your k-fold CSVs are named like fold_1/fold_data.csv, fold_2/fold_data.csv, etc.
# This path construction is crucial for the job array to work correctly.
FOLD_OUTPUT_DIR="${OUTPUT_BASE_DIR}/fold_${SLURM_ARRAY_TASK_ID}"
FOLD_CSV_PATH="${CSV_BASE_DIR}/fold_data_external_all_test.csv"
# FOLD_CSV_PATH="${CSV_BASE_DIR}/fold_${SLURM_ARRAY_TASK_ID}/fold_data_target.csv"

# --- Hyperparameters from README.md ---
# Note: Batch size and LR are adjusted for a single GPU setup
# BATCH_SIZE=24             # Smaller batch size for fine-tuning (FOR AMP ENABLED)
# BATCH_SIZE=48             # Smaller batch size for fine-tuning (FOR AMP ENABLED)
BATCH_SIZE=384             # Smaller batch size for fine-tuning (FOR AMP DISABLED)
UPDATE_FREQ=1        
# LEARNING_RATE=5e-5        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
# LEARNING_RATE=1e-4        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
# LEARNING_RATE=1e-5        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
LEARNING_RATE=1e-6        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
# LEARNING_RATE=5e-6        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
# LEARNING_RATE=5e-4       
# LEARNING_RATE=1e-5        # A smaller LR is safer for fine-tuning than the 5e-4 in the README
WARMUP_EPOCHS=1           #
# WARMUP_EPOCHS=5           #
EPOCHS=20                 # Start with 10 epochs and increase if needed (README uses 50)
LAYER_DECAY=1.00          # Disabled for SkinEHDLF due to hybrid architecture
DROP_PATH=0.4             #
# DROP_PATH=0.3             #
# WEIGHT_DECAY=0.05         #
WEIGHT_DECAY=0.1         #
# NOTE - Disabling mixup and cutmix for now
CUTMIX=0.0                #
MIXUP=0.0                 #
# MIXUP=0.8                 #
# CUTMIX=1.0                # *** NOTE - THIS COULD HARM PERFORMANCE IF THE DATA IS MULTIMODAL; IT CAN CAUSE THE MODEL TO LEARN SPOOFED FEATURES (ARTIFACTS) FROM MIXED SAMPLES ***
# MIXUP=0.8                 #
# CUTMIX=0.0                
SMOOTHING=0.0
REPROB=0.25               #
MONITOR="recall"     # Metric to monitor for saving best model (ONLY FOR MULTICLASS CLASSIFICATION; otherwise max AUC-ROC is used)

DROPOUT_RATE=0.5 # Adding dropout for fine-tuning to reduce overfitting

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


# --- Run the Fine-Tuning Script (loads best checkpoint weights produced by linear probing warmup) ---
${PYTHON} run_class_finetuning.py \
    --model ${MODEL} \
    --csv_path "${FOLD_CSV_PATH}" \
    --data_path "${FOLD_CSV_PATH}" \
    --batch_size ${BATCH_SIZE} \
    --root_path "${ROOT_PATH}" \
    --pretrained_checkpoint "${FOLD_OUTPUT_DIR}/checkpoint-best-stage-1-target.pth" \
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
    --smoothing ${SMOOTHING} \
    --reprob ${REPROB} \
    --monitor ${MONITOR} \
    --drop ${DROPOUT_RATE} \
    --clip_grad 1.0 \
    --wandb_name "Fold_${SLURM_ARRAY_TASK_ID}_Phase3_FT" \
    --aa '' \
    --mixup ${MIXUP} \
    --cutmix ${CUTMIX} \
    --mixup_prob 1.0 \
    --mixup_switch_prob 0.5 \
    --mixup_mode "batch" \
    --update_freq ${UPDATE_FREQ} \
    --val_batch_size 12 \
    --test_tta \
    --is_skinehdlf \
    --no_auto_resume \
    --no_class_weights \
    --resume "${FOLD_OUTPUT_DIR}/checkpoint-best-stage-1-target.pth" \
    --TTA \
    --eval_threshold 0.5 \
    --eval # This flag forces evaluation ONLY
    # --custom_majority_alpha 1.0 \
    # --custom_minority_alpha 5.0
    # --focal_loss \
    # --focal_loss_gamma 2.0 \
    # --TTA \
    # --focal_loss \
    # --focal_loss_gamma 4.0 \

    # NOTE - weights FLAG ENABLES THE WEIGHTED RANDOM SAMPLER

echo "Finished evaluation job for Fold ${SLURM_ARRAY_TASK_ID}"