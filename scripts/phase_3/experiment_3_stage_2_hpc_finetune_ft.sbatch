#!/bin/bash
#SBATCH --job-name=exp_3
#SBATCH --output=../../logs/phase_3/finetune_job_%A_%a.out
#SBATCH --error=../../logs/phase_3/finetune_job_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# NOTE - This script is meant to run multiple experiments with different hyperparameters. For the thesis, we will only run one experiment with the best hyperparameters found from our tuning process. To run multiple experiments, simply change the hyperparameters in this script and re-submit it to the cluster.


# --- Your Project Variables ---
PYTHON="python3"
MODEL="SkinEHDLF_Hybrid" # Base model to start fine-tuning from
NB_CLASSES=2
OUTPUT_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3"
CSV_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3" # IMPORTANT: Directory containing your fold_data.csv files
ROOT_PATH="/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/" # This is the same as phase 2 since images are unchanged
# Below is the pretrained checkpoint for the ENTIRE PanDerm dataset (DO NOT USE FOR THESIS)
# PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/panderm_ll_data6_checkpoint-499.pth"
# Below is the pretrained checkpoint for Imagenet, which the authors used for their pretraining prior to training on the entire dataset from their paper (USE FOR THESIS)
# NOTE - This cannot be used with a hybrid model like SkinEHDLF_Hybrid; need to use the SkinEHDLF-specific pretrained weights. This is handled by timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=pretrained, num_classes=0) and other backbone initializations in skin_ehdlf.py
# PRETRAINED_CHECKPOINT="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/pretrain_weight/vit_large_patch16_224.orig_in21k.bin"
NUM_WORKERS=8

# --- Define Fold-Specific Paths ---
# This assumes your k-fold CSVs are named like fold_1/fold_data.csv, fold_2/fold_data.csv, etc.
# This path construction is crucial for the job array to work correctly.
FOLD_OUTPUT_DIR="${OUTPUT_BASE_DIR}/experiment/fold_1"
FOLD_CSV_PATH="${CSV_BASE_DIR}/fold_1/fold_data_combined.csv"

# --- Hyperparameters from README.md ---
# Note: Batch size and LR are adjusted for a single GPU setup
# BATCH_SIZE=24             # Smaller batch size for fine-tuning (FOR AMP ENABLED)
# BATCH_SIZE=48             # Smaller batch size for fine-tuning (FOR AMP ENABLED)
BATCH_SIZE=64             # Smaller batch size for fine-tuning (FOR AMP DISABLED)
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
echo "Starting fine-tuning job for Fold 1 on $(hostname)"

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
    --pretrained_checkpoint "${OUTPUT_BASE_DIR}/fold_1/checkpoint-best-stage-1.pth" \
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
    --weights \
    --drop ${DROPOUT_RATE} \
    --clip_grad 1.0 \
    --wandb_name "EXPERIMENT_Fold_1_Phase3_FT" \
    --aa 'rand-m9-mstd0.5-inc1' \
    --stratify_source \
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
    --focal_loss \
    --focal_loss_gamma 2.0
    # --custom_majority_alpha 1.0 \
    # --custom_minority_alpha 5.0
    # --TTA \
    # --focal_loss \
    # --focal_loss_gamma 4.0 \

    # NOTE - weights FLAG ENABLES THE WEIGHTED RANDOM SAMPLER

echo "Finished fine-tuning experiment job for Fold 1"