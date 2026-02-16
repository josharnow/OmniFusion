#!/bin/bash
#SBATCH --job-name=Pprocess
#SBATCH --output=../../logs/phase_3/preprocessing_job_%A.out
#SBATCH --error=../../logs/phase_3/preprocessing_job_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8


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
echo "Starting preprocessing job for Fold ${SLURM_ARRAY_TASK_ID} on $(hostname)"

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
cd ./scripts/phase_3/utils


python3 phase_3_preprocessing_offline.py

echo "Finished offline image preprocessing."