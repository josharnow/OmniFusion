#!/bin/bash

# ================= CONFIGURATION =================
REMOTE_USER="ja50529n"
REMOTE_HOST="hpcmaster.seidenberg.pace.edu"
REMOTE_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/output/phase_3"
LOCAL_TARGET_DIR="./"
NUM_FOLDS=10
# =================================================

mkdir -p "$LOCAL_TARGET_DIR"

# Setup SSH Tunnel
SSH_SOCKET="/tmp/ssh_pace_csvs_$(date +%s)"

echo "🔐 Opening persistent connection..."
ssh -M -S "$SSH_SOCKET" -f -N "$REMOTE_USER@$REMOTE_HOST"

if [ $? -ne 0 ]; then
    echo "❌ Could not establish SSH connection."
    exit 1
fi

cleanup() {
    echo "🔒 Closing persistent connection..."
    ssh -S "$SSH_SOCKET" -O exit "$REMOTE_USER@$REMOTE_HOST" 2>/dev/null
    rm -f "$SSH_SOCKET"
}
trap cleanup EXIT

echo "--- 🔍 DOWNLOADING test.csv FROM ALL $NUM_FOLDS FOLDS ---"

for i in $(seq 1 $NUM_FOLDS); do
    REMOTE_PATH="$REMOTE_BASE_DIR/fold_$i/test.csv"
    LOCAL_FOLD_DIR="$LOCAL_TARGET_DIR/fold_$i"

    mkdir -p "$LOCAL_FOLD_DIR"
    echo -n "Pulling fold_$i/test.csv... "

    rsync -az -e "ssh -S $SSH_SOCKET" "$REMOTE_USER@$REMOTE_HOST:'$REMOTE_PATH'" "$LOCAL_FOLD_DIR/"

    if [ $? -eq 0 ]; then
        echo "✅ OK"
    else
        echo "❌ Failed (file may not exist yet)"
    fi
done

echo "--------------------------------"
echo "Done. Files saved to $LOCAL_TARGET_DIR/fold_*/test.csv"
