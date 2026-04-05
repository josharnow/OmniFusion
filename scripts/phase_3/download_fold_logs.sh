#!/bin/bash

# ================= CONFIGURATION =================
REMOTE_USER="ja50529n"
REMOTE_HOST="hpcmaster.seidenberg.pace.edu"
# The directory from your original request
REMOTE_BASE_DIR="/home/PACE/ja50529n/MS Thesis/Model/PanDerm/logs/phase_3"
LOCAL_TARGET_DIR="."
# =================================================

# 1. Get Prefix
read -p "Enter the file prefix to download: " PREFIX

if [ -z "$PREFIX" ]; then
    echo "Error: Prefix cannot be empty."
    exit 1
fi

mkdir -p "$LOCAL_TARGET_DIR"

# 2. Setup SSH Tunnel (Your Method)
SSH_SOCKET="/tmp/ssh_pace_sync_$(date +%s)"

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

echo "--- 🔍 CHECKING FILES ---"

# 3. GET FILE LIST
# We use 'ls -1' inside the directory to get simple filenames.
# We turn off colors (--color=never) to prevent control characters from breaking rsync.
FILE_LIST=$(ssh -S "$SSH_SOCKET" "$REMOTE_USER@$REMOTE_HOST" "cd \"$REMOTE_BASE_DIR\" && ls -1 --color=never ${PREFIX}* 2>/dev/null")

# Check if empty
if [ -z "$FILE_LIST" ]; then
    echo "❌ No files found matching '$PREFIX' in:"
    echo "   $REMOTE_BASE_DIR"
    exit 1
fi

echo "✅ Found files. Starting download..."

# 4. DOWNLOAD LOOP
# We set IFS to newline so the loop handles filenames with spaces correctly
IFS=$'\n'
for FILENAME in $FILE_LIST; do
    
    echo -n "Pulling $FILENAME... "
    
    # Construct the full remote path.
    # We wrap the whole path in single quotes so rsync handles the space in "MS Thesis"
    FULL_PATH="$REMOTE_BASE_DIR/$FILENAME"
    
    rsync -az -e "ssh -S $SSH_SOCKET" "$REMOTE_USER@$REMOTE_HOST:'$FULL_PATH'" "$LOCAL_TARGET_DIR/"
    
    if [ $? -eq 0 ]; then
        echo "✅ OK"
    else
        echo "❌ Failed"
    fi
done
unset IFS

echo "--------------------------------"
echo "Done."