# NOTE - Clears .pth files and .gpu file from all fold directories

# --- Change to the project root directory ---
cd ../..

cd output/phase_3

for fold_dir in fold_*; do
    echo "Cleaning $fold_dir"
    rm -f "$fold_dir"/*.pth
    rm -f "$fold_dir"/*.gpu*
    rm -f "$fold_dir"/val.csv
    rm -f "$fold_dir"/test.csv
done
echo "Fold cleanup complete."