import os
import glob

# Paths
SOURCE = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/"
DEST = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/preprocessed_images"

def check_missing():
    # 1. Get all source filenames (ignoring path)
    # This finds .jpg, .JPG, .png, .PNG, etc.
    src_files = set(os.listdir(SOURCE))
    dest_files = set(os.listdir(DEST))
    
    # 2. Filter out non-image files (like hidden . files)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    src_images = {f for f in src_files if os.path.splitext(f)[1].lower() in valid_exts}
    
    print(f"Total Source Images: {len(src_images)}")
    print(f"Total Destination Images: {len(dest_files)}")
    
    # 3. Find difference
    missing = src_images - dest_files
    print(f"Missing Count: {len(missing)}")
    
    if len(missing) > 0:
        print("\nFirst 10 missing files:")
        print(list(missing)[:10])
        
        # Analyze extensions of missing files
        missing_exts = {}
        for f in missing:
            ext = os.path.splitext(f)[1]
            missing_exts[ext] = missing_exts.get(ext, 0) + 1
        
        print("\nMissing file extensions breakdown:")
        print(missing_exts)

if __name__ == "__main__":
    check_missing()