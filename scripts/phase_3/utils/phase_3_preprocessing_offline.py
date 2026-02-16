import cv2
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/"
DEST_DIR = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/preprocessed_images"

class AdvancedSkinProcessing:
    """
    Implements specific dermatological image preprocessing steps:
    1. Artifact Removal (DullRazor algorithm for hair removal)
    2. Noise Reduction (Median Blurring)
    3. Contrast Enhancement (CLAHE in LAB color space)
    """
    
    def __init__(self):
        # Kernel for hair removal (Black-Hat transform)
        self.hair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        # CLAHE object for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Kernel size for Median Blur
        self.noise_kernel = 5

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Runs the full pipeline on a single BGR image (OpenCV standard).
        Returns a BGR image ready for saving.
        """
        # 1. Convert BGR -> RGB (Algorithms work best in RGB/Gray)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Artifact Removal (Hair)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.hair_kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        img_clean = cv2.inpaint(img_rgb, mask, 1, cv2.INPAINT_TELEA)
        
        # 3. Noise Reduction (Median Blur)
        img_smooth = cv2.medianBlur(img_clean, self.noise_kernel)
        
        # 4. Contrast Enhancement (CLAHE on L-channel of LAB)
        lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_final_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 5. Convert RGB -> BGR for saving
        return cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2BGR)

def process_single_file(file_path):
    """
    Helper function to process a single file (for parallel execution).
    """
    try:
        # Initialize processor locally to avoid pickling issues in multiprocessing
        processor = AdvancedSkinProcessing()
        
        # Read image
        img = cv2.imread(file_path)
        if img is None:
            return f"Skipped (Read Error): {file_path}"
        
        # Process
        processed_img = processor.process(img)
        
        # Construct output path
        filename = os.path.basename(file_path)
        save_path = os.path.join(DEST_DIR, filename)
        
        # Save
        cv2.imwrite(save_path, processed_img)
        return None # Success
        
    except Exception as e:
        return f"Error processing {file_path}: {str(e)}"

def main():
    # 1. Create Destination Directory
    if not os.path.exists(DEST_DIR):
        print(f"Creating directory: {DEST_DIR}")
        os.makedirs(DEST_DIR, exist_ok=True)
    
    # 2. Collect all images
    valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_paths = []
    for ext in valid_exts:
        # Use recursive=True if your images are in subfolders, 
        # but for this specific path, flat search is likely what you want.
        image_paths.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    
    total_files = len(image_paths)
    print(f"Found {total_files} images in {SOURCE_DIR}")
    
    if total_files == 0:
        print("No images found. Check your source path.")
        return

    # 3. Process in Parallel
    # Using ProcessPoolExecutor to speed up CPU-bound image processing
    # Adjust max_workers based on your CPU cores (e.g., 4, 8, or os.cpu_count())
    max_workers = min(32, os.cpu_count() + 4) 
    
    print(f"Starting processing with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show a progress bar
        results = list(tqdm(executor.map(process_single_file, image_paths), total=total_files))
        
    # 4. Report Results
    errors = [res for res in results if res is not None]
    
    print("\nProcessing Complete.")
    print(f"Successfully processed: {total_files - len(errors)}")
    print(f"Errors: {len(errors)}")
    
    if errors:
        print("\n--- Error Log ---")
        for err in errors[:10]: # Print first 10 errors
            print(err)
        if len(errors) > 10:
            print(f"...and {len(errors) - 10} more.")

if __name__ == "__main__":
    main()