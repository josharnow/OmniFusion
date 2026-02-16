import cv2
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/images/"
DEST_DIR = "/home/PACE/ja50529n/MS Thesis/Thesis Data/Skin Cancer Project/PanDerm & SkinEHDLF/phase_2/preprocessed_images"
TARGET_SIZE = (256, 256)

class AdvancedSkinProcessing:
    def __init__(self):
        # Hair Removal Kernel (11x11 for 256px images)
        self.hair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        
        # CLAHE object
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Bilateral Filter Settings (The "Optimization")
        # d=9: Diameter of each pixel neighborhood (9 is standard for noise removal)
        # sigmaColor=75: Mix pixels if colors are close (keeps edges sharp)
        # sigmaSpace=75: Mix pixels if they are close spatially
        self.bi_d = 9
        self.bi_sigmaColor = 75
        self.bi_sigmaSpace = 75

    def process(self, img_bgr: np.ndarray) -> np.ndarray:
        # 1. Resize
        img_resized = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # 2. Artifact (Hair) Removal
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.hair_kernel)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        img_clean = cv2.inpaint(img_rgb, mask, 1, cv2.INPAINT_TELEA)
        
        # 3. Noise Reduction -> CHANGED TO BILATERAL FILTER
        # Replaces cv2.medianBlur(img_clean, 3)
        img_smooth = cv2.bilateralFilter(img_clean, self.bi_d, self.bi_sigmaColor, self.bi_sigmaSpace)
        
        # 4. Contrast Enhancement (CLAHE on LAB)
        lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_final_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
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
        image_paths.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    
    total_files = len(image_paths)
    print(f"Found {total_files} images.")
    print(f"Processing Target Size: {TARGET_SIZE}")
    
    if total_files == 0:
        print("No images found. Check your source path.")
        return

    # 3. Process in Parallel
    # Using ProcessPoolExecutor to speed up CPU-bound image processing
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