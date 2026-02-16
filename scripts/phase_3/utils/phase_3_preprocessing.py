import cv2
import numpy as np
import os
import glob
from PIL import Image

class AdvancedSkinProcessing:
    """
    Implements specific dermatological image preprocessing steps:
    1. Artifact Removal (DullRazor algorithm for hair removal)
    2. Noise Reduction (Median Blurring)
    3. Contrast Enhancement (CLAHE in LAB color space)
    """
    
    def __init__(self, 
                 hair_removal_kernel=(17, 17), 
                 noise_kernel=5, 
                 clahe_clip=2.0, 
                 clahe_grid=(8, 8)):
        self.hair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, hair_removal_kernel)
        self.noise_kernel = noise_kernel
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    def remove_artifacts(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Implements the DullRazor algorithm to remove hair and dark artifacts.
        
        Args:
            img_rgb: Input image in RGB format (numpy array).
        Returns:
            Inpainted image with artifacts removed.
        """
        # 1. Convert to grayscale for feature detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 2. Black-Hat transform to isolate dark hair structures against light skin
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.hair_kernel)
        
        # 3. Thresholding to create a binary mask of the hairs
        # Threshold of 10 is standard for isolating distinct hairs
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # 4. Inpainting to replace hair pixels with neighboring skin data
        # radius=1, method=INPAINT_TELEA (Navier-Stokes based)
        inpainted = cv2.inpaint(img_rgb, mask, 1, cv2.INPAINT_TELEA)
        
        return inpainted

    def reduce_noise(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Applies Median Blurring to reduce salt-and-pepper noise while preserving edges.
        """
        return cv2.medianBlur(img_rgb, self.noise_kernel)

    def enhance_contrast(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        applied strictly to the Luminance (L) channel in LAB color space.
        """
        # 1. Convert to LAB color space
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        
        # 2. Split channels
        l, a, b = cv2.split(lab)
        
        # 3. Apply CLAHE to the L-channel
        l_enhanced = self.clahe.apply(l)
        
        # 4. Merge channels and convert back to RGB
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    def process_image(self, img: np.ndarray | Image.Image) -> np.ndarray:
        """
        Runs the full pipeline on a single image.
        Accepts PIL Image or Numpy Array (RGB).
        Returns Numpy Array (RGB).
        """
        # Convert PIL to Numpy if necessary
        if isinstance(img, Image.Image):
            img = np.array(img)
            
        # Ensure we are working with an RGB numpy array
        if len(img.shape) == 2:  # Grayscale check
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        # --- PIPELINE SEQUENCE ---
        # 1. Remove Artifacts (Hair) first so they don't get enhanced
        processed = self.remove_artifacts(img)
        
        # 2. Reduce Noise to smooth out inpainting residues
        processed = self.reduce_noise(processed)
        
        # 3. Enhance Contrast last on the clean image
        processed = self.enhance_contrast(processed)
        
        return processed

# --- USAGE EXAMPLES ---

if __name__ == "__main__":
    # Settings
    INPUT_DIR = "raw_data"        # Change this to your raw images folder
    OUTPUT_DIR = "phase_3_data"   # Change this to where you want clean images
    
    # Initialize Processor
    processor = AdvancedSkinProcessing()
    
    # Check if directories exist (create output if needed)
    if not os.path.exists(INPUT_DIR):
        print(f"Please create an input directory named '{INPUT_DIR}' or update the path.")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Get all images
        image_paths = glob.glob(os.path.join(INPUT_DIR, "*.*"))
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        count = 0
        for path in image_paths:
            if not any(path.lower().endswith(ext) for ext in valid_exts):
                continue
                
            try:
                # Read Image (OpenCV reads in BGR by default)
                bgr_img = cv2.imread(path)
                if bgr_img is None:
                    continue
                    
                # Convert BGR -> RGB for processing
                rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                
                # Process
                final_rgb = processor.process_image(rgb_img)
                
                # Convert RGB -> BGR for saving with OpenCV
                final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)
                
                # Save
                filename = os.path.basename(path)
                save_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(save_path, final_bgr)
                
                count += 1
                if count % 10 == 0:
                    print(f"Processed {count} images...")
                    
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
        print(f"Done. Processed {count} images to '{OUTPUT_DIR}'.")