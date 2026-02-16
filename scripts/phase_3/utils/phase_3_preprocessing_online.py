import cv2
import numpy as np
from PIL import Image

class AdvancedSkinProcessing:
    """
    Online Preprocessing Transform for PyTorch/Torchvision Pipelines.
    
    Implements:
    1. Resize to Target Size (default 256x256)
    2. Artifact Removal (DullRazor for hair)
    3. Smart Noise Reduction (Bilateral Filter)
    4. Contrast Enhancement (CLAHE in LAB color space)
    
    Usage:
        transform = transforms.Compose([
            AdvancedSkinProcessing(size=(256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            ...
        ])
    """
    
    def __init__(self, size=(256, 256)):
        self.target_size = size
        
        # --- PARAMETERS (Matched to Offline Script) ---
        
        # 1. Hair Removal Kernel
        # (11, 11) is tuned for 256px resolution
        self.hair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        
        # 2. CLAHE Settings (Contrast)
        # Clip Limit 2.0 prevents noise amplification
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # 3. Bilateral Filter Settings (Edge-Preserving Smooth)
        self.bi_d = 9
        self.bi_sigmaColor = 75
        self.bi_sigmaSpace = 75

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Preprocessed image.
        """
        # 1. Convert PIL (RGB) -> Numpy (RGB)
        # Note: OpenCV usually expects BGR, but since we are explicit with conversions
        # below, we can treat this array as RGB throughout to avoid channel swapping errors.
        img_np = np.array(img)
        
        # 2. Resize
        # We enforce resizing here to ensure the kernel sizes (11x11, etc.) 
        # operate on a consistent feature scale.
        img_resized = cv2.resize(img_np, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # 3. Artifact (Hair) Removal
        # Convert RGB -> Gray for detection
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        
        # Black-Hat transform to find dark hairs on light skin
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.hair_kernel)
        
        # Threshold to create mask (10 is standard)
        _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        
        # Inpaint (using the original RGB image)
        img_clean = cv2.inpaint(img_resized, mask, 1, cv2.INPAINT_TELEA)
        
        # 4. Noise Reduction (Bilateral Filter)
        # Preserves lesion borders while smoothing skin texture
        img_smooth = cv2.bilateralFilter(img_clean, self.bi_d, self.bi_sigmaColor, self.bi_sigmaSpace)
        
        # 5. Contrast Enhancement (CLAHE on L-channel)
        lab = cv2.cvtColor(img_smooth, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        lab_enhanced = cv2.merge((l_enhanced, a, b))
        img_final_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        # 6. Convert Numpy (RGB) -> PIL (RGB)
        return Image.fromarray(img_final_rgb)

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.target_size}, filter=Bilateral)'