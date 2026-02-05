from PIL import Image, ImageChops
import numpy as np
import cv2
import os

def analyze_physics(file_path):
    """
    Layer 3: Universal Signal Analysis.
    Checks for:
    1. ELA (Compression Consistency) - Finds pasted objects.
    2. High-Frequency Noise - Finds 'smooth' AI generation.
    
    Returns: Impact Score (+15 Real, -20 Fake)
    """
    impact = 0
    print(f"Layer 3 Analyzing Physics: {file_path}...")
    
    try:
        # ELA (Error Level Analysis)
        # AI edits often break the JPEG compression grid.
        original = Image.open(file_path).convert('RGB')
        
        # Save compressed version to memory
        from io import BytesIO
        buf = BytesIO()
        original.save(buf, 'JPEG', quality=90)
        buf.seek(0)
        compressed = Image.open(buf)
        
        # Get difference
        diff = ImageChops.difference(original, compressed)
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        print(f"   [Physics] ELA Score: {max_diff}")

        # Real camera JPEGs usually have low, uniform error (< 15)
        # Manipulated/AI images often spike > 25 due to mismatched compression
        if max_diff > 25:
            impact -= 20 # High Suspicion
        else:
            impact += 10 # Consistent structure

        # Noise Variance (The "Smoothness" Trap) 
        # AI models (Diffusion) struggle to generate chaotic sensor noise.
        img_cv = cv2.imread(file_path, 0) # Load Grayscale
        
        if img_cv is not None:
            # Calculate Laplacian Variance (Focus/Noise metric)
            noise_variance = cv2.Laplacian(img_cv, cv2.CV_64F).var()
            print(f"   [Physics] Noise Variance: {noise_variance:.2f}")
            
            if noise_variance < 50:
                impact -= 15 # Too smooth / blurry (Characteristic of AI)
            elif noise_variance > 300:
                impact += 5 # Natural sensor noise present
        
    except Exception as e:
        print(f"   [Physics Error] {e}")
        pass

    return impact

# ==========================================
# LOCAL TESTER
# ==========================================
if __name__ == "__main__":
    print("To test Layer 3, place a .jpg file here and run this script.")