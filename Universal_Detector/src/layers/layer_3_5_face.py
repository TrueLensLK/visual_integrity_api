import cv2
import numpy as np
import os

def analyze_face_consistency(image_path):
    """
    Layer 3.5: Face vs Background Consistency Check.
    Detects if the face has different noise/compression artifacts than the body.
    
    Returns: Score (-30 to +10)
    """
    try:
        # Load Image
        img = cv2.imread(image_path)
        if img is None: return 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect Face
        # We use the built-in OpenCV Face Detector 
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("   [Layer 3.5] No faces found. Skipping Face Check.")
            return 0 
            
        print(f"   [Layer 3.5] Found {len(faces)} face(s). Analyzing consistency...")

        # Analyze First Face Found 
        x, y, w, h = faces[0]
        
        # Face Region (The Suspect)
        face_roi = gray[y:y+h, x:x+w]
        
        # Background Region 
        # We take a patch from the top left 
        # Ensuring we don't grab the face itself
        bg_h, bg_w = 50, 50
        if y > 60: # If face is lower down, grab top left
            bg_roi = gray[0:50, 0:50]
        else: # If face is at top, grab bottom right
            h_img, w_img = gray.shape
            bg_roi = gray[h_img-60:h_img-10, w_img-60:w_img-10]

        # Compute Noise Variance 
        # High Variance = Sharp/Noisy (High Quality Camera)
        # Low Variance = Smooth/Blurry (AI Smoothing or Compression)
        face_noise = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        bg_noise = cv2.Laplacian(bg_roi, cv2.CV_64F).var()
        
        print(f"   [Face Check] Face Noise: {face_noise:.1f} | BG Noise: {bg_noise:.1f}")

        # Calculate Discrepancy 
        # A real photo has relatively consistent noise across the image.
        # A Face Swap often has a BLURRY face on a SHARP background
        
        ratio = face_noise / (bg_noise + 1e-5) # Avoid divide by zero
        
        # Logic: If ratio is wildly different (e.g., Face is 5x smoother than background)
        if ratio < 0.2: 
            # Face is WAY too smooth compared to background -> InSwapper/Filter
            print("   [Layer 3.5] Mismatch! Face is artificially smooth.")
            return -30 
        elif ratio > 5.0:
            # Face is WAY sharper than background -> Pasted High-Res Face
            print("   [Layer 3.5] Mismatch! Face is essentially pasted.")
            return -30
        
        # If passed
        return 10 # Bonus points for consistency

    except Exception as e:
        print(f"   [Layer 3.5 Error] {e}")
        return 0