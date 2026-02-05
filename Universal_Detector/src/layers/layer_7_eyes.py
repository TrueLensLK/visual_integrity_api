import cv2
import numpy as np
import math

def analyze_eyes(image_path):
    """
    Layer 6: Optical Physics (The "Soul" Check)
    Analyzes corneal reflections (glints) and pupil geometry.
    Real eyes reflect the same world. AI eyes often hallucinate.
    """
    print(f"👁️ Layer 6 Analyzing Eyes: {image_path}...")
    
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
    except (ImportError, AttributeError) as e:
        print(f"   [Physiology] Mediapipe not available: {str(e)}")
        return 0, "Eye Analysis Unavailable"
    
    # Initialize MediaPipe Face Mesh (High Definition)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.5
    ) as face_mesh:
        
        # 1. Load Image
        img = cv2.imread(image_path)
        if img is None:
            return 0, "No Image"
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)

        if not results.multi_face_landmarks:
            print("   [Physiology] No face detected for eye analysis.")
            return 0, "No Face Detected"

        face_landmarks = results.multi_face_landmarks[0]

        # --- INDICES FOR IRIS LANDMARKS (MediaPipe Standards) ---
        # Left Eye Iris: 474, 475, 476, 477 (Center is roughly average)
        # Right Eye Iris: 469, 470, 471, 472
        
        # Helper to get coordinate tuple
        def get_pt(idx):
            pt = face_landmarks.landmark[idx]
            return int(pt.x * w), int(pt.y * h)

        # 2. Extract Eye Regions
        # We use the pupil center landmarks provided by 'refine_landmarks=True'
        # Left Pupil Center: 468, Right Pupil Center: 473
        left_pupil = get_pt(468)
        right_pupil = get_pt(473)
        
        # Radius estimate (distance between pupil center and iris edge)
        left_rad = int(math.hypot(left_pupil[0] - get_pt(474)[0], left_pupil[1] - get_pt(474)[1]) * 1.5)
        right_rad = int(math.hypot(right_pupil[0] - get_pt(469)[0], right_pupil[1] - get_pt(469)[1]) * 1.5)

        # 3. Analyze Specular Highlights (The "Glint" Test)
        def get_glint_vector(pupil_center, radius, eye_name):
            # Crop the eye region
            x, y = pupil_center
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(w, x + radius), min(h, y + radius)
            
            eye_crop = img[y1:y2, x1:x2]
            if eye_crop.size == 0: return None
            
            # Convert to grayscale and find brightest spot (The Reflection)
            gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_eye)
            
            # Calculate vector from pupil center to the highlight
            # max_loc is relative to the crop
            # Center of crop is roughly (radius, radius)
            vec_x = max_loc[0] - radius
            vec_y = max_loc[1] - radius
            
            # Check if glint is bright enough (Real eyes are shiny)
            if max_val < 150:
                print(f"   [Physiology] {eye_name}: Eye too dark or matte (possible AI).")
                return None
                
            return (vec_x, vec_y)

        left_vec = get_glint_vector(left_pupil, left_rad, "Left Eye")
        right_vec = get_glint_vector(right_pupil, right_rad, "Right Eye")
        
        # 4. Compare Vectors (The "Consistency" Check)
        score = 0
        desc = "Neutral"
        
        if left_vec and right_vec:
            # Calculate direction difference
            dot_prod = left_vec[0]*right_vec[0] + left_vec[1]*right_vec[1]
            
            # If dot product is negative, reflections are in opposite directions (IMPOSSIBLE in physics)
            if dot_prod < 0:
                print(f"   [Physiology] Mismatched Reflections! Left:{left_vec}, Right:{right_vec}")
                score = -50
                desc = "Physics Violation: Mismatched Eye Reflections"
            else:
                print(f"   [Physiology] Consistent Reflections. Left:{left_vec}, Right:{right_vec}")
                score = 30
                desc = "Natural Eye Physics"
        else:
            # If eyes are "matte" (no reflection), it's suspicious for high-res portraits
            # But we are lenient because of bad lighting.
            score = -10 
            desc = "Dull/Matte Eyes (Unnatural)"

        print(f"   [Physiology] Score: {score} ({desc})")
        return score, desc