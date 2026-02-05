import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.insert(0, str(Path(__file__).parent / "Universal_Detector" / "src"))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import shutil
import os

# --- IMPORT LAYERS ---
from layers.layer_2_metadata import analyze_metadata
from layers.layer_3_physics import analyze_physics       # Digital Physics (ELA/Noise)
from layers.layer_3_5_face import analyze_face_consistency
from layers.layer_4_visual import predict_visuals        # Visual AI
from layers.layer_6_spectrum import analyze_spectrum     # Frequency Analysis
from layers.layer_7_eyes import analyze_eyes             # <--- NEW: Optical Physics (Eyes)
from layers.layer_5_judge import calculate_integrity     # The Judge

app = FastAPI()

# Create temp folder for uploads
os.makedirs("temp_uploads", exist_ok=True)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Master Pipeline:
    1. Layer 2: Metadata (Resolution Trap)
    2. Layer 3: Digital Physics (ELA/Noise)
    3. Layer 4: AI Visuals (ViT/EfficientNet)
    4. Layer 5: Face Consistency (Landmarks)
    5. Layer 6a: Spectrum Analysis (Frequency)
    6. Layer 6b: Optical Physics (Eye Reflections)
    7. The Judge: Final Verdict
    """
    try:
        # 1. Save the uploaded file temporarily
        file_path = f"temp_uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"\n--- 🔍 ANALYZING: {file.filename} ---")

        # 2. Run All Layers
        # We capture all evidence before sending it to the Judge
        meta_score, meta_data = analyze_metadata(file_path)
        physics_score = analyze_physics(file_path)
        face_score = analyze_face_consistency(file_path)
        visual_score = predict_visuals(file_path)
        spectrum_score, spectrum_desc = analyze_spectrum(file_path)
        
        # --- NEW: Run Eye Analysis ---
        eye_score, eye_desc = analyze_eyes(file_path)

        # 3. The Judge (Aggregation)
        # PASS ALL 6 SCORES TO THE JUDGE
        final_score, verdict, description = calculate_integrity(
            meta_score, 
            physics_score, 
            visual_score, 
            face_score, 
            spectrum_score, 
            eye_score  # <--- Critical Addition
        )
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

        # 4. Return JSON Result
        return {
            "filename": file.filename,
            "verdict": verdict,
            "confidence_score": final_score,
            "description": description,
            "details": {
                "metadata_score": meta_score,
                "physics_score": physics_score,
                "face_consistency_score": face_score,
                "ai_visual_score": visual_score,
                "spectral_score": spectrum_score,
                "eye_physics_score": eye_score,  # <--- Added to report
                "meta_tags": meta_data,
                "eye_details": eye_desc
            }
        }

    except Exception as e:
        print(f"Server Error: {e}")
        return {"error": str(e)}

@app.get("/")
def home():
    return HTMLResponse("""
    <div style="font-family: sans-serif; text-align: center; padding-top: 50px;">
        <h1>🕵️‍♂️ Universal Deepfake Detector</h1>
        <p><b>7-Layer Defense System Online</b></p>
        <p>Layers Active: Metadata, Digital Physics, Optical Physics (Eyes), Spectrum, Face, Visual AI</p>
        <p>Send a POST request to <code>/analyze</code> with an image file.</p>
    </div>
    """)