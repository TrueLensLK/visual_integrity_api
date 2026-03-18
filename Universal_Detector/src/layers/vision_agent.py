import base64
import json
import os
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import requests

# ============================================================================
# 1. DEFINE THE STRUCTURED OUTPUT (The JSON Schema)
# ============================================================================
class VisionForensicReport(BaseModel):
    visual_score: int = Field(
        description="Score from -50 (Definitively AI) to +50 (Definitively Real)"
    )
    confidence: float = Field(
        description="Confidence in the assessment from 0.0 to 1.0. Lower if blurry/compressed."
    )
    visual_uncertain: bool = Field(
        description="True if the image is too low-res or ambiguous to make a definitive call"
    )
    anomalies_detected: List[str] = Field(
        description="Specific AI artifacts found (e.g., 'melted text on sign', 'pupil asymmetry')"
    )
    authentic_elements: List[str] = Field(
        description="Elements that look physically and geometrically correct"
    )
    reasoning: str = Field(
        description="A concise 2-sentence explanation of why this score was chosen."
    )

# ============================================================================
# 2. THE SYSTEM PROMPT (The Detective's Brain)
# ============================================================================
VISION_AGENT_PROMPT = """
You are an expert Digital Forensic Vision Agent. Your job is to analyze images for semantic, physical, and geometric anomalies that indicate generative AI synthesis. 

Do NOT rely on the overall \"look\", \"quality\", or photorealism of the image, as modern AI can produce perfect textures. Instead, rigorously inspect these specific categories:

1. TYPOGRAPHY & SYMBOLS: Look for \"melted\" text, gibberish characters, or asymmetrical logos on clothing, street signs, or background objects.
2. ANATOMY & SYMMETRY: Inspect pupils (are they perfectly circular?), glasses frames (do they connect to the ears?), earrings (do they match?), and fingers/limbs (do they intersect impossibly?).
3. BACKGROUND CONTINUITY: Trace straight lines (fences, brick walls, horizons, window frames) behind subjects. Do they align perfectly when they emerge on the other side?
4. LIGHTING & PHYSICS: Check the shadows. Do the shadows cast by multiple objects align with a single light source? Are reflections in mirrors or water physically accurate?
5. MICRO-DETAILS: Look at hair strands, shoelaces, and buckles. Generative AI often blends distinct objects into one continuous fused texture.

Output a highly objective, structured analysis based strictly on these visual clues.
"""

# ============================================================================
# 3. THE AGENT EXECUTION FUNCTION
# ============================================================================
def encode_image_to_base64(image_path: str) -> str:
    """Helper function to convert local image to base64 for API transmission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_vision_agent(image_path: str) -> dict:
    """
    Executes the Vision Agent using Google Gemini API and your .env API key.
    """
    print(f"[Vision Agent] Inspecting semantics and geometry for: {image_path}")

    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in .env file.")

    base64_image = encode_image_to_base64(image_path)

    # Prepare Gemini API request
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=" + api_key
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [
                {"text": VISION_AGENT_PROMPT},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64_image
                }}
            ]}
        ],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
    except Exception as e:
        print(f"[Vision Agent] Connection failed: {e}")
        raise RuntimeError("Gemini API connection failed.")

    if response.status_code != 200:
        print(f"[Vision Agent] API Error: {response.status_code} {response.text}")
        raise RuntimeError("Gemini API call failed.")

    # Parse the JSON response
    try:
        gemini_response = response.json()
        # Gemini returns the text as a string, so parse it
        text = gemini_response["candidates"][0]["content"]["parts"][0]["text"]
        report = json.loads(text)
    except Exception as e:
        print(f"[Vision Agent] Failed to parse Gemini response: {e}")
        raise

    print(f"[Vision Agent] Analysis complete. Score: {report['visual_score']}")
    return report
