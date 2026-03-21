from typing import Dict, List, Tuple

def classify_origin(file_path: str, meta_score: float, spectrum_details: str = "", image_dims: Tuple[int, int] = None) -> Dict:
    """
    Classifies image origin to determine which forensic layers are trustworthy.
    
    Args:
        file_path: Absolute path to the image file
        meta_score: Score from Layer 2 (Metadata). 0 usually means missing/stripped.
        spectrum_details: Description string from Layer 6 (Spectrum), e.g. "JPEG: True (type: mozjpeg)..."
        image_dims: Optional (width, height) tuple from image loading
        
    Returns:
        Dictionary containing:
            - classification: "CAMERA_ORIGINAL", "LIKELY_WEB_SOURCED", or "WEB_SOURCED"
            - is_web_sourced: Boolean
            - trusted_layers: List of layer names to trust
            - unreliable_layers: List of layer names to zero out/downweight
            - reasoning: List of strings explaining the classification
    """
    
    # Initialize signals
    signals = {
        "mozjpeg": False,
        "stripped_exif": False,
        "standard_dims": False,
        "exact_aspect": False
    }
    
    reasoning = []
    
    # 1. MOZJPEG DETECTION (Strongest Signal)
    # Layer 6 puts this in the description string
    if "mozjpeg" in spectrum_details.lower():
        signals["mozjpeg"] = True
        reasoning.append("mozjpeg compression detected (Standard web encoder)")
        
    # 2. STRIPPED EXIF 
    # Layer 2 returns 0 or very low score for missing metadata
    # (Note: Camera originals can have stripped EXIF, but it's a supporting signal)
    if meta_score <= 5: 
        signals["stripped_exif"] = True
        reasoning.append("EXIF metadata missing or stripped")
        
    # 3. STANDARD WEB DIMENSIONS
    # Check if dimensions match common web presets
    if image_dims:
        w, h = image_dims
        web_presets = [
            (1920, 1080), (1280, 720), (1366, 768), (1440, 900), (1600, 900),
            (1024, 768), (800, 600), (640, 480), (1080, 1080), (1080, 1350), # IG
            (1080, 566), (1200, 630), (1200, 628) # FB/Twitter
        ]
        
        # Check exact matches or swaps (portrait/landscape)
        is_preset = (w, h) in web_presets or (h, w) in web_presets
        if is_preset:
            signals["standard_dims"] = True
            reasoning.append(f"Standard web dimensions ({w}x{h})")
            
        # Check exact aspect ratios (16:9, 4:3, 1:1)
        # 16:9 = 1.777..., 4:3 = 1.333...
        ratio = max(w, h) / min(w, h)
        if abs(ratio - 1.7777) < 0.01 or abs(ratio - 1.3333) < 0.01 or abs(ratio - 1.5) < 0.01 or abs(ratio - 1.0) < 0.01:
            signals["exact_aspect"] = True
            
    # === CLASSIFICATION LOGIC ===
    
    # DEFAULT: CAMERA_ORIGINAL
    classification = "CAMERA_ORIGINAL"
    is_web_sourced = False
    
    # WEB_SOURCED: mozjpeg OR (stripped_exif + standard_dims)
    if signals["mozjpeg"]:
        classification = "WEB_SOURCED"
        is_web_sourced = True
    elif signals["stripped_exif"] and signals["standard_dims"]:
        classification = "WEB_SOURCED"
        is_web_sourced = True
        reasoning.append("Combination of stripped EXIF and web dimensions is definitive")
        
    # LIKELY_WEB_SOURCED: stripped_exif but no other strong signals
    # or just standard dims
    elif (signals["stripped_exif"] or signals["standard_dims"]) and not is_web_sourced:
        classification = "LIKELY_WEB_SOURCED"
        is_web_sourced = True # Treat as web for safety, but maybe apply lighter penalties? 
                              # User prompt says: "This classification reduces trust... PRNU/Spectrum get reduced weight rather than zeroed."
                              # But for simplicity in is_web_sourced flag we set True, and let Judge handle nuance?
                              # The prompt says: "WEB_SOURCED ... zeroed out. LIKELY_WEB_SOURCED ... reduced weight."
                              # We will use the classification string in Judge to distinguish.
    
    # Define Trust Lists
    if classification == "WEB_SOURCED":
        # Trust ONLY visual content analysis
        trusted_layers = ["neural_network", "face_consistency", "artifacts", "physics", "c2pa"] 
        # Note: Physics is partial (Bayer only), Artifacts is partial.
        
        unreliable_layers = [
            "prnu", "spectrum", "watermark", "metadata", 
            "eye_physics", "shadow", "physical_continuity", "context"
        ]
        
    elif classification == "LIKELY_WEB_SOURCED":
        # Trust all but reduce confidence in hardware
        trusted_layers = ["neural_network", "face_consistency", "artifacts", "physics", "c2pa", "context"]
        unreliable_layers = ["prnu", "spectrum", "watermark", "metadata"] # Weakened
        
    else: # CAMERA_ORIGINAL
        trusted_layers = ["all"]
        unreliable_layers = []

    return {
        "classification": classification,
        "is_web_sourced": is_web_sourced,
        "trusted_layers": trusted_layers,
        "unreliable_layers": unreliable_layers,
        "reasoning": reasoning,
        "signals": signals
    }
