import os
import io
import json
from PIL import Image, ImageSequence, ImageCms
from pathlib import Path

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("[Warning] pillow-heif not installed. HEIC/AVIF support will be limited.")

# CONFIGURATION
TRIAGE_CONFIG = {
    "MIN_RES": 128,          
    "MAX_RES": 10000,        
    "MAX_FRAMES": 1,         
    "ALLOWED_FORMATS": {'JPEG', 'PNG', 'WEBP', 'TIFF', 'HEIC', 'AVIF', 'BMP'},
    "REQUIRE_ICC": False,
    "COMPRESSION_BPP_THRESHOLD": 2.0  # NEW: Bits-per-pixel threshold to flag heavy compression
}

MODE_DEPTH_MAP = {
    '1': 1, 'L': 8, 'P': 8, 'RGB': 8, 'RGBA': 8, 'CMYK': 8, 
    'I;16': 16, 'I;16L': 16, 'I;16B': 16, 'F': 32
}

def quick_check(file_path: str, config: dict = TRIAGE_CONFIG) -> dict:
    """
    Performs structural forensics to triage files before deep analysis.
    Adds Compression Triage to flag "Web Trash" for downstream math layers.
    """
    path = Path(file_path)
    if not path.exists():
        return {"status": "FAIL", "reason": "File not found"}

    try:
        # Get file size for compression math
        file_size_bytes = os.path.getsize(file_path)

        with Image.open(file_path) as img:
            # 1. FORMAT & INTEGRITY
            fmt = img.format
            if fmt not in config["ALLOWED_FORMATS"]:
                return {"status": "FAIL", "reason": f"Unsupported format: {fmt}"}
            
            # 2. ANIMATION DETECTION
            is_animated = getattr(img, "is_animated", False)
            n_frames = getattr(img, "n_frames", 1)
            if is_animated or n_frames > config["MAX_FRAMES"]:
                return {"status": "FAIL", "reason": f"Animation detected ({n_frames} frames)"}

            # 3. RESOLUTION & ASPECT
            w, h = img.size
            if w < config["MIN_RES"] or h < config["MIN_RES"]:
                return {"status": "FAIL", "reason": f"Resolution too low ({w}x{h})"}
            if w > config["MAX_RES"] or h > config["MAX_RES"]:
                return {"status": "FAIL", "reason": f"Resolution exceeds max ({w}x{h})"}
            
            # 4. COLOR PROFILE VALIDATION
            icc = img.info.get("icc_profile")
            profile_name = "None"
            if icc:
                try:
                    profile = ImageCms.getProfileDescription(io.BytesIO(icc)).strip()
                    profile_name = profile if profile else "Unknown"
                except:
                    profile_name = "Corrupt/Custom"
            elif config["REQUIRE_ICC"]:
                return {"status": "FAIL", "reason": "Missing color profile (ICC)"}

            # 5. BIT DEPTH ANALYSIS
            bit_depth = MODE_DEPTH_MAP.get(img.mode, "Unknown")

            # ---------------------------------------------------------
            # NEW: 6. COMPRESSION & DEGRADATION TRIAGE ("THE SIGNAL MUTE")
            # ---------------------------------------------------------
            is_degraded = False
            degradation_reasons = []

            # Calculate Bits Per Pixel (BPP). Low BPP = Heavy web compression.
            bpp = (file_size_bytes * 8) / (w * h)
            
            if bpp < config["COMPRESSION_BPP_THRESHOLD"]:
                is_degraded = True
                degradation_reasons.append(f"Low Bits-Per-Pixel ({bpp:.2f}) indicates heavy compression.")

            # Check for stripped EXIF data (common in social media/Google downloads)
            has_exif = 'exif' in img.info
            if not has_exif and fmt in ['JPEG', 'WEBP']:
                is_degraded = True
                degradation_reasons.append("Missing EXIF metadata (Likely web-scrubbed).")

            # Check for native WebP usage (Almost always lossy web format)
            if fmt == 'WEBP':
                is_degraded = True
                degradation_reasons.append("WebP format (Inherent loss of high-frequency data).")
            # ---------------------------------------------------------

            return {
                "status": "PASS",
                "is_degraded_signal": is_degraded,      # L5 CEA WILL READ THIS
                "degradation_reasons": degradation_reasons, # LLM JUDGE WILL READ THIS
                "details": {
                    "format": fmt,
                    "resolution": f"{w}x{h}",
                    "file_size_bytes": file_size_bytes,
                    "bits_per_pixel": round(bpp, 2),
                    "bit_depth": bit_depth,
                    "color_profile": profile_name,
                    "has_exif": has_exif
                }
            }

    except Exception as e:
        return {"status": "FAIL", "reason": f"Forensic Triage Error: {str(e)}"}