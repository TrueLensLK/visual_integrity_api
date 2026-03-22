"""
Layer 2: Metadata Forensics
"""

import os
from datetime import datetime
from typing import Tuple, Dict
from PIL import Image, ExifTags
from pathlib import Path

# --- AI Technical Signatures (found in XMP/IPTC/PNG text chunks) ---
AI_HIGH_CERTAINTY = {
    "workflow", "prompt", "negative_prompt", "model_hash", 
    "comfyui", "a1111", "automatic1111", "invokeai", "novelai"
}

AI_AMBIGUOUS = {
    "parameters", "sampler_name", "cfg_scale", "steps", "seed", 
    "denoising_strength", "k-diffusion", "latent-diffusion"
}

# --- Legitimate capture/editing software (not AI generators) ---
KNOWN_CAPTURE_SOFTWARE = {
    # Windows
    "microsoft windows", "windows camera", "windows 10", "windows 11",
    "snipping tool", "microsoft photos", "paint", "paint 3d",
    # Screen capture / recording
    "obs", "obs studio", "greenshot", "sharex", "lightshot", "snagit",
    # Mobile
    "samsung", "google camera", "apple", "ios", "android",
    "oneplus", "xiaomi", "huawei", "oppo", "vivo", "pixel",
    # Photo editors (light penalty, not AI)
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "darktable", "rawtherapee", "luminar", "on1",
    # Social / messaging (re-saved)
    "instagram", "whatsapp", "telegram", "signal", "snapchat",
    "facebook", "twitter", "tiktok",
    # Webcam
    "webcam", "logitech", "droidcam", "iriun", "epoccam",
}

# Software that's an editor (minor edit penalty)
EDITOR_SOFTWARE = {"photoshop", "lightroom", "gimp", "affinity", "capture one",
                   "darktable", "rawtherapee", "luminar", "on1"}


class MetadataForensics:
    """
    Layer 2: Metadata forensics.

    Score convention: -50 (strong AI) to +50 (strong real), 0 = neutral.
    Evidence-based: each positive/negative signal adjusts from 0.
    Missing data = neutral (0), NOT a penalty.
    """

    def __init__(self, image_path: str):
        self.path = Path(image_path)
        self.img = Image.open(image_path)
        self.format = (self.img.format or "").upper()
        self.info = self.img.info
        self.exif = self.img.getexif()
        self.filesize_kb = os.path.getsize(image_path) / 1024
        self.findings = []
        self.score = 0  # Start NEUTRAL — evidence moves it

    def analyze(self) -> Tuple[float, Dict]:
        all_metadata_text = str(self.info).lower() + str(list(self.exif.values())).lower()

        # 1. PROCESS LEGITIMATE SIGNALS FIRST
        # This gives us a "buffer" of authenticity.
        if bool(self.exif):
            self._analyze_exif()
        
        if self.format == "JPEG":
            self._check_thumbnail()

        # 2. NUANCED AI SIGNATURE SCAN
        # Instead of an override, we apply a penalty that can be offset.
        found_sigs = []
        ai_penalty = 0

        for sig in (AI_HIGH_CERTAINTY | AI_AMBIGUOUS):
            if sig in all_metadata_text:
                found_sigs.append(sig)
                if sig in AI_HIGH_CERTAINTY:
                    ai_penalty -= 45 # Very strong AI indicator
                else:
                    ai_penalty -= 25 # Could be an AI-edit or upscale

        # 3. APPLY PENALTY WITH REDEMPTION LOGIC
        if found_sigs:
            # If we found strong Camera Hardware (Make/Model), reduce the AI penalty
            # This identifies "AI-Edited Real Photos"
            has_hardware = any("hardware" in f.lower() for f in self.findings)
            
            if has_hardware:
                ai_penalty = max(-20, ai_penalty + 25) # "Pardon" the image
                self.findings.append(f"AI signals ({', '.join(found_sigs)}) detected but mitigated by Camera Hardware")
            else:
                self.findings.append(f"AI technical signatures found: {', '.join(found_sigs)}")
            
            self.score += ai_penalty

        # 4. STRIPPED METADATA DETECTION (Only if no AI sigs were found)
        elif not bool(self.exif) and not bool(self.info.get("xmp")):
            if self.filesize_kb > 500:
                self.score -= 20
                self.findings.append("Large file with completely stripped metadata")

        # Clamp to range
        self.score = max(-50, min(50, self.score))

        return self.score, {
            "verdict": self._get_verdict(),
            "score": self.score,
            "info": "; ".join(self.findings) if self.findings else "No metadata signals",
            "metadata_density": len(all_metadata_text)
        }
    def _analyze_exif(self):
        tags = {ExifTags.TAGS.get(k, k): v for k, v in self.exif.items()}

        # --- Timestamp validation ---
        dt_str = tags.get("DateTime") or tags.get("DateTimeOriginal")
        if dt_str:
            try:
                dt = datetime.strptime(str(dt_str), "%Y:%m:%d %H:%M:%S")
                if dt > datetime.now():
                    self.score -= 20
                    self.findings.append("Impossible Timestamp: future date")
                elif dt.year < 1995:
                    self.score -= 10
                    self.findings.append("Anachronistic Timestamp: predates digital era")
                else:
                    self.score += 8
                    self.findings.append(f"Valid timestamp: {dt_str}")
            except (ValueError, TypeError):
                pass

        # Subsecond data (cameras/phones embed milliseconds — AI tools don't)
        subsec = tags.get("SubsecTime") or tags.get("SubsecTimeOriginal")
        if subsec:
            self.score += 5
            self.findings.append(f"Subsecond timestamp present ({subsec}ms)")

        # --- GPS validation ---
        if "GPSInfo" in tags:
            gps_data = tags["GPSInfo"]
            if isinstance(gps_data, dict) and len(gps_data) >= 4:
                # Has lat+lon at minimum (keys 1-4)
                self.score += 12
                self.findings.append("GPS coordinates present (hardware signature)")
            elif gps_data:
                self.score += 6
                self.findings.append("Partial GPS data present")

        # --- Software analysis ---
        software = str(tags.get("Software", "")).lower().strip()
        if software:
            sw_lower = software

            # Check if it's a known AI generator (shouldn't be in Software tag,
            # but some tools label themselves)
            ai_in_software = any(ai in sw_lower for ai in
                                 ("midjourney", "dall-e", "dalle", "stable diffusion",
                                  "openai", "firefly", "flux", "novelai", "comfyui"))
            if ai_in_software:
                self.score -= 30
                self.findings.append(f"AI generator in Software tag: '{software}'")
            else:
                # Check if it's recognized legitimate software
                is_known = any(s in sw_lower for s in KNOWN_CAPTURE_SOFTWARE)
                is_editor = any(s in sw_lower for s in EDITOR_SOFTWARE)

                if is_editor:
                    self.score += 3  # Edited, but editing ≠ AI-generated
                    self.findings.append(f"Edited with: {software}")
                elif is_known:
                    self.score += 8
                    self.findings.append(f"Capture software: {software}")
                else:
                    self.score += 3  # Unknown software, small positive
                    self.findings.append(f"Software: {software}")

        # --- Hardware signature ---
        make = tags.get("Make", "")
        model = tags.get("Model", "")
        if make and model:
            self.score += 15
            self.findings.append(f"Camera hardware: {make} {model}")
        elif make or model:
            self.score += 8
            self.findings.append(f"Partial hardware: {make or model}")
        # NOTE: Missing Make/Model is NOT penalized — webcams, screenshots,
        # and social media images legitimately lack hardware identifiers.

        # --- Additional EXIF signals ---
        # Orientation tag (cameras set this based on accelerometer)
        if tags.get("Orientation"):
            self.score += 3
            self.findings.append("Orientation tag present (sensor-based)")

        # ExposureTime, FNumber, ISOSpeedRatings — strong camera indicators
        exposure_tags = sum(1 for t in ("ExposureTime", "FNumber",
                                         "ISOSpeedRatings", "FocalLength")
                           if tags.get(t))
        if exposure_tags >= 3:
            self.score += 10
            self.findings.append(f"Full exposure data ({exposure_tags}/4 tags)")
        elif exposure_tags >= 1:
            self.score += 4
            self.findings.append(f"Partial exposure data ({exposure_tags}/4 tags)")

    def _check_thumbnail(self):
        """Check for embedded EXIF thumbnail (JPEG only)."""
        try:
            has_app1 = hasattr(self.img, "applist") and \
                       any("APP1" in str(item) for item in self.img.applist)
        except (AttributeError, TypeError):
            has_app1 = False

        if not has_app1 and self.filesize_kb > 1000:
            self.score -= 5
            self.findings.append("Missing EXIF thumbnail (common in AI/screenshots)")

    def _get_verdict(self) -> str:
        if self.score >= 25:
            return "HIGHLY_LIKELY_REAL"
        if self.score >= 10:
            return "LIKELY_REAL"
        if self.score >= -10:
            return "INCONCLUSIVE"
        if self.score >= -25:
            return "SUSPICIOUS"
        return "LIKELY_AI"


# Entry point for the Layer
def analyze_metadata(image_path: str):
    engine = MetadataForensics(image_path)
    return engine.analyze()