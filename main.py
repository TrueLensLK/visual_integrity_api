"""
AI Image Detection System 
Multi-Layer Forensic Analysis Engine (Full Spectrum Edition)

SYSTEM ARCHITECTURE:
Layer 0:   C2PA Content Credentials (Cryptographic Truth)
Layer 1:   Quick Forensic Triage (File Validation)
Layer 2:   Metadata Analysis (EXIF/AI Signatures)
Layer 3:   Physics Analysis (ELA/Noise)
Layer 3.5: Face Consistency (Face vs Background)
Layer 4:   Neural Network Ensemble (SDXL-Detector + ViT + SigLIP2 + ConvNeXt + Swin + TTA)
Layer 5:   Master Judge (Weighted Multi-Layer Consensus)
Layer 6:   Spectrum Analysis (FFT Frequency Domain)
Layer 7:   Eye Reflection Physics (Optical Consistency)
Layer 8:   Watermark Detection (Visible, Stego, Hash, SynthID)
Layer 8.5: PRNU Sensor Fingerprint (Wavelet + Reference DB)
Layer 9:   Contextual Provenance (Reverse Image Search)
Layer 10:  Shadow Convergence Analysis (Light Source Consistency)
Layer 11:  Physical Continuity (Geometry)
Layer 12:  GAN/Diffusion Artifacts

FINAL BOSS (LLM + Adversarial Debate):
  - Gray zone (score 35-65)     → Single LLM call (Gemini/OpenRouter/Groq)
  - Contradiction detected       → Adversarial Debate:
      Prosecution (Gemini Vision) vs Defense (OpenRouter Vision)
      judged by Convergence Detector (Groq text), max 3 rounds
"""

import io
import os
import sys
import shutil
import uuid
import json
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from urllib.parse import urlparse

from dotenv import load_dotenv
load_dotenv()  # Load .env file for API keys

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, field_validator
import httpx
import requests

# --- PATH CONFIGURATION ---
# Add layers directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Universal_Detector", "src"))

# --- IMPORTS ---
try:
       from Universal_Detector.src.layers.layer_0_c2pa import verify_c2pa
       from Universal_Detector.src.layers.layer_1_triage import quick_check
       from Universal_Detector.src.layers.layer_2_metadata import analyze_metadata
       from Universal_Detector.src.layers.layer_3_physics import analyze_physics
       from Universal_Detector.src.layers.layer_3_5_face import analyze_face_consistency
       from Universal_Detector.src.layers.layer_4_visual import predict_visuals_detailed
       from Universal_Detector.src.layers.layer_5_judge import calculate_integrity
       from Universal_Detector.src.layers.layer_6_spectrum import analyze_spectrum
       from Universal_Detector.src.layers.layer_7_eyes import analyze_eyes
       from Universal_Detector.src.layers.layer_8_watermark import detect_watermarks
       from Universal_Detector.src.layers.layer_8_5_The_Sensor_Fingerprint import analyze_prnu
       from Universal_Detector.src.layers.layer_9_context import analyze_context
       from Universal_Detector.src.layers.layer_10_Shadow_Convergence import get_shadow_score
       from Universal_Detector.src.layers.layer_11_physical_continuity import get_physical_continuity_score
       from Universal_Detector.src.layers.layer_12_artifacts import analyze_artifacts
except ImportError as e:
    print(f"CRITICAL: Missing forensic layer modules. {e}")
    sys.exit(1)

# Import the new Case Builder and Modular Judge
try:
       from Universal_Detector.src.layers.forensic_case_builder import compile_case_file
       from Universal_Detector.src.layers.llm_judge import HybridJudge, LLMVerdict
except ImportError as e:
    print(f"CRITICAL: Missing core system modules (builder/judge). {e}")
    sys.exit(1)


@dataclass
class DetectionResult:
    """Standardized output format for detection results"""
    final_score: int  # 0-100 scale
    verdict: str  # "REAL", "AI-GENERATED", "AI-ENHANCED", or "EDITED"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    description: str
    layer_scores: Dict[str, float]
    layer_details: Dict[str, str]
    processing_time_ms: int
    warnings: list
    timestamp: str
    llm_reasoning: Optional[str] = None
    judge_source: str = "rule-based"
    debate_data: Optional[Dict] = None  # Populated when adversarial debate is used


class AIImageDetector:
    """
    Main orchestrator for the AI image detection system.
    Runs all layers in sequence and produces a final verdict.
    """
    
    def __init__(self, enable_gpu: bool = True, verbose: bool = True, enable_llm_judge: bool = True):
        self.enable_gpu = enable_gpu
        self.verbose = verbose
        self.warnings = []
        self.enable_llm_judge = enable_llm_judge
        
        # Initialize Hybrid Judge
        self.hybrid_judge = None
        if enable_llm_judge:
            try:
                self.hybrid_judge = HybridJudge(
                    enable_llm=True,
                    gemini_api_key=os.getenv("GOOGLE_AI_API_KEY") or os.getenv("GEMINI_API_KEY"),
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
                )
                self.log("LLM Final Boss enabled - Ready to review ambiguous cases", "INFO")
            except Exception as e:
                self.log(f"LLM Judge disabled: {e}", "WARN")
                self.hybrid_judge = None
        
    def log(self, message: str, level: str = "INFO"):
        """Internal logging function"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "[INFO]", "WARN": "[WARN]", "ERROR": "[ERROR]", "SUCCESS": "[OK]"}.get(level, "[INFO]")
            print(f"[{timestamp}] {prefix} {message}")
    
    def analyze_image(self, image_path: str) -> DetectionResult:
        start_time = datetime.now()
        self.warnings = []
        
        self.log(f"Starting analysis of: {os.path.basename(image_path)}", "INFO")
        self.log("=" * 60)
        
        # --- LAYER 1: TRIAGE ---
        triage_result = quick_check(image_path)
        if triage_result["status"] == "FAIL":
            return self._create_error_result(f"File validation failed: {triage_result['reason']}", start_time)
        
        # Global Flags
        is_jpeg = triage_result.get("details", {}).get("format", "").upper() in ("JPEG", "WEBP")
        if image_path.lower().endswith((".jpg", ".jpeg", ".webp")): is_jpeg = True

        layer_scores = {}
        layer_details = {}
        
        # --- LAYER 0: C2PA ---
        self.log("Layer 0: C2PA Content Credentials")
        c2pa_result = verify_c2pa(image_path)
        layer_details["c2pa"] = c2pa_result.get("message", "No credentials")

        # ====================================================================
        # === THE EARLY EXIT (SHORT-CIRCUIT) ===
        # ====================================================================
        if c2pa_result.get("is_ai_flagged") is True:
            reasons = ", ".join(c2pa_result.get("forensic_matches", []))
            self.log(f"C2PA hard-signature found! {reasons}", "ERROR")
            self.log("SHORT-CIRCUITING PIPELINE: Bypassing heavy layers to save compute.", "SUCCESS")
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return DetectionResult(
                final_score=5,  # 5/100 indicates heavily Fake in your scale
                verdict="AI-GENERATED",
                confidence="HIGH",
                description=f"Deterministic proof: C2PA Content Credentials confirm this is AI. ({reasons})",
                layer_scores={"c2pa": -100.0},  # Massive penalty score
                layer_details={"c2pa": c2pa_result.get("message")},
                processing_time_ms=processing_time,
                warnings=self.warnings + ["C2PA Early Exit triggered."],
                timestamp=datetime.now().isoformat(),
                llm_reasoning=None,
                judge_source="early-exit-c2pa"
            )
        else:
            self.log("C2PA clear. Proceeding to forensic layers.", "INFO")
        # ====================================================================

        # --- LAYER 2: METADATA ---
        self.log("Layer 2: Metadata")
        try:
            m_score, m_det = analyze_metadata(image_path)
            layer_scores["metadata"] = m_score
            layer_details["metadata"] = str(m_det)
        except Exception as e:
            layer_scores["metadata"] = 0
            layer_details["metadata"] = f"Error: {e}"

        # --- LAYER 3: PHYSICS ---
        self.log("Layer 3: Physics")
        has_bayer_pattern = False
        try:
            p_res = analyze_physics(image_path)
            layer_scores["physics"] = p_res["impact"] if isinstance(p_res, dict) else p_res
            findings = p_res.get("findings", []) if isinstance(p_res, dict) else []
            layer_details["physics"] = f"Findings: {'; '.join(findings)}" if findings else "Normal"
            # Extract Bayer pattern flag from physics details
            if isinstance(p_res, dict):
                details = p_res.get("details", {})
                has_bayer_pattern = bool(details.get("has_bayer_pattern", False))
        except Exception as e:
            layer_scores["physics"] = 0
            layer_details["physics"] = "Error"

        # --- LAYER 3.5: FACES ---
        self.log("Layer 3.5: Face Consistency")
        try:
            f_res = analyze_face_consistency(image_path)
            layer_scores["face_consistency"] = f_res["impact"] if isinstance(f_res, dict) else f_res
            layer_details["face_consistency"] = f"Faces: {f_res.get('face_count', 0)}"
        except Exception:
            layer_scores["face_consistency"] = 0

        # --- LAYER 4: VISUAL ENSEMBLE ---
        self.log("Layer 4: Neural Ensemble")
        visual_confidence = 1.0
        model_consensus = 0.0
        model_real_votes, model_ai_votes = 0, 0
        try:
            v_det = predict_visuals_detailed(image_path)
            visual_score = v_det.get("impact", 0)
            visual_confidence = v_det.get("confidence", 1.0)
            model_real_votes = v_det.get("real_votes", 0)
            model_ai_votes = v_det.get("ai_votes", 0)
            model_consensus = v_det.get("model_consensus", 0.0)
            
            layer_scores["neural_network"] = visual_score
            layer_details["neural_network"] = f"Votes: Real {model_real_votes} / AI {model_ai_votes}"
        except Exception:
            layer_scores["neural_network"] = 0
            self.warnings.append("Neural models unavailable")

        # --- LAYER 6: SPECTRUM ---
        self.log("Layer 6: Spectrum")
        try:
            s_score, s_desc = analyze_spectrum(image_path)
            layer_scores["spectrum"] = s_score
            layer_details["spectrum"] = s_desc
        except Exception: layer_scores["spectrum"] = 0

        # --- LAYER 7: EYES ---
        self.log("Layer 7: Eye Physics")
        try:
            e_score, e_desc = analyze_eyes(image_path)
            layer_scores["eye_physics"] = e_score
            layer_details["eye_physics"] = e_desc
        except Exception: layer_scores["eye_physics"] = 0

        # --- LAYER 8: WATERMARK ---
        self.log("Layer 8: Watermarks")
        try:
            w_score, w_desc = detect_watermarks(image_path, is_jpeg=is_jpeg)
            layer_scores["watermark"] = w_score
            layer_details["watermark"] = w_desc
        except Exception: layer_scores["watermark"] = 0

        # --- LAYER 8.5: PRNU ---
        self.log("Layer 8.5: PRNU Sensor Fingerprint")
        try:
            prnu_score, prnu_desc = analyze_prnu(image_path, is_jpeg_hint=is_jpeg)
            layer_scores["prnu"] = prnu_score
            layer_details["prnu"] = prnu_desc
        except Exception: layer_scores["prnu"] = 0

        # --- LAYER 9: CONTEXT ---
        self.log("Layer 9: Context")
        try:
            c_score, c_det = analyze_context(image_path)
            layer_scores["context"] = c_score
            layer_details["context"] = c_det.get("note", "")
        except Exception: layer_scores["context"] = 0

        # --- LAYER 10: SHADOW ---
        self.log("Layer 10: Shadow Convergence")
        try:
            sh_score = get_shadow_score(image_path)
            layer_scores["shadow"] = sh_score
            layer_details["shadow"] = f"Score: {sh_score}"
        except Exception: layer_scores["shadow"] = 0

        # --- LAYER 11: GEOMETRY ---
        self.log("Layer 11: Geometry")
        try:
            pc_score, pc_desc = get_physical_continuity_score(image_path)
            layer_scores["physical_continuity"] = pc_score
            layer_details["physical_continuity"] = pc_desc
        except Exception: layer_scores["physical_continuity"] = 0

        # --- LAYER 12: ARTIFACTS ---
        self.log("Layer 12: Artifacts")
        try:
            a_res = analyze_artifacts(image_path, is_jpeg=is_jpeg)
            layer_scores["artifacts"] = a_res["score"]
            layer_details["artifacts"] = a_res["description"]
        except Exception: layer_scores["artifacts"] = 0

        # ========================================
        # LAYER 5: MASTER JUDGE (Rule-Based)
        # ========================================
        self.log("Layer 5: Master Judge (Rule-Based)")
        final_score, verdict, description = calculate_integrity(
            c2pa_res=c2pa_result,
            meta_score=layer_scores.get("metadata", 0),
            physics_score=layer_scores.get("physics", 0),
            face_score=layer_scores.get("face_consistency", 0),
            visual_score=layer_scores.get("neural_network", 0),
            spectrum_score=layer_scores.get("spectrum", 0),
            eye_score=layer_scores.get("eye_physics", 0),
            watermark_score=layer_scores.get("watermark", 0),
            watermark_desc=layer_details.get("watermark", ""),
            prnu_score=layer_scores.get("prnu", 0),
            context_score=layer_scores.get("context", 0),
            context_details={"note": layer_details.get("context", "")},
            shadow_score=layer_scores.get("shadow", 0),
            shadow_desc=layer_details.get("shadow", ""),
            artifact_score=layer_scores.get("artifacts", 0),
            physical_continuity_score=layer_scores.get("physical_continuity", 0),
            visual_confidence=visual_confidence,
            is_jpeg=is_jpeg,
            visual_uncertain=False,
            model_real_votes=model_real_votes,
            model_ai_votes=model_ai_votes,
            model_count=5,
            model_consensus=model_consensus,
            has_bayer_pattern=has_bayer_pattern
        )

        rule_based_verdict = verdict
        rule_based_score = final_score
        rule_based_description = description
        judge_source = "rule-based"
        llm_reasoning = None
        llm_obj = None  # Track LLM/Debate result for debate_data extraction

        # ========================================
        # LLM FINAL BOSS
        # ========================================
        if self.hybrid_judge is not None:
            try:
                # 1. Compile Case File (Categorized)
                case_file = compile_case_file(
                    image_path=image_path,
                    layer_scores=layer_scores,
                    layer_details=layer_details,
                    rule_based_verdict=rule_based_verdict,
                    rule_based_score=rule_based_score,
                    rule_based_description=rule_based_description,
                    c2pa_result=c2pa_result,
                    is_jpeg=is_jpeg,
                    visual_confidence=visual_confidence,
                    model_consensus=model_consensus,
                    model_real_votes=model_real_votes,
                    model_ai_votes=model_ai_votes,
                    warnings=self.warnings
                )

                # 2. Consult Hybrid Judge
                self.log("Consulting LLM Judge for second opinion...", "INFO")
                final_verdict, f_score, f_desc, llm_obj = self.hybrid_judge.judge(
                    case_file=case_file,
                    rule_based_verdict=rule_based_verdict,
                    rule_based_score=rule_based_score,
                    rule_based_description=rule_based_description,
                    image_path=image_path
                )

                # 3. Apply Override if LLM intervened
                if llm_obj:
                    # NEW LOGIC: Do not allow LLM to override a Rule-Based Kill Switch
                    if "KILL SWITCH ACTIVATED" in rule_based_description and llm_obj.verdict != rule_based_verdict:
                        self.log(f"LLM attempted {llm_obj.verdict} override, but KILL SWITCH takes priority. Denied.", "WARN")
                        # Revert back to the mathematical rule-based scores
                        verdict = rule_based_verdict
                        final_score = rule_based_score
                        description = rule_based_description + f" [LLM override to {llm_obj.verdict} denied by Layer 5 Kill Switch]"
                        judge_source = "hybrid (rule-enforced)"
                        llm_reasoning = llm_obj.reasoning
                    else:
                        # Allow normal override for ambiguous cases
                        # Use the pre-calculated tuple values (not llm_obj attributes)
                        verdict = final_verdict
                        final_score = f_score
                        description = f_desc
                        is_debate = getattr(llm_obj, 'method', '') == 'adversarial_debate'
                        judge_source = "debate" if is_debate else "llm"
                        llm_reasoning = llm_obj.reasoning
                        label = "DEBATE" if is_debate else "LLM"
                        self.log(f"{label} OVERRIDE: {verdict}", "SUCCESS")
                else:
                    self.log("LLM agrees with Rule-Based verdict.", "INFO")

            except Exception as e:
                self.log(f"LLM Judge process failed: {e}", "WARN")

        # Final Cleanup
        confidence = self._calculate_confidence(final_score, layer_scores, c2pa_result)
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)

        # Extract debate data if adversarial debate was used
        debate_data = None
        if llm_obj and getattr(llm_obj, 'method', '') == 'adversarial_debate':
            debate_data = {
                "rounds_taken": llm_obj.rounds_taken,
                "debate_summary": llm_obj.debate_summary,
                "debate_history": llm_obj.debate_history
            }

        self.log(f"FINAL: {verdict} ({final_score}/100) - {judge_source}", "SUCCESS")
        
        return DetectionResult(
            final_score=final_score,
            verdict=verdict,
            confidence=confidence,
            description=description,
            layer_scores=layer_scores,
            layer_details=layer_details,
            processing_time_ms=processing_time,
            warnings=self.warnings.copy(),
            timestamp=datetime.now().isoformat(),
            llm_reasoning=llm_reasoning,
            judge_source=judge_source,
            debate_data=debate_data
        )

    def _calculate_confidence(self, final_score: int, layer_scores: Dict, c2pa_result: Dict) -> str:
        if c2pa_result.get("status") == "valid": return "HIGH"
        agreement_count = 0
        target = "REAL" if final_score > 50 else "FAKE"
        for s in layer_scores.values():
            if target == "REAL" and s > 10: agreement_count += 1
            if target == "FAKE" and s < -10: agreement_count += 1
        
        ratio = agreement_count / max(1, len(layer_scores))
        if ratio > 0.6: return "HIGH"
        if ratio > 0.4: return "MEDIUM"
        return "LOW"

    def _create_error_result(self, msg: str, start_time: datetime) -> DetectionResult:
        return DetectionResult(0, "ERROR", "N/A", msg, {}, {"error": msg}, 
                               int((datetime.now()-start_time).total_seconds()*1000), 
                               [msg], datetime.now().isoformat())


# ========================================
# CLI & API SETUP
# ========================================
def _sanitize(obj):
    """Deep clean object for JSON serialization (Fixes float32 errors)"""
    if isinstance(obj, dict): return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# --- CLI Execution ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)
    
    detector = AIImageDetector()
    res = detector.analyze_image(sys.argv[1])
    
    print(f"\n=== REPORT: {res.verdict} ===")
    print(f"Score: {res.final_score}")
    print(f"Source: {res.judge_source}")
    if res.llm_reasoning:
        print(f"LLM Reasoning: {res.llm_reasoning}")
    print("-" * 30)
    for k, v in res.layer_scores.items():
        print(f"{k:20}: {v:>6.1f}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and "uvicorn" not in sys.argv[0]:
        main()

# ===========================================================================
# /extract-phash bootstrap  –  must be defined before FastAPI app creation
# ===========================================================================
from contextlib import asynccontextmanager

from Universal_Detector.src.layers.phash_extractor import (
    extract_phash as _run_extract_phash,
    PhashResult,
    PHASH_DOWNLOAD_TIMEOUT as _PHASH_DOWNLOAD_TIMEOUT,
    PHASH_MAX_BYTES as _PHASH_MAX_BYTES,
)

# Module-level HTTP client singleton.
# Re-using a single AsyncClient across requests avoids per-request TCP
# handshake and TLS negotiation overhead — critical for a fast-path endpoint.
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Open / close the shared httpx client around the app lifetime."""
    global _http_client
    _http_client = httpx.AsyncClient(
        follow_redirects=True,
        timeout=_PHASH_DOWNLOAD_TIMEOUT,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    yield
    await _http_client.aclose()
    _http_client = None


# --- FastAPI App ---
app = FastAPI(title="AI Image Detection v6.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

api_detector = AIImageDetector(enable_llm_judge=os.getenv("ENABLE_LLM_JUDGE", "true").lower() == "true")

@app.post("/analyze")
async def api_analyze(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Invalid file type")
    
    temp_path = os.path.join("temp_uploads", f"{uuid.uuid4().hex}{os.path.splitext(file.filename)[1]}")
    os.makedirs("temp_uploads", exist_ok=True)
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = api_detector.analyze_image(temp_path)
        return _sanitize(asdict(result))
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.get("/health")
def health(): return {"status": "ok"}


# ---------------------------------------------------------------------------
# Request model for the URL-based analysis endpoint
# ---------------------------------------------------------------------------
_ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg", "image/png",
}
_MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB hard limit
_DOWNLOAD_TIMEOUT_SECONDS = 15


class AnalyzeUrlRequest(BaseModel):
    s3_url: HttpUrl  # Changed from 'url' to 's3_url' for consistency with /extract-phash

    @field_validator("s3_url")
    @classmethod
    def must_be_http_or_https(cls, v: HttpUrl) -> HttpUrl:
        if v.scheme not in ("http", "https"):
            raise ValueError("Only http/https URLs are supported.")
        return v


@app.post("/analyze-url")
async def api_analyze_url(body: AnalyzeUrlRequest):
    """
    Download an image from the provided URL and run the same multi-layer
    forensic analysis as the /analyze endpoint.

    - Validates that the URL is http/https.
    - Streams the response to check Content-Type and enforce a size limit
      before writing to disk (avoids downloading huge/non-image payloads).
    - Cleans up the temporary file regardless of success or failure.
    """
    url_str = str(body.s3_url)  # Changed from body.url to body.s3_url

    # Derive a safe file extension from the URL path (fallback to .jpg)
    url_path = urlparse(url_str).path
    _, ext = os.path.splitext(url_path)
    ext = ext.lower() if ext.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff") else ".jpg"

    temp_path = os.path.join("temp_uploads", f"{uuid.uuid4().hex}{ext}")
    os.makedirs("temp_uploads", exist_ok=True)

    try:
        # Stream the download so we can validate headers before buffering the body
        with requests.get(url_str, stream=True, timeout=_DOWNLOAD_TIMEOUT_SECONDS,
                          allow_redirects=True) as response:

            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: HTTP {response.status_code} from remote server."
                )

            # Validate Content-Type header
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
                raise HTTPException(
                    status_code=415,
                    detail=f"Remote URL does not point to a supported image. "
                           f"Content-Type received: '{content_type}'."
                )

            # Stream to disk while enforcing the size limit
            downloaded = 0
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    downloaded += len(chunk)
                    if downloaded > _MAX_IMAGE_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Remote image exceeds the maximum allowed size of "
                                   f"{_MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB."
                        )
                    f.write(chunk)

        if downloaded == 0:
            raise HTTPException(status_code=400, detail="Downloaded file is empty.")

        result = api_detector.analyze_image(temp_path)
        return _sanitize(asdict(result))

    except HTTPException:
        raise  # Re-raise FastAPI HTTP exceptions as-is
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out while downloading image from the provided URL "
                   f"(limit: {_DOWNLOAD_TIMEOUT_SECONDS}s)."
        )
    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Could not connect to remote server: {e}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)



# ===========================================================================
# /extract-phash  –  High-throughput perceptual hash endpoint
# ===========================================================================


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class ExtractPhashRequest(BaseModel):
    """Payload for the /extract-phash endpoint."""

    s3_url: str  # Pre-signed or public S3 URL (http/https)

    @field_validator("s3_url")
    @classmethod
    def _validate_url_scheme(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("s3_url must use http or https scheme.")
        if not parsed.netloc:
            raise ValueError("s3_url must be a fully qualified URL.")
        return v


class ExtractPhashResponse(BaseModel):
    """Hashes for the original and the horizontally mirrored image."""

    original_hash: str    # Binary string, e.g. "10110010…"
    mirrored_hash: str    # Binary string of the FLIP_LEFT_RIGHT variant
    hash_algorithm: str   # "pdq" | "phash"
    hash_bits: int        # Length of each binary string
    border_stripped: bool # Whether a uniform border was detected and removed


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post("/extract-phash", response_model=ExtractPhashResponse)
async def extract_phash(body: ExtractPhashRequest) -> ExtractPhashResponse:
    """
    Download an image from *s3_url* (entirely in memory – no disk I/O),
    apply forensic pre-processing mitigations, and return two PDQ/pHash
    binary strings:

    * **original_hash** – hash of the border-stripped image.
    * **mirrored_hash** – hash of the horizontally flipped variant.

    The two hashes allow upstream services to detect mirror-attack evasion
    by checking ``hamming(query, mirrored_hash)`` alongside the normal
    ``hamming(query, original_hash)``.

    All hashing logic lives in
    ``Universal_Detector/src/layers/phash_extractor.py``.
    """
    # ------------------------------------------------------------------
    # 1. Async stream download into memory (no disk I/O)
    #    Re-uses the module-level singleton client to avoid per-request
    #    TCP / TLS setup overhead.
    # ------------------------------------------------------------------
    buf = io.BytesIO()
    downloaded = 0

    assert _http_client is not None, "HTTP client not initialised (startup event missed)"

    try:
        async with _http_client.stream("GET", body.s3_url) as response:
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to download image: HTTP {response.status_code}.",
                )

            content_type = (
                response.headers.get("content-type", "")
                .split(";")[0]
                .strip()
                .lower()
            )
            if content_type and not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=415,
                    detail=f"URL does not point to a supported image. "
                           f"Content-Type received: '{content_type}'.",
                )

            async for chunk in response.aiter_bytes(chunk_size=65536):
                downloaded += len(chunk)
                if downloaded > _PHASH_MAX_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image exceeds the "
                               f"{_PHASH_MAX_BYTES // (1024 * 1024)} MB limit.",
                    )
                buf.write(chunk)

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {_PHASH_DOWNLOAD_TIMEOUT}s.",
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Network error: {exc}")

    if downloaded == 0:
        raise HTTPException(status_code=400, detail="Downloaded file is empty.")

    # ------------------------------------------------------------------
    # 2. Delegate all processing to the phash_extractor module.
    #    buf.getbuffer() returns a zero-copy memoryview — no second full
    #    allocation of the image data.
    # ------------------------------------------------------------------
    try:
        result: PhashResult = _run_extract_phash(buf.getbuffer())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return ExtractPhashResponse(
        original_hash=result.original_hash,
        mirrored_hash=result.mirrored_hash,
        hash_algorithm=result.hash_algorithm,
        hash_bits=result.hash_bits,
        border_stripped=result.border_stripped,
    )

