# DeepFake Detection System — Full Architecture Debug Prompt

> **Use this prompt when debugging false positives/negatives.** Paste it into your AI assistant with the JSON output from `/analyze` and it will have full context to diagnose issues.

---

## SYSTEM OVERVIEW

**What it is:** A FastAPI-based image forensic analysis engine that runs 12 detection layers in sequence on an uploaded image, then a master judge (Layer 5) combines all scores into a final verdict: `REAL`, `AI-GENERATED`, or `UNCERTAIN` with a 0-100 score.

**Stack:** Python 3.12 · FastAPI · PyTorch · OpenCV · MediaPipe · scipy · transformers (HuggingFace) · c2pa-python · Pillow

**Endpoint:** `POST /analyze` — upload an image file, returns JSON with `final_score`, `verdict`, `confidence`, `layer_scores`, `layer_details`, and `description`.

**Scoring convention (per layer):** Negative = AI evidence, Positive = Real evidence, Zero = neutral.  
**Final score (Judge):** 0 = definitely AI, 100 = definitely Real.

---

## EXECUTION PIPELINE (in order)

```
Layer 1  → Quick Triage (file validation, format, resolution)
           ↓ detects is_jpeg flag (JPEG/WebP)
Layer 0  → C2PA Cryptographic Credentials (verify c2pa signatures)
Layer 2  → Metadata Analysis (EXIF, AI signatures, timestamps)
Layer 3  → Physics Analysis (ELA ghosting, noise texture, edges)
Layer 3.5 → Face Consistency (face vs background noise, seam detection)
Layer 4  → Neural Network Ensemble (SDXL-Detector + ViT + SigLIP2 + ConvNeXt + Swin with TTA)
Layer 6  → Spectrum Analysis (FFT frequency domain, 7 sub-methods)
Layer 7  → Eye Reflection Physics (corneal glint vector matching)
Layer 8  → Watermark Detection (6 methods: visible, stego, hash, SynthID, FFT)
Layer 8.5 → PRNU Sensor Fingerprint (wavelet denoising, reference DB)
Layer 9  → Context Provenance (PLACEHOLDER — always returns 0)
Layer 10 → Shadow Convergence (light source consistency)
           ↓ all scores fed to:
Layer 5  → Master Judge v2.0 (weighted multi-layer consensus → final verdict)
           ↓ if contradiction or ambiguous:
Final Boss → LLM single call (gray zone) OR Adversarial Debate (contradictions)
```

---

## LAYER-BY-LAYER DETAILS

### Layer 0: C2PA Content Credentials
- **File:** `layer_0_c2pa.py` (148 lines)
- **What it does:** Checks for C2PA cryptographic manifests embedded in the image (used by Adobe, Google, Microsoft to sign content provenance).
- **Uses:** `c2pa-python` library (Rust bindings).
- **Has a `SignatureRegistry`** that caches known AI generators from c2pa.org and IPTC, refreshes every 24h.
- **Returns:** `{"status": "valid"|"tampered"|"none"|"error", "generator": "...", "is_ai_flagged": bool}`
- **Score impact:** If valid+AI → instant AI verdict at Layer 5. If valid+not-AI → instant REAL (score=100). If tampered → score=15 AI.

### Layer 1: Quick Triage
- **File:** `layer_1_triage.py` (~90 lines)
- **What it does:** Validates image format, resolution, animation, color profile, bit depth.
- **Allowed formats:** JPEG, PNG, WebP, TIFF, HEIC, AVIF, BMP
- **Rejects:** Animations, resolution <128px or >10000px
- **Returns:** `{"status": "PASS"|"FAIL", "details": {"format": "JPEG", ...}}`
- **The `format` field is used by main.py to set `is_jpeg = True` for JPEG/WebP images.**

### Layer 2: Metadata Analysis
- **File:** `layer_2_metadata.py` (~120 lines)
- **Starting score:** 50 (neutral-positive bias)
- **Checks:** EXIF data presence, AI technical signatures in metadata (prompt, seed, cfg_scale, etc.), timestamps, GPS data, camera hardware Make/Model, software used, embedded thumbnail, stripped metadata detection.
- **Score range:** 0 to ~100 (but passed to judge as raw score)
- **Known issue:** Score starts at 50, so even empty metadata starts biased toward "real". The `applist` thumbnail check can crash on PNG images.

### Layer 3: Physics Analysis (ELA + Noise)
- **File:** `layer_3_physics.py` (~130 lines)
- **Methods:** Multi-quality ELA (JPEG ghost detection at q=75,90,95), edge coherence (Canny density), noise texture analysis (Laplacian variance, simple Bayer pattern check).
- **Scoring:** Ghost detected → -25. Unnaturally smooth (var<15) → -15. Bayer pattern (var>100) → +15.
- **Returns:** `{"impact": score, "findings": [...], "details": {...}}`
- **Known issue:** `has_bayer = var > 100` is naive — just means sharp image, not actual Bayer detection.

### Layer 3.5: Face Consistency
- **File:** `layer_3_5_face.py` (~130 lines)
- **Face detection:** YuNet DNN (if .onnx available) or Haar cascade fallback.
- **Checks per face:** Noise ratio vs background (Laplacian variance), boundary seam detection (Sobel gradient).
- **Scoring:** Too smooth face (noise_ratio < 0.25) → -15. Too sharp (>4.0) → -20. Seam detected → -10. Max total: -40.
- **Known issue:** Seam threshold `mean(mag) > 50` is absolute, not relative. No color consistency check between face and background.

### Layer 4: Neural Network Ensemble
- **File:** `layer_4_visual.py` (~441 lines)
- **Models:**
  1. **SDXL Detector** — HuggingFace `Organika/sdxl-detector`, Stable Diffusion XL specific
  2. **ViT** — HuggingFace `prithivMLmods/Deep-Fake-Detector-v2-Model`, auto-detects fake label index
  3. **SigLIP2** — HuggingFace `haywoodsloan/ai-image-detector-deploy`, CLIP-based
  4. **ConvNeXt** — HuggingFace `umm-maybe/AI-image-detector`, modern CNN
  5. **Swin** — HuggingFace `microsoft/swin-tiny-patch4-window7-224`, hierarchical
- **Ensemble weights:** SDXL=15%, ViT=25%, SigLIP=25%, ConvNeXt=20%, Swin=15%
- **TTA:** 4 augmentations per model (original, H-flip, slight rotate, color jitter). Uses entropy of softmax outputs as uncertainty measure.
- **Confidence dampening:** If `overall_conf < 0.3`, score is dampened toward 0.
- **Model disagreement detection (G-4.2 fix):** When models differ by >25 points or have opposite signs → confidence penalized proportionally, `is_uncertain` forced True.
- **Returns via `predict_visuals_detailed()`:** `{"impact": score, "confidence": 0-1, "is_uncertain": bool, "dampening": float, "raw_score": float, "sdxl": {...}, "vit": {...}, "siglip": {...}, "convnext": {...}, "swin": {...}}`
- **Score range:** -50 to +50 (dampened)

### Layer 5: Master Judge v2.0
- **File:** `layer_5_judge.py` (~406 lines)
- **Input:** All layer scores + C2PA result + visual confidence + `is_jpeg` flag
- **Layer weights:**
  | Layer | Weight |
  |-------|--------|
  | visual (L4) | 0.22 |
  | spectrum (L6) | 0.15 |
  | prnu (L8.5) | 0.12 |
  | physics (L3) | 0.10 |
  | face (L3.5) | 0.10 |
  | eye (L7) | 0.08 |
  | watermark (L8) | 0.08 |
  | shadow (L10) | 0.07 |
  | metadata (L2) | 0.05 |
  | context (L9) | 0.03 |

- **Dynamic weight adjustments:**
  - If neural uncertain or confidence <0.4 → visual weight halved
  - If `is_jpeg=True` → spectrum ×0.6, prnu ×0.5, watermark ×0.6
  - If context score >60 → context boosted to 0.12

- **Decision tiers (in priority order):**
  1. **TIER 0:** Context ≥90 → REAL (trusted provenance)
  2. **TIER 1:** C2PA valid → instant verdict. C2PA tampered → AI (score=15)
  3. **TIER 1.5:** Watermark ≤ -45 **AND** ≥1 corroborating layer (visual/spectrum/prnu ≤ -20) → AI. Without corroboration → falls through.
  4. **TIER 2:** Eye ≤ -45 or Shadow ≤ -40 → AI (physical impossibility)
  5. **TIER 3:** Multi-layer agreement counting:
     - Strong AI = score ≤ -30, Moderate AI = score ≤ -15, Weak AI = score ≤ -5
     - Strong Real = score ≥ +20, Moderate Real = score ≥ +10
     - 3+ strong AI → definitive AI
     - 2 strong AI + moderate corroboration → AI
     - 1 strong + 2 moderate → AI
     - 1 strong alone + strong real contradicts → UNCERTAIN
     - 3+ moderate AI (no strong) → AI
     - 2 moderate → UNCERTAIN
  6. **TIER 4:** Weighted authenticity score (no significant AI signals):
     - Normalizes each score to 0-1 range, applies weights
     - ≥70 auth + 3 real layers → REAL (up to 95)
     - ≥60 + 2 real → REAL (75)
     - ≥55 → REAL (65)
     - ≥45 → UNCERTAIN (55)
     - <45 → UNCERTAIN (45)

### Layer 6: Spectrum Analysis (FFT)
- **File:** `layer_6_spectrum.py` (~829 lines)
- **7 sub-methods:** Periodic spike detection, directional spectrum, GAN checkerboard, upsampling patterns, patch-based FFT, multiband (RGB+YCbCr), JPEG encoder detection.
- **JPEG-aware:** Has two tiers:
  - **JPEG TIER 1** (lossy images): Only uses periodic spike count. Spike detection now masks out JPEG 8×8 block frequencies before counting (FP-4 fix). Uses `mean + 4*std` threshold. Scores: >1500 spikes → -30, >1200 → -15, >1000 → -5, else positive/neutral.
  - **Non-JPEG TIER 2-4:** Full analysis with directional, GAN, upscaling, patches, multiband.
- **Score range:** -50 to +35

### Layer 7: Eye Reflection Physics
- **File:** `layer_7_eyes.py` (~120 lines)
- **Uses:** MediaPipe FaceMesh with refined landmarks (iris landmarks 468, 473, etc.)
- **Method:** Finds brightest pixel (glint) in each eye crop, computes direction vector from pupil center. If dot product of left/right vectors is negative → mismatched reflections → physics violation.
- **Scoring:** Mismatch → -50. Matched → +30. Dark/matte eyes → -10. No face → 0.
- **Known issues:** Single-pixel glint is fragile, binary scoring (no gradient), single face only.

### Layer 8: Watermark Detection
- **File:** `layer_8_watermark.py` (~959 lines)
- **6 detection methods:**
  1. **Visible watermark scan** — edge density in 8 regions (4 corners + 4 edges + center). Thresholds: >22 density → -20 per region.
  2. **AI logo/text** — MSER text detection in corner quadrants. Threshold: >50 regions → -15 (raised from 8).
  3. **Steganography** — LSB analysis + chi-square test + DCT kurtosis. **Skipped entirely on JPEG** (FP-2 fix: JPEG randomizes LSBs).
  4. **Hash-based** — pHash comparison against known AI watermark patterns.
  5. **SynthID/Stable Signature** — wavelet cross-correlation (threshold: >0.82 → -30, >0.72 → -18), phase coherence between channels (threshold: >0.70 → -28, >0.55 → -15), latent-space periodicity at 64px/128px grids.
  6. **FFT invisible** — multi-scale patch FFT, spike ratio thresholds (>35 → -35, >25 → -18).
- **JPEG dampening:** If `is_jpeg=True`, SynthID scores dampened to 1/3 (max -10) unless cross-corr >0.85. FFT invisible scores dampened to 1/3 (max -8) unless ratio >80.
- **Combination logic:** If 3+ methods flag → worst_score × 1.2 + "Multiple watermarks" label.
- **Score range:** -60 to 0

### Layer 8.5: PRNU Sensor Fingerprint
- **File:** `layer_8_5_The_Sensor_Fingerprint.py` (~1020 lines)
- **Methods:** Wavelet-based denoising (multi-scale, replaces fastNlMeans), peak counting in noise residual FFT, noise variance, adaptive entropy (Freedman-Diaconis binning), spectral flatness, camera-specific PRNU validation via EXIF, reference pattern database (NCC comparison), flat-field estimation.
- **JPEG scoring (FP-5 fix — conservative):**
  - peak>2000 & var<1.5 & entropy<4.5 → -20 (very suspicious even for JPEG)
  - peak>1500 & var<2.0 & entropy<5.0 → -10 (borderline)
  - var>6 & entropy>6.0 → +15 (natural randomness)
  - else → 0 (neutral)
- **Non-JPEG scoring:** More aggressive, can reach -50 for strong synthetic patterns.
- **Score range:** -50 to +35

### Layer 9: Context Provenance (PLACEHOLDER)
- **File:** `layer_9_context.py` (~40 lines)
- **Always returns:** `(0, {"note": "Context lookup disabled"})` — no API configured.
- **Would need:** Reverse image search API (Google, TinEye), social media verification.

### Layer 10: Shadow Convergence
- **File:** `layer_10_Shadow_Convergence.py` (273 lines)
- **Method:** Detects shadow regions via HSV thresholding (dark + low saturation), estimates shadow direction per grid cell using gradient analysis, computes circular variance of directions.
- **Scoring:** Low variance (consistent shadows from one source) → positive. High variance (conflicting directions) → negative.
- **Score range:** ~-40 to +20

---

## `is_jpeg` FLAG PIPELINE

Detected in main.py after Layer 1 triage. Propagated to:
1. **Layer 8 (`detect_watermarks(is_jpeg=True)`)** → Skips LSB/stego, dampens SynthID and FFT invisible scores.
2. **Layer 8.5 (`analyze_prnu(is_jpeg_hint=True)`)** → Uses conservative JPEG scoring path.
3. **Layer 5 (`calculate_integrity(is_jpeg=True)`)** → Reduces weights for spectrum (×0.6), PRNU (×0.5), watermark (×0.6).

---

## FALSE POSITIVE FIXES ALREADY APPLIED

| ID | Fix | What Changed |
|----|-----|-------------|
| FP-1 | SynthID thresholds too low | Raised wavelet cross-corr from 0.65→0.82, phase coherence from 0.35→0.70 |
| FP-2 | LSB/chi² fires on JPEG | Stego detection skipped entirely for JPEG images |
| FP-3 | MSER text over-counting | Corner text threshold raised from 8→50 regions |
| FP-4 | FFT spike count on JPEG | Masks out 8×8 block frequencies, raised threshold from 3σ→4σ |
| FP-5 | PRNU too harsh on JPEG | Max JPEG penalty reduced from -35→-20, requires stronger evidence |
| FP-6 | Watermark instant verdict | Now requires corroboration from visual/spectrum/prnu (not just watermark alone) |
| G-4.2 | Model disagreement | When SDXL and ViT disagree by >25pts → confidence penalized, uncertain=True |
| A-1 | No JPEG awareness | Added `is_jpeg` flag passed from triage through watermark, PRNU, and judge |

---

## HOW TO DEBUG A FALSE POSITIVE

When the API returns a wrong verdict, check these fields in the JSON response:

1. **`layer_scores`** — Which layers contributed negative (AI) scores?
2. **`layer_details`** — What specific findings triggered each score?
3. **`description`** — Which judge tier made the decision?
4. **`verdict`** + **`final_score`** — How confident was the verdict?

**Common FP patterns to look for:**
- **Spectrum score very negative on JPEG** → Check if spike count is still too high after FP-4 fix
- **Watermark score very negative** → Check which sub-method triggered (SynthID? FFT? stego?)
- **PRNU score negative on JPEG** → Check if JPEG path is being used correctly
- **Neural models disagree** → Check `layer_details.neural_network` for confidence and uncertain flag
- **Metadata score low** → Layer 2 starts at 50 and subtracts; check if stripped metadata is penalizing real photos
- **Multiple moderate-AI layers** → Judge TIER 3 triggers with 3+ moderate signals — check if each is a genuine signal or noise

---

## KNOWN REMAINING GAPS

1. **Layer 9 is a placeholder** — always returns 0, wastes a weight slot
2. **Layer 2 starts at score=50** — biased starting point, can cause false metadata "real" signals
3. **Layer 3 Bayer detection** — `var > 100` is not real Bayer pattern detection
4. **Layer 3.5 seam threshold** — absolute threshold (50) doesn't adapt to image characteristics
5. **Layer 7 eyes** — single-pixel glint, binary scoring, single face only
6. **No video support** — system is image-only by design
7. **No batch/benchmark pipeline** — can't test against a labeled dataset automatically

---

## API USAGE

```bash
# Start server
python -m uvicorn main:app --reload

# Test with an image
curl -X POST http://localhost:8000/analyze -F "file=@photo.jpg"

# Health check
curl http://localhost:8000/health
```

**Response format:**
```json
{
  "final_score": 75,
  "verdict": "REAL",
  "confidence": "MEDIUM",
  "description": "Likely authentic - multiple positive signals",
  "layer_scores": { ... },
  "layer_details": { ... },
  "processing_time_ms": 4200,
  "warnings": [],
  "timestamp": "2026-02-09T...",
  "llm_reasoning": null,
  "judge_source": "rule-based",
  "debate_data": null
}
```

When adversarial debate is triggered, `debate_data` contains:
```json
{
  "debate_data": {
    "rounds_taken": 3,
    "debate_summary": {
      "prosecution_final_confidence": 0.85,
      "defense_final_confidence": 0.3,
      "winning_side": "prosecution",
      "key_turning_point": "Frequency artifacts inconsistent with JPEG compression",
      "processing_time_ms": 12000
    },
    "debate_history": [
      {
        "round": 1,
        "prosecution": { "confidence": 0.8, "evidence": [...], "summary": "..." },
        "defense": { "confidence": 0.7, "evidence": [...], "summary": "..." }
      }
    ]
  }
}
```

---

## FILE STRUCTURE

```
D:\DeepFake_Detection\
├── main.py                              # FastAPI app, AIImageDetector class, /analyze endpoint
├── requirements.txt
├── DEBUG_PROMPT.md                      # THIS FILE
└── Universal_Detector/
    └── src/
        └── layers/
            ├── layer_0_c2pa.py          # C2PA signatures (148 lines)
            ├── layer_1_triage.py        # File validation (~90 lines)
            ├── layer_2_metadata.py      # EXIF analysis (~120 lines)
            ├── layer_3_physics.py       # ELA + noise (~130 lines)
            ├── layer_3_5_face.py        # Face forensics (~130 lines)
            ├── layer_4_visual.py        # Neural ensemble (~441 lines)
            ├── layer_5_judge.py         # Master judge (~406 lines)
            ├── layer_6_spectrum.py      # FFT spectrum (~829 lines)
            ├── layer_7_eyes.py          # Eye reflections (~120 lines)
            ├── layer_8_watermark.py     # Watermark detection (~959 lines)
            ├── layer_8_5_The_Sensor_Fingerprint.py  # PRNU (~1020 lines)
            ├── layer_9_context.py       # Placeholder (~40 lines)
            ├── layer_10_Shadow_Convergence.py  # Shadows (273 lines)
            ├── debate/                       # Adversarial debate package
            │   ├── __init__.py              # Re-exports DebateOrchestrator
            │   ├── models.py                # Shared data classes & prompts
            │   ├── prosecution.py           # Prosecution agent (Gemini Vision)
            │   ├── defense.py               # Defense agent (OpenRouter Vision)
            │   ├── convergence.py           # Convergence judge (Groq text)
            │   └── orchestrator.py          # Debate flow controller
            ├── llm_judge.py                 # HybridJudge + ForensicAgent
            ├── forensic_case_builder.py      # Case file compiler
            └── (models auto-downloaded from HuggingFace)
```
