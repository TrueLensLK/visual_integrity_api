"""
Layer 5: Master Judge v3.3 - THE CONSENSUS + SAFETY EDITION

KEY PHILOSOPHY:
  1. Model Consensus has veto power over lone forensic signals
  2. PRNU synthetic grid needs corroboration OR high confidence
  3. Texture/compression issues get EDITED, not AI-GENERATED
  4. Hardware Veto protects real photos from neural network errors

VERSION HISTORY:
  v2.3 - Redemption Logic for authentic textures
  v3.0 - Hardware Veto + Lonewolf Rule + Edited Real Safety Net
  v3.1 - Model Consensus for compressed images
  v3.2 - Kill switches for definitive AI evidence
  v3.3 - FIXED: Consensus pre-check, PRNU corroboration, compressed detection
"""

from typing import Tuple, Dict

# ============================================================================
# USER-FRIENDLY DESCRIPTION GENERATOR
# ============================================================================
def _humanize_description(verdict: str, technical_desc: str, final_score: int) -> str:
    """
    Converts technical forensic descriptions into user-friendly language.

    Args:
        verdict: The verdict (REAL, AI-GENERATED, EDITED, etc.)
        technical_desc: The technical description from the analysis
        final_score: The authenticity score (0-100)

    Returns:
        A simple, non-technical description anyone can understand
    """

    # Base templates for each verdict type
    verdict_templates = {
        "REAL": {
            "high": "This image appears to be a genuine photograph taken with a camera.",
            "medium": "This image looks like a real photograph, though some quality is lost.",
            "low": "This image is likely a real photograph, but we couldn't verify all details."
        },
        "AI-GENERATED": {
            "high": "This image was likely created by artificial intelligence.",
            "medium": "This image shows signs of being AI-generated.",
            "low": "This image may have been created by AI, but we're not entirely certain."
        },
        "EDITED": {
            "default": "This image appears to be a real photograph that has been edited or modified."
        },
        "EDITED_REAL": {
            "default": "This is a real photograph that has been digitally enhanced or edited."
        },
        "AI-ENHANCED": {
            "default": "This appears to be a real photograph that has been enhanced using AI tools."
        },
        "LIKELY_REAL": {
            "default": "This image appears to be an authentic photograph."
        },
        "AMBIGUOUS_REQUIRES_AGENT": {
            "default": "This image requires additional analysis to determine its authenticity."
        },
        "ERROR": {
            "default": "We couldn't complete the analysis due to a technical issue."
        }
    }

    # Determine confidence level from score
    if final_score >= 75 or final_score <= 25:
        confidence = "high"
    elif final_score >= 60 or final_score <= 40:
        confidence = "medium"
    else:
        confidence = "low"

    # Get base description
    templates = verdict_templates.get(verdict, {"default": "Unable to determine the authenticity of this image."})

    if "default" in templates:
        base_desc = templates["default"]
    else:
        base_desc = templates.get(confidence, templates.get("medium", ""))

    # Add context based on technical description keywords
    context_additions = []

    tech_lower = technical_desc.lower()

    # C2PA (digital signature)
    if "c2pa" in tech_lower:
        if "verified" in tech_lower or "confirms" in tech_lower:
            if "ai" in tech_lower:
                return "This image has a verified digital signature confirming it was created with AI tools."
            else:
                return "This image has a verified digital signature from the original camera."
        elif "tampered" in tech_lower:
            return "This image's digital signature shows it has been tampered with."

    # Deepfake detection
    if "deepfake" in tech_lower:
        return "This image appears to contain a face that was digitally altered or generated."

    # Sensor/Camera detection
    if "sensor" in tech_lower and ("dna" in tech_lower or "verified" in tech_lower):
        context_additions.append("Our analysis detected authentic camera sensor patterns.")

    # AI watermark
    if "watermark" in tech_lower and "ai" in tech_lower:
        context_additions.append("We detected a hidden AI watermark in the image.")

    # Synthetic/Generated patterns
    if "synthetic" in tech_lower or "grid" in tech_lower:
        if verdict in ["AI-GENERATED", "EDITED"]:
            context_additions.append("We found patterns typically created by AI image generators.")

    # Model consensus
    if "consensus" in tech_lower or "models" in tech_lower:
        if "real" in tech_lower:
            context_additions.append("Multiple AI detection systems agree this looks authentic.")
        elif "ai" in tech_lower:
            context_additions.append("Multiple AI detection systems flagged this as potentially generated.")

    # Physical impossibilities
    if "impossible" in tech_lower:
        if "reflection" in tech_lower or "corneal" in tech_lower:
            context_additions.append("The reflections in the eyes don't match what we'd expect in a real photo.")
        elif "shadow" in tech_lower or "geometry" in tech_lower:
            context_additions.append("The shadows or shapes in this image don't follow natural physics.")
        elif "light" in tech_lower:
            context_additions.append("The lighting in this image doesn't look physically realistic.")

    # Compression/quality issues
    if "compress" in tech_lower or "jpeg" in tech_lower or "quality" in tech_lower:
        context_additions.append("Note: Image quality loss may have affected our analysis.")

    # Texture authenticity
    if "texture" in tech_lower and "authentic" in tech_lower:
        context_additions.append("The image textures match those of real photographs.")

    # Conflicting signals
    if "conflict" in tech_lower:
        base_desc = "This image shows mixed signals - some parts look real while others show signs of editing."

    # Enhancement/Upscaling
    if "upscal" in tech_lower or "enhanc" in tech_lower:
        context_additions.append("This image appears to have been enhanced or upscaled using AI.")

    # Insufficient evidence
    if "insufficient" in tech_lower:
        return "We couldn't find enough evidence to confidently determine if this image is real or AI-generated."

    # Build final description
    if context_additions:
        # Limit to 1-2 context points to keep it concise
        context = " " + context_additions[0]
        if len(context_additions) > 1:
            context += " " + context_additions[1]
        return base_desc + context

    return base_desc


def humanize_verdict_description(verdict: str, technical_desc: str, final_score: int) -> str:
    """
    Public function to convert technical forensic descriptions to user-friendly text.
    Ensures the output is concise (under 200 characters when possible).
    """
    friendly = _humanize_description(verdict, technical_desc, final_score)

    # Ensure it's not too long
    if len(friendly) > 250:
        # Truncate at last complete sentence within limit
        sentences = friendly.split('. ')
        result = ""
        for sentence in sentences:
            if len(result) + len(sentence) + 2 <= 250:
                result += sentence + ". "
            else:
                break
        friendly = result.strip()
        if not friendly.endswith('.'):
            friendly += "."

    return friendly


# ============================================================================
# LAYER WEIGHTS
# ============================================================================
LAYER_WEIGHTS = {
    "visual":    0.16,
    "spectrum":  0.14,
    "prnu":      0.13,
    "physics":   0.09,
    "face":      0.08,
    "physical_continuity": 0.08,
    "eye":       0.07,
    "watermark": 0.07,
    "shadow":    0.05,
    "artifacts": 0.06,
    "metadata":  0.04,
    "context":   0.03,
}

# Weights for compressed images (hardware evidence destroyed)
# v3.4: Bumped visual (0.15→0.20) and artifacts (0.03→0.08) since these
#        are the most reliable signals when hardware evidence is destroyed.
#        Trimmed face, eye, spectrum, geometry to compensate. Sum = 1.00.
COMPRESSED_LAYER_WEIGHTS = {
    "visual":    0.20,  # BOOSTED: Content-level analysis is strongest for web images
    "spectrum":  0.20,  # BOOSTED: Diffusion artifacts survive compression
    "prnu":      0.02,  # Almost ignore — destroyed by compression
    "physics":   0.15,  # BOOSTED: 3D physics cannot be faked by 2D generators
    "shadow":    0.12,  # BOOSTED: Inconsistent light vectors
    "face":      0.05,  # Reduced — face detection less reliable on compressed
    "physical_continuity": 0.08,
    "eye":       0.04,  # Reduced — reflections degraded by compression
    "watermark": 0.04,
    "artifacts": 0.08,  # BOOSTED: Spatial autocorrelation survives JPEG
    "metadata":  0.00,  # Dead on web
    "context":   0.02,
}


def calculate_integrity(
    c2pa_res: Dict,
    meta_score: float,
    physics_score: float,
    face_score: float,
    visual_score: float,
    spectrum_score: float,
    eye_score: float,
    watermark_score: float,
    watermark_desc: str,
    prnu_score: float,
    context_score: float,
    context_details: Dict,
    shadow_score: float = 0,
    shadow_desc: str = "Not analyzed",
    artifact_score: float = 0,
    physical_continuity_score: float = 0,
    visual_confidence: float = 1.0,
    visual_uncertain: bool = False,
    is_jpeg: bool = False,
    is_degraded_signal: bool = False,  # <-- ADD THIS NEW ARGUMENT FROM LAYER 1
    # Model consensus parameters
    model_real_votes: int = 0,
    model_ai_votes: int = 0,
    model_count: int = 0,
    model_consensus: float = 0.0,
    # NEW v3.3: PRNU confidence metrics
    prnu_flat_region_ratio: float = 1.0,  # 0-1, how much of image was flat regions
    prnu_details: Dict = None,
    # NEW v3.4: Bayer pattern flag (from physics layer details)
    has_bayer_pattern: bool = False,
) -> Tuple[int, str, str]:
    """
    Master Judge v3.3 - Makes final verdict with safety checks
    
    Returns:
        (final_score, verdict, description)
    """

    # 1. RAW SCORE COLLECTION
    raw_scores = {
        "visual": visual_score,
        "spectrum": spectrum_score,
        "prnu": prnu_score,
        "physics": physics_score,
        "face": face_score,
        "eye": eye_score,
        "watermark": watermark_score,
        "shadow": shadow_score,
        "artifacts": artifact_score,
        "metadata": meta_score,
        "context": context_score,
        "physical_continuity": physical_continuity_score,
    }

    # ========================================================================
    # UPSCALER / ENHANCEMENT EXCEPTION (PRIORITY CHECK)
    # ========================================================================
    # PRNU sees a synthetic grid, but physical/optical layers confirm real origin.
    # This pattern matches AI-upscaled or heavily enhanced real photographs.
    #
    # GUARD: Only fire if watermark is NOT also deeply negative.
    # When BOTH PRNU=-50 AND watermark=-50, the cause is more likely
    # editing/recompression artifacts creating false positives in both
    # pipelines — NOT genuine AI enhancement.  Let the full consensus
    # logic below handle that case with proper corroboration checks.
    #
    # FIX-9: Skip for JPEG/WebP images entirely. JPEG compression creates
    # PRNU synthetic grids (PCE > 10,000) on real photos, which is
    # indistinguishable from AI-enhancement at this layer. WhatsApp,
    # social media, and stock photo compression all trigger this.
    if (raw_scores.get('prnu', 0) <= -40
            and raw_scores.get('physics', 0) > 0
            and raw_scores.get('watermark', 0) > -40
            and not is_jpeg):  # JPEG compression creates false PRNU grids
        if raw_scores.get('eye', 0) > 0 or raw_scores.get('spectrum', 0) > 0:
            return (
                45,
                "AI-ENHANCED",
                "Authentic base photograph detected, but exhibits severe AI-upscaling or heavy digital enhancement (Synthetic 128px grid overriding natural sensor noise)."
            )

    # ========================================================================
    # FIX-10: JPEG + BAYER CAMERA GUARD (Before raw scores print)
    # ========================================================================
    # When a JPEG/WebP image shows real Bayer demosaicing, it is a real
    # camera photo with compression artifacts. JPEG compression creates
    # systematic false positives in spectrum and watermark layers.
    #
    # GUARD CONDITIONS (ALL must be true):
    #   - is_jpeg: Image is JPEG/WebP (compression artifacts present)
    #   - Bayer detected: physics > 0 OR has_bayer_pattern flag from layer 3
    #   - face_score > -20: No face anomaly (prevents guard on deepfakes)
    _has_bayer = physics_score > 0 or has_bayer_pattern
    if is_jpeg and _has_bayer and face_score > -20:
        print(f"\n[Judge] JPEG + BAYER CAMERA GUARD ACTIVE")
        print(f"  Physics: {physics_score:+.0f} (real Bayer CFA pattern detected)")
        print(f"  → Real camera photo with compression artifacts")
        
        if raw_scores["spectrum"] <= -20:
            old = raw_scores["spectrum"]
            raw_scores["spectrum"] = -15
            spectrum_score = -15
            print(f"  Spectrum: {old:+.0f} → -15 (JPEG multi-band bias)")
        
        if raw_scores["watermark"] <= -25:
            old = raw_scores["watermark"]
            raw_scores["watermark"] = -15
            watermark_score = -15
            print(f"  Watermark: {old:+.0f} → -15 (compression artifact)")

    print("\n[Judge] Raw Scores:")
    for layer, score in sorted(raw_scores.items(), key=lambda x: x[1]):
        indicator = "[!]" if score < -20 else "[+]" if score > 15 else "[ ]"
        print(f"  {layer:20s}: {score:+6.1f}  {indicator}")

    # ========================================================================
    # FIX-3: CORRECTED COMPRESSED IMAGE DETECTION
    # ========================================================================
    # Use the hard truth from Layer 1 Triage first, fallback to heuristics if needed
    is_compressed_image = is_degraded_signal or (
        (-15 < prnu_score < 10) and  # PRNU weak/neutral (not strongly AI)
        meta_score < 15 and           # Metadata stripped or minimal
        (spectrum_score < 15)         # Spectrum also affected by compression
    ) or (
        is_jpeg and _has_bayer and face_score > -20  # JPEG with real Bayer and no deepfake face
    )
    if is_compressed_image:
        print(f"\n[Judge] COMPRESSED IMAGE DETECTED:")
        print(f"  PRNU: {prnu_score:+.1f}")
        print(f"  Meta: {meta_score:+.1f}")
        print(f"  Spectrum: {spectrum_score:+.1f}")
        print(f"  → Hardware evidence may be degraded. Will rely on model consensus.")

    # ========================================================================
    # FIX-1: MODEL CONSENSUS PRE-CHECK (Before Kill Switches!)
    # ========================================================================
    # If 3+ models agree, this should influence kill switch decisions
    # Prevents lone forensic signals from overriding neural network consensus
    
    model_consensus_override = None
    
    if model_count >= 4:
        model_agreement_ratio = max(model_real_votes, model_ai_votes) / model_count
        majority_vote = "REAL" if model_real_votes > model_ai_votes else "AI"
        
        print(f"\n[Judge] MODEL CONSENSUS PRE-CHECK:")
        print(f"  Votes: {model_real_votes} REAL vs {model_ai_votes} AI (out of {model_count})")
        print(f"  Agreement: {model_agreement_ratio:.1%}")
        print(f"  Majority: {majority_vote}")
        
        # Strong consensus (4/5 or 5/5 agree)
        if model_agreement_ratio >= 0.80:  # 4/5 or 5/5
            print(f"  → STRONG consensus ({model_agreement_ratio:.0%})")
            
            if majority_vote == "REAL":
                # Count how many forensic layers strongly disagree
                strong_ai_signals = [k for k, s in raw_scores.items() 
                                    if s <= -30 and k != "context"]
                
                print(f"  Strong AI signals: {len(strong_ai_signals)} → {strong_ai_signals}")
                
                if len(strong_ai_signals) <= 1:
                    # Models say REAL, only 0-1 forensic layers disagree
                    print(f"  [+] Models outvote lone forensic signal(s)")
                    model_consensus_override = "REAL"
                elif len(strong_ai_signals) == 2:
                    # 2 forensic layers vs 4-5 models - this is a conflict
                    print(f"  [!] Conflict: 2 forensic layers vs {model_real_votes} models")
                    model_consensus_override = "UNCERTAIN"
                else:
                    # 3+ forensic layers agree on AI - trust forensics
                    print(f"  ! Multiple forensic layers confirm AI despite model consensus")
            
            elif majority_vote == "AI":
                # 4-5 models say AI
                strong_real_signals = [k for k, s in raw_scores.items() 
                                      if s >= 20 and k != "context"]
                
                if len(strong_real_signals) <= 1:
                    # Models say AI, forensics don't strongly contradict
                    print(f"  [+] Models agree on AI")
                    model_consensus_override = "AI"
        
        # Moderate consensus (3/5 agree)
        elif model_agreement_ratio >= 0.60:
            print(f"  → Moderate consensus ({model_agreement_ratio:.0%})")
            
            if majority_vote == "REAL":
                strong_ai_signals = [k for k, s in raw_scores.items() if s <= -30]
                
                if len(strong_ai_signals) == 1 and "prnu" in strong_ai_signals:
                    # PRNU is lone wolf vs 3 models saying REAL
                    print(f"  [!] PRNU is LONE AI signal vs {model_real_votes} models REAL")
                    print(f"  → Possible PRNU false positive (textured image?)")
                    model_consensus_override = "UNCERTAIN_PRNU_FP"

    # ========================================================================
    # TIER 0: C2PA CRYPTOGRAPHIC PROOF (Absolute Truth)
    # ========================================================================
    print(f"\n[Judge] Checking C2PA...")
    
    if c2pa_res.get("status") == "valid":
        if c2pa_res.get("is_ai_flagged"):
            print("  → C2PA confirms AI")
            return (0, "AI-GENERATED", f"C2PA confirms AI: {c2pa_res.get('ai_tool')}")
        
        print("  → C2PA confirms authentic")
        return (98, "REAL", "Cryptographically verified camera original")
    
    if c2pa_res.get("status") == "tampered":
        print("  → C2PA tampered")
        return (15, "AI-GENERATED", "C2PA signature tampering detected")

    # ========================================================================
    # TIER 1: KILL SWITCHES (With Safety Checks) - v3.3 FIXED
    # ========================================================================
    print(f"\n[Judge] Checking Kill Switches...")
    
    # ------------------------------------------------------------------------
    # FIX-2: KILL SWITCH 1 - PRNU Synthetic Grid (REQUIRES CORROBORATION)
    # ------------------------------------------------------------------------
    # FIX-7: If image is compressed/web-sourced, PRNU is unreliable.
    # Stock photo pipelines (resize + recompress + watermark) destroy sensor
    # fingerprints and create artificial grid patterns that mimic AI.
    # Skip kill switch entirely — weighted scoring uses 0.02 weight anyway.
    if prnu_score <= -45 and is_compressed_image:
        print(f"\n  [Kill Switch 1] PRNU synthetic grid SKIPPED (compressed image)")
        print(f"    PRNU score={prnu_score} but image is web/compressed")
        print(f"    → Stock/web processing destroys PRNU — false positive likely")
        print(f"    → Downgrading to -15 and flowing through weighted scoring")
        raw_scores["prnu"] = -15
        prnu_score = -15

    elif prnu_score <= -45:
        print(f"\n  [Kill Switch 1] PRNU reports synthetic grid (score={prnu_score})")

        # FIX-8: BAYER CONTRADICTION CHECK
        # If Layer 3 (physics) detected a real Bayer demosaicing pattern
        # (score > 0 OR has_bayer_pattern flag), that is strong evidence
        # of a REAL camera sensor. A real Bayer pattern and a synthetic
        # grid are physically contradictory — edited/recompressed real
        # images create periodic artifacts that mimic synthetic grids
        # while preserving genuine Bayer traces.
        if _has_bayer:
            print(f"    [!] BAYER CONTRADICTION: Physics={physics_score:+.0f} (Bayer={'flag' if has_bayer_pattern else 'score'}) vs PRNU={prnu_score:+.0f} (synthetic grid)")
            print(f"    → Real Bayer demosaicing is incompatible with synthetic grid")
            print(f"    → Likely editing/recompression artifact. Downgrading PRNU to -15")
            raw_scores["prnu"] = -15
            prnu_score = -15

        # FIX-5: Check PRNU confidence (flat region ratio)
        elif prnu_flat_region_ratio < 0.20:
            print(f"    [!] SAFETY CHECK: Flat regions only {prnu_flat_region_ratio:.1%}")
            print(f"    → Insufficient flat regions for reliable PRNU")
            print(f"    → Image too textured (fur, fabric, foliage?)")
            print(f"    → DOWNGRADING: -50 → -20 (suspicious but unreliable)")
            
            # Override PRNU score
            raw_scores["prnu"] = -20
            prnu_score = -20
            
            # Return EDITED if this was the main evidence
            if model_consensus_override == "UNCERTAIN_PRNU_FP":
                return (
                    50,
                    "EDITED",
                    f"PRNU false positive likely (only {prnu_flat_region_ratio:.1%} flat regions, texture interference)"
                )
        
        else:
            # Sufficient flat regions - PRNU is reliable
            # But still check for corroboration
            
            other_ai_signals = [k for k, s in raw_scores.items() 
                               if k != "prnu" and s <= -25]
            
            print(f"    Corroborating AI signals: {len(other_ai_signals)} → {other_ai_signals}")
            
            # Check model consensus override
            if model_consensus_override == "REAL":
                print(f"    [!] MODEL CONSENSUS OVERRIDE: {model_real_votes}/{model_count} models say REAL")
                print(f"    → PRNU synthetic grid vs model consensus = CONFLICT")
                return (
                    45,
                    "EDITED",
                    f"PRNU synthetic grid conflicts with model consensus ({model_real_votes}/{model_count} REAL)"
                )
            
            elif model_consensus_override == "UNCERTAIN_PRNU_FP":
                print(f"    [!] PRNU is LONE signal vs {model_real_votes} model votes REAL")
                return (
                    50,
                    "EDITED",
                    "PRNU lone wolf vs neural network consensus - possible false positive"
                )
            
            # No model consensus override - check traditional corroboration
            elif len(other_ai_signals) >= 1:
                # PRNU + at least 1 other layer = definitive
                print(f"    [+] CORROBORATED by {other_ai_signals[0]}")
                print(f"    → KILL SWITCH ACTIVATED")
                return (
                    5,
                    "AI-GENERATED",
                    f"Synthetic grid (PRNU) corroborated by {', '.join(other_ai_signals)}"
                )
            
            elif prnu_score == -50:
                # Maximum PRNU confidence without corroboration
                # Be cautious but still flag
                print(f"    ! Maximum PRNU confidence but NO corroboration")
                print(f"    → Continuing to weighted analysis with heavy penalty")
                # Don't kill switch - let it flow through weighted scoring
    
    # ------------------------------------------------------------------------
    # KILL SWITCH 2: Artifact Grid Pattern
    # ------------------------------------------------------------------------
    if artifact_score <= -40:
        print(f"\n  [Kill Switch 2] Artifact grid detected (score={artifact_score})")
        
        # Check model consensus
        if model_consensus_override == "REAL":
            print(f"    [!] MODEL CONSENSUS says REAL - artifact may be compression")
            # Downgrade instead of kill switch
            raw_scores["artifacts"] = -20
        else:
            print(f"    → KILL SWITCH ACTIVATED")
            return (8, "AI-GENERATED", "GAN/Diffusion synthesis grid detected")
    
    # ------------------------------------------------------------------------
    # KILL SWITCH 3: Physical Continuity Violations
    # ------------------------------------------------------------------------
    if physical_continuity_score <= -40:
        print(f"\n  [Kill Switch 3] Impossible geometry (score={physical_continuity_score})")
        
        # This is very reliable - don't override
        # But check if it's outdoor natural lighting (false positive)
        if "outdoor" in str(raw_scores) or "natural lighting" in str(raw_scores):
            print(f"    [!] May be outdoor scene with natural ambient light")
            raw_scores["physical_continuity"] = -15
        else:
            print(f"    → KILL SWITCH ACTIVATED")
            return (10, "AI-GENERATED", "Physically impossible geometry")
    
    # ------------------------------------------------------------------------
    # KILL SWITCH 4: Strong Watermark
    # ------------------------------------------------------------------------
    if watermark_score <= -45:
        print(f"\n  [Kill Switch 4] Strong watermark (score={watermark_score})")
        
        # Check if it's a photographer watermark vs AI watermark
        if "photographer" in watermark_desc.lower() or "photo" in watermark_desc.lower():
            print(f"    [i] Photographer watermark detected (not AI)")
            raw_scores["watermark"] = 0
        else:
            # Need corroboration
            other_ai = [k for k, s in raw_scores.items() 
                       if k != "watermark" and s <= -20]
            
            if len(other_ai) >= 1:
                print(f"    [+] Corroborated by {other_ai}")
                print(f"    → KILL SWITCH ACTIVATED")
                return (5, "AI-GENERATED", f"AI watermark ({watermark_desc})")
            else:
                print(f"    ! No corroboration - continuing")
    
    # ------------------------------------------------------------------------
    # KILL SWITCH 5: Deepfake Signature (Real sensor + AI face)
    # ------------------------------------------------------------------------
    if prnu_score >= 20 and face_score <= -25:
        print(f"\n  [Kill Switch 5] Deepfake signature")
        print(f"    PRNU={prnu_score} (real sensor) + Face={face_score} (anomalous)")
        print(f"    → KILL SWITCH ACTIVATED")
        return (15, "AI-GENERATED", "Deepfake: Real camera base with AI face")
    

    # ------------------------------------------------------------------------
    # NEW: KILL SWITCH 6: SOTA Physics Override & Agentic Handoff
    # ------------------------------------------------------------------------
    if model_consensus_override == "REAL":
        # The visual models think it's perfectly real. 
        # But if the invisible physical/frequency layers scream fake, it's a SOTA deepfake.
        sota_giveaways = []
        if spectrum_score <= -30: sota_giveaways.append("Diffusion Spectrum Roll-off")
        if shadow_score <= -30: sota_giveaways.append("Impossible 3D Light Vectors")
        
        if len(sota_giveaways) > 0:
            print(f"\n  [Kill Switch 6] SOTA Generator Suspected!")
            print(f"    Visual models fooled (REAL), but physics say FAKE: {sota_giveaways}")
            print(f"    → TRIGGERING AGENTIC LLM ESCALATION")
            
            # Instead of guessing, we hand this edge-case directly to the VLM Tool Agent
            return (
                30, 
                "AMBIGUOUS_REQUIRES_AGENT", 
                f"SOTA Conflict: Visual consensus is REAL, but Physics failed ({', '.join(sota_giveaways)}). Escalate to LLM Agent with Tool Nodes."
            )

    print(f"\n  [+] No kill switches activated")

    # ========================================================================
    # SCORE ADJUSTMENTS & DAMPENING
    # ========================================================================
    effective_scores = raw_scores.copy()
    
    # Hardware Veto (Physical DNA proves real)
    is_physically_real = prnu_score >= 20 and spectrum_score >= 18
    
    if is_physically_real:
        print(f"\n[Judge] HARDWARE VETO ACTIVE")
        print(f"  PRNU: {prnu_score:+.0f}")
        print(f"  Spectrum: {spectrum_score:+.0f}")
        print(f"  → Physical sensor DNA verified")
        
        # Cap AI signals
        if effective_scores["visual"] < -5:
            effective_scores["visual"] = -5
        if effective_scores["artifacts"] < -10:
            effective_scores["artifacts"] = -10
    
    # Lonewolf Rule
    ai_indicators = [k for k, s in effective_scores.items() if s <= -25 and k != "context"]
    real_indicators = [k for k, s in effective_scores.items() if s >= 15 and k != "context"]
    
    if len(ai_indicators) == 1 and "visual" in ai_indicators:
        rest_score = sum(s for k, s in effective_scores.items() 
                        if k not in ["visual", "context"])
        
        if rest_score > -10:
            print(f"\n[Judge] LONEWOLF REJECTION: Visual NN alone, rest={rest_score:+.0f}")
            effective_scores["visual"] *= 0.3
    
    # Texture Authenticity (mild redemption)
    # For compressed/web images where PRNU is unreliable (downgraded to -15),
    # spectrum alone can verify authentic camera texture.
    if is_compressed_image and prnu_score == -15 and spectrum_score >= 10:
        is_texture_authentic = True
    else:
        is_texture_authentic = spectrum_score >= 10 and prnu_score >= 5
    
    if is_texture_authentic and not is_physically_real:
        print(f"\n[Judge] [+] Texture looks authentic (mild redemption)")
        if effective_scores["metadata"] < -15:
            effective_scores["metadata"] = -15
        if effective_scores["shadow"] < -15:
            effective_scores["shadow"] = -15
    
    # JPEG Dampening
    if is_jpeg:
        print(f"\n[Judge] JPEG dampening applied")
        for key in ["spectrum", "prnu", "watermark", "physics", "face"]:
            if -30 < effective_scores[key] < 0:
                old = effective_scores[key]
                effective_scores[key] *= 0.6
                if old != effective_scores[key]:
                    print(f"  {key}: {old:+.1f} → {effective_scores[key]:+.1f}")
    
    # ========================================================================
    # WEIGHT SELECTION
    # ========================================================================
    if is_compressed_image:
        weights = dict(COMPRESSED_LAYER_WEIGHTS)
        print(f"\n[Judge] Using COMPRESSED weights (visual={weights['visual']*100:.0f}%)")
    else:
        weights = dict(LAYER_WEIGHTS)
    
    if visual_uncertain or visual_confidence < 0.4:
        weights["visual"] *= 0.5
    
    if is_jpeg:
        weights["spectrum"] *= 0.7
        weights["prnu"] *= 0.6

    # ========================================================================
    # TIER 2: HARDWARE VETO VERDICT
    # ========================================================================
    if is_physically_real:
        any_ai_signals = sum(1 for s in effective_scores.values() if s <= -15)
        
        if any_ai_signals >= 2:
            return (70, "EDITED_REAL", "Physical camera DNA with AI post-processing")
        else:
            return (90, "REAL", "Physical sensor DNA verified (PRNU + Spectrum)")

    # ========================================================================
    # TIER 3: MODEL CONSENSUS VERDICT (For compressed images)
    # ========================================================================
    if is_compressed_image and model_count >= 4:
        print(f"\n[Judge] MODEL CONSENSUS DECISION:")
        
        # Strong REAL consensus
        if model_real_votes >= 4 and model_consensus >= 0.65:
            ai_artifacts = [k for k, s in effective_scores.items() 
                          if s <= -25 and k in ["spectrum", "face", "eye", "watermark"]]
            
            if len(ai_artifacts) == 0:
                print(f"  [+] {model_real_votes}/{model_count} models say REAL, no AI artifacts")
                return (72, "LIKELY_REAL", f"Model consensus ({model_real_votes}/{model_count} REAL)")
            else:
                print(f"  ! Models say REAL but {len(ai_artifacts)} artifacts found: {ai_artifacts}")
        
        # Strong AI consensus
        elif model_ai_votes >= 4:
            print(f"  [-] {model_ai_votes}/{model_count} models say AI")
            return (20, "AI-GENERATED", f"Model consensus ({model_ai_votes}/{model_count} AI)")
        
        # Moderate REAL consensus
        elif model_real_votes >= 3 and visual_score > 10:
            ai_artifacts = [k for k, s in effective_scores.items() 
                          if s <= -20 and k in ["spectrum", "face", "eye", "watermark"]]
            
            if len(ai_artifacts) == 0:
                print(f"  [ ] {model_real_votes}/{model_count} models REAL, no artifacts")
                return (65, "LIKELY_REAL", f"Moderate consensus ({model_real_votes}/{model_count} REAL)")

    # ========================================================================
    # TIER 4: WATERMARK WITH CORROBORATION
    # ========================================================================
    if effective_scores["watermark"] <= -35:
        other_ai = [k for k, s in effective_scores.items() 
                   if k != "watermark" and s <= -20]
        
        if len(other_ai) >= 1:
            return (5, "AI-GENERATED", f"Watermark + {len(other_ai)} corroboration(s)")
        elif is_texture_authentic:
            print(f"[Judge] Watermark in real-texture image dismissed")
        elif is_compressed_image and model_real_votes >= 3:
            # For compressed/JPEG images, watermark detection is prone to false
            # positives from compression artifacts. If model consensus leans REAL
            # and there's no corroboration, treat as EDITED rather than AI.
            print(f"[Judge] [!] Lone watermark on compressed image with {model_real_votes}/{model_count} models REAL")
            print(f"    → Downgrading from AI-GENERATED to EDITED")
            return (55, "EDITED", f"Watermark FP likely (compressed + {model_real_votes}/{model_count} models REAL)")
        else:
            return (25, "AI-GENERATED", "AI watermark (no corroboration but no real texture)")

    # ========================================================================
    # TIER 5: PHYSICAL IMPOSSIBILITIES (with corroboration)
    # ========================================================================
    ai_indicators = [k for k, s in effective_scores.items() if s <= -25 and k != "context"]
    
    if effective_scores["eye"] <= -45 and len(ai_indicators) >= 2:
        return (8, "AI-GENERATED", "Impossible corneal reflections (corroborated)")
    
    if effective_scores["shadow"] <= -40 and len(ai_indicators) >= 2:
        return (12, "AI-GENERATED", "Impossible shadow geometry (corroborated)")

    # ========================================================================
    # TIER 6: EDITED REAL SAFETY NET
    # ========================================================================
    if is_texture_authentic:
        strong_ai = [k for k, s in effective_scores.items() if s <= -30 and k != "context"]
        
        if len(strong_ai) == 1:
            return (70, "EDITED_REAL", f"Authentic base + {strong_ai[0]} modification")
        elif len(strong_ai) == 0:
            return (85, "REAL", "Authentic texture, no AI signatures")

    # ========================================================================
    # TIER 7: WEIGHTED AUTHENTICITY SCORE
    # ========================================================================
    print(f"\n[Judge] Computing weighted authenticity...")
    
    weighted_sum = 0.0
    active_weight_sum = 0.0
    skipped_layers = []
    
    for layer, score in effective_scores.items():
        w = weights.get(layer, 0.05)
        
        if score == 0:
            skipped_layers.append(layer)
            continue
        
        norm = max(0.0, min(1.0, (score + 50) / 100.0))
        weighted_sum += norm * w
        active_weight_sum += w
    
    if skipped_layers:
        print(f"  Skipped neutral: {', '.join(skipped_layers)}")
    
    if active_weight_sum > 0:
        final_auth = (weighted_sum / active_weight_sum) * 100
    else:
        return (50, "EDITED", "All layers neutral - insufficient signal")
    
    strong_ai_count = sum(1 for s in effective_scores.values() if s <= -20)
    strong_real_count = sum(1 for s in effective_scores.values() if s >= 15)
    
    print(f"  Final auth: {final_auth:.1f}%")
    print(f"  AI signals: {strong_ai_count}")
    print(f"  Real signals: {strong_real_count}")



    # TIER 8: FINAL VERDICT MAPPING
    # REAL verdict
    if final_auth >= 65:
        return (int(final_auth), "REAL", "Strong multi-layer authenticity")
    
    if final_auth >= 55 and strong_real_count >= 2:
        return (int(final_auth), "REAL", "Multiple forensic layers indicate camera source")
    
    # AI verdict (requires 2+ signals)
    if final_auth <= 35 and strong_ai_count >= 2:
        return (int(final_auth), "AI-GENERATED", f"Consensus AI detection ({strong_ai_count} signals)")
    
    if final_auth <= 45 and strong_ai_count >= 3:
        return (int(final_auth), "AI-GENERATED", f"Strong AI consensus ({strong_ai_count} signals)")
    
    # Conflicting signals
    if 35 < final_auth < 60 and strong_ai_count > 0 and strong_real_count > 0:
        return (int(final_auth), "EDITED", "Conflicting forensic signals")
    
    # Single AI signal without corroboration
    if strong_ai_count == 1 and strong_real_count == 0:
        return (int(final_auth), "EDITED", "Single AI indicator needs corroboration")
    
    # Weak signals: use count as tie-breaker
    if strong_ai_count > strong_real_count:
        return (int(final_auth), "EDITED", "Slight AI lean but insufficient evidence")
    elif strong_real_count > strong_ai_count:
        return (int(final_auth), "REAL", "More real indicators than AI")
    
    return (int(final_auth), "EDITED", "Insufficient forensic evidence")


if __name__ == "__main__":
    print("=" * 70)
    print("Master Judge v3.3 - CONSENSUS + SAFETY EDITION")
    print("=" * 70)
    print("\nKEY FIXES:")
    print("   Model consensus pre-check before kill switches")
    print("   PRNU kill switch requires corroboration")
    print("   Fixed compressed image detection")
    print("   Kill switches can be overridden by strong consensus")
    print("   PRNU confidence check (flat region ratio)")
    print("=" * 70)

    # === Example: How to wire the Vision Agent into Layer 5 ===
    from vision_agent import run_vision_agent
    # Replace 'suspect_image.jpg' with your test image path
    vision_results = run_vision_agent("suspect_image.jpg")

    # Example: Mock data for other forensic layers
    prnu_data = {"score": -10}
    spectrum_data = {"score": -5}
    # ... add other mock or real layer outputs as needed ...

    # Call Layer 5 with Vision Agent results
    final_score, verdict, description = calculate_integrity(
        c2pa_res={},
        meta_score=0,
        physics_score=0,
        face_score=0,
        visual_score=vision_results["visual_score"],
        spectrum_score=spectrum_data["score"],
        eye_score=0,
        watermark_score=0,
        watermark_desc="",
        prnu_score=prnu_data["score"],
        context_score=0,
        context_details={},
        shadow_score=0,
        shadow_desc="",
        artifact_score=0,
        physical_continuity_score=0,
        visual_confidence=vision_results["confidence"],
        visual_uncertain=vision_results["visual_uncertain"],
        is_jpeg=False,
        is_degraded_signal=False,
        model_real_votes=0,
        model_ai_votes=0,
        model_count=0,
        model_consensus=0.0,
        prnu_flat_region_ratio=1.0,
        prnu_details=None
    )

    print(f"\nFINAL VERDICT: {verdict} ({final_score}/100)")
    print(f"Explanation: {description}")

