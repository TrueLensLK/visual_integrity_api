"""
Forensic Case Builder (Full Spectrum Edition)
Compiles ALL layer results into a categorized "Case File" for the LLM Judge.
Ensures no layer is hidden, grouping them by forensic domain.
"""

import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import os
import json

# --- CONFIGURATION: Define which layers belong to which category ---
LAYER_CATEGORIES = {
    "Cryptographic": ["c2pa", "metadata", "exif", "signature"],
    "Physics & Sensor": ["prnu", "noise", "bayer", "cfa", "sensor_pattern", "chromatic_aberration"],
    "Pixel & Compression": ["ela", "jpeg_ghost", "error_level", "quality", "double_compression", "blocking"],
    "Semantic & Visual": ["face_consistency", "eyes", "lighting", "shadows", "geometry", "reflection"],
    "Neural Classifiers": ["ensemble", "neural_network", "deep_learning", "ai_probability"]
}

def get_category(layer_name: str) -> str:
    """Find the category for a given layer name."""
    layer_lower = layer_name.lower()
    for category, keywords in LAYER_CATEGORIES.items():
        if any(k in layer_lower for k in keywords):
            return category
    return "Other / Unclassified"

def generate_case_id(image_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    path_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]
    return f"CASE-{timestamp}-{path_hash.upper()}"

def extract_image_context(image_path: str, layer_details: Dict[str, Any]) -> str:
    """Extract contextual description."""
    context_parts = []
    
    # Check faces
    face_detail = layer_details.get("face_consistency", "")
    if "Faces:" in str(face_detail):
        try:
            face_part = str(face_detail).split("Faces:")[1].split("|")[0].strip()
            if face_part.isdigit() and int(face_part) > 0:
                context_parts.append(f"Contains {face_part} face(s)")
        except: pass
    
    if not context_parts:
        context_parts.append("General Scene")
    return "; ".join(context_parts)

def classify_evidence_strength(score: float) -> str:
    if score <= -40: return "STRONG FAKE"
    elif score <= -20: return "MODERATE FAKE"
    elif score <= -5: return "WEAK FAKE"
    elif score < 5: return "NEUTRAL"
    elif score < 20: return "WEAK REAL"
    elif score < 40: return "MODERATE REAL"
    else: return "STRONG REAL"

def identify_contradictions(layer_scores: Dict[str, float], layer_details: Dict[str, str], is_jpeg: bool = False, effective_scores: Optional[Dict[str, float]] = None) -> list:
    """Identify conflicts between categories."""
    contradictions = []
    
    # Calculate category averages (Using RAW scores)
    cat_scores = {}
    for layer, score in layer_scores.items():
        cat = get_category(layer)
        if cat not in cat_scores: cat_scores[cat] = []
        cat_scores[cat].append(score)
    
    avgs = {k: sum(v)/len(v) for k, v in cat_scores.items() if v}
    
    # Check 1: Physics (Real) vs Visuals (Fake)
    if avgs.get("Physics & Sensor", 0) > 20 and avgs.get("Semantic & Visual", 0) < -20:
        contradictions.append({
            "type": "PHYSICS_VS_VISUAL",
            "note": "Camera sensor noise is present (REAL), but visual geometry looks wrong (FAKE). Possible 'Cheapfake' or edited real photo."
        })

    # Check 2: The Resizing Paradox (PRNU says Fake, Others say Real)
    prnu = layer_scores.get("prnu", 0)
    nn = layer_scores.get("neural_network", 0)
    
    # Logic Update: If image is JPEG compressed, PRNU grids are often artifacts
    if prnu < -30 and nn > 20 and is_jpeg:
        contradictions.append({
            "type": "SOCIAL_MEDIA_COMPRESSION",
            "note": "High PRNU/Grid score (-30 or worse) in a compressed JPEG/Social Media image. "
                    "This is a COMMON FALSE POSITIVE. Resizing algorithms create grid-like artifacts "
                    "that mimic AI generation. If the image visually looks real (people, nature), "
                    "DISREGARD the PRNU signal."
        })
    elif prnu < -30 and nn > 30:
         contradictions.append({
            "type": "RESIZING_ARTIFACT",
            "note": "PRNU indicates a grid (Fake), but Neural Networks see a natural image. Likely a resized/screenshot real image."
        })

    # NEW CHECK: Suppression Detection (Raw vs Effective)
    # If a score was significantly dampened by the Rule-Based Judge, flag it for the LLM.
    if effective_scores:
        for layer, raw_score in layer_scores.items():
            eff_score = effective_scores.get(layer, raw_score)
            if raw_score <= -30 and eff_score > -15:
                # Strong FAKE signal was suppressed
                contradictions.append({
                    "type": "SUPPRESSED_SIGNAL",
                    "note": f"The '{layer}' signal was dampened from {raw_score} (STRONG FAKE) to {eff_score} by the Rule-Based Judge. "
                            f"Investigate if this suppression was valid (e.g. compression artifact) or a missed detection."
                })

    # Check 3: PRNU Synthetic Grid vs Bayer Pattern (Physics)
    # Bayer demosaicing is a hardware artifact from real camera sensors.
    # However, high-quality AI can sometimes mimic periodic patterns or checkerboard artifacts.
    physics = layer_scores.get("physics", 0)
    if prnu < -40 and physics > 10:
        contradictions.append({
            "type": "BAYER_PRNU_CONTRADICTION",
            "note": f"PRNU indicates a strong synthetic grid (score={prnu}), but physics detected Bayer-like patterns (score=+{physics}). "
                    f"While Bayer usage suggests a real sensor, sophisticated AI (GANs) can generate checkerboard artifacts "
                    f"that mimic this. Do NOT automatically dismiss the PRNU score if the image shows other AI signs."
        })

    return contradictions

def determine_reliability(layer_name: str, score: float, detail: str, context_dict: Dict) -> Tuple[str, str]:
    """
    Determine the reliability of a forensic signal based on context flags vs score.
    Returns: (reliability_level, reason)
    Levels: HIGH, MEDIUM, LOW
    """
    reliability = "MEDIUM"
    reason = "Standard analysis"
    detail_lower = detail.lower()
    
    # Context extraction
    is_jpeg = context_dict.get("is_jpeg", False)
    face_count = context_dict.get("face_count", 0)
    
    # --- PRNU Rules ---
    if layer_name == "prnu":
        # Attempt to parse PCE and Flat Ratio from detail string
        # Expected format: "PCE: 12345 (interpretation) | Flat: 85.0% ..." or similar
        import re
        pce_match = re.search(r'PCE[:=]\s?([\d,]+)', detail)
        flat_match = re.search(r'Flat[:=]\s?([\d\.]+)%', detail)
        
        pce = float(pce_match.group(1).replace(',', '')) if pce_match else 0
        flat_regions = float(flat_match.group(1)) if flat_match else 0.0
        
        # Rule 1: Compression Artifact Range
        if 10000 <= pce <= 100000 and flat_regions > 60 and is_jpeg:
            reliability = "LOW"
            reason = f"PCE={pce:,.0f} in compression artifact range (10K-100K) on image with {flat_regions}% flat regions."
        
        # Rule 2: Genuine Synthetic Grid
        elif pce > 100000 and flat_regions < 60:
            reliability = "HIGH"
            reason = f"PCE={pce:,.0f} far exceeds compression range. Genuine synthetic grid."
            
        # Rule 3: Intermediate
        elif 3000 <= pce <= 10000:
            reliability = "MEDIUM"
            reason = f"PCE={pce:,.0f} in intermediate range."
            
        # Rule 4: Camera Original Low PCE
        elif "camera original" in detail_lower and pce < 3000:
             reliability = "HIGH"
             reason = "Genuine camera original with expected low PCE."
             
    # --- Watermark Rules ---
    elif layer_name == "watermark":
        if "visible" in detail_lower and ("text" in detail_lower or "logo" in detail_lower):
            reliability = "LOW"
            reason = "Brand mark or text likely triggered false positive."
        elif "synthid" in detail_lower and score <= -40:
            reliability = "HIGH"
            reason = "Strong SynthID detection."
        elif is_jpeg and score > -30:
            reliability = "LOW"
            reason = "Signal potentially dampened by JPEG compression."
        elif "multiple" in detail_lower and "confirmed" in detail_lower:
            reliability = "HIGH"
            reason = "Multiple watermark types confirmed."

    # --- Shadow Rules ---
    elif layer_name == "shadow":
        phys_cont = context_dict.get("physical_continuity_detail", "").lower()
        if face_count == 0 and ("ambient" in phys_cont or "spread" in phys_cont):
            reliability = "LOW"
            reason = "Indoor multi-source lighting on faceless image is ambiguous."
        elif "outdoor" in detail_lower and "dominant" in detail_lower:
            reliability = "HIGH"
            reason = "Outdoor scene with clear dominant shadow direction."
        elif "indoor" in detail_lower:
            reliability = "MEDIUM"
            reason = "Indoor lighting analysis."

    # --- Spectrum Rules ---
    elif layer_name == "spectrum":
        if is_jpeg and "suspicious" in detail_lower:
             reliability = "LOW"
             reason = "MozJPEG artifact, not AI manipulation."
        elif ("spike" in detail_lower and "1000" in detail_lower) and ("text" in detail_lower or "logo" in detail_lower):
             reliability = "LOW"
             reason = "High frequency text patterns mimicking AI spikes."
        elif not is_jpeg and "suspicious" in detail_lower:
             reliability = "HIGH"
             reason = "Suspicious patches in non-JPEG image."
             
    # --- Visual Rules (Including Ateeqq) ---
    elif layer_name == "visual" or layer_name == "ensemble":
        # Check breakdown if available in context
        breakdown = context_dict.get("model_breakdown", {})
        sdxl_data = breakdown.get("sdxl", {})
        ateeqq_data = breakdown.get("ateeqq", {}) # NEW: Ateeqq instead of SigLIP

        # SDXL Reliability
        if sdxl_data:
            conf = sdxl_data.get("conf", 0)
            if conf is None: conf = 0
            
            if conf < 0.60:
                reliability = "LOW"
                reason = f"SDXL confidence {conf:.2f} below 0.60 floor (treating as noise)."
            elif conf > 0.80:
                reliability = "HIGH"
                reason = f"SDXL confidence {conf:.2f} in specialist authority range."
            else:
                reliability = "MEDIUM"
                reason = f"SDXL confidence {conf:.2f} in intermediate range."

        # Ateeqq Reliability
        if ateeqq_data:
             conf = ateeqq_data.get("conf", 0)
             if conf is None: conf = 0
             if conf > 0.85:
                 reliability = "HIGH"
                 reason += f" | Ateeqq (Generative Expert) high confidence {conf:.2f}"

    return reliability, reason

def compile_case_file(
    image_path: str,
    layer_scores: Dict[str, float],
    layer_details: Dict[str, str],
    rule_based_verdict: str,
    rule_based_score: int,
    rule_based_description: str,
    c2pa_result: Optional[Dict] = None,
    image_description: Optional[str] = None,
    is_jpeg: bool = False,
    visual_confidence: float = 1.0,
    model_consensus: float = 0.0,
    model_real_votes: int = 0,
    model_ai_votes: int = 0,
    warnings: Optional[list] = None,
    effective_scores: Optional[Dict[str, float]] = None,
    model_breakdown: Optional[Dict[str, Any]] = None,
    face_count: int = 0  # NEW: Need face count for reliability rules
) -> Dict[str, Any]:
    
    case_id = generate_case_id(image_path)
    if not image_description:
        image_description = extract_image_context(image_path, layer_details)
    
    # Context dictionary for reliability checker
    context_dict = {
        "is_jpeg": is_jpeg,
        "face_count": face_count,
        "physical_continuity_detail": layer_details.get("physical_continuity", ""),
        "model_breakdown": model_breakdown or {}
    }

    # 1. Build Categorized Evidence
    evidence_by_category = {cat: [] for cat in LAYER_CATEGORIES.keys()}
    evidence_by_category["Other / Unclassified"] = []
    
    # Flatten strictly for the 'raw' list, but organize strictly for the prompt
    all_evidence_flat = []
    
    for layer_name, score in layer_scores.items():
        eff_score = effective_scores.get(layer_name, score) if effective_scores else score
        detail = layer_details.get(layer_name, "No detail provided")
        
        # NEW: Determine Reliability
        reliability, rel_reason = determine_reliability(layer_name, score, detail, context_dict)
        
        item = {
            "layer": layer_name,
            "raw_score": score,
            "effective_score": eff_score,
            "score": eff_score, # Backward compatibility for formatters
            "strength": classify_evidence_strength(eff_score), # Use EFFECTIVE strength here
            "detail": detail,
            "reliability": reliability,      # NEW FIELD
            "reliability_reason": rel_reason # NEW FIELD
        }
        
        # Add suppression flag if difference is significant
        if abs(score - eff_score) > 10:
            item["suppressed"] = True
            item["raw_strength"] = classify_evidence_strength(score)
        
        cat = get_category(layer_name)
        if cat in evidence_by_category:  # Safe append
            evidence_by_category[cat].append(item)
        all_evidence_flat.append(item)

    # 3. Detect Neural Outlier (Swin on Face=0)
    neural_outlier_warning = None
    neural_score = layer_scores.get('neural_network', 0)
    
    # Check if Swin corrupted the aggregate score (e.g., aggregate is +40 but most models are negative)
    # This happens if Swin returns +50 while others return -10, averaging out to positive.
    # Logic: If 3+ models say AI (model_ai_votes >= 3) but the aggregate score is positive,
    # OR if Swin contributes highly positive score while face count is 0.
    
    # We can infer Swin's influence if we see high disagreement and specific conditions
    if model_ai_votes >= 3 and neural_score > 0:
         neural_outlier_warning = (
            f"CRITICAL: The aggregate neural_network score ({neural_score:+.1f}) may be INVALID. "
            f"3/{model_ai_votes + model_real_votes} models say FAKE, yet the total score is POSITIVE. "
            f"This suggests a face-swap model (Swin) defaulted to REAL on a faceless image. "
            f"TRUST THE VOTE COUNT (3+ votes FAKE), NOT THE AGGREGATE SCORE."
        )

    # 4. Compile Final Case File
    return {
        "case_id": case_id,
        "timestamp": datetime.now().isoformat(),
        "image_info": {
            "filename": os.path.basename(image_path),
            "context": image_description,
            "is_compressed": is_jpeg,
            "face_count": face_count
        },
        # CRITICAL: Expose raw scores at root so LLM Judge can find them easily
        "layer_scores": layer_scores,
        "effective_scores": effective_scores or layer_scores, # Expose effective as well
        "evidence_by_category": evidence_by_category,
        "all_evidence": all_evidence_flat,
        "cryptographic": c2pa_result or {},
        "contradictions": identify_contradictions(layer_scores, layer_details, is_jpeg=is_jpeg, effective_scores=effective_scores),
        "neural_consensus": {
            "real_votes": model_real_votes,
            "ai_votes": model_ai_votes,
            "confidence": visual_confidence,
            "outlier_warning": neural_outlier_warning, # NEW FIELD
            "true_ai_votes": model_ai_votes,
            "true_real_votes": model_real_votes,
            "model_breakdown": model_breakdown if model_breakdown else {} # Added expert breakdown
        },
        "rule_based": {
            "verdict": rule_based_verdict,
            "score": rule_based_score,
            "description": rule_based_description
        },
        "warnings": warnings or []
    }


def case_file_to_prompt_string(case_file: Dict[str, Any]) -> str:
    """
    Converts case file to a formatted string that FORCES the LLM to look at ALL layers.
    """
    lines = []
    lines.append(f"=== FORENSIC CASE FILE: {case_file['case_id']} ===")
    lines.append(f"Image: {case_file['image_info']['filename']}")
    lines.append(f"Context: {case_file['image_info']['context']}")
    lines.append("")

    # 1. Critical Veto Section
    lines.append("=== CRITICAL CHECKS (VETO POWER) ===")
    c2pa = case_file.get('cryptographic', {})
    if c2pa.get('status') == 'valid':
        lines.append(f"C2PA SIGNATURE: VALID ({c2pa.get('generator', 'Unknown')}) -> TRUST THIS.")
    else:
        lines.append(f"C2PA Signature: None/Invalid")
        
    physics_layers = case_file['evidence_by_category'].get('Physics & Sensor', [])
    # Check strictly for Bayer patterns in details
    bayer_found = any("bayer" in e['detail'].lower() for e in physics_layers)
    if bayer_found:
        lines.append("HARDWARE: Bayer/CFA Pattern Detected (Real hardware artifact, but check for AI mimicry).")
    lines.append("")

    # 2. Categorized Dashboard (The "Full Spectrum" View)
    lines.append("===FULL LAYER TELEMETRY (Categorized) ===")
    lines.append("Instruction: Review consistency within each category.")
    
    for category, items in case_file['evidence_by_category'].items():
        if not items: continue
        
        # Calculate category average for quick insight
        # Use effective score for average
        avg_score = sum(i.get('effective_score', i.get('score', 0)) for i in items) / len(items)
        status = "REAL" if avg_score > 10 else ("FAKE" if avg_score < -10 else "NEUTRAL")
        
        lines.append(f"\n--- {category.upper()} (Trend: {status}) ---")
        for item in items:
            # Display effective score primarily, but show raw if significantly different
            eff_val = item.get('effective_score', item.get('score', 0))
            raw_val = item.get('raw_score', eff_val)
            
            score_display = f"{eff_val:+05.1f}"
            suppressed_marker = ""
            
            if abs(eff_val - raw_val) > 10:
                score_display += f" (Raw: {raw_val:+05.1f})"
                suppressed_marker = "[SUPPRESSED]"
            
            icon = "✅" if eff_val > 25 else ("❌" if eff_val < -25 else "⚫")
            
            # RELIABILITY DISPLAY FIX
            reliability = item.get('reliability', 'MEDIUM')
            rel_reason = item.get('reliability_reason', '')
            
            rel_tag = ""
            if reliability != "MEDIUM":
                rel_tag = f" | RELIABILITY: {reliability}"
                
            line_detail = item['detail']
            
            # Format main line
            lines.append(f"  {icon} {suppressed_marker} [{item['layer'].ljust(15)}] Score: {score_display}{rel_tag} | {line_detail}")
            
            # Sub-line for reliability reason if not standard/medium
            if reliability != "MEDIUM" and rel_reason:
                lines.append(f"     Reason: {rel_reason}")

    # 3. Neural Consensus
    lines.append(f"\n===  VISUAL CONSENSUS ===")
    nn = case_file['neural_consensus']
    lines.append(f"Models voting REAL: {nn['real_votes']}")
    lines.append(f"Models voting FAKE: {nn['ai_votes']}")
    lines.append(f"Visual Confidence:  {nn['confidence']:.1%}")
    if nn.get('outlier_warning'):
        lines.append(f"\n{nn['outlier_warning']}\n")

    # 4. Contradictions
    if case_file.get('contradictions'):
        lines.append("\n=== CONTRADICTIONS DETECTED ===")
        for c in case_file['contradictions']:
            lines.append(f"Type: {c['type']}")
            lines.append(f"Note: {c['note']}")

    # 5. Rule-Based Judge Pre-Assessment
    rb = case_file.get('rule_based', {})
    if rb.get('verdict'):
        lines.append(f"\n=== RULE-BASED JUDGE PRE-ASSESSMENT ===")
        lines.append(f"Verdict: {rb['verdict']} (Score: {rb.get('score', 'N/A')}/100)")
        lines.append(f"Reasoning: {rb.get('description', 'N/A')}")
        lines.append("")
        lines.append("IMPORTANT: The rule-based judge has already analyzed these scores with:")
        lines.append("  - JPEG compression dampening (reduces unreliable forensic signals)")
        lines.append("  - Bayer contradiction detection (PRNU synthetic grid vs real sensor pattern)")
        lines.append("  - False positive mitigation (lone signals without corroboration)")
        lines.append("  - Model consensus weighting")
        lines.append("If the rule-based judge DISMISSED a score (e.g., downgraded PRNU due to")
        lines.append("Bayer contradiction), do NOT use that dismissed score as primary evidence.")

    return "\n".join(lines)