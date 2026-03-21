"""
Forensic Case Builder (Full Spectrum Edition)
Compiles ALL layer results into a categorized "Case File" for the LLM Judge.
Ensures no layer is hidden, grouping them by forensic domain.
"""

import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
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
    model_breakdown: Optional[Dict[str, Any]] = None  # Added
) -> Dict[str, Any]:
    
    case_id = generate_case_id(image_path)
    if not image_description:
        image_description = extract_image_context(image_path, layer_details)
    
    # 1. Build Categorized Evidence
    evidence_by_category = {cat: [] for cat in LAYER_CATEGORIES.keys()}
    evidence_by_category["Other / Unclassified"] = []
    
    # Flatten strictly for the 'raw' list, but organize strictly for the prompt
    all_evidence_flat = []
    
    for layer_name, score in layer_scores.items():
        eff_score = effective_scores.get(layer_name, score) if effective_scores else score
        
        item = {
            "layer": layer_name,
            "raw_score": score,
            "effective_score": eff_score,
            "score": eff_score, # Backward compatibility for formatters
            "strength": classify_evidence_strength(eff_score), # Use EFFECTIVE strength here
            "detail": layer_details.get(layer_name, "No detail provided")
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
            f"⚠️ CRITICAL: The aggregate neural_network score ({neural_score:+.1f}) may be INVALID. "
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
            "is_compressed": is_jpeg
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
            lines.append(f"  {icon} {suppressed_marker} [{item['layer'].ljust(15)}] Score: {score_display} | {item['detail']}")

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