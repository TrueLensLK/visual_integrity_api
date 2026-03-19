import os

import os
import sys
import time
import glob
import json
import math
from datetime import datetime
from tqdm import tqdm  # pip install tqdm (for progress bars)
from tabulate import tabulate # pip install tabulate (for nice tables)

# --- IMPORT YOUR PIPELINE ---
# Dynamically add Universal_Detector/src to sys.path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "Universal_Detector", "src"))
from Universal_Detector.src.layers.forensic_case_builder import compile_case_file
from Universal_Detector.src.layers.llm_judge import HybridJudge
# --- CONFIGURATION ---
DATASET_PATH = "validation_dataset"
ENABLE_LLM = False  # Set to True to test the full "Final Boss" pipeline

def run_validation():
    print("STARTING VALIDATION RUN...")
    print(f"Dataset: {DATASET_PATH}")
    
    # 1. Initialize the Pipeline
    judge = HybridJudge(enable_llm=ENABLE_LLM)

    # Import detection layers (reuse main.py logic)
    import importlib
    layers_dir = os.path.join(os.path.dirname(__file__), "Universal_Detector", "src", "layers")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Universal_Detector", "src"))
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

    # 2. Load Images
    real_images = glob.glob(os.path.join(DATASET_PATH, "real", "*"))
    fake_images = glob.glob(os.path.join(DATASET_PATH, "fake", "*"))
    all_files = [(img, "REAL") for img in real_images] + [(img, "AI-GENERATED") for img in fake_images]

    if not all_files:
        print("No images found! Check your folder structure.")
        return

    print(f"Found {len(real_images)} Real and {len(fake_images)} Fake images.")

    results = []
    detailed_results = []  # For the report document
    correct_count = 0
    tp = 0 # True Positives (Fake detected as Fake)
    tn = 0 # True Negatives (Real detected as Real)
    fp = 0 # False Positives (Real detected as Fake)
    fn = 0 # False Negatives (Fake detected as Real)
    
    # Per-class tracking
    real_scores = []
    fake_scores = []
    processing_times = []
    start_run_time = time.time()

    # 3. Process Loop
    for image_path, ground_truth in tqdm(all_files, desc="Analyzing"):
        filename = os.path.basename(image_path)
        try:
            # --- FULL DETECTION PIPELINE (mirrors main.py) ---
            triage_result = quick_check(image_path)
            if triage_result["status"] == "FAIL":
                raise Exception(f"File validation failed: {triage_result['reason']}")

            is_jpeg = triage_result.get("details", {}).get("format", "").upper() in ("JPEG", "WEBP")
            if image_path.lower().endswith((".jpg", ".jpeg", ".webp")):
                is_jpeg = True

            layer_scores = {}
            layer_details = {}

            c2pa_result = verify_c2pa(image_path)
            layer_details["c2pa"] = c2pa_result.get("message", "No credentials")

            try:
                m_score, m_det = analyze_metadata(image_path)
                layer_scores["metadata"] = m_score
                layer_details["metadata"] = str(m_det)
            except Exception as e:
                layer_scores["metadata"] = 0
                layer_details["metadata"] = f"Error: {e}"

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

            try:
                f_res = analyze_face_consistency(image_path)
                layer_scores["face_consistency"] = f_res["impact"] if isinstance(f_res, dict) else f_res
                layer_details["face_consistency"] = f"Faces: {f_res.get('face_count', 0)}"
            except Exception:
                layer_scores["face_consistency"] = 0

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

            try:
                s_score, s_desc = analyze_spectrum(image_path)
                layer_scores["spectrum"] = s_score
                layer_details["spectrum"] = s_desc
            except Exception:
                layer_scores["spectrum"] = 0

            try:
                e_score, e_desc = analyze_eyes(image_path)
                layer_scores["eye_physics"] = e_score
                layer_details["eye_physics"] = e_desc
            except Exception:
                layer_scores["eye_physics"] = 0

            try:
                w_score, w_desc = detect_watermarks(image_path, is_jpeg=is_jpeg)
                layer_scores["watermark"] = w_score
                layer_details["watermark"] = w_desc
            except Exception:
                layer_scores["watermark"] = 0

            prnu_details_dict = {}
            try:
                prnu_score, prnu_desc, prnu_details_dict = analyze_prnu(image_path, is_jpeg_hint=is_jpeg)
                layer_scores["prnu"] = prnu_score
                layer_details["prnu"] = prnu_desc
            except Exception:
                layer_scores["prnu"] = 0

            context_data_dict = {}
            try:
                c_score, context_data_dict = analyze_context(image_path)
                layer_scores["context"] = c_score
                layer_details["context"] = context_data_dict.get("note", "")
            except Exception:
                layer_scores["context"] = 0

            try:
                sh_score = get_shadow_score(image_path)
                layer_scores["shadow"] = sh_score
                layer_details["shadow"] = f"Score: {sh_score}"
            except Exception:
                layer_scores["shadow"] = 0

            try:
                pc_score, pc_desc = get_physical_continuity_score(image_path)
                layer_scores["physical_continuity"] = pc_score
                layer_details["physical_continuity"] = pc_desc
            except Exception:
                layer_scores["physical_continuity"] = 0

            try:
                a_res = analyze_artifacts(image_path, is_jpeg=is_jpeg)
                layer_scores["artifacts"] = a_res["score"]
                layer_details["artifacts"] = a_res["description"]
            except Exception:
                layer_scores["artifacts"] = 0

            # --- LAYER 5: MASTER JUDGE (Rule-Based) ---
            print(f"[DEBUG] {filename} layer_scores: {layer_scores}")
            final_score, verdict, description, effective_scores = calculate_integrity(
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
                prnu_details=prnu_details_dict,
                context_score=layer_scores.get("context", 0),
                context_details=context_data_dict,
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

            # --- Compile Case File for LLM Judge ---
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
                warnings=[],
                effective_scores=effective_scores
            )

            # --- Run Final Judge (LLM/Hybrid) ---
            final_verdict, final_score, final_desc, _ = judge.judge(
                case_file,
                rule_based_verdict,
                rule_based_score,
                rule_based_description,
                image_path=image_path
            )

            # --- GRADE THE RESULT ---
            # Map multi-class verdicts to binary classification
            if final_verdict in ("REAL", "LIKELY_REAL", "EDITED_REAL"):
                system_said = "REAL"
            elif final_verdict in ("AI-GENERATED", "AI-ENHANCED"):
                system_said = "AI-GENERATED"
            else:
                # For EDITED, AMBIGUOUS, etc. - use score (>50 = REAL-leaning)
                system_said = "REAL" if final_score > 50 else "AI-GENERATED"
            truth = "REAL" if "REAL" in ground_truth.upper() else "AI-GENERATED"
            is_correct = (system_said == truth)
            if is_correct: correct_count += 1

            if truth == "AI-GENERATED" and system_said == "AI-GENERATED": tp += 1
            if truth == "REAL" and system_said == "REAL": tn += 1
            if truth == "REAL" and system_said == "AI-GENERATED": fp += 1
            if truth == "AI-GENERATED" and system_said == "REAL": fn += 1

            # Track per-class scores
            if truth == "REAL":
                real_scores.append(final_score)
            else:
                fake_scores.append(final_score)

            results.append([
                filename,
                truth,
                system_said,
                f"{final_score}%",
                "OK" if is_correct else "FAIL",
                final_desc[:50] + "..."
            ])

            # Detailed record for report
            detailed_results.append({
                "filename": filename,
                "ground_truth": truth,
                "system_verdict": system_said,
                "raw_verdict": final_verdict,
                "score": final_score,
                "correct": is_correct,
                "reasoning": final_desc,
                "layer_scores": {k: float(v) for k, v in layer_scores.items()},
            })

            time.sleep(2)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append([filename, ground_truth, "ERROR", "0%", "FAIL", str(e)])

    # 4. Calculate Metrics
    total_run_time = time.time() - start_run_time
    total = len(all_files)
    accuracy = (correct_count / total) * 100 if total > 0 else 0
    
    # Avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity (True Negative Rate) - How well we identify REAL images
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Negative Predictive Value - When we say REAL, how often are we right?
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Balanced Accuracy (handles class imbalance)
    balanced_accuracy = ((recall + specificity) / 2) * 100
    
    # Matthews Correlation Coefficient (best single metric for binary classification)
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 1
    mcc = mcc_num / mcc_den
    
    # False Positive Rate & False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Per-class accuracy
    real_accuracy = (tn / len(real_images) * 100) if len(real_images) > 0 else 0
    fake_accuracy = (tp / len(fake_images) * 100) if len(fake_images) > 0 else 0
    
    # Score statistics
    avg_real_score = sum(real_scores) / len(real_scores) if real_scores else 0
    avg_fake_score = sum(fake_scores) / len(fake_scores) if fake_scores else 0
    
    # Cohen's Kappa (agreement beyond chance)
    p_observed = accuracy / 100
    p_expected_pos = ((tp + fp) / total) * ((tp + fn) / total) if total > 0 else 0
    p_expected_neg = ((tn + fn) / total) * ((tn + fp) / total) if total > 0 else 0
    p_expected = p_expected_pos + p_expected_neg
    cohens_kappa = (p_observed - p_expected) / (1 - p_expected) if (1 - p_expected) > 0 else 0

    # 5. Print Report
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    headers = ["Filename", "Ground Truth", "System Verdict", "Score", "Correct?", "Reasoning"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    print("\n" + "="*60)
    print("PRIMARY METRICS")
    print("="*60)
    print(f"  Accuracy:           {accuracy:.1f}%  ({correct_count}/{total})")
    print(f"  Balanced Accuracy:  {balanced_accuracy:.1f}%")
    print(f"  Precision:          {precision:.4f}  (When we say FAKE, how often correct)")
    print(f"  Recall/Sensitivity: {recall:.4f}  (How many FAKEs we catch)")
    print(f"  F1-Score:           {f1_score:.4f}  (Harmonic mean of Precision & Recall)")
    print(f"  MCC:               {mcc:+.4f}  (Matthews Correlation Coefficient)")
    print(f"  Cohen's Kappa:     {cohens_kappa:+.4f}  (Agreement beyond chance)")
    
    print(f"\n{'='*60}")
    print("SECONDARY METRICS")
    print("="*60)
    print(f"  Specificity (TNR):  {specificity:.4f}  (How many REALs we correctly classify)")
    print(f"  NPV:               {npv:.4f}  (When we say REAL, how often correct)")
    print(f"  False Positive Rate: {fpr:.4f}  (Real flagged as Fake)")
    print(f"  False Negative Rate: {fnr:.4f}  (Fake missed as Real)")
    
    print(f"\n{'='*60}")
    print("PER-CLASS BREAKDOWN")
    print("="*60)
    print(f"  Real Images:  {tn}/{len(real_images)} correct  ({real_accuracy:.1f}%)")
    print(f"  Fake Images:  {tp}/{len(fake_images)} correct  ({fake_accuracy:.1f}%)")
    print(f"  Avg Real Score:  {avg_real_score:.1f}/100  (higher = more REAL)")
    print(f"  Avg Fake Score:  {avg_fake_score:.1f}/100  (lower = more FAKE)")
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX")
    print("="*60)
    cm_headers = ["", "Predicted FAKE", "Predicted REAL"]
    cm_data = [
        ["Actual FAKE", f"TP = {tp}", f"FN = {fn}"],
        ["Actual REAL", f"FP = {fp}", f"TN = {tn}"],
    ]
    print(tabulate(cm_data, headers=cm_headers, tablefmt="grid"))
    
    print(f"\n  True Positives  (Fake → Fake): {tp}")
    print(f"  True Negatives  (Real → Real): {tn}")
    print(f"  False Positives (Real → Fake): {fp}  ← 'False Alarm'")
    print(f"  False Negatives (Fake → Real): {fn}  ← 'Missed Detection'")
    
    print(f"\n{'='*60}")
    print("PERFORMANCE")
    print("="*60)
    print(f"  Total Runtime:      {total_run_time:.1f}s")
    print(f"  Avg per Image:      {total_run_time/total:.1f}s")
    print(f"  Images Processed:   {total}")
    print(f"  Dataset:            {len(real_images)} Real + {len(fake_images)} Fake")
    
    # ========================================================================
    # 6. Generate Markdown Validation Report Document
    # ========================================================================
    report_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_filename = f"VALIDATION_REPORT_{report_timestamp}.md"
    
    # Build results table rows
    table_rows = ""
    for r in detailed_results:
        icon = "OK" if r["correct"] else "FAIL"
        reasoning_short = r["reasoning"][:80].replace("|", "\\|")
        table_rows += f"| {r['filename'][:45]} | {r['ground_truth']} | {r['system_verdict']} | {r['raw_verdict']} | {r['score']}% | {icon} | {reasoning_short}... |\n"
    
    # Build per-image layer score breakdown
    layer_breakdown = ""
    for r in detailed_results:
        icon = "OK" if r["correct"] else "FAIL"
        layer_breakdown += f"\n### [{icon}] `{r['filename']}`\n"
        layer_breakdown += f"- **Ground Truth:** {r['ground_truth']}  |  **System:** {r['system_verdict']} ({r['raw_verdict']})  |  **Score:** {r['score']}%\n"
        layer_breakdown += f"- **Reasoning:** {r['reasoning']}\n"
        layer_breakdown += f"- **Layer Scores:**\n"
        for layer, score in sorted(r["layer_scores"].items(), key=lambda x: x[1]):
            indicator = "[!]" if score < -20 else "[+]" if score > 15 else "[ ]"
            layer_breakdown += f"  - {indicator} `{layer}`: {score:+.1f}\n"
    
    # Build error analysis
    fp_list = [r for r in detailed_results if r["ground_truth"] == "REAL" and not r["correct"]]
    fn_list = [r for r in detailed_results if r["ground_truth"] == "AI-GENERATED" and not r["correct"]]
    
    error_analysis = ""
    if fp_list:
        error_analysis += "\n### False Positives (Real images flagged as Fake)\n\n"
        for r in fp_list:
            error_analysis += f"- **{r['filename']}** — Score: {r['score']}% — {r['reasoning'][:100]}\n"
    if fn_list:
        error_analysis += "\n### False Negatives (Fake images missed as Real)\n\n"
        for r in fn_list:
            error_analysis += f"- **{r['filename']}** — Score: {r['score']}% — {r['reasoning'][:100]}\n"
    if not fp_list and not fn_list:
        error_analysis = "\n> **Perfect classification** — No errors detected.\n"
    
    report_content = f"""# DeepFake Detection — Validation Report

**Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}  
**Dataset:** `{DATASET_PATH}/` ({len(real_images)} Real + {len(fake_images)} Fake = {total} images)  
**LLM Judge:** {"Enabled" if ENABLE_LLM else "Disabled"} ({"Available" if judge.agent and (judge.agent.gemini_api_key or judge.agent.groq_api_key or judge.agent.openrouter_api_key) else "No API keys — rule-based fallback"})  
**Runtime:** {total_run_time:.1f}s ({total_run_time/total:.1f}s per image)

---

## Summary

| Metric | Value |
|--------|-------|
| **Accuracy** | **{accuracy:.1f}%** ({correct_count}/{total}) |
| **Balanced Accuracy** | {balanced_accuracy:.1f}% |
| **Precision** | {precision:.4f} |
| **Recall (Sensitivity)** | {recall:.4f} |
| **Specificity (TNR)** | {specificity:.4f} |
| **F1-Score** | **{f1_score:.4f}** |
| **MCC** | {mcc:+.4f} |
| **Cohen's Kappa** | {cohens_kappa:+.4f} |
| **NPV** | {npv:.4f} |
| **False Positive Rate** | {fpr:.4f} |
| **False Negative Rate** | {fnr:.4f} |

### Per-Class Performance

| Class | Correct | Total | Accuracy | Avg Score |
|-------|---------|-------|----------|-----------|
| Real Images | {tn} | {len(real_images)} | {real_accuracy:.1f}% | {avg_real_score:.1f}/100 |
| Fake Images | {tp} | {len(fake_images)} | {fake_accuracy:.1f}% | {avg_fake_score:.1f}/100 |

---

## Confusion Matrix

|  | **Predicted FAKE** | **Predicted REAL** |
|--|-------------------:|-------------------:|
| **Actual FAKE** | TP = {tp} | FN = {fn} |
| **Actual REAL** | FP = {fp} | TN = {tn} |

- **True Positives (TP={tp}):** Fake images correctly identified as Fake
- **True Negatives (TN={tn}):** Real images correctly identified as Real
- **False Positives (FP={fp}):** Real images incorrectly flagged as Fake ← *False Alarms*
- **False Negatives (FN={fn}):** Fake images incorrectly passed as Real ← *Missed Detections*

---

## Detailed Results

| Filename | Truth | Verdict | Raw Verdict | Score | OK? | Reasoning |
|----------|-------|---------|-------------|-------|-----|-----------|
{table_rows}

---

## Error Analysis
{error_analysis}

---

## Metric Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP+TN) / Total | Overall correctness |
| **Balanced Accuracy** | (Sensitivity+Specificity) / 2 | Handles class imbalance |
| **Precision** | TP / (TP+FP) | When system says FAKE, how often right |
| **Recall (Sensitivity)** | TP / (TP+FN) | What % of actual fakes are caught |
| **Specificity (TNR)** | TN / (TN+FP) | What % of actual reals are correctly cleared |
| **F1-Score** | 2·(P·R)/(P+R) | Harmonic mean of Precision & Recall |
| **NPV** | TN / (TN+FN) | When system says REAL, how often right |
| **MCC** | (TP·TN−FP·FN)/√(...) | Best single metric for binary classification (−1 to +1) |
| **Cohen's Kappa** | (p_o−p_e)/(1−p_e) | Agreement corrected for chance (0=chance, 1=perfect) |
| **FPR** | FP / (FP+TN) | False alarm rate |
| **FNR** | FN / (FN+TP) | Miss rate |

---

## Per-Image Layer Score Breakdown
{layer_breakdown}

---

*Report generated by `validate.py` — DeepFake Detection System*
"""

    report_path = os.path.join(os.path.dirname(__file__), report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n{'='*60}")
    print(f"REPORT SAVED: {report_filename}")
    print("="*60)

if __name__ == "__main__":
    run_validation()