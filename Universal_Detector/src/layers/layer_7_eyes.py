"""
Layer 7: Optical Physics — Corneal Reflection Forensics v2.0
=============================================================
Real eyes reflect the same physical world.  AI eyes hallucinate.

ANALYSIS PIPELINE (per eye):
  7.1  Specular highlight detection via adaptive thresholding
  7.2  Glint counting — both eyes should have the SAME number of glints
  7.3  Glint morphology — real reflections are small, round, bright;
       AI reflections are blobby, irregular, scattered
  7.4  Relative glint position — glints should be in similar relative
       positions within each iris (same light → same geometry)
  7.5  Intensity profile — real specular highlights have sharp falloff;
       AI glints often have soft/gradient edges

MULTI-FACE:
  For multi-face images, each face pair is checked for cross-face
  reflection consistency: all faces lit by the same scene should
  share roughly the same number and relative position of glints.

SCORING: -50 (impossible optics) to +30 (consistent reflections)

DEPENDENCIES: mediapipe (FaceMesh with iris landmarks), opencv, numpy
"""

import cv2
import numpy as np
import math
from typing import Tuple, List, Dict, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

# Minimum eye crop size (pixels) to attempt analysis
MIN_EYE_CROP_PX = 12

# Adaptive threshold for specular highlights
GLINT_BRIGHTNESS_PERCENTILE = 97  # top 3 % of iris intensity
GLINT_MIN_BRIGHTNESS = 180        # absolute floor (0-255)

# Morphology
GLINT_MAX_AREA_RATIO = 0.25       # a single glint can't cover > 25 % of iris
GLINT_MIN_AREA_PX = 2             # ignore sub-2-px noise dots
GLINT_CIRCULARITY_THRESH = 0.35   # below this → irregular blob

# Cross-eye matching
POSITION_TOLERANCE = 0.35         # normalised-distance tolerance for matched glints
COUNT_MISMATCH_PENALTY = -20      # different glint counts
SHAPE_MISMATCH_PENALTY = -15      # same count but very different shapes


# ============================================================================
# HELPER: EXTRACT EYE CROPS FROM FACE MESH
# ============================================================================

def _get_eye_data(face_landmarks, img, h: int, w: int) -> Optional[Dict]:
    """
    Extract left and right eye crops + metadata from MediaPipe FaceMesh.

    Uses refined iris landmarks (468-477) available when refine_landmarks=True.

    Returns dict with keys:
        left_crop, right_crop, left_center, right_center,
        left_radius, right_radius
    or None if eyes are too small / off-frame.
    """
    def pt(idx):
        lm = face_landmarks.landmark[idx]
        return int(lm.x * w), int(lm.y * h)

    # Iris landmarks (refined):
    #   Left eye:  468 (center), 469-471 (iris ring)
    #   Right eye: 473 (center), 474-476 (iris ring)
    left_center = pt(468)
    right_center = pt(473)

    left_rad = int(math.hypot(
        left_center[0] - pt(469)[0],
        left_center[1] - pt(469)[1]
    ) * 1.8)  # pad 80 % beyond iris edge

    right_rad = int(math.hypot(
        right_center[0] - pt(474)[0],
        right_center[1] - pt(474)[1]
    ) * 1.8)

    if left_rad < MIN_EYE_CROP_PX // 2 or right_rad < MIN_EYE_CROP_PX // 2:
        return None

    def _crop(center, radius):
        x, y = center
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w, x + radius), min(h, y + radius)
        crop = img[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    lc = _crop(left_center, left_rad)
    rc = _crop(right_center, right_rad)
    if lc is None or rc is None:
        return None

    return {
        "left_crop": lc, "right_crop": rc,
        "left_center": left_center, "right_center": right_center,
        "left_radius": left_rad, "right_radius": right_rad,
    }


# ============================================================================
# 7.1 + 7.2: SPECULAR HIGHLIGHT DETECTION & COUNTING
# ============================================================================

def _detect_glints(eye_crop_bgr: np.ndarray) -> List[Dict]:
    """
    Detect specular highlights (glints) inside an eye crop.

    Method:
      1. Convert to grayscale.
      2. Compute adaptive brightness threshold (top N-percentile pixels).
      3. Threshold + morphological open to remove noise.
      4. Connected-component analysis → contours.
      5. For each contour: area, circularity, centroid, mean intensity.

    Returns list of dicts, one per glint:
        {centroid_norm, area_ratio, circularity, peak_intensity, contour}
    """
    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
    eh, ew = gray.shape
    total_px = eh * ew

    if total_px < MIN_EYE_CROP_PX ** 2:
        return []

    # Adaptive threshold: use percentile of THIS eye's intensity
    pct_val = float(np.percentile(gray, GLINT_BRIGHTNESS_PERCENTILE))
    thresh_val = max(GLINT_MIN_BRIGHTNESS, pct_val)

    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

    # Clean up with morphological open (remove single-px noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    glints = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < GLINT_MIN_AREA_PX:
            continue
        area_ratio = area / total_px
        if area_ratio > GLINT_MAX_AREA_RATIO:
            continue  # too big — probably iris reflection or sclera

        # Circularity = 4π·area / perimeter²  (1.0 = perfect circle)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter + 1e-6)

        # Centroid (normalised to 0-1 within crop)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"] / ew
        cy = M["m01"] / M["m00"] / eh

        # Peak intensity inside the glint contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        peak_intensity = float(np.max(gray[mask > 0])) if np.any(mask > 0) else 0

        glints.append({
            "centroid_norm": (cx, cy),
            "area_ratio": area_ratio,
            "circularity": circularity,
            "peak_intensity": peak_intensity,
            "contour": cnt,
        })

    # Sort by brightness (strongest first)
    glints.sort(key=lambda g: g["peak_intensity"], reverse=True)
    return glints


# ============================================================================
# 7.3: GLINT MORPHOLOGY SCORING
# ============================================================================

def _score_morphology(glints: List[Dict]) -> Tuple[float, str]:
    """
    Score a single eye's glint morphology.

    Returns (penalty, description).
    penalty is 0 (fine) to -25 (very bad).
    """
    if not glints:
        return -5, "no glints detected (matte/dark eye)"

    # Average circularity of the top-3 glints (ignore tiny stragglers)
    top = glints[:3]
    avg_circ = np.mean([g["circularity"] for g in top])
    max_area = max(g["area_ratio"] for g in top)

    findings = []
    penalty = 0.0

    if avg_circ < GLINT_CIRCULARITY_THRESH:
        penalty -= 12
        findings.append(f"irregular glint shapes (circ={avg_circ:.2f})")

    if max_area > 0.15:
        penalty -= 8
        findings.append(f"oversized highlight ({max_area:.1%} of iris)")

    # Check for scattered micro-glints (AI smears light across the iris)
    if len(glints) > 6:
        penalty -= 10
        findings.append(f"too many highlights ({len(glints)})")

    if not findings:
        desc = f"{len(glints)} clean glint(s), circ={avg_circ:.2f}"
    else:
        desc = "; ".join(findings)

    return penalty, desc


# ============================================================================
# 7.4: CROSS-EYE GLINT CONSISTENCY
# ============================================================================

def _compare_eyes(left_glints: List[Dict], right_glints: List[Dict]) -> Tuple[float, str]:
    """
    Compare specular highlights between left and right eye.
    """
    n_left = len(left_glints)
    n_right = len(right_glints)

    # ── A: Count match ──
    if n_left == 0 and n_right == 0:
        return -5, "Both eyes matte (no glints)"

    if n_left == 0 or n_right == 0:
        return -20, f"Asymmetric: L={n_left} R={n_right} glints (one eye matte)"

    count_diff = abs(n_left - n_right)
    if count_diff > 2:
        return COUNT_MISMATCH_PENALTY, f"Glint count mismatch: L={n_left} R={n_right}"

    # ── B: Position matching (take up to 3 strongest per eye) ──
    top_l = left_glints[:3]
    top_r = right_glints[:3]

    # For each left glint, find closest right glint (mirrored x)
    matched = 0
    position_errors = []
    intensity_ratios = []
    circ_diffs = []

    used_r: set = set()
    for lg in top_l:
        lx, ly = lg["centroid_norm"]
        # Mirror x for right eye comparison (light source is same direction
        # from camera, but eyes are mirror-symmetric in face plane)
        mirrored_x = 1.0 - lx
        best_dist = float("inf")
        best_idx = -1
        for j, rg in enumerate(top_r):
            if j in used_r:
                continue
            rx, ry = rg["centroid_norm"]
            dist = math.hypot(mirrored_x - rx, ly - ry)
            if dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx >= 0 and best_dist < POSITION_TOLERANCE:
            matched += 1
            used_r.add(best_idx)
            position_errors.append(best_dist)

            rg = top_r[best_idx]
            # ── C: Intensity ratio ──
            i_ratio = min(lg["peak_intensity"], rg["peak_intensity"]) / (
                max(lg["peak_intensity"], rg["peak_intensity"]) + 1e-6
            )
            intensity_ratios.append(i_ratio)

            # ── D: Circularity diff ──
            circ_diffs.append(abs(lg["circularity"] - rg["circularity"]))

    # ── Scoring ──
    score = 0.0
    findings = []

    n_expected = min(len(top_l), len(top_r))
    match_ratio = matched / max(n_expected, 1)

    if match_ratio >= 0.8:
        score += 15
        findings.append(f"{matched}/{n_expected} glints matched across eyes")
    elif match_ratio >= 0.5:
        score += 5
        findings.append(f"{matched}/{n_expected} glints partially matched")
    else:
        score -= 18
        findings.append(f"Poor cross-eye match ({matched}/{n_expected})")

    # Position accuracy bonus/penalty
    if position_errors:
        avg_pos_err = np.mean(position_errors)
        if avg_pos_err < 0.10:
            score += 10
            findings.append(f"glint positions tightly matched (err={avg_pos_err:.2f})")
        elif avg_pos_err > 0.25:
            score -= 10
            findings.append(f"glint positions loosely matched (err={avg_pos_err:.2f})")

    # Intensity consistency
    if intensity_ratios:
        avg_ir = np.mean(intensity_ratios)
        if avg_ir < 0.50:
            score -= 8
            findings.append(f"brightness mismatch (ratio={avg_ir:.2f})")
        elif avg_ir > 0.85:
            score += 5
            findings.append(f"brightness consistent (ratio={avg_ir:.2f})")

    # Circularity consistency
    if circ_diffs:
        avg_cd = np.mean(circ_diffs)
        if avg_cd > 0.30:
            score -= 8
            findings.append(f"shape mismatch (Δcirc={avg_cd:.2f})")

    # Count penalty (small)
    if count_diff == 1:
        score -= 5
        findings.append(f"glint count off by 1 (L={n_left} R={n_right})")
    elif count_diff == 2:
        score -= 10
        findings.append(f"glint count off by 2 (L={n_left} R={n_right})")

    description = "; ".join(findings) if findings else "Cross-eye comparison complete"
    return score, description


# ============================================================================
# 7.5: INTENSITY PROFILE (EDGE SHARPNESS)
# ============================================================================

def _score_edge_sharpness(eye_crop_bgr: np.ndarray, glints: List[Dict]) -> Tuple[float, str]:
    """
    Real specular highlights have sharp intensity falloff (point-source
    reflection off a curved cornea).  AI-generated highlights often have
    soft/gradient edges because diffusion models blur high-frequency details.
    """
    if not glints:
        return 0, ""

    gray = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.Laplacian(gray, cv2.CV_64F)
    abs_grad = np.abs(grad)

    sharpness_scores = []
    for g in glints[:3]:
        cnt = g["contour"]
        # Dilate contour slightly to get boundary ring
        mask_inner = np.zeros_like(gray)
        cv2.drawContours(mask_inner, [cnt], -1, 255, -1)
        mask_outer = cv2.dilate(mask_inner, np.ones((3, 3), np.uint8), iterations=1)
        boundary = mask_outer - mask_inner

        if np.sum(boundary > 0) < 3:
            continue

        edge_strength = float(np.mean(abs_grad[boundary > 0]))
        sharpness_scores.append(edge_strength)

    if not sharpness_scores:
        return 0, ""

    avg_sharp = np.mean(sharpness_scores)

    if avg_sharp > 40:
        return 5, f"sharp glint edges ({avg_sharp:.0f})"
    elif avg_sharp < 12:
        return -10, f"soft/blurry glint edges ({avg_sharp:.0f})"
    else:
        return 0, f"moderate glint edges ({avg_sharp:.0f})"


# ============================================================================
# MULTI-FACE CROSS-CONSISTENCY
# ============================================================================

def _cross_face_consistency(all_face_data: List[Dict]) -> Tuple[float, str]:
    """
    If multiple faces are in the same scene, ALL should have consistent
    glint patterns (same light sources illuminate everyone).

    Checks glint count variance across faces.
    """
    if len(all_face_data) < 2:
        return 0, ""

    counts = []
    for fd in all_face_data:
        lc = len(fd.get("left_glints", []))
        rc = len(fd.get("right_glints", []))
        counts.append((lc + rc) / 2.0)  # average glints per face

    if not counts:
        return 0, ""

    count_std = float(np.std(counts))
    count_mean = float(np.mean(counts))

    if count_std > 1.5 and count_mean > 0:
        return -15, f"Cross-face glint inconsistency (μ={count_mean:.1f}, σ={count_std:.1f})"
    elif count_std < 0.5 and count_mean > 0:
        return 8, f"Cross-face glints consistent ({len(all_face_data)} faces)"
    return 0, ""


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def analyze_eyes(image_path: str) -> Tuple[float, str]:

    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
    except (ImportError, AttributeError) as e:
        return 0, f"Mediapipe not available: {str(e)}"

    img = cv2.imread(image_path)
    if img is None:
        return 0, "No Image"

    h, w = img.shape[:2]
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(rgb_img)
        if not results.multi_face_landmarks:
            return 0, "No Face Detected"

        total_score = 0.0
        all_findings = []
        all_face_data = []

        for fi, face_landmarks in enumerate(results.multi_face_landmarks):
            eye_data = _get_eye_data(face_landmarks, img, h, w)
            if eye_data is None:
                all_findings.append(f"Face {fi}: eyes too small/off-frame")
                continue

            # ── Detect glints ──
            left_glints = _detect_glints(eye_data["left_crop"])
            right_glints = _detect_glints(eye_data["right_crop"])

            face_record = {
                "left_glints": left_glints,
                "right_glints": right_glints,
            }
            all_face_data.append(face_record)

            # ── 7.3: Morphology per eye ──
            l_morph_score, l_morph_desc = _score_morphology(left_glints)
            r_morph_score, r_morph_desc = _score_morphology(right_glints)
            morph_score = (l_morph_score + r_morph_score) / 2.0
            total_score += morph_score

            if l_morph_desc or r_morph_desc:
                all_findings.append(
                    f"Face {fi} morph: L=[{l_morph_desc}] R=[{r_morph_desc}]"
                )

            # ── 7.4: Cross-eye consistency ──
            cross_score, cross_desc = _compare_eyes(left_glints, right_glints)
            total_score += cross_score
            if cross_desc:
                all_findings.append(f"Face {fi}: {cross_desc}")

            # ── 7.5: Edge sharpness ──
            l_sharp, l_sdesc = _score_edge_sharpness(eye_data["left_crop"], left_glints)
            r_sharp, r_sdesc = _score_edge_sharpness(eye_data["right_crop"], right_glints)
            total_score += (l_sharp + r_sharp) / 2.0
            for sd in (l_sdesc, r_sdesc):
                if sd:
                    all_findings.append(f"Face {fi}: {sd}")

        # ── Multi-face cross-consistency ──
        if len(all_face_data) >= 2:
            xf_score, xf_desc = _cross_face_consistency(all_face_data)
            total_score += xf_score
            if xf_desc:
                all_findings.append(xf_desc)

        # Clamp to scoring range
        final_score = int(max(-50, min(30, total_score)))

        description = "; ".join(all_findings) if all_findings else "Eye analysis complete"
        return final_score, description


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        score, desc = analyze_eyes(sys.argv[1])
        print(f"\n=== Layer 7: Corneal Reflection Forensics v2.0 ===")
        print(f"Score: {score}")
        print(f"Description: {desc}")
    else:
        print("Usage: python layer_7_eyes.py <image_path>")
        print("Requires: mediapipe (pip install mediapipe)")
