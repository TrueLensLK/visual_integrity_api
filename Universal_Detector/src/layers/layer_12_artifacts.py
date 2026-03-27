"""
Layer 12: GAN & Diffusion Artifacts (Spatial & Co-occurrence)
FOCUS: Checkerboard patterns from Transposed Convolutions & Upsampling Grids.

While Layer 6 looks at the Frequency Domain (FFT), Layer 12 looks at the
SPATIAL Domain and Pixel Co-occurrence logic to find:
1. Checkerboard Artifacts (periodic pixel intensity changes).
2. Grid-like dependencies in pixel neighborhoods (Autocorrelation).
3. Resampling Traces (interpolation artifacts).

VERSION: 2.0 - FALSE POSITIVE REDUCTION
- Higher thresholds to avoid flagging natural textures
- Multi-scale analysis to distinguish AI grids from natural patterns
- JPEG/compression-aware detection
"""

import cv2
import numpy as np
from scipy.signal import convolve2d

def calculate_spatial_autocorrelation(img_gray: np.ndarray) -> dict:
    """
    Detects periodic repetitions in the spatial domain using 
    fast autocorrelation via FFT.
    
    Real photos have a smooth central peak in autocorrelation.
    GANs/Diffusion models often have secondary peaks (echoes) 
    due to fixed-grid upsampling.
    
    Returns dict with echo score and analysis details.
    """
    # 1. Calculate periodic component using FFT -> IFFT
    f = np.fft.fft2(img_gray)
    p = np.abs(f)**2
    # Autocorrelation is IFFT of Power Spectrum
    ac = np.real(np.fft.ifft2(p))
    ac = np.fft.fftshift(ac)
    
    # Normalize
    center_val = ac[ac.shape[0]//2, ac.shape[1]//2]
    if center_val > 0:
        ac /= center_val
    
    # 2. Analyze the shape of the central peak
    h, w = ac.shape
    cy, cx = h//2, w//2
    
    # Look for "Periodic Echoes" (secondary peaks) outside the main lobe
    # Mask out the center (DC component + immediate correlation)
    # Use larger mask to avoid false positives from natural correlation
    mask_r = 10  # Increased from 5 to reduce FP
    ac_masked = ac.copy()
    ac_masked[cy-mask_r:cy+mask_r, cx-mask_r:cx+mask_r] = 0
    
    # Find max peak in the rest of the map
    max_echo = np.max(ac_masked)
    
    # Count significant echoes (for AI, there are usually multiple regular echoes)
    echo_threshold = 0.15
    echo_count = np.sum(ac_masked > echo_threshold)
    
    # Check for REGULAR GRID pattern (AI artifact signature)
    # AI upsampling creates echoes at regular intervals (e.g., every 2, 4, or 8 pixels)
    # Natural textures have irregular patterns
    
    # Sample along horizontal and vertical axes
    h_slice = ac_masked[cy, cx+mask_r:min(cx+100, w)]
    v_slice = ac_masked[cy+mask_r:min(cy+100, h), cx]
    
    # Measure regularity of peaks
    h_peaks = np.where(h_slice > 0.1)[0]
    v_peaks = np.where(v_slice > 0.1)[0]
    
    is_regular_grid = False
    if len(h_peaks) >= 3:
        h_diffs = np.diff(h_peaks)
        if len(h_diffs) > 0 and np.std(h_diffs) < 2:  # Very regular spacing
            is_regular_grid = True
    if len(v_peaks) >= 3:
        v_diffs = np.diff(v_peaks)
        if len(v_diffs) > 0 and np.std(v_diffs) < 2:
            is_regular_grid = True
    
    return {
        "max_echo": float(max_echo),
        "echo_count": int(echo_count),
        "is_regular_grid": is_regular_grid
    }

def detect_checkerboard_spatial(img_gray: np.ndarray) -> float:
    """
    Uses a Laplacian-like kernel specifically tuned for 
    high-frequency checkerboard toggles (0-255-0-255).
    """
    # Kernel to find 1-pixel alternating patterns
    k1 = np.array([[ 1, -1],
                   [-1,  1]])
    
    conv = convolve2d(img_gray, k1, mode='valid')
    
    # High variance in this convolution map means strong local checkerboarding
    score = np.var(conv)
    return float(score)

def analyze_artifacts(file_path: str, is_jpeg: bool = False) -> dict:
    """
    Main artifact analysis with FALSE POSITIVE REDUCTION.
    
    Key changes from v1.0:
    - Much higher thresholds (real photos often have natural textures)
    - Require REGULAR GRID pattern, not just high echo
    - Multi-factor decision (echo + regularity + checkerboard)
    - v2.1: JPEG-aware - ignores 8x8 block boundary grids from JPEG compression
    """
    try:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"score": 0, "description": "Image load failed"}

        # Resize for speed if huge
        h, w = img.shape
        if h > 512 and w > 512:
            cy, cx = h//2, w//2
            img = img[cy-256:cy+256, cx-256:cx+256]
            
        img_f = img.astype(np.float32) / 255.0

        # Analysis 1: Autocorrelation (Grid Echoes)
        ac_result = calculate_spatial_autocorrelation(img_f)
        echo_score = ac_result["max_echo"]
        echo_count = ac_result["echo_count"]
        is_regular_grid = ac_result["is_regular_grid"]
        
        # Analysis 2: Checkerboard Variance
        checker_var = detect_checkerboard_spatial(img_f)
        
        score = 0
        desc = []
        
        # ================================================================
        # JPEG COMPRESSION HANDLING (v2.1)
        # ================================================================
        # JPEG uses 8x8 DCT blocks, which creates natural "grids" at 8-pixel
        # intervals. This is NOT an AI artifact - it's compression artifact.
        # AI upsampling typically creates 2x, 4x grids (not 8x8).
        # For JPEG images, we HEAVILY discount grid patterns.
        # ================================================================
        
        jpeg_discount = 0.7 if is_jpeg else 1.0  # 70% discount for JPEG grid signals
        
        # ================================================================
        # CONSERVATIVE SCORING LOGIC (v2.1 - JPEG aware)
        # ================================================================
        # Philosophy: Only flag as AI if we see CLEAR synthetic patterns
        #             Natural textures (brick, fabric, foliage) can have echoes
        #             but they're not REGULAR GRIDS
        #             JPEG compression creates 8x8 block grids - ignore those
        # ================================================================
        
        # Thresholds - MUCH higher to avoid false positives
        # Real photos with textures can have echoes up to 0.4
        # For JPEG, even 0.9+ echoes can be compression artifacts
        ECHO_SUSPICIOUS = 0.45 if not is_jpeg else 0.85  # Higher threshold for JPEG
        ECHO_STRONG = 0.60 if not is_jpeg else 0.95
        CHECKER_SUSPICIOUS = 0.08  # Was 0.05/0.10
        CHECKER_STRONG = 0.15
        
        # Decision logic: Need MULTIPLE indicators or VERY strong single indicator
        ai_indicators = 0
        
        # For JPEG: Regular grid patterns are expected from 8x8 blocks, so ignore them
        if is_jpeg:
            is_regular_grid = False  # JPEG naturally has regular 8x8 grids - not AI
            desc.append("JPEG grid ignored")
        
        # Check 1: High echo with regular grid pattern (strongest AI signal)
        if echo_score > ECHO_SUSPICIOUS and is_regular_grid:
            ai_indicators += 2
            desc.append(f"Regular grid pattern (echo={echo_score:.2f})")
        elif echo_score > ECHO_STRONG and not is_jpeg:
            # Very high echo even without regularity check (not for JPEG)
            ai_indicators += 1
            desc.append(f"High autocorrelation (echo={echo_score:.2f})")
        
        # Check 2: Checkerboard patterns (still valid for JPEG - AI checkerboard is different from JPEG)
        if checker_var > CHECKER_STRONG:
            ai_indicators += 2
            desc.append(f"Strong checkerboard (var={checker_var:.3f})")
        elif checker_var > CHECKER_SUSPICIOUS:
            ai_indicators += 1
            desc.append(f"Mild checkerboard (var={checker_var:.3f})")
        
        # Check 3: Many echo peaks (skip for JPEG - compression creates many echoes)
        if echo_count > 50 and is_regular_grid and not is_jpeg:
            ai_indicators += 1
            desc.append(f"Multiple regular echoes ({echo_count})")
        
        # ================================================================
        # SCORING: Require strong evidence before flagging
        # ================================================================
        if ai_indicators >= 3:
            # Strong multi-factor evidence
            score = -45
        elif ai_indicators == 2:
            # Moderate evidence
            score = -25
        elif ai_indicators == 1:
            # Weak/single indicator - could be false positive
            score = -10
        else:
            score = 0
            desc.append("No synthesis artifacts detected")
        
        # Clamp score
        final_score = max(-50, min(0, score))
        
        return {
            "score": final_score,
            "description": ", ".join(desc) if desc else "Clean",
            "details": {
                "autocorrelation_echo": round(echo_score, 4),
                "echo_count": echo_count,
                "is_regular_grid": is_regular_grid,
                "checkerboard_var": round(checker_var, 4),
                "ai_indicators": ai_indicators
            }
        }
        
    except Exception as e:
        return {"score": 0, "description": "Error", "details": str(e)}
