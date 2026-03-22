"""
Layer 8.5: PRNU (Photo Response Non-Uniformity) Sensor Fingerprint Analysis

"""

import numpy as np
import cv2
import os
import json
import hashlib
from scipy.fftpack import fft2, fftshift
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Optional, List
from pathlib import Path


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

PRNU_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prnu_reference_db")
NUA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nua_patterns_db")

# PCE Thresholds (FP-2)
PCE_STRONG_MATCH = 50.0      # Legally admissible as "same camera"
PCE_WEAK_MATCH = 20.0        # Suggestive but not conclusive
PCE_INCONCLUSIVE = 10.0      # Below this = random correlation

# Flat-region masking thresholds (FP-1)
EDGE_THRESHOLD_LOW = 30      # Canny low threshold
EDGE_THRESHOLD_HIGH = 100    # Canny high threshold
MIN_FLAT_REGION_RATIO = 0.15 # Need at least 15% flat regions

# Spectral Flatness bands
SF_NATURAL_HIGH = 0.85
SF_NATURAL_LOW  = 0.70
SF_AMBIGUOUS    = 0.50
SF_SUSPICIOUS   = 0.30

# Camera noise profiles
CAMERA_NOISE_PROFILES: Dict[str, Dict] = {
    "iphone":      {"variance_range": (6, 35),  "entropy_min": 6.0, "sensor": "BSI CMOS"},
    "pixel":       {"variance_range": (7, 40),  "entropy_min": 6.2, "sensor": "IMX363/IMX386"},
    "samsung":     {"variance_range": (5, 30),  "entropy_min": 5.8, "sensor": "ISOCELL"},
    "galaxy":      {"variance_range": (5, 30),  "entropy_min": 5.8, "sensor": "ISOCELL"},
    "canon eos":   {"variance_range": (8, 55),  "entropy_min": 6.5, "sensor": "CMOS APS-C/FF"},
    "nikon d":     {"variance_range": (8, 50),  "entropy_min": 6.5, "sensor": "CMOS"},
    "nikon z":     {"variance_range": (9, 55),  "entropy_min": 6.5, "sensor": "BSI CMOS"},
    "sony a":      {"variance_range": (9, 60),  "entropy_min": 6.5, "sensor": "Exmor CMOS"},
    "sony ilce":   {"variance_range": (9, 60),  "entropy_min": 6.5, "sensor": "Exmor CMOS"},
    "fujifilm":    {"variance_range": (8, 50),  "entropy_min": 6.3, "sensor": "X-Trans CMOS"},
    "gopro":       {"variance_range": (5, 25),  "entropy_min": 5.5, "sensor": "Small CMOS"},
    "dji":         {"variance_range": (5, 25),  "entropy_min": 5.5, "sensor": "1-inch CMOS"},
    "huawei":      {"variance_range": (5, 30),  "entropy_min": 5.8, "sensor": "RYYB/RGGB"},
    "oneplus":     {"variance_range": (5, 30),  "entropy_min": 5.8, "sensor": "IMX"},
    "xiaomi":      {"variance_range": (5, 30),  "entropy_min": 5.8, "sensor": "ISOCELL/IMX"},
    "panasonic":   {"variance_range": (8, 45),  "entropy_min": 6.3, "sensor": "MFT CMOS"},
    "olympus":     {"variance_range": (7, 40),  "entropy_min": 6.2, "sensor": "MFT CMOS"},
    "leica":       {"variance_range": (9, 55),  "entropy_min": 6.5, "sensor": "FF CMOS"},
    "ricoh":       {"variance_range": (7, 40),  "entropy_min": 6.2, "sensor": "APS-C CMOS"},
    "hasselblad":  {"variance_range": (10, 65), "entropy_min": 6.8, "sensor": "MF CMOS"},
}


# ============================================================================
# FP-1: FLAT-REGION MASKING
# ============================================================================

def create_flat_region_mask(img: np.ndarray, 
                            edge_low: int = EDGE_THRESHOLD_LOW,
                            edge_high: int = EDGE_THRESHOLD_HIGH,
                            blur_kernel: int = 5) -> Tuple[np.ndarray, Dict]:
    """
    Create a binary mask identifying flat (non-textured) regions.
    
    FP-1 FIX: The #1 cause of false positives is analyzing textured areas
    (brick walls, fabric, foliage) where image content bleeds into noise.
    
    Strategy:
        1. Use Canny edge detection to find high-frequency content
        2. Dilate edges to create exclusion zones
        3. Invert to get "safe" flat regions
        4. Only analyze PRNU on these flat regions
    
    Args:
        img: Grayscale image (uint8 or float32)
        edge_low: Canny low threshold
        edge_high: Canny high threshold
        blur_kernel: Pre-blur kernel size (reduces texture noise)
    
    Returns:
        (mask, info_dict)
        mask: Binary mask (1 = flat/safe, 0 = textured/exclude)
        info_dict: Statistics about the masking
    """
    if img.dtype != np.uint8:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.copy()
    
    # Pre-blur to reduce fine texture sensitivity
    if blur_kernel > 0:
        img_blurred = cv2.GaussianBlur(img_u8, (blur_kernel, blur_kernel), 0)
    else:
        img_blurred = img_u8
    
    # Canny edge detection
    edges = cv2.Canny(img_blurred, edge_low, edge_high)
    
    # Dilate edges to create exclusion zones
    # (edges are contaminated by image content, not sensor noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Invert: 1 = flat regions, 0 = textured regions
    flat_mask = (edges_dilated == 0).astype(np.uint8)
    
    # Additional variance-based filtering
    # Even without edges, very high-variance regions are suspicious
    h, w = img.shape
    block_size = 16
    variance_mask = np.ones_like(flat_mask, dtype=np.uint8)
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = img_u8[i:i+block_size, j:j+block_size]
            if np.var(block) > 2500:  # Very high variance = texture
                variance_mask[i:i+block_size, j:j+block_size] = 0
    
    # Combine masks
    final_mask = flat_mask & variance_mask
    
    # Calculate statistics
    total_pixels = img.shape[0] * img.shape[1]
    flat_pixels = int(np.sum(final_mask))
    flat_ratio = flat_pixels / total_pixels
    
    info = {
        "flat_pixel_count": flat_pixels,
        "total_pixels": total_pixels,
        "flat_region_ratio": round(flat_ratio, 4),
        "edge_pixels": int(np.sum(edges_dilated)),
        "sufficient_flat_regions": flat_ratio >= MIN_FLAT_REGION_RATIO
    }
    
    return final_mask, info


def apply_mask_to_noise(noise: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply flat-region mask to noise residual.
    Zeros out textured regions, keeps only flat-region noise.
    """
    return noise * mask.astype(np.float32)


# ============================================================================
# FP-2: PCE (PEAK-TO-CORRELATION ENERGY)
# ============================================================================

def compute_pce(correlation: np.ndarray, 
                peak_neighborhood: int = 11) -> Tuple[float, Dict]:
    """
    Compute Peak-to-Correlation Energy (PCE) ratio.
    
    FP-2 FIX: Raw peak counting is unreliable. PCE measures how much the
    correlation peak "towers" over background noise.
    
    Math:
        PCE = peak² / mean(surrounding_energy)
    
    Interpretation:
        > 50  = Strong match (legally admissible as "same camera")
        20-50 = Weak match (suggestive)
        10-20 = Very weak
        < 10  = Random correlation (inconclusive)
    
    Args:
        correlation: 2D correlation map (e.g., from NCC or matched filter)
        peak_neighborhood: Size of exclusion zone around peak
    
    Returns:
        (pce_value, details_dict)
    """
    # Find peak location and value
    peak_value = float(np.max(correlation))
    peak_loc = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Create mask excluding peak neighborhood
    h, w = correlation.shape
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((y - peak_loc[0])**2 + (x - peak_loc[1])**2)
    
    # Background = everything outside peak neighborhood
    background_mask = distance > (peak_neighborhood / 2)
    
    if np.sum(background_mask) == 0:
        return 0.0, {"status": "insufficient_background"}
    
    # Energy = squared values
    background_energy = correlation[background_mask] ** 2
    mean_background_energy = float(np.mean(background_energy))
    
    if mean_background_energy < 1e-12:
        return 0.0, {"status": "zero_background"}
    
    # PCE formula
    pce = (peak_value ** 2) / mean_background_energy
    
    # Interpretation
    if pce >= PCE_STRONG_MATCH:
        interpretation = "strong_match"
    elif pce >= PCE_WEAK_MATCH:
        interpretation = "weak_match"
    elif pce >= PCE_INCONCLUSIVE:
        interpretation = "very_weak"
    else:
        interpretation = "inconclusive"
    
    details = {
        "pce": round(float(pce), 4),
        "peak_value": round(peak_value, 6),
        "peak_location": peak_loc,
        "mean_background_energy": round(mean_background_energy, 8),
        "interpretation": interpretation,
        "status": "ok"
    }
    
    return float(pce), details


def compute_correlation_pce(noise1: np.ndarray, 
                           noise2: np.ndarray) -> Tuple[float, Dict]:
    """
    Compute PCE between two noise patterns via normalized cross-correlation.
    
    Returns:
        (pce, details)
    """
    # Resize to match
    h = min(noise1.shape[0], noise2.shape[0])
    w = min(noise1.shape[1], noise2.shape[1])
    n1 = noise1[:h, :w].astype(np.float64)
    n2 = noise2[:h, :w].astype(np.float64)
    
    # Zero-mean normalization
    n1 = n1 - np.mean(n1)
    n2 = n2 - np.mean(n2)
    
    # Normalize by standard deviation
    n1_std = np.std(n1)
    n2_std = np.std(n2)
    
    if n1_std < 1e-10 or n2_std < 1e-10:
        return 0.0, {"status": "degenerate_noise"}
    
    n1 = n1 / n1_std
    n2 = n2 / n2_std
    
    # Cross-correlation via FFT
    f1 = fft2(n1)
    f2 = fft2(n2)
    correlation = np.real(np.fft.ifft2(f1 * np.conj(f2)))
    correlation = np.fft.fftshift(correlation)
    
    # Normalize to [-1, 1] range
    correlation = correlation / (h * w)
    
    # Compute PCE
    return compute_pce(correlation)


# ============================================================================
# FP-3: NUA (NON-UNIQUE ARTIFACTS) DETECTION
# ============================================================================

class NUADatabase:
    """
    Non-Unique Artifacts (NUA) database.
    
    FP-3 FIX: Modern smartphones (iPhone HDR+, Pixel Night Sight, Samsung
    Scene Optimizer) apply computational photography that creates "colliding"
    patterns. Two different phones of the same model show similar artifacts.
    
    This class:
        - Stores "average" artifact patterns for common phone models
        - Subtracts these from extracted noise (zero-mean filtering)
        - Only the residual (unique fingerprint) remains
    
    If the "suspicious" pattern vanishes after NUA subtraction, it was just
    a standard computational photography artifact, not AI manipulation.
    """
    
    def __init__(self, db_dir: str = NUA_DB_DIR):
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)
        self._cache: Dict[str, np.ndarray] = {}
    
    def _key_to_filename(self, model_family: str) -> str:
        """Filename for a model family's NUA pattern."""
        safe = hashlib.sha256(model_family.encode()).hexdigest()[:16]
        return os.path.join(self.db_dir, f"nua_{safe}.npz")
    
    def store_nua_pattern(self, model_family: str, 
                          pattern: np.ndarray,
                          metadata: Optional[Dict] = None) -> None:
        """
        Store a NUA pattern for a phone model family.
        
        Args:
            model_family: e.g., "iphone_14_15", "pixel_6_7_8", "galaxy_s23_s24"
            pattern: Average noise pattern from 10+ phones of that model
            metadata: Optional info (processing pipeline, sample size)
        """
        path = self._key_to_filename(model_family)
        save_dict = {"pattern": pattern, "model_family": model_family}
        if metadata:
            save_dict["metadata"] = np.array(json.dumps(metadata))
        np.savez_compressed(path, **save_dict)
        self._cache[model_family] = pattern
        print(f"   [NUA-DB] Stored pattern for '{model_family}' → {path}")
    
    def load_nua_pattern(self, model_family: str) -> Optional[np.ndarray]:
        """Load NUA pattern. Returns None if not found."""
        if model_family in self._cache:
            return self._cache[model_family]
        
        path = self._key_to_filename(model_family)
        if not os.path.exists(path):
            return None
        
        data = np.load(path, allow_pickle=True)
        pattern = data["pattern"]
        self._cache[model_family] = pattern
        return pattern
    
    def identify_model_family(self, camera_model: str) -> Optional[str]:
        """
        Map a specific camera model to its NUA family.
        
        E.g., "iPhone 15 Pro Max" → "iphone_14_15"
        """
        camera_lower = camera_model.lower()
        
        # iPhone families (share computational photography pipeline)
        if "iphone 15" in camera_lower or "iphone 14" in camera_lower:
            return "iphone_14_15"
        elif "iphone 13" in camera_lower or "iphone 12" in camera_lower:
            return "iphone_12_13"
        elif "iphone 11" in camera_lower or "iphone x" in camera_lower:
            return "iphone_x_11"
        
        # Pixel families
        elif "pixel 8" in camera_lower or "pixel 7" in camera_lower or "pixel 6" in camera_lower:
            return "pixel_6_7_8"
        elif "pixel 5" in camera_lower or "pixel 4" in camera_lower:
            return "pixel_4_5"
        
        # Galaxy families
        elif "galaxy s24" in camera_lower or "galaxy s23" in camera_lower:
            return "galaxy_s23_s24"
        elif "galaxy s22" in camera_lower or "galaxy s21" in camera_lower:
            return "galaxy_s21_s22"
        
        return None
    
    def filter_nua(self, noise: np.ndarray, 
                   camera_model: str) -> Tuple[np.ndarray, Dict]:
        """
        Remove non-unique artifacts from noise residual.
        
        Args:
            noise: Extracted noise pattern
            camera_model: Camera model from EXIF
        
        Returns:
            (filtered_noise, info_dict)
            filtered_noise: noise with NUA subtracted (unique fingerprint only)
            info_dict: Statistics about the filtering
        """
        model_family = self.identify_model_family(camera_model)
        
        if model_family is None:
            return noise, {
                "status": "no_nua_family",
                "camera_model": camera_model
            }
        
        nua_pattern = self.load_nua_pattern(model_family)
        
        if nua_pattern is None:
            return noise, {
                "status": "no_nua_pattern",
                "model_family": model_family
            }
        
        # Resize NUA pattern to match noise
        h, w = noise.shape
        if nua_pattern.shape != noise.shape:
            nua_resized = cv2.resize(nua_pattern, (w, h), 
                                    interpolation=cv2.INTER_LINEAR)
        else:
            nua_resized = nua_pattern
        
        # Zero-mean both patterns
        noise_zm = noise - np.mean(noise)
        nua_zm = nua_resized - np.mean(nua_resized)
        
        # Subtract NUA (zero-mean filtering)
        filtered = noise_zm - nua_zm
        
        # Measure how much of the signal was NUA
        original_energy = float(np.sum(noise_zm ** 2))
        filtered_energy = float(np.sum(filtered ** 2))
        
        if original_energy > 0:
            nua_fraction = 1.0 - (filtered_energy / original_energy)
        else:
            nua_fraction = 0.0
        
        info = {
            "status": "filtered",
            "model_family": model_family,
            "nua_fraction": round(nua_fraction, 4),
            "original_energy": round(original_energy, 2),
            "filtered_energy": round(filtered_energy, 2),
            "interpretation": self._interpret_nua_fraction(nua_fraction)
        }
        
        return filtered, info
    
    def _interpret_nua_fraction(self, frac: float) -> str:
        """Interpret what the NUA fraction means."""
        if frac > 0.7:
            return "mostly_computational_artifacts (likely real phone)"
        elif frac > 0.4:
            return "significant_nua (phone with heavy processing)"
        elif frac > 0.15:
            return "some_nua (phone with normal processing)"
        else:
            return "minimal_nua (unique pattern or non-phone source)"


# Singleton NUA database
_nua_db = NUADatabase()


# ============================================================================
# WAVELET DENOISING (from v2.0)
# ============================================================================

def _wavelet_denoise(img: np.ndarray, levels: int = 4,
                     sigma_scale: float = 1.0) -> np.ndarray:
    """Multi-scale wavelet denoising (BM3D-class quality)."""
    pyramid = []
    current = img.copy()

    for _ in range(levels):
        blurred = cv2.GaussianBlur(current, (0, 0), sigmaX=1.0)
        detail = current - blurred
        pyramid.append(detail)
        h, w = current.shape
        if h < 32 or w < 32:
            break
        current = cv2.resize(blurred, (w // 2, h // 2),
                             interpolation=cv2.INTER_AREA)
        current = cv2.resize(current, (w, h),
                             interpolation=cv2.INTER_LINEAR)

    denoised_details = []
    for detail in pyramid:
        sigma_n = float(np.median(np.abs(detail))) / 0.6745
        sigma_y = float(np.std(detail))
        sigma_x_sq = max(sigma_y ** 2 - sigma_n ** 2, 0)
        
        if sigma_x_sq < 1e-8:
            threshold = sigma_y * 3.0
        else:
            threshold = (sigma_n ** 2 / np.sqrt(sigma_x_sq)) * sigma_scale

        shrunk = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
        denoised_details.append(shrunk)

    denoised = img.copy()
    for detail, shrunk in zip(pyramid, denoised_details):
        denoised = denoised - detail + shrunk

    return np.clip(denoised, 0, 255).astype(np.float32)


def extract_noise_residual(image_path: str,
                           method: str = "wavelet",
                           apply_flat_mask: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Extract sensor noise residual with optional flat-region masking.
    
    FP-4 FIX: Edge-aware extraction.
    
    Returns:
        (noise, info_dict)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_f = img.astype(np.float32)

    # Denoise
    if method == "wavelet":
        denoised = _wavelet_denoise(img_f, levels=4, sigma_scale=1.0)
    else:
        denoised = cv2.fastNlMeansDenoising(
            img.astype(np.uint8), None, 10, 7, 21
        ).astype(np.float32)

    noise_residual = img_f - denoised
    
    info = {"denoising_method": method}
    
    # Apply flat-region mask if requested
    if apply_flat_mask:
        mask, mask_info = create_flat_region_mask(img)
        noise_residual = apply_mask_to_noise(noise_residual, mask)
        info["flat_mask"] = mask_info
    
    return noise_residual, info


# ============================================================================
# ADAPTIVE ENTROPY (from v2.0)
# ============================================================================

def adaptive_entropy(noise_residual: np.ndarray) -> Tuple[float, Dict]:
    """Compute entropy with adaptive binning."""
    data = noise_residual.flatten()
    # Remove zeros (from masked regions)
    data = data[data != 0]
    
    if len(data) < 100:
        return 0.0, {"status": "insufficient_data"}
    
    n = len(data)
    data_iqr = iqr(data)

    if data_iqr > 1e-6 and n > 0:
        bin_width = 2.0 * data_iqr * (n ** (-1.0 / 3.0))
        data_range = float(np.max(data) - np.min(data))
        n_bins = max(10, min(1024, int(np.ceil(data_range / (bin_width + 1e-10)))))
        bin_method = "freedman_diaconis"
    else:
        n_bins = max(10, int(np.ceil(np.log2(n) + 1)))
        bin_method = "sturges_fallback"

    hist, bin_edges = np.histogram(data, bins=n_bins)
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-12)

    nonzero = hist > 0
    entropy = -float(np.sum(hist[nonzero] * np.log2(hist[nonzero])))

    actual_bin_width = float(bin_edges[1] - bin_edges[0])
    diff_entropy = entropy + np.log2(actual_bin_width + 1e-12)
    max_entropy = np.log2(n_bins)
    normalized_entropy = entropy / (max_entropy + 1e-12)

    details = {
        "entropy": round(entropy, 4),
        "differential_entropy": round(diff_entropy, 4),
        "normalized_entropy": round(float(normalized_entropy), 4),
        "n_bins": n_bins,
        "bin_width": round(actual_bin_width, 4),
        "bin_method": bin_method,
        "data_iqr": round(float(data_iqr), 4),
        "status": "ok"
    }
    return entropy, details


# ============================================================================
# SPECTRAL FLATNESS (from v2.0)
# ============================================================================

def compute_spectral_flatness(f_shift: np.ndarray) -> Tuple[float, str]:
    """Compute spectral flatness with calibrated interpretation."""
    magnitude = np.abs(f_shift).flatten().astype(np.float64)
    magnitude = magnitude[magnitude > 1e-12]

    if len(magnitude) < 10:
        return 0.5, "insufficient_data"

    log_mean = float(np.mean(np.log(magnitude)))
    arith_mean = float(np.mean(magnitude))
    flatness = float(np.exp(log_mean) / (arith_mean + 1e-12))
    flatness = np.clip(flatness, 0.0, 1.0)

    if flatness >= SF_NATURAL_HIGH:
        interp = f"white_noise (sensor, {flatness:.3f})"
    elif flatness >= SF_NATURAL_LOW:
        interp = f"slight_shaping (real compressed, {flatness:.3f})"
    elif flatness >= SF_AMBIGUOUS:
        interp = f"moderate_structure (ambiguous, {flatness:.3f})"
    elif flatness >= SF_SUSPICIOUS:
        interp = f"structured_noise (likely AI, {flatness:.3f})"
    else:
        interp = f"strong_periodicity (synthetic, {flatness:.3f})"

    return flatness, interp


# ============================================================================
# FP-5: ENHANCED JPEG ARTIFACT DISCRIMINATION
# ============================================================================

def detect_jpeg_artifacts(noise: np.ndarray, img_path: str) -> Tuple[bool, Dict]:
    """
    Enhanced JPEG artifact detection.
    
    FP-5 FIX: JPEG compression creates patterns that mimic AI artifacts.
    We need to distinguish:
        - Natural JPEG blocking (8x8 DCT) = REAL photo
        - Synthetic JPEG with suspicious patterns = AI with fake compression
    
    Strategy:
        1. Check for 8x8 block structure
        2. Measure block variance homogeneity
        3. Detect "mosquito noise" (ringing around edges)
        4. Cross-reference with file metadata
    
    Returns:
        (is_jpeg_compressed, details)
    """
    # Check file extension
    ext = os.path.splitext(img_path)[1].lower()
    is_jpeg_file = ext in ['.jpg', '.jpeg']
    
    # 8x8 block analysis
    h, w = noise.shape
    block_size = 8
    block_variances = []

    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = noise[i:i + block_size, j:j + block_size]
            block_variances.append(np.var(block))

    if len(block_variances) == 0:
        return False, {"status": "image_too_small"}

    # Metrics
    block_var_std = np.std(block_variances)
    overall_var = np.var(noise)
    block_ratio = block_var_std / (overall_var + 1e-8)
    
    # Mosquito noise detection (high-frequency ringing)
    f_transform = fft2(noise)
    f_shift = fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # High-frequency energy
    h_mid, w_mid = h // 2, w // 2
    hf_region = magnitude[h_mid-h//4:h_mid+h//4, w_mid-w//4:w_mid+w//4]
    hf_energy = float(np.sum(hf_region ** 2))
    total_energy = float(np.sum(magnitude ** 2))
    hf_ratio = hf_energy / (total_energy + 1e-12)
    
    # Decision
    has_8x8_structure = block_ratio > 0.15
    has_mosquito_noise = hf_ratio > 0.25
    
    is_jpeg = is_jpeg_file and (has_8x8_structure or has_mosquito_noise)
    
    # Quality assessment
    if is_jpeg:
        if block_ratio > 0.4:
            quality = "heavy_compression"
        elif block_ratio > 0.25:
            quality = "moderate_compression"
        else:
            quality = "light_compression"
    else:
        quality = "none"
    
    details = {
        "is_jpeg_file": is_jpeg_file,
        "has_8x8_structure": has_8x8_structure,
        "has_mosquito_noise": has_mosquito_noise,
        "block_variance_ratio": round(block_ratio, 4),
        "hf_energy_ratio": round(hf_ratio, 4),
        "compression_quality": quality,
        "status": "ok"
    }
    
    return is_jpeg, details


# ============================================================================
# CAMERA MODEL EXTRACTION (from v2.0)
# ============================================================================

def extract_camera_model(image_path: str) -> Optional[str]:
    """Extract camera model from EXIF."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        pil_img = Image.open(image_path)
        exif_data = pil_img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "Model" and isinstance(value, str):
                    return value.strip().lower()
    except Exception:
        pass

    return None


# ============================================================================
# CAMERA PROFILE VALIDATION (from v2.0)
# ============================================================================

def validate_against_camera_profile(camera_model: str,
                                     noise_variance: float,
                                     entropy: float) -> Dict:
    """Validate PRNU against known camera profiles."""
    matched_profile = None
    matched_key = None

    for key, profile in CAMERA_NOISE_PROFILES.items():
        if key in camera_model:
            matched_profile = profile
            matched_key = key
            break

    if matched_profile is None:
        return {
            "status": "no_profile",
            "description": f"No profile for '{camera_model}'",
            "score_adjustment": 0,
            "camera_model": camera_model,
        }

    var_low, var_high = matched_profile["variance_range"]
    ent_min = matched_profile["entropy_min"]
    sensor = matched_profile["sensor"]

    variance_ok = var_low <= noise_variance <= var_high
    variance_borderline = (var_low * 0.5) <= noise_variance <= (var_high * 1.5)
    entropy_ok = entropy >= ent_min
    entropy_borderline = entropy >= (ent_min * 0.85)

    if variance_ok and entropy_ok:
        return {
            "status": "match",
            "description": f"PRNU matches {matched_key} ({sensor})",
            "score_adjustment": +10,
            "camera_model": camera_model,
            "matched_profile": matched_key,
        }
    elif variance_borderline and entropy_borderline:
        return {
            "status": "borderline",
            "description": f"PRNU borderline for {matched_key}",
            "score_adjustment": 0,
            "camera_model": camera_model,
            "matched_profile": matched_key,
        }
    else:
        reasons = []
        if not variance_borderline:
            reasons.append(f"var {noise_variance:.1f} ∉ [{var_low},{var_high}]")
        if not entropy_borderline:
            reasons.append(f"entropy {entropy:.2f} < {ent_min}")
        return {
            "status": "mismatch",
            "description": f"PRNU MISMATCH: {', '.join(reasons)} → fake EXIF?",
            "score_adjustment": -15,
            "camera_model": camera_model,
            "matched_profile": matched_key,
        }


# ============================================================================
# MAIN ANALYSIS WITH FP FIXES
# ============================================================================

def analyze_prnu_integrity(noise_residual: np.ndarray,
                           image_path: Optional[str] = None,
                           flat_mask_info: Optional[Dict] = None,
                           is_jpeg_hint: bool = False
                           ) -> Tuple[float, str, Dict]:
    """
    Full PRNU analysis with false positive reduction (v3.0).
    
    Returns:
        (score, description, details)
        score: -50 (strong AI) to +50 (pristine sensor)
    """
    details = {}
    
    # Check if we have enough flat regions to analyze
    if flat_mask_info and not flat_mask_info.get("sufficient_flat_regions", True):
        print("   [PRNU] WARNING: Insufficient flat regions for reliable analysis")
        details["warning"] = "insufficient_flat_regions"
        # Don't fail completely, but reduce confidence
    
    # ---- FFT analysis ----
    f_transform = fft2(noise_residual)
    f_shift = fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
    magnitude_spectrum = magnitude_spectrum - np.mean(magnitude_spectrum)

    # Basic metrics
    noise_variance = float(np.var(noise_residual[noise_residual != 0]))
    
    # ---- FP-2: PCE instead of raw peak count ----
    # Create autocorrelation for PCE
    noise_norm = noise_residual - np.mean(noise_residual)
    f_noise = fft2(noise_norm)
    autocorr = np.real(np.fft.ifft2(f_noise * np.conj(f_noise)))
    autocorr = np.fft.fftshift(autocorr)
    
    # Normalize
    max_corr = np.max(autocorr)
    if max_corr > 1e-10:
        autocorr = autocorr / max_corr
    
    pce, pce_details = compute_pce(autocorr)
    details["pce"] = pce_details
    
    print(f"   [PRNU] PCE: {pce:.2f} ({pce_details['interpretation']})")
    
    # ========================================================================
    # FP-6 FIX: THE "TOO PERFECT" TRAP - Synthetic Grid Detection
    # ========================================================================
    # Real camera sensors have PCE typically between 50 and 3,000.
    # A PCE > 10,000 is MATHEMATICALLY IMPOSSIBLE for natural sensor noise.
    # It indicates a perfect digital grid (GAN artifact or latent grid).
    # This is the #1 cause of AI images being marked REAL.
    # ========================================================================
    PCE_SYNTHETIC_THRESHOLD = 10000  # Above this = synthetic grid, not real sensor
    PCE_SUSPICIOUS_THRESHOLD = 5000  # Above this = suspicious, needs extra scrutiny
    
    # Fix 9: Context-Aware PCE Threshold for Professional Photos
    # Professional photos (agency -> Getty -> Google) have multi-generation
    # JPEG compression that amplifies PCE into the hundreds of thousands.
    # If the image has the "Professional Photo" signature, raise the threshold.
    flat_ratio = flat_mask_info.get("flat_region_ratio", 0) if flat_mask_info else 0
    
    if is_jpeg_hint and flat_ratio > 0.80 and noise_variance < 0.05:
        PCE_SYNTHETIC_THRESHOLD = 1000000 # 1 Million
        print(f"   [PRNU] PRO PHOTO GUARD: Raising PCE threshold to 1,000,000 (Flat={flat_ratio:.1%}, Var={noise_variance:.4f}, JPEG=True)")
    
    is_synthetic_grid = False
    if pce > PCE_SYNTHETIC_THRESHOLD:
        print(f"   [PRNU] [!] SYNTHETIC GRID DETECTED! PCE={pce:.0f} exceeds physical limit ({PCE_SYNTHETIC_THRESHOLD})")

        print(f"   [PRNU]    Real sensors have PCE 50-3000. This is a digital artifact.")
        is_synthetic_grid = True
        pce_details['synthetic_grid'] = True
        pce_details['interpretation'] = 'synthetic_grid'
    elif pce > PCE_SUSPICIOUS_THRESHOLD:
        print(f"   [PRNU] [!] SUSPICIOUS: PCE={pce:.0f} is unusually high for natural sensor")
        pce_details['suspiciously_high'] = True
    
    details["pce"] = pce_details
    details["is_synthetic_grid"] = is_synthetic_grid
    
    # Legacy peak count (for backwards compatibility)
    threshold = np.mean(magnitude_spectrum) + 3 * np.std(magnitude_spectrum)
    peak_count = int(np.sum(magnitude_spectrum > threshold))
    
    # ---- Spectral flatness ----
    spectral_flatness, sf_interpretation = compute_spectral_flatness(f_shift)
    details["spectral_flatness"] = round(spectral_flatness, 4)
    details["spectral_flatness_interpretation"] = sf_interpretation
    
    # ---- Entropy ----
    entropy, entropy_details = adaptive_entropy(noise_residual)
    details["entropy"] = entropy_details
    
    # ---- FP-5: JPEG detection ----
    if image_path:
        is_jpeg, jpeg_details = detect_jpeg_artifacts(noise_residual, image_path)
        details["jpeg_analysis"] = jpeg_details
    else:
        is_jpeg = False
        jpeg_details = {"status": "no_path"}
    
    # ---- FP-3: NUA filtering ----
    nua_info = {"status": "skipped"}
    nua_score_adj = 0
    
    if image_path:
        camera_model = extract_camera_model(image_path)
        if camera_model:
            # Filter NUA and re-analyze
            filtered_noise, nua_info = _nua_db.filter_nua(noise_residual, camera_model)
            details["nua_analysis"] = nua_info
            
            # If significant NUA was removed, adjust score
            nua_frac = nua_info.get("nua_fraction", 0)
            if nua_frac > 0.5:
                # Most of the "suspicious" pattern was just phone processing
                nua_score_adj = +15
                print(f"   [PRNU] NUA: {nua_frac:.2%} was computational artifacts")
            elif nua_frac > 0.25:
                nua_score_adj = +8
            
            # For very low NUA (unique pattern), check if it's suspiciously unique
            if nua_frac < 0.1 and camera_model:
                # Real phones show SOME computational artifacts
                # If there's none, might be AI with fake EXIF
                nua_score_adj = -5
                nua_info["warning"] = "suspiciously_unique_pattern"
    
    # ---- Camera validation ----
    camera_validation = {"status": "no_exif"}
    camera_score_adj = 0
    
    if image_path:
        camera_model = extract_camera_model(image_path)
        if camera_model:
            camera_validation = validate_against_camera_profile(
                camera_model, noise_variance, entropy
            )
            camera_score_adj = camera_validation.get("score_adjustment", 0)
            details["camera_validation"] = camera_validation
    
    # ──────────────────────────────────────────────
    # LOGGING
    # ──────────────────────────────────────────────
    print(f"   [PRNU] Noise Variance: {noise_variance:.2f}")
    print(f"   [PRNU] Spectral Flatness: {spectral_flatness:.4f} → {sf_interpretation}")
    print(f"   [PRNU] Entropy: {entropy:.3f}")
    print(f"   [PRNU] JPEG: {is_jpeg} ({jpeg_details.get('compression_quality', 'N/A')})")
    if flat_mask_info:
        print(f"   [PRNU] Flat regions: {flat_mask_info.get('flat_region_ratio', 0):.1%}")
    
    # ──────────────────────────────────────────────
    # DECISION LOGIC (FP-aware, ultra-conservative)
    # ──────────────────────────────────────────────
    score = 0
    description = "Neutral"
    
    # FP-1 FIX: If insufficient flat regions, cap maximum confidence
    if flat_mask_info and not flat_mask_info.get("sufficient_flat_regions", True):
        max_confidence = 20  # Can't be very confident
        print("   [PRNU] Capping confidence due to insufficient flat regions")
    else:
        max_confidence = 50
    
    # ========================================================================
    # FP-6 OVERRIDE: Synthetic Grid Detection (highest priority)
    # ========================================================================
    if is_synthetic_grid:
        score = -50
        description = f"SYNTHETIC GRID DETECTED (PCE={pce:.0f} exceeds physical limit 10,000)"
        # Skip all other scoring - this is definitive AI evidence
        print(f"   [PRNU] Base: -50 (synthetic grid override)")
        details["synthetic_grid_override"] = True
        details["base_score"] = score
        details["final_score"] = score
        return score, description, details
    
    # Check for suspiciously high PCE (5000-10000 range)
    if pce > PCE_SUSPICIOUS_THRESHOLD:
        # High but not impossible - could be edited/enhanced real photo
        # Cap positive score and add warning
        max_confidence = min(max_confidence, 10)
        print(f"   [PRNU] Capping confidence to {max_confidence} due to suspicious PCE")
    
    if is_jpeg:
        # FP-5 FIX: JPEG path - more nuanced scoring
        # JPEG compression destroys fine noise but PCE can still be reliable
        
        # TIER 1: Very high PCE = definitive sensor signature (can't fake this)
        if pce >= PCE_STRONG_MATCH:  # PCE >= 1000
            # Even in JPEG, very high PCE is trustworthy
            score = 30
            description = f"JPEG with strong sensor signature (PCE={pce:.0f})"
        
        # TIER 2: Moderate PCE with supporting evidence
        elif pce >= PCE_WEAK_MATCH:  # PCE >= 50
            if noise_variance > 2 or entropy > 4:
                score = 20
                description = f"JPEG real photo (PCE={pce:.0f})"
            else:
                score = 10
                description = f"JPEG with sensor signature (PCE={pce:.0f})"
        
        # TIER 3: Weak but present PCE
        elif pce > PCE_INCONCLUSIVE:  # PCE > 10
            if noise_variance > 5 and entropy > 5.8:
                score = 15
                description = "JPEG real photo (weak sensor signature)"
            else:
                score = 5
                description = f"JPEG with faint sensor (PCE={pce:.0f})"
        
        # TIER 4: Suspicious uniformity (AI-like)
        elif pce < 5 and noise_variance < 2 and entropy < 5.0:
            score = -15
            description = "JPEG with suspicious uniformity"
        
        # TIER 5: Neutral (can't determine)
        else:
            score = 0
            description = "JPEG compression (neutral)"
    
    else:
        # Non-JPEG path - use PCE as primary metric
        
        # FP-2 FIX: PCE-based decision
        if pce >= PCE_STRONG_MATCH:
            # Very strong correlation peak = real sensor
            score = 40
            description = f"Strong sensor signature (PCE={pce:.1f})"
        
        elif pce >= PCE_WEAK_MATCH:
            # Weak but significant correlation
            if noise_variance > 8 and entropy > 6.2:
                score = 25
                description = f"Natural sensor (PCE={pce:.1f}, high randomness)"
            else:
                score = 10
                description = f"Weak sensor signature (PCE={pce:.1f})"
        
        elif pce >= PCE_INCONCLUSIVE:
            # Very weak correlation - check other metrics
            if spectral_flatness >= SF_NATURAL_HIGH and entropy > 6.5:
                score = 15
                description = "Flat spectrum (sensor-like despite low PCE)"
            elif spectral_flatness < SF_SUSPICIOUS:
                score = -20
                description = f"Structured spectrum (PCE={pce:.1f}, flatness={spectral_flatness:.3f})"
            else:
                score = 0
                description = f"Inconclusive (PCE={pce:.1f})"
        
        else:
            # PCE < 10 = essentially random
            if peak_count > 800 and noise_variance < 4 and entropy < 5.5:
                score = -40
                description = "AI indicators: high peaks + low randomness"
            elif spectral_flatness < SF_SUSPICIOUS:
                score = -30
                description = f"Strong periodicity (flatness={spectral_flatness:.3f})"
            elif spectral_flatness < SF_AMBIGUOUS and peak_count > 400:
                score = -20
                description = "Moderate synthetic patterns"
            elif noise_variance > 10 and entropy > 6.5:
                # High randomness despite low PCE - might be heavily processed real
                score = 10
                description = "High randomness (processed real?)"
            else:
                score = -10
                description = "Low correlation (borderline)"
    
    # ---- Apply adjustments ----
    base_score = score
    score += nua_score_adj
    score += camera_score_adj
    
    # Apply flat-region confidence cap
    score = max(-50, min(max_confidence, score))
    
    if nua_score_adj != 0 or camera_score_adj != 0:
        adj_parts = []
        if nua_score_adj != 0:
            adj_parts.append(f"nua:{nua_score_adj:+d}")
        if camera_score_adj != 0:
            adj_parts.append(f"camera:{camera_score_adj:+d}")
        description += f" [adj: {', '.join(adj_parts)}]"
    
    print(f"   [PRNU] Base: {base_score:+d} → Final: {score:+d}")
    
    # ---- Build details ----
    details.update({
        "peak_count": peak_count,
        "noise_variance": round(noise_variance, 3),
        "score_adjustments": {
            "nua": nua_score_adj,
            "camera": camera_score_adj,
        },
        "base_score": base_score,
        "final_score": score,
    })
    
    if flat_mask_info:
        details["flat_mask"] = flat_mask_info
    
    return score, description, details


# ============================================================================
# PUBLIC API
# ============================================================================

def get_prnu_score(file_path: str, use_flat_masking: bool = True, is_jpeg_hint: bool = False) -> Tuple[float, str, Dict]:
    """
    Main PRNU analysis with false positive reduction.
    
    Args:
        file_path: Path to image
        use_flat_masking: Enable FP-1 flat-region masking (recommended)
    
    Returns:
        (score, description, details)
    """
    try:
        print(f"   [PRNU v3.0] Starting analysis...")
        print(f"   [PRNU] Flat masking: {use_flat_masking}")
        
        noise, extract_info = extract_noise_residual(
            file_path, 
            method="wavelet",
            apply_flat_mask=use_flat_masking
        )
        
        flat_mask_info = extract_info.get("flat_mask")
        
        # Auto-detect JPEG if hint not provided
        is_jpeg = is_jpeg_hint or file_path.lower().endswith(('.jpg', '.jpeg', '.webp'))
        
        score, desc, details = analyze_prnu_integrity(
            noise, 
            image_path=file_path,
            flat_mask_info=flat_mask_info,
            is_jpeg_hint=is_jpeg
        )
        
        print(f"   [PRNU] Score: {score:+.0f} | {desc}")
        return score, desc, details

    except Exception as e:
        print(f"   [PRNU Error] {e}")
        import traceback
        traceback.print_exc()
        return 0, f"PRNU analysis failed: {str(e)}", {}


def analyze_prnu(image_path: str, is_jpeg_hint: bool = False) -> Tuple[float, str, Dict]:
    """Wrapper for compatibility."""
    return get_prnu_score(image_path, use_flat_masking=True, is_jpeg_hint=is_jpeg_hint)


# ============================================================================
# CLI TESTING
# ============================================================================

if __name__ == "__main__":
    import sys

    print(f"\n{'=' * 70}")
    print("Layer 8.5: PRNU Sensor Fingerprint Analysis v3.0")
    print("FALSE POSITIVE REDUCTION EDITION")
    print(f"{'=' * 70}\n")

    if len(sys.argv) > 1:
        score, desc = analyze_prnu(sys.argv[1])

        print(f"\n{'=' * 70}")
        print(f"FINAL SCORE: {score:+.0f}")
        print(f"DESCRIPTION: {desc}")
        print(f"{'=' * 70}\n")

        if score > 20:
            print("[REAL] Strong evidence of camera sensor")
        elif score > 5:
            print("[REAL] Likely real, some compression")
        elif score > -10:
            print("[NEUTRAL] Indeterminate")
        elif score > -30:
            print("[SUSPICIOUS] Possible AI (needs corroboration)")
        else:
            print("[AI] Strong synthetic signatures")

        print(f"\n{'=' * 70}")
        print("v3.0 FALSE POSITIVE FIXES:")
        print("=" * 70)
        print("  FP-1  Flat-region masking (exclude textured areas)")
        print("  FP-2  PCE instead of raw peak counting")
        print("  FP-3  NUA filtering (remove computational artifacts)")
        print("  FP-4  Edge-aware noise extraction")
        print("  FP-5  Enhanced JPEG discrimination")
        print("=" * 70)

    else:
        print("Usage:")
        print("  python layer_8_5_enhanced.py <image_path>")
        print()
        print("v3.0 Improvements (False Positive Reduction):")
        print("  - Flat-region masking: Excludes textured areas from analysis")
        print("  - PCE metric: Peak-to-Correlation Energy (replaces raw peaks)")
        print("  - NUA filtering: Removes smartphone computational artifacts")
        print("  - Edge-aware extraction: Better noise isolation")
        print("  - JPEG discrimination: Distinguishes real JPEG from fake")