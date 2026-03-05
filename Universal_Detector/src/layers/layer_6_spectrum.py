"""
Layer 6: Frequency Domain Forensics (FFT) - COMPLETE VERSION
Fixes all 7 identified gaps + ultra-conservative for web images

VERSION: 2.0 - Production Ready
DATE: February 2026

IMPROVEMENTS:
[+] L6-1: Multi-band analysis (R,G,B + Y,Cb,Cr channels)
[+] L6-2: Windowing function (Hann/Hamming to reduce edge artifacts)
[+] L6-3: Directional spectrum analysis (detects axis-aligned grids)
[+] L6-4: Patch-based FFT (detects local manipulation)
[+] L6-5: Advanced JPEG detection (multiple encoder signatures)
[+] L6-6: GAN checkerboard detection (transposed convolution artifacts)
[+] L6-7: AI upsampling pattern detection (Real-ESRGAN, Topaz, etc.)

SCORING: -50 (Strong AI) to +35 (Natural)
"""

import numpy as np
import cv2
from scipy.fftpack import fft2, fftshift
from scipy.signal import find_peaks
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_windowing(img: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """
    L6-2: Apply window function to reduce FFT edge artifacts
    
    Args:
        img: Grayscale image
        window_type: 'hann', 'hamming', or 'blackman'
    
    Returns:
        Windowed image
    """
    h, w = img.shape
    
    if window_type == 'hann':
        window_y = np.hanning(h)
        window_x = np.hanning(w)
    elif window_type == 'hamming':
        window_y = np.hamming(h)
        window_x = np.hamming(w)
    elif window_type == 'blackman':
        window_y = np.blackman(h)
        window_x = np.blackman(w)
    else:
        return img
    
    # Create 2D window
    window_2d = np.sqrt(np.outer(window_y, window_x))
    
    return img * window_2d


def advanced_jpeg_detection(magnitude_spectrum: np.ndarray, img_shape: Tuple[int, int]) -> Dict:
    """
    L6-5: Advanced JPEG artifact detection for multiple JPEG libraries
    
    Detects signatures from:
    - Standard libjpeg (8, 16, 24, 32 pixel blocks)
    - Progressive JPEG (4, 8, 12, 16)
    - MozJPEG (8, 16, 24, 32, 40)
    - WebP JPEG mode (8, 16, 32)
    """
    h, w = img_shape
    center_y, center_x = h // 2, w // 2
    
    jpeg_signatures = {
        'standard': [8, 16, 24, 32],
        'progressive': [4, 8, 12, 16],
        'mozjpeg': [8, 16, 24, 32, 40],
        'webp': [8, 16, 32],
    }
    
    detected_type = None
    max_score = 0
    
    threshold = np.mean(magnitude_spectrum) + 2 * np.std(magnitude_spectrum)
    
    for lib_name, offsets in jpeg_signatures.items():
        score = 0
        
        for offset in offsets:
            # Check horizontal
            if center_x + offset < w:
                if magnitude_spectrum[center_y, center_x + offset] > threshold:
                    score += 1
            
            # Check vertical
            if center_y + offset < h:
                if magnitude_spectrum[center_y + offset, center_x] > threshold:
                    score += 1
        
        if score > max_score:
            max_score = score
            detected_type = lib_name
    
    is_jpeg = max_score >= 3
    
    return {
        'is_jpeg': is_jpeg,
        'type': detected_type if is_jpeg else None,
        'confidence': max_score,
        'score': max_score
    }


def detect_periodic_spikes(magnitude_spectrum: np.ndarray) -> Tuple[int, float]:
    """
    Detect periodic frequency spikes (general AI artifacts)
    
    FP-4 FIX: Exclude the DC-adjacent cross (JPEG 8x8 block artifacts)
    and use a higher threshold (mean + 4*std instead of 3*std) to reduce
    false positives on compressed images.
    
    Returns:
        spike_count: Number of peaks above threshold
        spike_strength: Average strength of peaks
    """
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    
    # Create a mask that excludes JPEG block-boundary frequencies
    # JPEG creates periodic spikes at multiples of h/8, w/8 from center
    mask = np.ones_like(magnitude_spectrum, dtype=bool)
    for mult in range(1, 12):
        # Exclude cross-shaped regions at JPEG block frequencies
        fy = cy + (h * mult) // 8 if (h * mult) // 8 < h // 2 else 0
        fy2 = cy - (h * mult) // 8 if (h * mult) // 8 < h // 2 else 0
        fx = cx + (w * mult) // 8 if (w * mult) // 8 < w // 2 else 0
        fx2 = cx - (w * mult) // 8 if (w * mult) // 8 < w // 2 else 0
        for yy in [fy, fy2]:
            if 0 < yy < h:
                mask[max(0, yy-1):min(h, yy+2), :] = False
        for xx in [fx, fx2]:
            if 0 < xx < w:
                mask[:, max(0, xx-1):min(w, xx+2)] = False
    
    # Also exclude DC component area
    dc_r = max(5, min(h, w) // 50)
    y_g, x_g = np.ogrid[:h, :w]
    mask[(y_g - cy)**2 + (x_g - cx)**2 < dc_r**2] = False
    
    filtered = magnitude_spectrum[mask]
    threshold = np.mean(filtered) + 4 * np.std(filtered)  # Raised from 3→4 std
    spike_count = int(np.sum(filtered > threshold))
    
    peaks_above = filtered[filtered > threshold]
    spike_strength = float(np.mean(peaks_above - threshold)) if len(peaks_above) > 0 else 0.0
    
    return spike_count, spike_strength


def analyze_directional_spectrum(magnitude_spectrum: np.ndarray) -> Dict:
    """
    L6-3: Detect directional patterns (AI upsampling grids are axis-aligned)
    
    AI upsampling creates periodic patterns along horizontal and vertical axes.
    This detects those patterns that radial averaging would miss.
    
    Returns:
        Dict with score and description
    """
    h, w = magnitude_spectrum.shape
    
    # Horizontal profile (average along columns)
    horizontal_profile = np.mean(magnitude_spectrum, axis=0)
    
    # Vertical profile (average along rows)
    vertical_profile = np.mean(magnitude_spectrum, axis=1)
    
    def detect_periodicity(profile):
        """Find evenly-spaced peaks (periodicity indicator)"""
        try:
            peaks, properties = find_peaks(
                profile, 
                height=np.mean(profile) + 2*np.std(profile),
                distance=5  # Minimum distance between peaks
            )
            
            if len(peaks) < 3:
                return 0
            
            # Check if peaks are evenly spaced
            peak_diffs = np.diff(peaks)
            periodicity_score = np.std(peak_diffs) / (np.mean(peak_diffs) + 1e-8)
            
            # Low std/mean ratio = highly periodic = AI grid
            if periodicity_score < 0.3:
                return len(peaks)
            
            return 0
        except:
            return 0
    
    h_periodicity = detect_periodicity(horizontal_profile)
    v_periodicity = detect_periodicity(vertical_profile)
    
    # AI grids have strong periodicity in BOTH directions
    if h_periodicity >= 6 and v_periodicity >= 6:
        return {
            'score': -50,
            'description': f'Strong AI Grid: H={h_periodicity} V={v_periodicity} periodic peaks',
            'h_peaks': h_periodicity,
            'v_peaks': v_periodicity
        }
    elif h_periodicity >= 4 and v_periodicity >= 4:
        return {
            'score': -35,
            'description': f'AI Grid Pattern: H={h_periodicity} V={v_periodicity} peaks',
            'h_peaks': h_periodicity,
            'v_peaks': v_periodicity
        }
    elif h_periodicity >= 3 or v_periodicity >= 3:
        return {
            'score': -20,
            'description': f'Directional patterns: H={h_periodicity} V={v_periodicity}',
            'h_peaks': h_periodicity,
            'v_peaks': v_periodicity
        }
    
    return {
        'score': 0,
        'description': 'No strong directional patterns',
        'h_peaks': h_periodicity,
        'v_peaks': v_periodicity
    }


def detect_gan_checkerboard(magnitude_spectrum: np.ndarray) -> Dict:
    """
    L6-6: Detect GAN checkerboard pattern from transposed convolution
    
    GANs using stride-2 transposed convolution create characteristic checkerboard
    artifacts at specific frequencies (half the image size).
    
    Pattern appears as strong peaks at:
    - Frequency = N/2 (cardinal directions)
    - Moderate peaks at diagonals
    
    Returns:
        Dict with score and description
    """
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Check half-frequency regions (where GAN artifacts concentrate)
    try:
        half_freq_regions = [
            magnitude_spectrum[center_y, min(center_x + w//4, w-1)],  # Right
            magnitude_spectrum[center_y, max(center_x - w//4, 0)],    # Left
            magnitude_spectrum[min(center_y + h//4, h-1), center_x],  # Down
            magnitude_spectrum[max(center_y - h//4, 0), center_x],    # Up
        ]
        
        # Check diagonal regions (complete checkerboard)
        diagonal_regions = [
            magnitude_spectrum[min(center_y + h//4, h-1), min(center_x + w//4, w-1)],
            magnitude_spectrum[max(center_y - h//4, 0), min(center_x + w//4, w-1)],
            magnitude_spectrum[min(center_y + h//4, h-1), max(center_x - w//4, 0)],
            magnitude_spectrum[max(center_y - h//4, 0), max(center_x - w//4, 0)],
        ]
    except IndexError:
        return {'score': 0, 'description': 'Image too small for GAN detection'}
    
    avg_baseline = np.mean(magnitude_spectrum)
    std_baseline = np.std(magnitude_spectrum)
    
    # Count strong peaks
    axis_peaks = sum(1 for val in half_freq_regions if val > avg_baseline + 4*std_baseline)
    diag_peaks = sum(1 for val in diagonal_regions if val > avg_baseline + 3*std_baseline)
    
    # GAN checkerboard: strong axis peaks + moderate diagonal peaks
    if axis_peaks >= 3 and diag_peaks >= 2:
        return {
            'score': -50,
            'description': f'GAN Checkerboard Pattern (axis:{axis_peaks} diag:{diag_peaks})'
        }
    elif axis_peaks >= 3:
        return {
            'score': -40,
            'description': f'Strong GAN upsampling artifacts (axis:{axis_peaks})'
        }
    elif axis_peaks >= 2 and diag_peaks >= 2:
        return {
            'score': -30,
            'description': f'Likely GAN patterns (axis:{axis_peaks} diag:{diag_peaks})'
        }
    elif axis_peaks >= 2:
        return {
            'score': -20,
            'description': 'Possible GAN artifacts'
        }
    
    return {'score': 0, 'description': 'No GAN checkerboard'}


def detect_ai_upscaling(magnitude_spectrum: np.ndarray, img_shape: Tuple[int, int]) -> Dict:
    """
    L6-7: Detect AI upscaling signatures (Real-ESRGAN, Topaz, Waifu2x, etc.)
    
    AI upscalers boost high frequencies unnaturally and create ringing artifacts.
    
    Signatures:
    - Extreme HF boost (8-10x vs natural 2-3x)
    - Ringing artifacts near edges
    - Oversharpening patterns
    
    Returns:
        Dict with score and description
    """
    h, w = img_shape
    
    # Calculate frequency distribution
    flat_spectrum = magnitude_spectrum.flatten()
    sorted_spectrum = np.sort(flat_spectrum)
    
    # Top 1% frequencies (AI upscalers artificially boost these)
    top_1_percent = sorted_spectrum[-int(len(sorted_spectrum)*0.01):]
    top_5_percent = sorted_spectrum[-int(len(sorted_spectrum)*0.05):]
    median_freq = np.median(flat_spectrum)
    
    boost_ratio_1 = np.mean(top_1_percent) / (median_freq + 1e-8)
    boost_ratio_5 = np.mean(top_5_percent) / (median_freq + 1e-8)
    
    # Check for ringing artifacts (oversharpening)
    center_y, center_x = h // 2, w // 2
    ring_size = min(10, h//4, w//4)
    
    try:
        ring_region = magnitude_spectrum[
            center_y-ring_size:center_y+ring_size,
            center_x-ring_size:center_x+ring_size
        ]
        ring_variance = np.var(ring_region)
    except:
        ring_variance = 0
    
    # AI upscaler detection
    # Natural images: boost ~ 2-3x
    # AI upscaled: boost > 5-10x
    
    if boost_ratio_1 > 10 and ring_variance > 150:
        return {
            'score': -50,
            'description': f'Extreme AI Upscaling (boost:{boost_ratio_1:.1f}x ringing:{ring_variance:.0f})'
        }
    elif boost_ratio_1 > 8:
        return {
            'score': -45,
            'description': f'Strong AI Upscaler signature (boost:{boost_ratio_1:.1f}x)'
        }
    elif boost_ratio_1 > 6 and boost_ratio_5 > 3.5:
        return {
            'score': -35,
            'description': f'Likely AI upscaled (boost:{boost_ratio_1:.1f}x)'
        }
    elif boost_ratio_1 > 5 and ring_variance > 100:
        return {
            'score': -30,
            'description': f'Possible upscaling with ringing'
        }
    elif boost_ratio_1 > 4.5:
        return {
            'score': -20,
            'description': f'Borderline HF boost (boost:{boost_ratio_1:.1f}x)'
        }
    
    return {
        'score': 0,
        'description': f'Natural frequency distribution (boost:{boost_ratio_1:.1f}x)'
    }


def patch_based_fft(img: np.ndarray, patch_size: int = 128) -> Dict:
    """
    L6-4: Patch-based FFT analysis (detects localized AI manipulation)
    
    Face swaps and local AI edits show up in patch-level frequency analysis.
    This divides the image into overlapping patches and analyzes each.
    
    Returns:
        Dict with score and description
    """
    h, w = img.shape
    
    # Image too small for patch analysis
    if h < patch_size or w < patch_size:
        return {
            'score': 0,
            'description': 'Image too small for patch analysis',
            'variance': 0
        }
    
    patch_scores = []
    suspicious_patches = []
    
    stride = patch_size // 2  # 50% overlap
    
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = img[y:y+patch_size, x:x+patch_size]
            
            # FFT of this patch
            f = fft2(patch)
            fshift = fftshift(f)
            magnitude = np.abs(fshift)
            
            # Count periodic spikes in this patch
            threshold = np.mean(magnitude) + 3 * np.std(magnitude)
            spike_count = int(np.sum(magnitude > threshold))
            
            patch_scores.append(spike_count)
            
            # Flag suspicious patches (AI artifacts)
            if spike_count > 60:
                suspicious_patches.append((x, y, spike_count))
    
    if len(patch_scores) < 4:
        return {'score': 0, 'description': 'Insufficient patches', 'variance': 0}
    
    # Calculate variance across patches
    variance = float(np.var(patch_scores))
    avg_spikes = float(np.mean(patch_scores))
    max_spikes = float(np.max(patch_scores))
    
    # High variance = inconsistent patches = localized manipulation
    if variance > 300 and len(suspicious_patches) > 5:
        return {
            'score': -40,
            'description': f'Localized AI manipulation: {len(suspicious_patches)} suspicious patches (var:{variance:.0f})',
            'variance': variance,
            'suspicious_count': len(suspicious_patches)
        }
    elif variance > 200 and len(suspicious_patches) > 3:
        return {
            'score': -30,
            'description': f'Likely local edits (var:{variance:.0f})',
            'variance': variance,
            'suspicious_count': len(suspicious_patches)
        }
    elif variance > 150:
        return {
            'score': -15,
            'description': f'Inconsistent patches (var:{variance:.0f})',
            'variance': variance,
            'suspicious_count': len(suspicious_patches)
        }
    elif avg_spikes < 25 and variance < 50:
        return {
            'score': 20,
            'description': 'Consistent natural texture across patches',
            'variance': variance,
            'suspicious_count': 0
        }
    
    return {
        'score': 0,
        'description': f'Neutral patch variance (var:{variance:.0f})',
        'variance': variance,
        'suspicious_count': len(suspicious_patches)
    }


def analyze_single_channel(channel: np.ndarray, apply_window: bool = True) -> float:
    """
    Analyze a single color channel with FFT
    
    Returns:
        Score for this channel (-50 to +35)
    """
    # Apply windowing to reduce edge artifacts
    if apply_window:
        channel = apply_windowing(channel, 'hann')
    
    # Compute FFT
    f = fft2(channel)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Detect periodic spikes
    spike_count, spike_strength = detect_periodic_spikes(magnitude_spectrum)
    
    # Simple scoring for channel
    if spike_count > 800 and spike_strength > 15:
        return -40
    elif spike_count > 600:
        return -30
    elif spike_count > 400:
        return -15
    elif spike_count < 200:
        return 15
    else:
        return 0


def analyze_multiband_spectrum(img_color: np.ndarray) -> Dict:
    """
    L6-1: Multi-band FFT analysis (RGB + YCbCr channels)
    
    AI tools often manipulate color channels differently, leaving
    inconsistencies that only show up in per-channel analysis.
    
    Returns:
        Dict with overall score and channel details
    """
    # Convert to different color spaces
    rgb = img_color
    ycbcr = cv2.cvtColor(img_color, cv2.COLOR_BGR2YCrCb)
    
    results = {}
    
    # Analyze RGB channels
    for i, name in enumerate(['R', 'G', 'B']):
        channel = rgb[:,:,2-i]  # OpenCV is BGR
        score = analyze_single_channel(channel)
        results[name] = score
    
    # Analyze YCbCr channels
    for i, name in enumerate(['Y', 'Cb', 'Cr']):
        channel = ycbcr[:,:,i]
        score = analyze_single_channel(channel)
        results[name] = score
    
    # Check for channel inconsistency (AI signature)
    rgb_scores = [results['R'], results['G'], results['B']]
    rgb_variance = float(np.var(rgb_scores))
    
    # High variance = channels processed differently = AI
    if rgb_variance > 150:
        return {
            'score': -35,
            'description': f'Inconsistent RGB processing (var:{rgb_variance:.0f}) - AI signature',
            'rgb_variance': rgb_variance,
            'channel_scores': results
        }
    
    # Chroma channels often have AI artifacts
    if results['Cb'] < -35 or results['Cr'] < -35:
        return {
            'score': -30,
            'description': f'AI artifacts in chroma (Cb:{results["Cb"]:.0f} Cr:{results["Cr"]:.0f})',
            'rgb_variance': rgb_variance,
            'channel_scores': results
        }
    
    # Average all channel scores
    avg_score = float(np.mean(list(results.values())))
    
    return {
        'score': avg_score,
        'description': f'Multi-band analysis (avg:{avg_score:.0f})',
        'rgb_variance': rgb_variance,
        'channel_scores': results
    }


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_spectrum(image_path: str) -> Tuple[float, str]:
    """
    Layer 6: Complete Frequency Domain Forensics
    
    Combines all 7 detection methods:
    1. Multi-band analysis (RGB + YCbCr)
    2. Windowed FFT (reduces edge artifacts)
    3. Directional spectrum (axis-aligned grids)
    4. Patch-based FFT (local manipulation)
    5. Advanced JPEG detection (multiple encoders)
    6. GAN checkerboard detection
    7. AI upscaling detection
    
    Returns:
        (score, description)
        score: -50 (Strong AI) to +35 (Natural)
    """
    try:
        # Load image (color for multi-band)
        img_color = cv2.imread(image_path)
        if img_color is None:
            return 0, "Error loading image"
        
        # Also load grayscale for main analysis
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        h, w = img_gray.shape
        
        # Apply windowing to reduce edge artifacts (L6-2)
        img_windowed = apply_windowing(img_gray, 'hann')
        
        # Compute FFT on windowed image
        f = fft2(img_windowed)
        fshift = fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # ===== RUN ALL DETECTION METHODS =====
        
        # 1. L6-5: Advanced JPEG detection
        jpeg_result = advanced_jpeg_detection(magnitude_spectrum, (h, w))
        
        # 2. Basic spike detection
        spike_count, spike_strength = detect_periodic_spikes(magnitude_spectrum)
        
        # 3. L6-3: Directional spectrum analysis
        directional_result = analyze_directional_spectrum(magnitude_spectrum)
        
        # 4. L6-6: GAN checkerboard detection
        gan_result = detect_gan_checkerboard(magnitude_spectrum)
        
        # 5. L6-7: AI upscaling detection
        upscale_result = detect_ai_upscaling(magnitude_spectrum, (h, w))
        
        # 6. L6-4: Patch-based analysis
        patch_result = patch_based_fft(img_gray, patch_size=min(128, h//3, w//3))
        
        # 7. L6-1: Multi-band analysis
        multiband_result = analyze_multiband_spectrum(img_color)
        
        # 8. Radial frequency profile (original method)
        center_x, center_y = w // 2, h // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        r = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        r = r.astype(int)
        
        tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-8)
        radial_profile = radial_profile / (np.max(radial_profile) + 1e-8)
        
        high_freq_energy = float(np.mean(radial_profile[-int(len(radial_profile)*0.2):]))
        
        # ===== LOGGING =====
        
        print(f"   [Spectrum] High Freq Energy: {high_freq_energy:.4f}")
        print(f"   [Spectrum] Spikes: {spike_count} (strength: {spike_strength:.2f})")
        print(f"   [Spectrum] JPEG: {jpeg_result['is_jpeg']} (type: {jpeg_result.get('type', 'N/A')})")
        print(f"   [Spectrum] Directional: {directional_result['description']}")
        print(f"   [Spectrum] GAN Check: {gan_result['description']}")
        print(f"   [Spectrum] Upscaling: {upscale_result['description']}")
        print(f"   [Spectrum] Patches: {patch_result['description']}")
        print(f"   [Spectrum] Multi-band: {multiband_result['description']}")
        
        # ===== DECISION TREE (ULTRA-CONSERVATIVE FOR WEB IMAGES) =====
        
        # TIER 0: Critical AI signatures (immediate verdict)
        if gan_result['score'] <= -40:
            return gan_result['score'], gan_result['description']
        
        if upscale_result['score'] <= -45:
            return upscale_result['score'], upscale_result['description']
        
        if directional_result['score'] <= -45:
            return directional_result['score'], directional_result['description']
        
        if patch_result['score'] <= -35:
            return patch_result['score'], patch_result['description']
        
        # TIER 1: JPEG compression handling (be VERY lenient)
        if jpeg_result['is_jpeg']:
            print(f"   [Spectrum] [i] JPEG compression detected ({jpeg_result['type']})")
            
            # For JPEG images, be ultra-conservative
            if spike_count > 1500:
                return -30, f"Excessive patterns even for JPEG ({spike_count} spikes)"
            elif spike_count > 1200:
                return -15, f"Heavy JPEG with concerning patterns ({spike_count})"
            elif spike_count > 1000:
                return -5, f"Heavy JPEG compression ({spike_count} spikes)"
            else:
                # Normal JPEG compression
                if high_freq_energy > 0.22:
                    return 20, f"Normal JPEG with natural texture ({spike_count} spikes)"
                elif high_freq_energy > 0.16:
                    return 10, f"JPEG compressed, acceptable texture"
                else:
                    return 0, f"JPEG compressed ({spike_count} spikes, neutral)"
        
        # TIER 2: Non-JPEG AI detection (stricter)
        
        # Strong multi-method agreement = high confidence AI
        ai_signals = []
        if directional_result['score'] < -20:
            ai_signals.append('directional')
        if gan_result['score'] < -20:
            ai_signals.append('GAN')
        if upscale_result['score'] < -20:
            ai_signals.append('upscaling')
        if patch_result['score'] < -20:
            ai_signals.append('patches')
        if multiband_result['score'] < -25:
            ai_signals.append('multiband')
        
        # Multiple methods agree = AI
        if len(ai_signals) >= 3:
            combined_score = min(
                directional_result['score'],
                gan_result['score'],
                upscale_result['score'],
                patch_result['score']
            )
            return combined_score, f"Multiple AI signatures: {', '.join(ai_signals)}"
        
        elif len(ai_signals) >= 2:
            return -35, f"AI detected by: {', '.join(ai_signals)}"
        
        elif len(ai_signals) == 1:
            # Single method - be more conservative
            method_scores = {
                'directional': directional_result['score'],
                'GAN': gan_result['score'],
                'upscaling': upscale_result['score'],
                'patches': patch_result['score'],
                'multiband': multiband_result['score']
            }
            return method_scores[ai_signals[0]], f"AI signature in {ai_signals[0]}"
        
        # TIER 3: Traditional spike analysis (fallback)
        
        if spike_count > 800 and spike_strength > 15:
            return -40, f"Strong periodic grid ({spike_count} spikes, strength:{spike_strength:.1f})"
        
        elif spike_count > 600 and spike_strength > 12:
            return -30, f"AI patterns ({spike_count} spikes, strength:{spike_strength:.1f})"
        
        elif spike_count > 500 and spike_strength > 10:
            return -20, f"Moderate AI patterns ({spike_count} spikes)"
        
        elif spike_count > 400:
            return -10, f"Borderline patterns ({spike_count} spikes)"
        
        # Smoothness check (AI lacks texture)
        elif high_freq_energy < 0.08:
            return -35, "Extreme smoothing (AI blur)"
        
        elif high_freq_energy < 0.14:
            return -20, "Low texture detail (AI characteristic)"
        
        # TIER 4: Natural detection
        
        # Pristine natural spectrum
        if high_freq_energy > 0.28 and spike_count < 200:
            return 35, "Pristine natural spectrum (camera sensor noise)"
        
        elif high_freq_energy > 0.24 and spike_count < 250:
            return 30, "Strong natural sensor noise"
        
        elif high_freq_energy > 0.20 and spike_count < 300:
            return 25, "Natural texture (typical real photo)"
        
        elif high_freq_energy > 0.18 and spike_count < 400:
            return 15, "Natural texture (likely real)"
        
        elif high_freq_energy > 0.16:
            return 5, "Moderate natural texture"
        
        # Default: Neutral
        return 0, f"Neutral spectrum (spikes:{spike_count} HF:{high_freq_energy:.3f})"
    
    except Exception as e:
        print(f"   [Spectrum Error] {e}")
        import traceback
        traceback.print_exc()
        return 0, f"Analysis error: {str(e)}"


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        print(f"\n{'='*70}")
        print("Layer 6: Frequency Domain Forensics v2.0 - COMPLETE")
        print(f"{'='*70}\n")
        
        score, desc = analyze_spectrum(sys.argv[1])
        
        print(f"\n{'='*70}")
        print(f"FINAL SCORE: {score:+.0f}")
        print(f"DESCRIPTION: {desc}")
        print(f"{'='*70}\n")
        
        # Interpretation
        if score > 20:
            print("[+] Strong evidence of natural image")
        elif score > 5:
            print("[+] Likely natural image")
        elif score > -15:
            print("[ ] Neutral or web-compressed (inconclusive)")
        elif score > -30:
            print("[!] AI patterns detected (needs corroboration)")
        else:
            print("[-] Strong AI signatures detected")
        
        print("\n" + "="*70)
        print("IMPROVEMENTS IN v2.0:")
        print("="*70)
        print("[+] Multi-band analysis (RGB + YCbCr)")
        print("[+] Windowing function (Hann window)")
        print("[+] Directional spectrum (H/V grid detection)")
        print("[+] Patch-based FFT (local manipulation)")
        print("[+] Advanced JPEG detection (4 encoder types)")
        print("[+] GAN checkerboard detection")
        print("[+] AI upscaling detection (Real-ESRGAN, etc.)")
        print("="*70)
    
    else:
        print("Usage: python layer_6_spectrum.py <image_path>")
        print("\nFeatures:")
        print("  • Multi-band FFT (RGB + YCbCr channels)")
        print("  • Directional analysis (detects axis-aligned grids)")
        print("  • Patch-based detection (localized manipulation)")
        print("  • GAN checkerboard detection")
        print("  • AI upscaling signatures")
        print("  • Advanced JPEG artifact handling")
        print("  • Ultra-conservative for web images")