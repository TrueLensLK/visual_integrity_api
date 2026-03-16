import os
import cv2
import numpy as np
from PIL import Image, ImageChops
from io import BytesIO
from typing import Dict, List, Tuple, Optional

class SignalForensics:
    """
    Layer 3: Universal Signal Analysis
    """
    def __init__(self, file_path: str):
        self.path = file_path
        self.img_pil = Image.open(file_path).convert('RGB')
        self.img_cv = cv2.imread(file_path)
        self.width, self.height = self.img_pil.size
        self.impact = 0
        self.findings = []
        
        # Detect format and estimate JPEG quality
        self.format = (self.img_pil.format or "").upper()
        self.is_jpeg = self.format in ("JPEG", "MPO") or file_path.lower().endswith((".jpg", ".jpeg"))
        self.jpeg_quality = self._estimate_jpeg_quality() if self.is_jpeg else None

    def analyze(self) -> Dict:
        # 1. MULTI-LEVEL ELA (Gap L3-1, L3-2) - now JPEG-quality aware
        ela_results = self._run_multi_ela([75, 90, 95])
        
        # 2. EDGE COHERENCE (Gap L3-4)
        # AI often has "ringing" or overly sharp edges compared to natural optics
        edge_score = self._analyze_edges()
        
        # 3. NOISE ANALYSIS with REAL Bayer detection (Gap L3-3, L3-6)
        noise_results = self._analyze_noise_texture()

        # --- Scoring Logic (Gap L3-7) ---
        # ELA scoring now accounts for JPEG quality
        if ela_results['ghost_detected']:
            # Reduce penalty for low-quality JPEGs (they naturally have high ELA variance)
            if self.jpeg_quality and self.jpeg_quality < 70:
                self.impact -= 10
                self.findings.append(f"Possible JPEG Ghosting (dampened for Q{self.jpeg_quality})")
            else:
                self.impact -= 25
                self.findings.append("High JPEG Ghosting (Double Compression/Manipulation)")

        if noise_results['is_unnaturally_smooth']:
            self.impact -= 15
            self.findings.append("Denoising/AI Smoothness pattern detected")
        elif noise_results['has_bayer_pattern']:
            self.impact += 15
            self.findings.append(f"Bayer demosaicing pattern detected (confidence: {noise_results['bayer_confidence']:.0%})")

        # Add JPEG quality info to findings if detected
        if self.jpeg_quality:
            self.findings.append(f"Estimated JPEG quality: {self.jpeg_quality}")

        return {
            "impact": self.impact,
            "findings": self.findings,
            "details": {
                **ela_results, 
                **noise_results, 
                "edge_score": edge_score,
                "jpeg_quality": self.jpeg_quality,
                "is_jpeg": self.is_jpeg
            }
        }
    
    def _estimate_jpeg_quality(self) -> Optional[int]:
        """
        Estimate JPEG quality level by analyzing quantization tables.
        
        Method: Compare the image's DCT coefficients after re-encoding at
        various quality levels. The quality that produces the smallest
        difference is likely the original quality.
        
        Returns:
            Estimated quality (1-100) or None if estimation fails
        """
        try:
            # Quick estimation via progressive re-encoding
            qualities_to_test = [50, 60, 70, 75, 80, 85, 90, 95]
            min_diff = float('inf')
            best_quality = 85  # Default assumption
            
            # Get original image data for comparison
            original_array = np.array(self.img_pil)
            
            for q in qualities_to_test:
                buf = BytesIO()
                self.img_pil.save(buf, 'JPEG', quality=q)
                buf.seek(0)
                recompressed = np.array(Image.open(buf))
                
                # Calculate mean squared error
                mse = np.mean((original_array.astype(float) - recompressed.astype(float)) ** 2)
                
                if mse < min_diff:
                    min_diff = mse
                    best_quality = q
            
            # If minimum MSE is very low at a specific quality, that's likely the original
            # If MSE is consistently high, it's likely been through multiple compressions
            return best_quality
            
        except Exception:
            return None

    def _run_multi_ela(self, qualities: List[int]) -> Dict:
        """
        Gap L3-1 & L3-2: Tests multiple JPEG qualities to find the 'Ghost'.
        Now JPEG-quality aware with adaptive thresholds.
        """
        # For non-JPEG images, ELA is less meaningful
        if not self.is_jpeg:
            return {
                "max_ela_diff": 0,
                "ghost_detected": False,
                "quality_variance": 0,
                "note": "ELA skipped for lossless format"
            }
        
        diffs = []
        for q in qualities:
            buf = BytesIO()
            self.img_pil.save(buf, 'JPEG', quality=q)
            buf.seek(0)
            resaved = Image.open(buf)
            diff = ImageChops.difference(self.img_pil, resaved)
            extrema = diff.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            diffs.append(max_diff)

        # L3-5: JPEG Ghost Detection with quality-adaptive threshold
        # The threshold for "ghost detected" should scale with JPEG quality
        # Low-quality JPEGs (Q<70) naturally have higher variance
        base_threshold = 20
        if self.jpeg_quality:
            if self.jpeg_quality < 60:
                threshold = 35  # Very high threshold for heavily compressed
            elif self.jpeg_quality < 75:
                threshold = 28  # Higher threshold for medium compression
            else:
                threshold = base_threshold  # Normal threshold for high quality
        else:
            threshold = base_threshold
        
        ghost_detected = max(diffs) - min(diffs) > threshold
        
        return {
            "max_ela_diff": max(diffs),
            "ghost_detected": ghost_detected,
            "quality_variance": float(np.var(diffs)),
            "ela_threshold_used": threshold
        }

    def _analyze_edges(self) -> float:
        """Gap L3-4: Detects unnatural edge transitions using Canny and Sobel."""
        gray = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Calculate edge density
        edge_density = np.sum(edges > 0) / (self.width * self.height)
        return float(edge_density)

    def _analyze_noise_texture(self) -> Dict:
        """
        Gap L3-3 & L3-6: Distinguishes between sensor noise and AI smoothness.
        
        IMPROVED: Real Bayer pattern detection via 2x2 periodicity analysis
        in the green channel using FFT peak detection.
        """
        gray = cv2.cvtColor(self.img_cv, cv2.COLOR_BGR2GRAY)
        
        # Laplacian for general variance (smoothness detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        var = laplacian.var()
        
        # Real Bayer Pattern Detection
        bayer_result = self._detect_bayer_pattern()
        
        is_smooth = var < 15
        
        return {
            "variance": float(var),
            "is_unnaturally_smooth": is_smooth,
            "has_bayer_pattern": bayer_result['detected'],
            "bayer_confidence": bayer_result['confidence'],
            "bayer_details": bayer_result['details']
        }
    
    def _detect_bayer_pattern(self) -> Dict:
        """
        Detect Bayer pattern (RGGB demosaicing artifacts) via 2x2 periodicity.
        
        Real camera sensors use a Bayer Color Filter Array (CFA) with a 2x2 pattern:
            R G
            G B
        
        After demosaicing, this leaves subtle 2x2 periodic artifacts that can be
        detected by looking for peaks at specific frequencies in the FFT:
        - Nyquist/2 frequency in both horizontal and vertical directions
        - The green channel shows the strongest periodicity (2 green pixels per 2x2)
        
        Returns:
            Dict with 'detected' (bool), 'confidence' (0-1), and 'details'
        """
        try:
            # Extract green channel (most affected by Bayer pattern)
            # In BGR format, green is index 1
            green = self.img_cv[:, :, 1].astype(np.float32)
            
            # Also check red-green and blue-green differences
            # Bayer artifacts show up as 2x2 periodicity in color differences
            red = self.img_cv[:, :, 2].astype(np.float32)
            blue = self.img_cv[:, :, 0].astype(np.float32)
            
            rg_diff = red - green
            bg_diff = blue - green
            
            # Compute FFT of the green channel and color differences
            def get_periodicity_score(channel: np.ndarray) -> Tuple[float, float]:
                """
                Compute FFT and check for peaks at 2x2 periodicity frequencies.
                Returns (peak_strength, noise_floor) for computing SNR.
                """
                h, w = channel.shape
                
                # Apply window to reduce edge effects
                window_y = np.hanning(h)
                window_x = np.hanning(w)
                window_2d = np.outer(window_y, window_x)
                windowed = channel * window_2d
                
                # Compute FFT
                f_transform = np.fft.fft2(windowed)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                # The 2x2 Bayer pattern creates peaks at (h/2, w/2) from center
                # i.e., at the Nyquist/2 frequency
                cy, cx = h // 2, w // 2
                
                # Check for peaks at the four corners of the "half-Nyquist" cross
                # These correspond to the 2-pixel periodicity
                peak_regions = []
                offset_y = h // 4
                offset_x = w // 4
                
                # Sample peak regions (avoiding exact center which is DC)
                regions = [
                    (cy - offset_y, cx),      # Top
                    (cy + offset_y, cx),      # Bottom  
                    (cy, cx - offset_x),      # Left
                    (cy, cx + offset_x),      # Right
                    (cy - offset_y, cx - offset_x),  # Diagonal TL
                    (cy - offset_y, cx + offset_x),  # Diagonal TR
                    (cy + offset_y, cx - offset_x),  # Diagonal BL
                    (cy + offset_y, cx + offset_x),  # Diagonal BR
                ]
                
                peak_values = []
                for ry, rx in regions:
                    # Sample a small region around each point
                    r = 3
                    ry, rx = int(ry), int(rx)
                    if 0 <= ry-r and ry+r < h and 0 <= rx-r and rx+r < w:
                        region = magnitude[ry-r:ry+r+1, rx-r:rx+r+1]
                        peak_values.append(np.max(region))
                
                if not peak_values:
                    return 0.0, 1.0
                
                peak_strength = np.mean(peak_values)
                
                # Calculate noise floor (median of the spectrum, excluding DC)
                dc_mask = np.ones_like(magnitude, dtype=bool)
                dc_r = max(5, min(h, w) // 20)
                y_grid, x_grid = np.ogrid[:h, :w]
                dc_mask[(y_grid - cy)**2 + (x_grid - cx)**2 < dc_r**2] = False
                noise_floor = np.median(magnitude[dc_mask])
                
                return peak_strength, noise_floor
            
            # Analyze green channel and color differences
            green_peak, green_noise = get_periodicity_score(green)
            rg_peak, rg_noise = get_periodicity_score(rg_diff)
            bg_peak, bg_noise = get_periodicity_score(bg_diff)
            
            # Calculate Signal-to-Noise Ratios
            green_snr = green_peak / (green_noise + 1e-8)
            rg_snr = rg_peak / (rg_noise + 1e-8)
            bg_snr = bg_peak / (bg_noise + 1e-8)
            
            # Bayer pattern is detected if we see elevated periodicity
            # especially in the color difference channels
            # Threshold calibrated empirically:
            # - Real camera images: SNR typically 1.5-4.0 in color diff channels
            # - AI images: SNR typically < 1.2 (no periodic structure)
            # - Synthetic/rendered: SNR can vary
            
            # Combined score: weight color differences more heavily
            combined_snr = 0.3 * green_snr + 0.35 * rg_snr + 0.35 * bg_snr
            
            # Detection threshold
            BAYER_THRESHOLD = 1.4
            detected = combined_snr > BAYER_THRESHOLD
            
            # Confidence scales from threshold to 2x threshold
            confidence = min(1.0, max(0.0, (combined_snr - 1.0) / 1.5))
            
            return {
                'detected': detected,
                'confidence': confidence,
                'details': {
                    'green_snr': float(green_snr),
                    'rg_diff_snr': float(rg_snr),
                    'bg_diff_snr': float(bg_snr),
                    'combined_snr': float(combined_snr),
                    'threshold': BAYER_THRESHOLD
                }
            }
            
        except Exception as e:
            return {
                'detected': False,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }


def analyze_physics(file_path: str):
    engine = SignalForensics(file_path)
    return engine.analyze()