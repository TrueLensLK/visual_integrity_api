import cv2
import numpy as np
import os
import warnings
from typing import Dict, List, Tuple, Optional

"""
Layer 3.5: Face Consistency & Face-Swap Detection
"""

# ============================================================================
# FACE DETECTOR INITIALIZATION WITH CLEAR WARNINGS
# ============================================================================

_FACE_DETECTOR = None
_HAAR_CASCADE = None
USE_DNN = False

# Try to find YuNet model in multiple locations
_YUNET_MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), "face_detection_yunet_2023mar.onnx"),
    os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"),
    "face_detection_yunet_2023mar.onnx",
]

def _init_face_detector():
    """Initialize face detector with clear warnings about fallback."""
    global _FACE_DETECTOR, _HAAR_CASCADE, USE_DNN
    
    # Try YuNet first (more accurate)
    for model_path in _YUNET_MODEL_PATHS:
        if os.path.exists(model_path):
            try:
                _FACE_DETECTOR = cv2.FaceDetectorYN.create(model_path, "", (320, 320))
                USE_DNN = True
                print(f"[Layer 3.5] YuNet face detector loaded: {model_path}")
                return
            except Exception as e:
                print(f"[Layer 3.5] YuNet load failed: {e}")
    
    # Fallback to Haar cascade with clear warning
    USE_DNN = False
    _HAAR_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    warning_msg = (
        "[Layer 3.5] WARNING: YuNet model (face_detection_yunet_2023mar.onnx) not found!\n"
        "  → Falling back to Haar cascade (less accurate, misses profile/small faces)\n"
        "  → To fix: Download YuNet from OpenCV Zoo:\n"
        "    https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet\n"
        f"  → Place in: {os.path.dirname(__file__)}"
    )
    print(warning_msg)
    warnings.warn("YuNet face detector not available, using Haar cascade fallback", UserWarning)

# Initialize on module load
_init_face_detector()

class FaceForensics:
    """
    Enhanced face forensics with face-swap detection capabilities.
    """
    def __init__(self, img_path: str):
        self.img = cv2.imread(img_path)
        if self.img is None: raise ValueError("Image load failed")
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.gray.shape
        self.impact = 0
        self.findings = []
        self.detector_type = "YuNet" if USE_DNN else "Haar (fallback)"

    def analyze(self) -> Dict:
        # Log detector type
        self.findings.append(f"Detector: {self.detector_type}")
        
        # 1. MULTI-FACE DETECTION (Gap L3.5-2)
        faces = self._get_faces()
        if faces is None or len(faces) == 0:
            return {"impact": 0, "face_count": 0, "findings": self.findings, "status": "No faces detected"}

        total_face_impact = 0
        
        # 2. DYNAMIC BACKGROUND SAMPLING (Gap L3.5-3)
        bg_stats = self._get_background_reference(faces)

        face_data = []  # Store per-face analysis for cross-comparison
        
        for i, (x, y, fw, fh) in enumerate(faces):
            face_roi = self.gray[y:y+fh, x:x+fw]
            face_color = self.img[y:y+fh, x:x+fw]
            
            face_analysis = {
                'index': i,
                'bbox': (x, y, fw, fh),
                'penalties': []
            }
            
            # 3. NOISE & BLUR CONSISTENCY (Gap L3.5-5)
            f_noise = cv2.Laplacian(face_roi, cv2.CV_64F).var()
            noise_ratio = f_noise / (bg_stats['noise_var'] + 1e-5)
            face_analysis['noise_ratio'] = noise_ratio
            
            if noise_ratio < 0.25:
                total_face_impact -= 15
                self.findings.append(f"Face {i}: AI-Denoising mismatch (too smooth)")
                face_analysis['penalties'].append('smooth')
            elif noise_ratio > 4.0:
                total_face_impact -= 20
                self.findings.append(f"Face {i}: Resolution mismatch (appears pasted)")
                face_analysis['penalties'].append('resolution')
            
            # 4. BOUNDARY ARTIFACTS (Gap L3.5-6) - Enhanced
            seam_result = self._detect_seam_enhanced(x, y, fw, fh)
            if seam_result['has_seam']:
                total_face_impact -= seam_result['penalty']
                self.findings.append(f"Face {i}: {seam_result['description']}")
                face_analysis['penalties'].append('seam')
            
            # 5. NEW: FACE-SWAP DETECTION - Warping Artifacts
            warp_result = self._detect_warping_artifacts(face_roi)
            if warp_result['detected']:
                total_face_impact -= warp_result['penalty']
                self.findings.append(f"Face {i}: {warp_result['description']}")
                face_analysis['penalties'].append('warping')
            
            # 6. NEW: FACE-SWAP DETECTION - Skin Tone Mismatch
            skin_result = self._detect_skin_tone_mismatch(x, y, fw, fh)
            if skin_result['mismatch']:
                total_face_impact -= skin_result['penalty']
                self.findings.append(f"Face {i}: {skin_result['description']}")
                face_analysis['penalties'].append('skin_tone')
            face_analysis['skin_tone'] = skin_result.get('face_tone')
            
            # 7. NEW: FACE-SWAP DETECTION - GAN Boundary Artifacts
            gan_result = self._detect_gan_boundary_artifacts(x, y, fw, fh)
            if gan_result['detected']:
                total_face_impact -= gan_result['penalty']
                self.findings.append(f"Face {i}: {gan_result['description']}")
                face_analysis['penalties'].append('gan_boundary')
            
            face_data.append(face_analysis)
        
        # 8. NEW: Multi-face cross-comparison (lighting consistency)
        if len(face_data) > 1:
            cross_result = self._cross_compare_faces(face_data)
            if cross_result['inconsistent']:
                total_face_impact -= cross_result['penalty']
                self.findings.append(cross_result['description'])

        self.impact = max(-50, total_face_impact)
        return {
            "impact": self.impact,
            "face_count": len(faces),
            "findings": self.findings,
            "detector": self.detector_type,
            "face_details": face_data
        }

    def _get_faces(self) -> List[Tuple[int, int, int, int]]:
        if USE_DNN and _FACE_DETECTOR is not None:
            _FACE_DETECTOR.setInputSize((self.w, self.h))
            _, faces = _FACE_DETECTOR.detect(self.img)
            return [tuple(f[:4].astype(int)) for f in faces] if faces is not None else []
        else:
            faces = _HAAR_CASCADE.detectMultiScale(self.gray, 1.1, 5)
            if len(faces) == 0:
                return []
            return [tuple(f) for f in faces]

    def _get_background_reference(self, faces) -> Dict:
        """Gap L3.5-3: Samples non-face regions to get a baseline for sensor noise."""
        mask = np.ones((self.h, self.w), dtype=np.uint8) * 255
        for (x, y, w, h) in faces:
            # Mask out faces + 20px buffer
            cv2.rectangle(mask, (x-20, y-20), (x+w+20, y+h+20), 0, -1)
        
        # Sample noise from the unmasked (actual background) areas
        bg_pixels = self.gray[mask > 0]
        if len(bg_pixels) < 100:
            return {'noise_var': 50.0}  # Default if not enough background
        bg_noise = cv2.Laplacian(self.gray, cv2.CV_64F)[mask > 0].var()
        return {'noise_var': bg_noise}

    def _detect_seam_enhanced(self, x: int, y: int, w: int, h: int) -> Dict:
        """
        Enhanced seam detection: checks for sharp gradients at face boundary.
        Now also checks for color discontinuities (not just grayscale edges).
        """
        pad = 8
        y1, y2 = max(0, y-pad), min(self.h, y+h+pad)
        x1, x2 = max(0, x-pad), min(self.w, x+w+pad)
        
        roi_outer = self.gray[y1:y2, x1:x2]
        
        # Edge gradient analysis
        grad_x = cv2.Sobel(roi_outer, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_outer, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(grad_x, grad_y)
        
        # Calculate boundary vs interior gradient ratio
        boundary_mask = np.zeros_like(roi_outer, dtype=bool)
        boundary_mask[:pad, :] = True  # Top
        boundary_mask[-pad:, :] = True  # Bottom
        boundary_mask[:, :pad] = True  # Left
        boundary_mask[:, -pad:] = True  # Right
        
        boundary_grad = np.mean(mag[boundary_mask]) if np.any(boundary_mask) else 0
        interior_grad = np.mean(mag[~boundary_mask]) if np.any(~boundary_mask) else 1
        
        grad_ratio = boundary_grad / (interior_grad + 1e-5)
        
        # Also check color channel discontinuity
        roi_color = self.img[y1:y2, x1:x2]
        color_discontinuity = self._check_color_discontinuity(roi_color, pad)
        
        if grad_ratio > 3.0 and color_discontinuity > 0.15:
            return {
                'has_seam': True,
                'penalty': 15,
                'description': f'Strong digital seam (grad_ratio={grad_ratio:.1f}, color_disc={color_discontinuity:.2f})'
            }
        elif grad_ratio > 2.0 or color_discontinuity > 0.12:
            return {
                'has_seam': True,
                'penalty': 8,
                'description': f'Possible seam artifact (grad_ratio={grad_ratio:.1f})'
            }
        
        return {'has_seam': False, 'penalty': 0, 'description': ''}
    
    def _check_color_discontinuity(self, roi_color: np.ndarray, pad: int) -> float:
        """Check for LAB color space discontinuity at boundary."""
        try:
            lab = cv2.cvtColor(roi_color, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Compare boundary colors to interior colors
            h, w = lab.shape[:2]
            if h < pad*3 or w < pad*3:
                return 0.0
            
            boundary = np.concatenate([
                lab[:pad, :].reshape(-1, 3),
                lab[-pad:, :].reshape(-1, 3),
                lab[:, :pad].reshape(-1, 3),
                lab[:, -pad:].reshape(-1, 3)
            ])
            interior = lab[pad:-pad, pad:-pad].reshape(-1, 3)
            
            if len(interior) == 0:
                return 0.0
            
            # Calculate mean color difference (in LAB space, perceptually uniform)
            boundary_mean = np.mean(boundary, axis=0)
            interior_mean = np.mean(interior, axis=0)
            
            # Delta E approximation
            delta_e = np.sqrt(np.sum((boundary_mean - interior_mean) ** 2)) / 100.0
            return float(delta_e)
        except:
            return 0.0

    def _detect_warping_artifacts(self, face_roi: np.ndarray) -> Dict:
        """
        Detect face warping artifacts from face-swap algorithms.
        
        Face swaps often use thin-plate spline or similar warping which creates:
        1. Unnatural curvature in straight lines (especially around edges)
        2. Local distortion patterns
        3. Inconsistent perspective
        """
        try:
            if face_roi.size < 100:
                return {'detected': False, 'penalty': 0, 'description': ''}
            
            # 1. Detect straight line distortion using Hough lines
            edges = cv2.Canny(face_roi, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                    minLineLength=20, maxLineGap=5)
            
            if lines is None or len(lines) < 3:
                # Not enough lines to analyze
                return {'detected': False, 'penalty': 0, 'description': ''}
            
            # 2. Calculate line angle variance
            # In a natural face, detected lines should have consistent angles
            # Warping creates inconsistent angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            
            angle_variance = np.var(angles)
            
            # 3. Check for unusual curvature using local binary patterns
            # Warping creates unusual texture patterns
            lbp_score = self._compute_lbp_uniformity(face_roi)
            
            # High angle variance + low LBP uniformity = warping
            if angle_variance > 2000 and lbp_score < 0.3:
                return {
                    'detected': True,
                    'penalty': 12,
                    'description': f'Warping artifacts detected (angle_var={angle_variance:.0f}, lbp={lbp_score:.2f})'
                }
            elif angle_variance > 1500 or lbp_score < 0.25:
                return {
                    'detected': True,
                    'penalty': 6,
                    'description': 'Possible warping distortion'
                }
            
            return {'detected': False, 'penalty': 0, 'description': ''}
            
        except Exception:
            return {'detected': False, 'penalty': 0, 'description': ''}
    
    def _compute_lbp_uniformity(self, roi: np.ndarray) -> float:
        """
        Compute Local Binary Pattern uniformity score.
        Natural faces have higher uniformity; GAN/warped faces have lower.
        """
        try:
            if len(roi.shape) == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            h, w = roi.shape
            if h < 10 or w < 10:
                return 0.5
            
            # Simple LBP computation
            lbp = np.zeros((h-2, w-2), dtype=np.uint8)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = roi[i, j]
                    code = 0
                    code |= (roi[i-1, j-1] > center) << 7
                    code |= (roi[i-1, j] > center) << 6
                    code |= (roi[i-1, j+1] > center) << 5
                    code |= (roi[i, j+1] > center) << 4
                    code |= (roi[i+1, j+1] > center) << 3
                    code |= (roi[i+1, j] > center) << 2
                    code |= (roi[i+1, j-1] > center) << 1
                    code |= (roi[i, j-1] > center) << 0
                    lbp[i-1, j-1] = code
            
            # Count uniform patterns (patterns with <= 2 bit transitions)
            uniform_count = 0
            total = lbp.size
            for val in lbp.flatten():
                # Count bit transitions
                binary = bin(val)[2:].zfill(8)
                transitions = sum(binary[i] != binary[i+1] for i in range(7))
                transitions += binary[7] != binary[0]  # Circular
                if transitions <= 2:
                    uniform_count += 1
            
            return uniform_count / total if total > 0 else 0.5
            
        except:
            return 0.5

    def _detect_skin_tone_mismatch(self, x: int, y: int, w: int, h: int) -> Dict:
        """
        Detect skin tone mismatch between face and surrounding areas (neck, ears).
        Face swaps often have color/tone inconsistencies at boundaries.
        """
        try:
            # Get face region in LAB color space
            face_region = self.img[y:y+h, x:x+w]
            face_lab = cv2.cvtColor(face_region, cv2.COLOR_BGR2LAB)
            
            # Get surrounding regions (below face = neck area, sides = ears)
            neck_y1 = min(y + h, self.h - 1)
            neck_y2 = min(y + h + h//3, self.h)
            neck_region = self.img[neck_y1:neck_y2, x:x+w] if neck_y2 > neck_y1 else None
            
            # Left side (potential ear/hair)
            left_x1 = max(0, x - w//4)
            left_region = self.img[y:y+h, left_x1:x] if left_x1 < x else None
            
            # Right side
            right_x2 = min(x + w + w//4, self.w)
            right_region = self.img[y:y+h, x+w:right_x2] if x+w < right_x2 else None
            
            # Extract skin-like pixels using simple color thresholding
            def get_skin_tone(region):
                if region is None or region.size < 50:
                    return None
                lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
                # Simple skin detection: moderate L, slight red bias (a), slight yellow (b)
                l, a, b = cv2.split(lab)
                skin_mask = (l > 50) & (l < 220) & (a > 120) & (a < 180) & (b > 120) & (b < 180)
                skin_pixels = lab[skin_mask]
                if len(skin_pixels) < 10:
                    return None
                return np.mean(skin_pixels, axis=0)
            
            face_tone = get_skin_tone(face_region)
            if face_tone is None:
                return {'mismatch': False, 'penalty': 0, 'description': '', 'face_tone': None}
            
            # Compare with surrounding regions
            mismatches = []
            for name, region in [('neck', neck_region), ('left', left_region), ('right', right_region)]:
                tone = get_skin_tone(region)
                if tone is not None:
                    # Delta E in LAB space
                    delta = np.sqrt(np.sum((face_tone - tone) ** 2))
                    if delta > 25:  # Noticeable difference
                        mismatches.append((name, delta))
            
            if len(mismatches) >= 2:
                return {
                    'mismatch': True,
                    'penalty': 15,
                    'description': f'Skin tone mismatch with {", ".join(m[0] for m in mismatches)}',
                    'face_tone': face_tone.tolist()
                }
            elif len(mismatches) == 1:
                return {
                    'mismatch': True,
                    'penalty': 8,
                    'description': f'Slight skin tone mismatch ({mismatches[0][0]}: ΔE={mismatches[0][1]:.0f})',
                    'face_tone': face_tone.tolist()
                }
            
            return {'mismatch': False, 'penalty': 0, 'description': '', 'face_tone': face_tone.tolist()}
            
        except Exception:
            return {'mismatch': False, 'penalty': 0, 'description': '', 'face_tone': None}

    def _detect_gan_boundary_artifacts(self, x: int, y: int, w: int, h: int) -> Dict:
        """
        Detect GAN-specific boundary artifacts around face edges.
        
        GANs (especially face-swap GANs like SimSwap, FaceShifter) leave
        characteristic high-frequency artifacts at blend boundaries:
        1. Ringing/halo effects
        2. Checkerboard patterns from upsampling
        3. Unnatural frequency distribution at edges
        """
        try:
            # Expand slightly to capture boundary
            pad = 15
            y1, y2 = max(0, y - pad), min(self.h, y + h + pad)  
            x1, x2 = max(0, x - pad), min(self.w, x + w + pad)
            
            roi = self.gray[y1:y2, x1:x2].astype(np.float32)
            
            if roi.size < 200:
                return {'detected': False, 'penalty': 0, 'description': ''}
            
            # 1. Check for high-frequency ringing at boundary
            # Apply high-pass filter
            kernel_hp = np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]], dtype=np.float32)
            hp_response = cv2.filter2D(roi, cv2.CV_32F, kernel_hp)
            
            # Create boundary mask (ring around face)
            rh, rw = roi.shape
            inner_mask = np.zeros((rh, rw), dtype=bool)
            inner_y1, inner_y2 = pad, rh - pad
            inner_x1, inner_x2 = pad, rw - pad
            if inner_y2 > inner_y1 and inner_x2 > inner_x1:
                inner_mask[inner_y1:inner_y2, inner_x1:inner_x2] = True
            
            boundary_mask = ~inner_mask
            
            # Compare high-frequency energy at boundary vs interior
            boundary_hf = np.mean(np.abs(hp_response[boundary_mask])) if np.any(boundary_mask) else 0
            interior_hf = np.mean(np.abs(hp_response[inner_mask])) if np.any(inner_mask) else 1
            
            hf_ratio = boundary_hf / (interior_hf + 1e-5)
            
            # 2. Check for checkerboard pattern using FFT
            f = np.fft.fft2(roi)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            cy, cx = rh // 2, rw // 2
            # Checkerboard creates peaks at half-Nyquist
            qy, qx = rh // 4, rw // 4
            
            if qy > 2 and qx > 2:
                corner_energy = (
                    magnitude[cy-qy-1:cy-qy+2, cx-qx-1:cx-qx+2].mean() +
                    magnitude[cy-qy-1:cy-qy+2, cx+qx-1:cx+qx+2].mean() +
                    magnitude[cy+qy-1:cy+qy+2, cx-qx-1:cx-qx+2].mean() +
                    magnitude[cy+qy-1:cy+qy+2, cx+qx-1:cx+qx+2].mean()
                ) / 4
                avg_energy = np.median(magnitude)
                checker_ratio = corner_energy / (avg_energy + 1e-5)
            else:
                checker_ratio = 1.0
            
            # Detection logic
            if hf_ratio > 2.5 and checker_ratio > 3.0:
                return {
                    'detected': True,
                    'penalty': 18,
                    'description': f'GAN boundary artifacts (HF_ratio={hf_ratio:.1f}, checker={checker_ratio:.1f})'
                }
            elif hf_ratio > 2.0 or checker_ratio > 2.5:
                return {
                    'detected': True,
                    'penalty': 10,
                    'description': f'Possible GAN artifacts at face boundary'
                }
            
            return {'detected': False, 'penalty': 0, 'description': ''}
            
        except Exception:
            return {'detected': False, 'penalty': 0, 'description': ''}

    def _cross_compare_faces(self, face_data: List[Dict]) -> Dict:
        """
        Cross-compare multiple faces for consistency.
        In real photos, all faces have similar:
        - Noise characteristics (same camera)
        - Lighting direction
        - Color temperature
        """
        if len(face_data) < 2:
            return {'inconsistent': False, 'penalty': 0, 'description': ''}
        
        try:
            # Compare noise ratios between faces
            noise_ratios = [f['noise_ratio'] for f in face_data if 'noise_ratio' in f]
            if len(noise_ratios) >= 2:
                noise_variance = np.var(noise_ratios)
                noise_ratio_range = max(noise_ratios) / (min(noise_ratios) + 1e-5)
                
                if noise_ratio_range > 3.0:
                    return {
                        'inconsistent': True,
                        'penalty': 12,
                        'description': f'Multi-face noise inconsistency (range={noise_ratio_range:.1f}x)'
                    }
            
            # Compare skin tones between faces
            skin_tones = [f['skin_tone'] for f in face_data if f.get('skin_tone') is not None]
            if len(skin_tones) >= 2:
                # Calculate pairwise color differences
                max_delta = 0
                for i in range(len(skin_tones)):
                    for j in range(i+1, len(skin_tones)):
                        delta = np.sqrt(np.sum((np.array(skin_tones[i]) - np.array(skin_tones[j])) ** 2))
                        max_delta = max(max_delta, delta)
                
                if max_delta > 35:  # Significant color temperature difference
                    return {
                        'inconsistent': True,
                        'penalty': 10,
                        'description': f'Multi-face color inconsistency (ΔE={max_delta:.0f})'
                    }
            
            return {'inconsistent': False, 'penalty': 0, 'description': ''}
            
        except:
            return {'inconsistent': False, 'penalty': 0, 'description': ''} 

def analyze_face_consistency(image_path: str):
    engine = FaceForensics(image_path)
    return engine.analyze()