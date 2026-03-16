"""
Layer 11: Physical Continuity Analysis (Geometric Truth)
VERSION: 1.0

AI generators excel at textures but fail at projective geometry.
This layer detects violations of mathematical 3D space constraints.

SUB-ANALYZERS:
  11.1: Vanishing Point Coherence - Parallel lines must converge correctly
  11.2: Multi-Object Shadow Vectors - Multiple shadows must share light source

SCORING: -50 (impossible physics) to +30 (consistent geometry)
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import math


# ============================================================================
# 11.1: VANISHING POINT ANALYSIS
# ============================================================================

class VanishingPointAnalyzer:
    """
    Detects vanishing point consistency using Line Segment Detector (LSD).
    
    In real photographs:
    - Parallel lines in 3D converge to a vanishing point in 2D
    - All horizontal parallels converge to one point on the horizon
    - AI often creates lines that don't converge properly
    
    Method:
    1. Detect line segments using LSD
    2. Cluster lines by angle (group parallel-ish lines)
    3. For each cluster, compute intersection points (candidate VPs)
    4. Check if VPs from different clusters are geometrically consistent
    """
    
    def __init__(self, min_line_length: int = 30, angle_tolerance: float = 5.0):
        """
        Args:
            min_line_length: Minimum line segment length in pixels
            angle_tolerance: Degrees tolerance for grouping parallel lines
        """
        self.min_line_length = min_line_length
        self.angle_tolerance = angle_tolerance
        self.lsd = cv2.createLineSegmentDetector(0)
    
    def analyze(self, img: np.ndarray) -> Dict:
        """
        Analyze vanishing point consistency.
        
        Returns:
            Dict with score, description, and analysis details
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        h, w = gray.shape
        
        # Detect line segments
        lines = self._detect_lines(gray)
        
        if len(lines) < 6:
            return {
                "score": 0,
                "description": "Insufficient lines for VP analysis",
                "line_count": len(lines),
                "vp_clusters": 0
            }
        
        # Cluster lines by angle
        clusters = self._cluster_lines_by_angle(lines)
        
        # Filter to significant clusters (at least 3 lines)
        significant_clusters = {k: v for k, v in clusters.items() if len(v) >= 3}
        
        if len(significant_clusters) < 2:
            return {
                "score": 0,
                "description": "Insufficient parallel line groups",
                "line_count": len(lines),
                "vp_clusters": len(significant_clusters)
            }
        
        # Compute vanishing points for each cluster
        vanishing_points = []
        for angle, cluster_lines in significant_clusters.items():
            vp = self._compute_vanishing_point(cluster_lines, (w, h))
            if vp is not None:
                vanishing_points.append({
                    "angle": angle,
                    "vp": vp,
                    "line_count": len(cluster_lines)
                })
        
        if len(vanishing_points) < 2:
            return {
                "score": 0,
                "description": "Could not compute vanishing points",
                "line_count": len(lines),
                "vp_clusters": len(significant_clusters)
            }
        
        # Analyze VP consistency
        consistency = self._analyze_vp_consistency(vanishing_points, (w, h))
        
        # Scoring based on consistency
        score, description = self._compute_score(consistency, vanishing_points)
        
        return {
            "score": score,
            "description": description,
            "line_count": len(lines),
            "vp_clusters": len(vanishing_points),
            "consistency": consistency
        }
    
    def _detect_lines(self, gray: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """Detect line segments using LSD."""
        lines_raw = self.lsd.detect(gray)[0]
        
        if lines_raw is None:
            return []
        
        lines = []
        for line in lines_raw:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length >= self.min_line_length:
                lines.append((x1, y1, x2, y2))
        
        return lines
    
    def _cluster_lines_by_angle(self, lines: List[Tuple]) -> Dict[int, List[Tuple]]:
        """Group lines by their angle (within tolerance)."""
        clusters = defaultdict(list)
        
        for line in lines:
            x1, y1, x2, y2 = line
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Normalize to 0-180 (direction doesn't matter)
            angle = angle % 180
            
            # Quantize to bins
            bin_angle = int(angle / self.angle_tolerance) * self.angle_tolerance
            clusters[bin_angle].append(line)
        
        return dict(clusters)
    
    def _compute_vanishing_point(
        self, 
        lines: List[Tuple], 
        img_size: Tuple[int, int]
    ) -> Optional[Tuple[float, float]]:
        """
        Compute vanishing point for a cluster of parallel lines using RANSAC.
        """
        if len(lines) < 2:
            return None
        
        w, h = img_size
        
        # Compute all pairwise intersections
        intersections = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                pt = self._line_intersection(line1, line2)
                if pt is not None:
                    x, y = pt
                    # Filter points that are too far (likely parallel lines)
                    if abs(x) < w * 10 and abs(y) < h * 10:
                        intersections.append(pt)
        
        if len(intersections) < 3:
            return None
        
        # Use RANSAC to find consensus VP
        best_vp = None
        best_inliers = 0
        threshold = max(w, h) * 0.05  # 5% of image size
        
        np.random.seed(42)  # Reproducibility
        for _ in range(min(100, len(intersections) * 2)):
            # Random sample
            idx = np.random.randint(0, len(intersections))
            candidate = intersections[idx]
            
            # Count inliers
            inliers = 0
            for pt in intersections:
                dist = math.sqrt((pt[0] - candidate[0])**2 + (pt[1] - candidate[1])**2)
                if dist < threshold:
                    inliers += 1
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_vp = candidate
        
        # Require consensus
        if best_inliers < len(intersections) * 0.3:
            return None
        
        return best_vp
    
    def _line_intersection(
        self, 
        line1: Tuple, 
        line2: Tuple
    ) -> Optional[Tuple[float, float]]:
        """Compute intersection of two line segments (extended to infinity)."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:
            return None  # Parallel lines
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        
        return (px, py)
    
    def _analyze_vp_consistency(
        self, 
        vanishing_points: List[Dict], 
        img_size: Tuple[int, int]
    ) -> Dict:
        """
        Analyze geometric consistency of vanishing points.
        
        In real scenes:
        - VPs for horizontal lines should lie near the horizon
        - VPs for vertical lines go to infinity (above/below)
        - The horizon line connects horizontal VPs
        """
        w, h = img_size
        
        # Separate horizontal-ish (0-30°, 150-180°) from vertical-ish (60-120°)
        horizontal_vps = []
        vertical_vps = []
        other_vps = []
        
        for vp_data in vanishing_points:
            angle = vp_data["angle"]
            vp = vp_data["vp"]
            
            if angle < 30 or angle > 150:
                horizontal_vps.append(vp)
            elif 60 < angle < 120:
                vertical_vps.append(vp)
            else:
                other_vps.append(vp)
        
        consistency = {
            "horizontal_vps": len(horizontal_vps),
            "vertical_vps": len(vertical_vps),
            "horizon_consistency": None,
            "vertical_consistency": None
        }
        
        # Check horizontal VP alignment (should form a horizon line)
        if len(horizontal_vps) >= 2:
            # Compute variance in y-coordinate (should be similar for horizon)
            y_coords = [vp[1] for vp in horizontal_vps]
            y_variance = np.var(y_coords) / (h ** 2)  # Normalized
            consistency["horizon_consistency"] = 1.0 - min(1.0, y_variance * 10)
        
        # Check vertical line VPs (should be far above/below, or at infinity)
        if len(vertical_vps) >= 1:
            # Vertical VPs should have extreme y values
            y_coords = [abs(vp[1]) for vp in vertical_vps]
            avg_y_distance = np.mean(y_coords)
            # Good if VP is far from image center
            consistency["vertical_consistency"] = min(1.0, avg_y_distance / (h * 2))
        
        return consistency
    
    def _compute_score(
        self, 
        consistency: Dict, 
        vanishing_points: List[Dict]
    ) -> Tuple[float, str]:
        """Compute final score based on VP consistency analysis."""
        
        # Start neutral
        score = 0
        findings = []
        
        # Count total supporting lines across all VP clusters
        total_lines = sum(vp.get("line_count", 0) for vp in vanishing_points)
        
        # Horizon consistency
        if consistency["horizon_consistency"] is not None:
            hc = consistency["horizon_consistency"]
            if hc > 0.8:
                score += 15
                findings.append("Horizon VPs aligned")
            elif hc < 0.3:
                # FIX: Dampen penalty when few lines detected.
                # Natural/wildlife scenes have very few straight lines (grass,
                # animals, foliage). Forcing VP analysis on < 15 lines produces
                # unreliable results — the "misalignment" is often just noise
                # from curved natural edges being misread as lines.
                if total_lines < 15:
                    dampened = int(-25 * 0.3)  # -7 instead of -25
                    score += dampened
                    findings.append(f"Horizon VPs misaligned ({hc:.2f}) [dampened: only {total_lines} lines]")
                else:
                    score -= 25
                    findings.append(f"Horizon VPs misaligned ({hc:.2f})")
        
        # Multiple VPs found = structured scene
        if len(vanishing_points) >= 3:
            score += 10
            findings.append(f"{len(vanishing_points)} VP clusters found")
        
        # Clamp score
        score = max(-50, min(30, score))
        
        description = "; ".join(findings) if findings else "VP analysis complete"
        return score, description


# ============================================================================
# 11.2: MULTI-OBJECT SHADOW VECTOR ANALYSIS
# ============================================================================

class MultiShadowAnalyzer:
    """
    Analyzes shadow direction consistency across multiple objects.
    
    In real photographs:
    - All shadows must originate from the same light source(s)
    - Shadow length ratios should match object height ratios
    - AI often creates shadows pointing in different directions
    
    Method:
    1. Detect objects (using contours or simple blob detection)
    2. For each object, detect its shadow region
    3. Estimate shadow direction vector for each object
    4. Compare vectors - they should be parallel (single light) or
       show consistent multiple-source pattern
    """
    
    def __init__(self):
        self.min_object_area = 500  # Minimum contour area
    
    def analyze(self, img: np.ndarray) -> Dict:
        """
        Analyze shadow consistency across multiple detected objects.
        """
        h, w = img.shape[:2]
        
        # Detect shadow regions
        shadow_mask = self._detect_shadows(img)
        
        # Detect potential objects (bright regions above shadows)
        object_mask = self._detect_objects(img, shadow_mask)
        
        # Find object-shadow pairs
        pairs = self._find_object_shadow_pairs(img, object_mask, shadow_mask)
        
        if len(pairs) < 2:
            return {
                "score": 0,
                "description": "Insufficient object-shadow pairs",
                "pair_count": len(pairs)
            }
        
        # Compute shadow direction for each pair
        shadow_vectors = []
        for obj_centroid, shadow_centroid in pairs:
            # Vector from object to shadow
            vec = (
                shadow_centroid[0] - obj_centroid[0],
                shadow_centroid[1] - obj_centroid[1]
            )
            magnitude = math.sqrt(vec[0]**2 + vec[1]**2)
            if magnitude > 10:  # Minimum shadow length
                angle = math.atan2(vec[1], vec[0])
                shadow_vectors.append({
                    "angle": angle,
                    "magnitude": magnitude,
                    "object": obj_centroid,
                    "shadow": shadow_centroid
                })
        
        if len(shadow_vectors) < 2:
            return {
                "score": 0,
                "description": "Could not compute shadow vectors",
                "pair_count": len(pairs)
            }
        
        # Analyze vector consistency
        angles = [sv["angle"] for sv in shadow_vectors]
        consistency = self._compute_angular_consistency(angles)
        
        # Scoring
        score, description = self._compute_score(consistency, shadow_vectors)
        
        return {
            "score": score,
            "description": description,
            "pair_count": len(shadow_vectors),
            "consistency": consistency,
            "shadow_vectors": shadow_vectors
        }
    
    def _detect_shadows(self, img: np.ndarray) -> np.ndarray:
        """Detect shadow regions using LAB color space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        
        # Shadows are dark but not completely black
        # Use adaptive threshold for varying lighting
        shadow_mask = cv2.adaptiveThreshold(
            l_chan, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            51, 10
        )
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        return shadow_mask
    
    def _detect_objects(self, img: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
        """Detect bright foreground objects (non-shadow regions with edges)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect object boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges_dilated, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create object mask from significant contours
        object_mask = np.zeros_like(gray)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_object_area:
                cv2.drawContours(object_mask, [cnt], -1, 255, -1)
        
        # Remove shadow regions from object mask
        object_mask = cv2.bitwise_and(object_mask, cv2.bitwise_not(shadow_mask))
        
        return object_mask
    
    def _find_object_shadow_pairs(
        self, 
        img: np.ndarray,
        object_mask: np.ndarray, 
        shadow_mask: np.ndarray
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Find pairs of objects and their corresponding shadows."""
        h, w = img.shape[:2]
        
        # Find object contours
        obj_contours, _ = cv2.findContours(
            object_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Find shadow contours
        shadow_contours, _ = cv2.findContours(
            shadow_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        pairs = []
        
        for obj_cnt in obj_contours:
            if cv2.contourArea(obj_cnt) < self.min_object_area:
                continue
            
            # Object centroid
            M = cv2.moments(obj_cnt)
            if M["m00"] == 0:
                continue
            obj_cx = int(M["m10"] / M["m00"])
            obj_cy = int(M["m01"] / M["m00"])
            
            # Find nearest shadow below or beside the object
            best_shadow = None
            best_distance = float('inf')
            
            for shadow_cnt in shadow_contours:
                if cv2.contourArea(shadow_cnt) < self.min_object_area * 0.3:
                    continue
                
                # Shadow centroid
                M_s = cv2.moments(shadow_cnt)
                if M_s["m00"] == 0:
                    continue
                shadow_cx = int(M_s["m10"] / M_s["m00"])
                shadow_cy = int(M_s["m01"] / M_s["m00"])
                
                # Shadow should be below or to the side of object (not above)
                # Allow some tolerance for perspective
                if shadow_cy < obj_cy - h * 0.1:
                    continue
                
                # Distance
                dist = math.sqrt((obj_cx - shadow_cx)**2 + (obj_cy - shadow_cy)**2)
                
                # Shadow shouldn't be too far
                if dist < best_distance and dist < max(w, h) * 0.4:
                    best_distance = dist
                    best_shadow = (shadow_cx, shadow_cy)
            
            if best_shadow is not None:
                pairs.append(((obj_cx, obj_cy), best_shadow))
        
        return pairs
    
    def _compute_angular_consistency(self, angles: List[float]) -> Dict:
        """Compute consistency metrics for shadow angles."""
        if len(angles) < 2:
            return {"variance": 0, "mean_angle": 0}
        
        angles_arr = np.array(angles)
        
        # Circular mean
        cos_sum = np.sum(np.cos(angles_arr))
        sin_sum = np.sum(np.sin(angles_arr))
        mean_angle = math.atan2(sin_sum, cos_sum)
        
        # Circular variance (0 = consistent, 1 = chaotic)
        R = math.sqrt(cos_sum**2 + sin_sum**2) / len(angles_arr)
        circular_variance = 1 - R
        
        # Angular spread (max deviation from mean)
        deviations = []
        for angle in angles:
            diff = abs(angle - mean_angle)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            deviations.append(diff)
        max_deviation = max(deviations)
        
        return {
            "circular_variance": circular_variance,
            "mean_angle": mean_angle,
            "max_deviation_deg": math.degrees(max_deviation),
            "spread_deg": math.degrees(max_deviation) * 2
        }
    
    def _compute_score(
        self, 
        consistency: Dict, 
        shadow_vectors: List[Dict]
    ) -> Tuple[float, str]:
        """Compute final score based on shadow vector consistency."""
        
        score = 0
        findings = []
        
        circ_var = consistency.get("circular_variance", 0.5)
        spread = consistency.get("spread_deg", 90)
        
        # ================================================================
        # SPREAD-BASED SCORING (Primary - Physics Constraint)
        # ================================================================
        # A shadow spread > 180° means light is coming from opposite directions
        # simultaneously, which is PHYSICALLY IMPOSSIBLE in a single photograph.
        # This is a hard fail that overrides circular variance scoring.
        # ================================================================
        
        if spread > 300:
            # Near-omnidirectional lighting — suspicious but could be outdoor
            # ambient. Contour-based shadow detection in open natural scenes
            # (grass, savanna, forest) picks up every dark patch as a "shadow"
            # creating garbage pairs with random directions.
            # FIX: If many pairs but high spread, it's likely noisy detection
            # in an outdoor scene, not truly impossible physics.
            pair_count = len(shadow_vectors)
            if pair_count >= 5 and spread > 330:
                # Many pairs + near-omnidirectional = noisy scene, not AI
                score = -10
                findings.append(f"Shadow spread {spread:.0f}° with {pair_count} pairs "
                               f"(likely outdoor ambient lighting, dampened)")
            else:
                score = -50
                findings.append(f"IMPOSSIBLE: Shadow spread {spread:.0f}° (light from all directions)")
            # Don't early return — let circular variance refine
        
        elif spread > 180:
            # Light from opposite directions — suspicious
            pair_count = len(shadow_vectors)
            if pair_count >= 5:
                score = -15
                findings.append(f"Shadow spread {spread:.0f}° with {pair_count} pairs "
                               f"(ambiguous, dampened)")
            else:
                score = -45
                findings.append(f"IMPOSSIBLE: Shadow spread {spread:.0f}° (opposing light sources)")
        
        elif spread > 120:
            # Very wide spread - suspicious but could be strong ambient
            score -= 30
            findings.append(f"Suspicious shadow spread {spread:.0f}° (likely AI)")
        
        elif spread > 90:
            # Wide spread - multiple light sources, still possible
            score -= 15
            findings.append(f"Wide shadow spread {spread:.0f}° (multiple sources?)")
        
        else:
            # Spread <= 90° - physically reasonable
            # Now use circular variance for fine-grained scoring
            if circ_var < 0.1:
                score += 25
                findings.append("Perfect shadow alignment")
            elif circ_var < 0.25:
                score += 15
                findings.append("Consistent single light source")
            elif circ_var < 0.4:
                score += 5
                findings.append("Multiple light sources (normal)")
            elif circ_var < 0.6:
                score -= 10
                findings.append(f"Moderately inconsistent shadows (var={circ_var:.2f})")
            else:
                score -= 20
                findings.append(f"Chaotic shadow directions (var={circ_var:.2f})")
        
        # Bonus for multiple verified shadows (more data = more confidence)
        if len(shadow_vectors) >= 4:
            if score > 0:
                score += 5
                findings.append(f"{len(shadow_vectors)} shadows analyzed")
            elif score < -30:
                # Multiple shadows all showing impossible physics = worse
                score -= 5
                findings.append(f"{len(shadow_vectors)} shadows confirm violation")
        
        # Clamp
        score = max(-50, min(30, score))
        
        description = "; ".join(findings) if findings else "Shadow analysis complete"
        return score, description


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_physical_continuity(image_path: str) -> Dict:
    """
    Layer 11: Full physical continuity analysis.
    
    Combines:
    - 11.1: Vanishing Point Coherence
    - 11.2: Multi-Object Shadow Vectors
    
    Returns:
        Dict with combined score, individual sub-scores, and description
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "score": 0,
                "description": "Image load failed",
                "vp_result": None,
                "shadow_result": None
            }
        
        h, w = img.shape[:2]
        
        # Skip very small images
        if w < 200 or h < 200:
            return {
                "score": 0,
                "description": "Image too small for geometric analysis",
                "vp_result": None,
                "shadow_result": None
            }
        
        # 11.1: Vanishing Point Analysis
        vp_analyzer = VanishingPointAnalyzer()
        vp_result = vp_analyzer.analyze(img)
        
        # 11.2: Multi-Shadow Analysis
        shadow_analyzer = MultiShadowAnalyzer()
        shadow_result = shadow_analyzer.analyze(img)
        
        # Combine scores
        # Weight VP slightly higher as it's more reliable
        vp_score = vp_result["score"]
        shadow_score = shadow_result["score"]
        
        # Only count scores from analyzers that had enough data
        active_scores = []
        active_weights = []
        
        if vp_result.get("vp_clusters", 0) >= 2:
            active_scores.append(vp_score)
            active_weights.append(0.6)
        
        if shadow_result.get("pair_count", 0) >= 2:
            active_scores.append(shadow_score)
            active_weights.append(0.4)
        
        if not active_scores:
            combined_score = 0
            description = "Insufficient geometric features"
        else:
            # Normalize weights
            total_weight = sum(active_weights)
            combined_score = sum(s * w for s, w in zip(active_scores, active_weights)) / total_weight
            combined_score = int(round(combined_score))
            
            # Build description
            desc_parts = []
            if vp_result.get("vp_clusters", 0) >= 2:
                desc_parts.append(f"VP: {vp_result['description']}")
            if shadow_result.get("pair_count", 0) >= 2:
                desc_parts.append(f"Shadows: {shadow_result['description']}")
            description = " | ".join(desc_parts) if desc_parts else "Analysis complete"
        
        return {
            "score": combined_score,
            "description": description,
            "vp_result": vp_result,
            "shadow_result": shadow_result
        }
        
    except Exception as e:
        return {
            "score": 0,
            "description": f"Analysis error: {str(e)}",
            "vp_result": None,
            "shadow_result": None
        }


# Convenience function for integration
def get_physical_continuity_score(image_path: str) -> Tuple[float, str]:
    """
    Simple interface returning (score, description).
    """
    result = analyze_physical_continuity(image_path)
    return result["score"], result["description"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_physical_continuity(sys.argv[1])
        print(f"\n=== Layer 11: Physical Continuity ===")
        print(f"Combined Score: {result['score']}")
        print(f"Description: {result['description']}")
        
        if result['vp_result']:
            print(f"\n11.1 Vanishing Points:")
            print(f"  Score: {result['vp_result']['score']}")
            print(f"  Lines: {result['vp_result']['line_count']}")
            print(f"  VP Clusters: {result['vp_result']['vp_clusters']}")
        
        if result['shadow_result']:
            print(f"\n11.2 Multi-Shadow:")
            print(f"  Score: {result['shadow_result']['score']}")
            print(f"  Pairs: {result['shadow_result']['pair_count']}")
    else:
        print("Usage: python layer_11_physical_continuity.py <image_path>")
