"""
Layer 10: Shadow Convergence Analysis
"""

import cv2


def get_shadow_score(image_path: str, is_jpeg: bool = False) -> float:
    """
    Shadow-only score extracted from Layer 11's MultiShadowAnalyzer.

    Returns a single float in the range [-50, +30].
    Falls back to 0 (neutral) on any error.
    """
    try:
        from .layer_11_physical_continuity import MultiShadowAnalyzer

        img = cv2.imread(image_path)
        if img is None:
            return 0

        h, w = img.shape[:2]
        if w < 200 or h < 200:
            return 0  # too small for geometric analysis

        analyzer = MultiShadowAnalyzer()
        result = analyzer.analyze(img)

        pair_count = result.get("pair_count", 0)
        if pair_count < 2:
            return 0  # not enough data for a shadow-only verdict

        return float(result.get("score", 0))

    except Exception:
        return 0