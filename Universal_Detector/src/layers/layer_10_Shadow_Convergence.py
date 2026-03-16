"""
Layer 10: Shadow Convergence Analysis
VERSION: 3.0 - Unified Delegation

Prior to v3.0, this module contained a shallow shadow analysis that overlapped
with Layer 11's superior MultiShadowAnalyzer (contour-based object-shadow
pair detection, angular spread, circular variance).

v3.0 eliminates the duplication by delegating entirely to Layer 11's shadow
sub-analyzer.  The `get_shadow_score()` interface is preserved so main.py
and Layer 5 continue to receive an independent shadow score.

WHY KEEP THIS MODULE?
  Layer 5 Judge uses `shadow_score` and `physical_continuity_score` as two
  separate inputs with independent weights (shadow: 0.05/0.12, physical_
  continuity: 0.08).  `physical_continuity_score` is a *combined* VP + Shadow
  score.  This module provides a *shadow-only* signal so the Judge can
  reason about shadow physics in isolation.
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