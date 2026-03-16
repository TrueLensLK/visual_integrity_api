"""
Perceptual Hash Extractor  –  Fast-Path Forensic Fingerprinting
===============================================================
Implements the two forensic mitigations identified in adversarial
perceptual-hash research:

  Mitigation 1 – Border Stripping
      A uniform border (black letterboxing, white padding, etc.) shifts
      the Hamming distance distribution of standard pHash by up to 40 %,
      making near-duplicate detection unreliable.  We detect and crop such
      borders using pixel-variance analysis before hashing.

  Mitigation 2 – Mirror Defense
      Horizontally flipping an image entirely breaks a standard pHash
      lookup.  By always storing *both* the original and mirrored hash,
      upstream Hamming-distance queries catch mirror-attack evasion.

Public API
----------
    extract_phash(image_bytes: bytes) -> PhashResult

Constants
---------
    BORDER_TOLERANCE   – Max per-channel std-dev to classify a row/col as
                         "uniform".  15 absorbs JPEG ringing in solid borders.
    PHASH_MAX_BYTES    – Hard download-size cap (bytes).
    PHASH_DOWNLOAD_TIMEOUT – httpx timeout in seconds.

Hash libraries (priority order)
--------------------------------
    1. pdqhash  – Facebook PDQ, 256-bit.  Most robust against rescaling.
    2. imagehash.phash – DCT-based, 64-bit.  Widely available fallback.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Optional hash libraries (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import pdqhash as _pdqhash_lib  # type: ignore
    _PDQ_AVAILABLE = True
except ImportError:
    _PDQ_AVAILABLE = False

try:
    import imagehash as _imagehash_lib  # type: ignore
    _IMAGEHASH_AVAILABLE = True
except ImportError:
    _IMAGEHASH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BORDER_TOLERANCE: int = 15          # pixel std-dev threshold for border detection
PHASH_MAX_BYTES: int = 20 * 1024 * 1024   # 20 MB
PHASH_DOWNLOAD_TIMEOUT: int = 10    # seconds


# ---------------------------------------------------------------------------
# Public result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhashResult:
    """Immutable result returned by :func:`extract_phash`."""

    original_hash: str    # Binary string  e.g. "10110010…"
    mirrored_hash: str    # Binary string of the FLIP_LEFT_RIGHT variant
    hash_algorithm: str   # "pdq" | "phash"
    hash_bits: int        # Length of each binary string (256 for PDQ, 64 for phash)
    border_stripped: bool # True when a uniform border was detected and removed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _strip_uniform_borders(img_array: np.ndarray, tolerance: int = BORDER_TOLERANCE) -> np.ndarray:
    """
    Crop uniform-coloured borders from an RGB uint8 NumPy array.

    Implementation is **fully vectorised**: per-channel standard deviations
    are computed for every row and every column in a single NumPy call
    (no Python-level loops), making this suitable for millisecond fast-path
    execution even on high-resolution images.

    A row or column is considered "uniform" when the maximum per-channel
    standard deviation across all its pixels is at or below *tolerance*.
    A value of 15 comfortably absorbs JPEG compression ringing around
    solid-colour padding without trimming real content.

    Args:
        img_array:  H×W×C uint8 NumPy array (RGB).
        tolerance:  Per-channel std-dev ceiling for "uniform" classification.

    Returns:
        Cropped uint8 array.  Returns the original unchanged when no border
        is detected or when the entire image appears uniform.
    """
    if img_array.ndim != 3 or img_array.shape[2] < 3:
        return img_array  # grayscale / single-channel – skip

    h, w = img_array.shape[:2]

    # ---- vectorised row uniformity: shape (H,) -------------------------
    # std over all pixels in each row, per channel → (H, C); take max over C
    row_std_max = np.max(
        np.std(img_array.reshape(h, -1, img_array.shape[2]).astype(np.float32), axis=1),
        axis=1,
    )  # shape: (H,)

    # ---- vectorised column uniformity: shape (W,) ----------------------
    col_std_max = np.max(
        np.std(img_array.transpose(1, 0, 2).reshape(w, -1, img_array.shape[2]).astype(np.float32), axis=1),
        axis=1,
    )  # shape: (W,)

    uniform_rows = row_std_max <= tolerance   # bool array (H,)
    uniform_cols = col_std_max <= tolerance   # bool array (W,)

    # Find the first / last non-uniform row/col
    non_uniform_rows = np.where(~uniform_rows)[0]
    non_uniform_cols = np.where(~uniform_cols)[0]

    # Safety guard: if the whole image is uniform, return it as-is
    if non_uniform_rows.size == 0 or non_uniform_cols.size == 0:
        return img_array

    top    = int(non_uniform_rows[0])
    bottom = int(non_uniform_rows[-1])
    left   = int(non_uniform_cols[0])
    right  = int(non_uniform_cols[-1])

    return img_array[top : bottom + 1, left : right + 1]


def _hash_to_binary_string(pil_image: Image.Image) -> str:
    """
    Compute a perceptual hash and return it as a ``'0'``/``'1'`` binary string.

    The binary-string format lets upstream Node.js / SQL services execute
    native bitwise XOR + POPCOUNT Hamming distance queries without any
    additional encoding.

    Priority:
      1. ``pdqhash`` (Facebook PDQ, 256-bit) – most robust against rescaling.
      2. ``imagehash.phash`` (DCT-based, 64-bit) – widely available fallback.

    Raises:
        RuntimeError: When neither hash library is installed.
    """
    if _PDQ_AVAILABLE:
        arr = np.array(pil_image.convert("RGB"), dtype=np.uint8)
        hash_vector, _quality = _pdqhash_lib.compute(arr)
        return "".join(str(int(b)) for b in hash_vector)

    if _IMAGEHASH_AVAILABLE:
        h = _imagehash_lib.phash(pil_image)
        bits = np.array(h.hash).flatten()  # numpy bool array
        return "".join("1" if b else "0" for b in bits)

    raise RuntimeError(
        "No perceptual-hash library is available. "
        "Install 'pdqhash' (preferred) or 'imagehash'."
    )


# Pillow ≥ 9.1 moved transpose constants into Image.Transpose
_FLIP_LEFT_RIGHT = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def extract_phash(image_bytes: Union[bytes, memoryview, bytearray]) -> PhashResult:
    """
    Apply forensic pre-processing mitigations and compute perceptual hashes.

    Accepts ``bytes``, ``memoryview``, or ``bytearray`` so callers can pass
    ``buf.getbuffer()`` directly from a ``BytesIO`` stream, avoiding an
    extra full-buffer copy (important for the 20 MB size limit).

    Pipeline
    --------
    1. Decode *image_bytes* into a Pillow image (entirely in memory).
    2. **Mitigation 1** – Strip uniform borders via :func:`_strip_uniform_borders`
       (vectorised NumPy — no Python loops).
    3. **Mitigation 2** – Generate a horizontally mirrored copy with
       ``Image.FLIP_LEFT_RIGHT``.
    4. Hash both variants with PDQ (or phash fallback) → binary strings.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, WebP, …).

    Returns:
        :class:`PhashResult` containing both hashes as binary strings plus
        metadata about which algorithm was used and whether a border was found.

    Raises:
        ValueError:  When *image_bytes* is empty or Pillow cannot decode it.
        RuntimeError: When no hash library is installed.
    """
    if not image_bytes:
        raise ValueError("image_bytes must not be empty.")

    # 1. Decode
    try:
        buf = io.BytesIO(image_bytes)
        pil_image = Image.open(buf)
        pil_image.load()  # force full decode while buf is still live
    except Exception as exc:
        raise ValueError(f"Cannot decode image: {exc}") from exc

    pil_rgb = pil_image.convert("RGB")

    # 2. Border stripping
    img_array: np.ndarray = np.array(pil_rgb, dtype=np.uint8)
    stripped_array: np.ndarray = _strip_uniform_borders(img_array, tolerance=BORDER_TOLERANCE)
    border_stripped: bool = stripped_array.shape != img_array.shape
    content_image: Image.Image = Image.fromarray(stripped_array)

    # 3. Mirror variant
    mirrored_image: Image.Image = content_image.transpose(_FLIP_LEFT_RIGHT)

    # 4. Hash generation
    original_hash = _hash_to_binary_string(content_image)
    mirrored_hash = _hash_to_binary_string(mirrored_image)

    algorithm = "pdq" if _PDQ_AVAILABLE else "phash"

    return PhashResult(
        original_hash=original_hash,
        mirrored_hash=mirrored_hash,
        hash_algorithm=algorithm,
        hash_bits=len(original_hash),
        border_stripped=border_stripped,
    )
