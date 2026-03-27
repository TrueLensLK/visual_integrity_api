"""
Layer 4: Neural Network Ensemble v5.0 (6-Model Consensus)
============================================================
Combines SIX specialized deep learning models for robust detection:

  4.1 SDXL Detector    → Stable Diffusion XL specific
  4.2 ViT Transformer  → Macro-geometry
  4.3 Deepfake Expert  → Face-Swap specialist (Deepfake-v1) - *Runs if Face > 0*
  4.4 Ateeqq           → Midjourney/DALLE Expert (Generative Artifacts)
  4.5 ConvNeXt         → Modern CNN
  4.6 Swin Transformer → Hierarchical vision transformer

WHY 5 MODELS?
- Works on WhatsApp/Google compressed images (content-level analysis)
- If 4/5 models agree on REAL → strong consensus even without PRNU
- Each model sees different AI artifacts

CHANGES (v4.0):
- Replaced EfficientNet-B0 (trained on older GAN data) with Organika/sdxl-detector
  which specifically targets Stable Diffusion XL outputs and newer generators
- Removed dependency on local efficientnet_b0.pth weights file
- Removed efficientnet-pytorch dependency

IMPROVEMENTS:
- Test-Time Augmentation (TTA) for robustness
- Entropy-based uncertainty scoring
- Confidence dampening when models disagree
- 5-model consensus with weighted voting
- Auto-detect fake/real label indices
- Model consensus for compressed images (no hardware evidence)
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
except ImportError:
    print("[Layer 4] Transformers library not found. Install with `pip install transformers`.")
    exit(1)

try:
    from transformers import ConvNextForImageClassification
    HAS_CONVNEXT = True
except ImportError:
    HAS_CONVNEXT = False
    print("[Layer 4] Warning: ConvNextForImageClassification not available")

try:
    from transformers import SwinForImageClassification
    HAS_SWIN = True
except ImportError:
    HAS_SWIN = False
    print("[Layer 4] Warning: SwinForImageClassification not available")

import os
from typing import Tuple, Dict, Optional

# ==========================================
# CONFIGURATION - 6 MODEL ENSEMBLE (v5.0)
# ==========================================
# 1. SDXL Detector (HuggingFace) - Stable Diffusion XL specialist
#    Replaces EfficientNet-B0 which was trained on older GAN data
SDXL_MODEL_NAME = "Organika/sdxl-detector"

# 2. ViT Transformer Settings (HuggingFace) - Deepfake specialist
VIT_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

# 3. Deepfake Face-Swap Expert (HuggingFace) - Face-Swap specialist
#    New for v5.0 - Only runs when face is detected
DEEPFAKE_MODEL_NAME = "prithivMLmods/deepfake-detector-model-v1"

# 4. Ateeqq AI vs Human Detector (HuggingFace) - Broad Spectrum (Midjourney v6, SD 3.5, GPT-4o)
ATEEQQ_MODEL_NAME = "Ateeqq/ai-vs-human-image-detector"

# 5. ConvNeXt AI Detector (HuggingFace) - Modern CNN for AI detection
CONVNEXT_MODEL_NAME = "umm-maybe/AI-image-detector"

# 6. Swin Transformer (HuggingFace) - Another deepfake detector
SWIN_MODEL_NAME = "Wvolf/ViT_Deepfake_Detection"  # Alternative Swin-based detector

# 7. Ensemble weights (sum = 1.0) - Spread across 6 models
#    NOTE: If no face is detected, W_DEEPFAKE is redistributed or model skipped
W_SDXL      = 0.15
W_VIT       = 0.15
W_DEEPFAKE  = 0.20  # Strong weight for face swaps
W_ATEEQQ    = 0.22  # High weight for Midjourney/DALLE-3 coverage
W_CONVNEXT  = 0.13
W_SWIN      = 0.15

# ==========================================
# GLOBAL MODEL CACHE (Singleton pattern)
# ==========================================
_models_loaded = False
_device = None
_sdxl_model = None
_sdxl_processor = None
_sdxl_fake_idx = 0  # Auto-detected on load
_vit_model = None
_vit_processor = None
_vit_fake_idx = 1  # Auto-detected on load
_deepfake_model = None      # New
_deepfake_processor = None  # New
_deepfake_fake_idx = 0      # New
_ateeqq_model = None
_ateeqq_processor = None
_ateeqq_fake_idx = 0  # Auto-detected on load
_convnext_model = None
_convnext_processor = None
_convnext_fake_idx = 0  # Auto-detected on load
_swin_model = None
_swin_processor = None
_swin_fake_idx = 0  # Auto-detected on load


def _detect_fake_index(model, model_name: str) -> int:
    """Auto-detect which index is the 'fake' class from model config."""
    id2label = getattr(model.config, 'id2label', None)
    if id2label:
        for idx, label in id2label.items():
            label_lower = str(label).lower()
            if any(kw in label_lower for kw in ['fake', 'ai', 'generated', 'synthetic', 'deepfake']):
                print(f"[Layer 4] {model_name}: Detected fake class at index {idx} ('{label}')")
                return int(idx)
    return 0  # Default fallback


def _load_models():
    """
    Load all neural network models with caching.
    Models are loaded ONCE and reused for all subsequent predictions.
    """
    global _models_loaded, _device
    global _sdxl_model, _sdxl_processor, _sdxl_fake_idx
    global _vit_model, _vit_processor, _vit_fake_idx
    global _deepfake_model, _deepfake_processor, _deepfake_fake_idx
    global _ateeqq_model, _ateeqq_processor, _ateeqq_fake_idx
    global _convnext_model, _convnext_processor, _convnext_fake_idx
    global _swin_model, _swin_processor, _swin_fake_idx
    
    if _models_loaded:
        return _device
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Layer 4] Initializing 6-Model Ensemble v5.0 on {_device}...")

    # --- 4.1: Load SDXL Detector (Replaces EfficientNet-B0) ---
    try:
        print(f"[Layer 4.1] Loading SDXL Detector: {SDXL_MODEL_NAME}...")
        _sdxl_processor = AutoImageProcessor.from_pretrained(SDXL_MODEL_NAME)
        _sdxl_model = AutoModelForImageClassification.from_pretrained(SDXL_MODEL_NAME)
        _sdxl_model.to(_device).eval()
        _sdxl_fake_idx = _detect_fake_index(_sdxl_model, "SDXL-Detector")
        print("[Layer 4.1] SDXL Detector loaded")
    except Exception as e:
        print(f"[Layer 4.1 Error] SDXL Detector Load Failed: {e}")
        _sdxl_model = None
        _sdxl_processor = None

    # --- 4.2: Load ViT Transformer (Macro-geometry detector) ---
    try:
        print(f"[Layer 4.2] Loading ViT: {VIT_MODEL_NAME}...")
        _vit_processor = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME)
        _vit_model = AutoModelForImageClassification.from_pretrained(VIT_MODEL_NAME)
        _vit_model.to(_device).eval()
        _vit_fake_idx = _detect_fake_index(_vit_model, "ViT")
        print("[Layer 4.2] ViT Transformer loaded")
    except Exception as e:
        print(f"[Layer 4.2 Error] ViT Load Failed: {e}")
        _vit_model = None
        _vit_processor = None

    # --- 4.3: Load Deepfake Face-Swap Expert (New v5.0) ---
    try:
        print(f"[Layer 4.3] Loading Deepfake Expert: {DEEPFAKE_MODEL_NAME}...")
        _deepfake_processor = AutoImageProcessor.from_pretrained(DEEPFAKE_MODEL_NAME)
        _deepfake_model = AutoModelForImageClassification.from_pretrained(DEEPFAKE_MODEL_NAME)
        _deepfake_model.to(_device).eval()
        _deepfake_fake_idx = _detect_fake_index(_deepfake_model, "Deepfake-Expert")
        print("[Layer 4.3] Deepfake Expert loaded")
    except Exception as e:
        print(f"[Layer 4.3 Error] Deepfake Expert Load Failed: {e}")
        _deepfake_model = None
        _deepfake_processor = None

    # --- 4.3: Load Ateeqq (Midjourney/DALLE Expert) ---
    try:
        print(f"[Layer 4.3] Loading Ateeqq: {ATEEQQ_MODEL_NAME}...")
        _ateeqq_processor = AutoImageProcessor.from_pretrained(ATEEQQ_MODEL_NAME)
        _ateeqq_model = AutoModelForImageClassification.from_pretrained(ATEEQQ_MODEL_NAME)
        _ateeqq_model.to(_device).eval()
        # Ateeqq uses strict labels: 0=FAKE (AI), 1=REAL (Human) typically
        # But we auto-detect to be safe
        _ateeqq_fake_idx = _detect_fake_index(_ateeqq_model, "Ateeqq")
        print("[Layer 4.3] Ateeqq loaded")
    except Exception as e:
        print(f"[Layer 4.3 Error] Ateeqq Load Failed: {e}")
        _ateeqq_model = None
        _ateeqq_processor = None

    # --- 4.4: Load ConvNeXt AI Detector ---
    try:
        print(f"[Layer 4.4] Loading ConvNeXt: {CONVNEXT_MODEL_NAME}...")
        _convnext_processor = AutoImageProcessor.from_pretrained(CONVNEXT_MODEL_NAME)
        _convnext_model = AutoModelForImageClassification.from_pretrained(CONVNEXT_MODEL_NAME)
        _convnext_model.to(_device).eval()
        _convnext_fake_idx = _detect_fake_index(_convnext_model, "ConvNeXt")
        print("[Layer 4.4] ConvNeXt loaded")
    except Exception as e:
        print(f"[Layer 4.4 Error] ConvNeXt Load Failed: {e}")
        _convnext_model = None
        _convnext_processor = None

    # --- 4.5: Load Swin Transformer Detector ---
    try:
        print(f"[Layer 4.5] Loading Swin: {SWIN_MODEL_NAME}...")
        _swin_processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_NAME)
        _swin_model = AutoModelForImageClassification.from_pretrained(SWIN_MODEL_NAME)
        _swin_model.to(_device).eval()
        _swin_fake_idx = _detect_fake_index(_swin_model, "Swin")
        print("[Layer 4.5] Swin Transformer loaded")
    except Exception as e:
        print(f"[Layer 4.5 Error] Swin Load Failed: {e}")
        _swin_model = None
        _swin_processor = None

    _models_loaded = True
    model_count = sum([
        _sdxl_model is not None, 
        _vit_model is not None, 
        _ateeqq_model is not None,
        _convnext_model is not None,
        _swin_model is not None
    ])
    print(f"[Layer 4] Ensemble v5.0 ready: {model_count}/5 models loaded")
    return _device


def _get_tta_transforms():
    """
    Test-Time Augmentation transforms.
    Running multiple augmented versions makes the model more robust
    against adversarial attacks and reduces lucky/unlucky predictions.
    """
    return [
        lambda img: img,                                    # Original
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),   # Horizontal flip
        lambda img: img.rotate(3),                          # Slight rotation +3°
        lambda img: img.rotate(-3),                         # Slight rotation -3°
    ]


def _calculate_entropy(probs: np.ndarray) -> float:
    """
    Calculate Shannon entropy of probability distribution.
    Higher entropy = more uncertain prediction.
    Max entropy for 2 classes = ln(2) ≈ 0.693
    """
    return -np.sum(probs * np.log(probs + 1e-10))


def _run_sdxl_tta(image: Image.Image, device: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run SDXL Detector with Test-Time Augmentation.
    Specifically trained to detect Stable Diffusion XL outputs.
    
    Returns: (score, confidence) or (None, None) if model unavailable
    """
    if _sdxl_model is None or _sdxl_processor is None:
        return None, None
    
    tta_transforms = _get_tta_transforms()
    all_probs = []
    
    try:
        for aug_fn in tta_transforms:
            img_aug = aug_fn(image)
            inputs = _sdxl_processor(images=img_aug, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = _sdxl_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Average TTA predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Use auto-detected fake index
        fake_idx = _sdxl_fake_idx
        real_idx = 1 - fake_idx
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]
        
        # Calculate score: -50 (fake) to +50 (real)
        score = (prob_real - prob_fake) * 50
        
        # Calculate confidence from entropy
        entropy = _calculate_entropy(avg_probs)
        confidence = 1.0 - (entropy / 0.693)  # Normalize by max entropy
        
        # FIX 2: SDXL Confidence Floor at 0.60 (treat weak signals as neutral)
        if abs(confidence) < 0.60:
            print(f"   [4.1 SDXL] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
                  f"-> score={score:+.1f} (conf={confidence:.2f})")
            print(f"   [4.1 SDXL] LOW CONFIDENCE ({confidence:.2f} < 0.60) — treating as NEUTRAL, score zeroed")
            return 0.0, 0.0

        print(f"   [4.1 SDXL] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
              f"-> score={score:+.1f} (conf={confidence:.2f})")
        
        return score, confidence
        
    except Exception as e:
        print(f"   [4.1 SDXL Error] {e}")
        return None, None


def _run_vit_tta(image: Image.Image, device: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run ViT Transformer with Test-Time Augmentation.
    
    Returns: (score, confidence) or (None, None) if model unavailable
    """
    if _vit_model is None or _vit_processor is None:
        return None, None
    
    tta_transforms = _get_tta_transforms()
    all_probs = []
    
    try:
        for aug_fn in tta_transforms:
            img_aug = aug_fn(image)
            inputs = _vit_processor(images=img_aug, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = _vit_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Average TTA predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Use auto-detected fake index
        fake_idx = _vit_fake_idx
        real_idx = 1 - fake_idx
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]
        
        # Calculate score: -50 (fake) to +50 (real)
        score = (prob_real - prob_fake) * 50
        
        # Calculate confidence from entropy
        entropy = _calculate_entropy(avg_probs)
        confidence = 1.0 - (entropy / 0.693)
        
        print(f"   [4.2 ViT] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
              f"-> score={score:+.1f} (conf={confidence:.2f})")
        
        return score, confidence
        
    except Exception as e:
        print(f"   [4.2 ViT Error] {e}")
        return None, None


def _run_deepfake_tta(image: Image.Image, device: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run Deepfake Face-Swap Expert with Test-Time Augmentation.
    Specialized for face swaps and deepfakes. Only runs if faces are detected.
    
    Returns: (score, confidence) or (None, None) if model unavailable
    """
    if _deepfake_model is None or _deepfake_processor is None:
        return None, None
    
    tta_transforms = _get_tta_transforms()
    all_probs = []
    
    try:
        for aug_fn in tta_transforms:
            img_aug = aug_fn(image)
            inputs = _deepfake_processor(images=img_aug, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = _deepfake_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Average TTA predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Use auto-detected fake index
        fake_idx = _deepfake_fake_idx
        real_idx = 1 - fake_idx
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]
        
        # Calculate score: -50 (fake) to +50 (real)
        score = (prob_real - prob_fake) * 50
        
        # Calculate confidence from entropy
        entropy = _calculate_entropy(avg_probs)
        confidence = 1.0 - (entropy / 0.693)
        
        print(f"   [4.3 Deepfake] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
              f"-> score={score:+.1f} (conf={confidence:.2f})")
        
        return score, confidence
        
    except Exception as e:
        print(f"   [4.3 Deepfake Error] {e}")
        return None, None


def _run_ateeqq_tta(image: Image.Image, device: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run Ateeqq AI vs Human Detector with Test-Time Augmentation.
    Specialist for Midjourney v6, SD 3.5, and DALL-E 3.
    
    Returns: (score, confidence) or (None, None) if model unavailable
    """
    if _ateeqq_model is None or _ateeqq_processor is None:
        return None, None
    
    tta_transforms = _get_tta_transforms()
    all_probs = []
    
    try:
        for aug_fn in tta_transforms:
            img_aug = aug_fn(image)
            inputs = _ateeqq_processor(images=img_aug, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = _ateeqq_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Average TTA predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Use auto-detected fake index
        fake_idx = _ateeqq_fake_idx
        real_idx = 1 - fake_idx
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]
        
        # Calculate score: -50 (fake) to +50 (real)
        score = (prob_real - prob_fake) * 50
        
        # Calculate confidence from entropy
        entropy = _calculate_entropy(avg_probs)
        confidence = 1.0 - (entropy / 0.693)
        
        print(f"   [4.3 Ateeqq] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
              f"-> score={score:+.1f} (conf={confidence:.2f})")
        
        return score, confidence
        
    except Exception as e:
        print(f"   [4.3 Ateeqq Error] {e}")
        return None, None


def _run_convnext_tta(image: Image.Image, device: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run ConvNeXt AI Detector with Test-Time Augmentation.
    ConvNeXt is a modern CNN architecture that excels at detecting AI artifacts.
    
    Returns: (score, confidence) or (None, None) if model unavailable
    """
    if _convnext_model is None or _convnext_processor is None:
        return None, None
    
    tta_transforms = _get_tta_transforms()
    all_probs = []
    
    try:
        for aug_fn in tta_transforms:
            img_aug = aug_fn(image)
            inputs = _convnext_processor(images=img_aug, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = _convnext_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Average TTA predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Use auto-detected fake index
        fake_idx = _convnext_fake_idx
        real_idx = 1 - fake_idx
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]
        
        # Calculate score: -50 (fake) to +50 (real)
        score = (prob_real - prob_fake) * 50
        
        # Calculate confidence from entropy
        entropy = _calculate_entropy(avg_probs)
        confidence = 1.0 - (entropy / 0.693)
        
        print(f"   [4.4 ConvNeXt] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
              f"-> score={score:+.1f} (conf={confidence:.2f})")
        
        return score, confidence
        
    except Exception as e:
        print(f"   [4.4 ConvNeXt Error] {e}")
        return None, None


def _run_swin_tta(image: Image.Image, device: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run Swin Transformer with Test-Time Augmentation.
    Swin uses hierarchical feature maps and shifted windows for multi-scale analysis.
    
    Returns: (score, confidence) or (None, None) if model unavailable
    """
    if _swin_model is None or _swin_processor is None:
        return None, None
    
    tta_transforms = _get_tta_transforms()
    all_probs = []
    
    try:
        for aug_fn in tta_transforms:
            img_aug = aug_fn(image)
            inputs = _swin_processor(images=img_aug, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = _swin_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Average TTA predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Use auto-detected fake index
        fake_idx = _swin_fake_idx
        real_idx = 1 - fake_idx
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]
        
        # Calculate score: -50 (fake) to +50 (real)
        score = (prob_real - prob_fake) * 50
        
        # Calculate confidence from entropy
        entropy = _calculate_entropy(avg_probs)
        confidence = 1.0 - (entropy / 0.693)
        
        print(f"   [4.5 Swin] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
              f"-> score={score:+.1f} (conf={confidence:.2f})")
        
        return score, confidence
        
    except Exception as e:
        print(f"   [4.5 Swin Error] {e}")
        return None, None


# ... (Keep previous imports and _load_models logic) ...

def predict_visuals(file_path: str) -> float:
    """
    Layer 4 v3.0: 5-Model Consensus Ensemble. 
    Focus: Works on compressed images (WhatsApp, Google) via content-level analysis.
    """
    try:
        device = _load_models()
        image = Image.open(file_path).convert('RGB')
        
        # 1. PRE-FLIGHT QUALITY CHECK
        is_low_quality = image.size[0] < 500 or image.size[1] < 500
        
        # Run ALL 5 Sub-Layers
        s1, c1 = _run_sdxl_tta(image, device)           # SDXL/Diffusion
        s2, c2 = _run_vit_tta(image, device)           # Geometry
        s3, c3 = _run_ateeqq_tta(image, device)        # Midjourney/DALLE Expert
        s4, c4 = _run_convnext_tta(image, device)      # Modern CNN
        s5, c5 = _run_swin_tta(image, device)          # Hierarchical ViT
        
        scores = [s1, s2, s3, s4, s5]
        confs = [c1, c2, c3, c4, c5]
        model_names = ['SDXL-Detector', 'ViT', 'Ateeqq', 'ConvNeXt', 'Swin']
        
        # Filter out None results
        valid_scores = [(s, c, n) for s, c, n in zip(scores, confs, model_names) if s is not None]
        
        if not valid_scores:
            return 0.0
        
        n_models = len(valid_scores)
        scores_only = [v[0] for v in valid_scores]
        confs_only = [v[1] for v in valid_scores]
        names_only = [v[2] for v in valid_scores]
        
        # 2. ADAPTIVE WEIGHTING
        current_weights = [W_SDXL, W_VIT, W_ATEEQQ, W_CONVNEXT, W_SWIN]
        if is_low_quality:
            print("   [Layer 4] ! Low Quality detected. Reducing SDXL weight.")
            current_weights = [0.05, 0.25, 0.25, 0.25, 0.20]
        
        # Normalize weights for available models
        available_weights = [current_weights[i] for i, s in enumerate(scores) if s is not None]
        total_w = sum(available_weights)
        available_weights = [w / total_w for w in available_weights]

        # 3. COUNT VOTES
        ai_signals = sum(1 for s in scores_only if s < -12)
        real_signals = sum(1 for s in scores_only if s > 12)
        
        # Calculate raw weighted score
        raw_score = sum(s * w for s, w in zip(scores_only, available_weights))
        
        # --- 5-MODEL CONSENSUS LOGIC ---
        
        # Case A: Strong REAL consensus (4+ models agree)
        if real_signals >= 4 and n_models >= 4:
            avg_conf = np.mean(confs_only)
            if avg_conf >= 0.55:
                raw_score = max(raw_score, 25.0)  # Boost to REAL
                print(f"   [Layer 4] [+] STRONG REAL CONSENSUS: {real_signals}/{n_models} models agree")
        
        # Case B: Strong AI consensus (4+ models agree)
        elif ai_signals >= 4 and n_models >= 4:
            avg_conf = np.mean(confs_only)
            if avg_conf >= 0.55:
                raw_score = min(raw_score, -25.0)  # Confirm AI
                print(f"   [Layer 4] [-] STRONG AI CONSENSUS: {ai_signals}/{n_models} models agree")
        
        # Case C: Only one model thinks it's AI (The "Loner" Case) - FP protection
        elif ai_signals == 1 and raw_score < 0:
            raw_score = max(5.0, raw_score + 25.0) 
            print("   [Layer 4] ! FP MITIGATION: Single-model AI signal rejected.")

        # Case D: Significant Disagreement (some Real vs some Fake)
        elif ai_signals >= 2 and real_signals >= 2:
            raw_score = raw_score * 0.3
            print("   [Layer 4] ! CONFLICT: Models disagree. Score dampened to neutral.")
        
        # Case E: Weak consensus (3 models agree)
        elif real_signals >= 3 and n_models >= 4:
            print(f"   [Layer 4] [ ] Moderate REAL consensus: {real_signals}/{n_models}")
        elif ai_signals >= 3 and n_models >= 4:
            print(f"   [Layer 4] [ ] Moderate AI consensus: {ai_signals}/{n_models}")

        # 4. SQUASHING EXTREMES
        final_score = np.clip(raw_score, -50, 50)
        
        # Final Confidence calculation
        avg_conf = np.mean(confs_only)
        
        print(f"   [L4 Final] Score: {final_score:+.1f} | Conf: {avg_conf:.2f} | Models: {n_models}/5")
        print(f"   [L4 Votes] AI: {ai_signals}, REAL: {real_signals}")
        return round(float(final_score), 2)

    except Exception as e:
        print(f"   [Layer 4 Error] {e}")
        return 0.0
    
def predict_visuals_detailed(file_path: str, face_count: int = 0) -> Dict:
    """
    Extended version returning full analysis details.
    Useful for debugging and explainability.
    Returns model consensus info for Judge layer.
    """
    try:
        device = _load_models()
        image = Image.open(file_path).convert('RGB')
        
        # Run base models
        sdxl_score, sdxl_conf = _run_sdxl_tta(image, device)
        vit_score, vit_conf = _run_vit_tta(image, device)
        ateeqq_score, ateeqq_conf = _run_ateeqq_tta(image, device)
        convnext_score, convnext_conf = _run_convnext_tta(image, device)
        swin_score, swin_conf = _run_swin_tta(image, device)
        
        # Run Deepfake Expert ONLY if faces are detected
        deepfake_score, deepfake_conf = None, None
        if face_count > 0:
            print(f"   [Layer 4] Face detected ({face_count}). Running Deepfake Expert.")
            deepfake_score, deepfake_conf = _run_deepfake_tta(image, device)
        else:
            print(f"   [Layer 4] No faces detected. Skipping Deepfake Expert.")

        # --- FIX: Null out Swin if no face detected (it defaults to REAL on scenery) ---
        if face_count == 0:
            if swin_score is not None:
                print(f"   [Layer 4] ! NO FACE: Nulling Swin score ({swin_score:+.1f}) — face-swap model invalid on faceless image")
                swin_score = None
                swin_conf = None
        
        # Calculate ensemble
        scores, confidences, weights, model_names = [], [], [], []
        
        if sdxl_score is not None:
            scores.append(sdxl_score)
            confidences.append(sdxl_conf)
            weights.append(W_SDXL)
            model_names.append("sdxl")
        if vit_score is not None:
            scores.append(vit_score)
            confidences.append(vit_conf)
            weights.append(W_VIT)
            model_names.append("vit")
        if deepfake_score is not None:
            scores.append(deepfake_score)
            confidences.append(deepfake_conf)
            weights.append(W_DEEPFAKE)
            model_names.append("deepfake")
        if ateeqq_score is not None:
            scores.append(ateeqq_score)
            confidences.append(ateeqq_conf)
            weights.append(W_ATEEQQ)
            model_names.append("ateeqq")
        if convnext_score is not None:
            scores.append(convnext_score)
            confidences.append(convnext_conf)
            weights.append(W_CONVNEXT)
            model_names.append("convnext")
        if swin_score is not None:
            scores.append(swin_score)
            confidences.append(swin_conf)
            weights.append(W_SWIN)
            model_names.append("swin")
        
        if not scores:
            return {"impact": 0, "confidence": 0, "is_uncertain": True, 
                    "findings": ["No models available"], "model_consensus": 0, "total_models": 0}
        
        n_models = len(scores)
        
        # ================================================================
        # 6-MODEL CONSENSUS LOGIC (v5.0)
        # ================================================================

        # Fix 1: Distance-Based Outlier Detection REMOVED
        # (Outlier logic was deleting high-confidence signals incorrectly)

        # 1. Confidence-weighted scoring
        conf_weights = []
        for i, (s, c, w) in enumerate(zip(scores, confidences, weights)):
            # Give bonus to high confidence predictions
            conf_factor = 0.3 + 0.7 * c 
            conf_weights.append(w * conf_factor)
            
        total_cw = sum(conf_weights)
        if total_cw > 0:
            conf_weights = [cw / total_cw for cw in conf_weights]
        else:
            conf_weights = [1.0 / n_models] * n_models
        
        raw_score = sum(s * cw for s, cw in zip(scores, conf_weights))
        overall_conf = sum(c * cw for c, cw in zip(confidences, conf_weights))
        
        # 2. High-confidence override (ONLY specialized models)
        SPECIALIZED_MODELS = ["vit", "deepfake", "ateeqq", "convnext", "swin"]
        high_conf_override = False
        for i, c in enumerate(confidences):
            if c > 0.85 and model_names[i] in SPECIALIZED_MODELS:
                # If specific expert is VERY sure, and others are uncertain
                others_uncertain = all(confidences[j] < 0.40 for j in range(n_models) if j != i)
                if others_uncertain:
                    # Trust the expert
                    raw_score = scores[i]
                    overall_conf = confidences[i] * 0.90
                    high_conf_override = True
                    break
        
        # 3. Consensus check
        ai_votes = sum(1 for s in scores if s < -10)
        real_votes = sum(1 for s in scores if s > 10)
        
        # NEW: Model consensus ratio for Judge layer
        model_consensus = max(ai_votes, real_votes) / n_models if n_models > 0 else 0
        consensus_direction = "REAL" if real_votes > ai_votes else "AI" if ai_votes > real_votes else "NEUTRAL"
        
        # Build strict model breakdown for Judge/Case File
        model_breakdown = {
            "sdxl": {"score": sdxl_score, "conf": sdxl_conf} if sdxl_score is not None else None,
            "vit": {"score": vit_score, "conf": vit_conf} if vit_score is not None else None,
            "deepfake_expert": {"score": deepfake_score, "conf": deepfake_conf} if deepfake_score is not None else None,
            "ateeqq": {"score": ateeqq_score, "conf": ateeqq_conf} if ateeqq_score is not None else None,
            "convnext": {"score": convnext_score, "conf": convnext_conf} if convnext_score is not None else None,
            "swin": {"score": swin_score, "conf": swin_conf} if swin_score is not None else None
        }

        # 4. Disagreement handling
        
        # 4. Disagreement handling
        model_disagreement = False
        if n_models >= 2:
            max_gap = max(scores) - min(scores)
            signs = [s > 0 for s in scores]
            sign_disagree = len(set(signs)) > 1
            
            if sign_disagree or max_gap > 40:
                model_disagreement = True
                # Penalize score if disagreement is high
                if raw_score < 0:
                    uncertainty_pull = min(0.6, max_gap / 100.0)
                    raw_score = raw_score * (1 - uncertainty_pull)
                
                disagreement_penalty = min(0.5, max_gap / 100.0)
                overall_conf *= (1.0 - disagreement_penalty)
        
        # 5. AI consensus requirement
        # If score is very AI-heavy, ensure we have at least partial consensus
        if raw_score < -20:
            if ai_votes < 2 and overall_conf < 0.6:
                 # Protection against single-model false positives
                raw_score *= 0.6 
        
        # 6. REAL consensus boost
        if real_votes >= 4 and overall_conf >= 0.60:
            raw_score = max(raw_score, 25.0)
        
        # 7. Final dampening based on confidence
        min_confidence = 0.30
        if overall_conf < min_confidence:
            dampening = max(0.2, overall_conf / min_confidence)
        else:
            dampening = 1.0
        
        final_score = raw_score * dampening
        
        # Build findings list
        findings = [
            f"Ensemble confidence: {overall_conf*100:.1f}%",
            f"TTA augmentations: 4 per model",
            f"Models used: {n_models}/6",
            f"AI votes: {ai_votes}/{n_models}, Real votes: {real_votes}/{n_models}",
            f"Model consensus: {model_consensus*100:.0f}% ({consensus_direction})"
        ]
        if model_disagreement:
            findings.append(f"Model disagreement detected")
        if high_conf_override:
            findings.append("High-confidence override active")
        if real_votes >= 4:
            findings.append("STRONG REAL CONSENSUS")
        if ai_votes >= 4:
            findings.append("STRONG AI CONSENSUS")
        
        result = {
            "impact": round(final_score, 2),
            "raw_score": round(raw_score, 2),
            "confidence": round(overall_conf, 2),
            "dampening": round(dampening, 2),
            "is_uncertain": overall_conf < 0.45 or model_disagreement,
            "model_count": n_models,
            "ai_votes": ai_votes,
            "real_votes": real_votes,
            "model_consensus": round(model_consensus, 2),
            "consensus_direction": consensus_direction,
            "model_disagreement": model_disagreement,
            "high_conf_override": high_conf_override,
            "model_breakdown": model_breakdown,
            "findings": findings
        }
        
        # Add individual model results
        if sdxl_score is not None:
            result["sdxl"] = {"score": round(sdxl_score, 2), "confidence": round(sdxl_conf, 2)}
        if vit_score is not None:
            result["vit"] = {"score": round(vit_score, 2), "confidence": round(vit_conf, 2)}
        if deepfake_score is not None:
            result["deepfake"] = {"score": round(deepfake_score, 2), "confidence": round(deepfake_conf, 2)}
        if ateeqq_score is not None:
            result["ateeqq"] = {"score": round(ateeqq_score, 2), "confidence": round(ateeqq_conf, 2)}
        if convnext_score is not None:
            result["convnext"] = {"score": round(convnext_score, 2), "confidence": round(convnext_conf, 2)}
        if swin_score is not None:
            result["swin"] = {"score": round(swin_score, 2), "confidence": round(swin_conf, 2)}
        
        return result
        
    except Exception as e:
        print(f"Error in predict_visuals_detailed: {e}")
        return {"impact": 0, "confidence": 0, "is_uncertain": True,
                "findings": [f"Error: {str(e)}"], "model_consensus": 0, "total_models": 0}


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        print(f"\n{'='*60}")
        print("Layer 4: 5-Model Neural Network Ensemble v4.0")
        print(f"{'='*60}")
        print("Models:")
        print(f"  • 4.1 SDXL Detector (SDXL/diffusion) - {W_SDXL*100:.0f}%")
        print(f"  • 4.2 ViT Transformer (macro-geometry) - {W_VIT*100:.0f}%")
        print(f"  • 4.3 Ateeqq (Midjourney/DALLE Expert) - {W_ATEEQQ*100:.0f}%")
        print(f"  • 4.4 ConvNeXt (modern CNN) - {W_CONVNEXT*100:.0f}%")
        print(f"  • 4.5 Swin Transformer (hierarchical) - {W_SWIN*100:.0f}%")
        print(f"{'='*60}\n")
        
        result = predict_visuals_detailed(sys.argv[1])
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Final Score: {result['impact']:+.1f}")
        print(f"Raw Score:   {result.get('raw_score', 'N/A')}")
        print(f"Confidence:  {result['confidence']*100:.1f}%")
        print(f"Uncertain:   {result['is_uncertain']}")
        print(f"Consensus:   {result.get('model_consensus', 0)*100:.0f}% ({result.get('consensus_direction', 'N/A')})")
        print(f"{'='*60}")
        
        if result.get('sdxl'):
            print(f"SDXL Detect: score={result['sdxl']['score']:+.1f}, "
                  f"conf={result['sdxl']['confidence']:.2f}")
        if result.get('vit'):
            print(f"ViT:          score={result['vit']['score']:+.1f}, "
                  f"conf={result['vit']['confidence']:.2f}")
        if result.get('ateeqq'):
            print(f"Ateeqq:       score={result['ateeqq']['score']:+.1f}, "
                  f"conf={result['ateeqq']['confidence']:.2f}")
        if result.get('convnext'):
            print(f"ConvNeXt:     score={result['convnext']['score']:+.1f}, "
                  f"conf={result['convnext']['confidence']:.2f}")
        if result.get('swin'):
            print(f"Swin:         score={result['swin']['score']:+.1f}, "
                  f"conf={result['swin']['confidence']:.2f}")
        
        print(f"{'='*60}")
        
        # Interpretation
        score = result['impact']
        consensus = result.get('model_consensus', 0)
        real_votes = result.get('real_votes', 0)
        ai_votes = result.get('ai_votes', 0)
        
        if result['is_uncertain']:
            print("[ ] Model is UNCERTAIN - take result with caution")
        elif real_votes >= 4:
            print(f"[+] STRONG CONSENSUS: {real_votes}/5 models say REAL")
        elif ai_votes >= 4:
            print(f"[-] STRONG CONSENSUS: {ai_votes}/5 models say AI")
        elif score > 15:
            print("[+] Neural networks indicate REAL image")
        elif score > -15:
            print("[ ] Neural networks are INCONCLUSIVE")
        else:
            print("[-] Neural networks indicate AI-GENERATED")
    else:
        print("Usage: python layer_4_visual.py <image_path>")
        print("\n5-Model Ensemble v4.0:")
        print("  • 4.1 SDXL Detector: Stable Diffusion XL specific (newer generators)")
        print("  • 4.2 ViT Transformer: Macro-geometry (semantic inconsistencies)")
        print("  • 4.3 Ateeqq: Midjourney/DALLE Expert (Generative Artifacts)")
        print("  • 4.4 ConvNeXt: Modern CNN for AI artifacts")
        print("  • 4.5 Swin Transformer: Hierarchical multi-scale analysis")
        print("\nFeatures:")
        print("  • Test-Time Augmentation (4 augments: original, flip, ±3° rotation)")
        print("  • Entropy-based uncertainty scoring")
        print("  • Confidence dampening for uncertain/disagreeing predictions")
        print("  • 5-model consensus for WhatsApp/Google images")
        print(f"\nWeights: SDXL={W_SDXL:.0%}, ViT={W_VIT:.0%}, Ateeqq={W_ATEEQQ:.0%}, ConvNeXt={W_CONVNEXT:.0%}, Swin={W_SWIN:.0%}")
        print("\nRequirements:")
        print("  pip install torch torchvision transformers numpy")
        print("\nModels:")
        print(f"  • SDXL Detector: {SDXL_MODEL_NAME}")
        print(f"  • ViT Transformer: {VIT_MODEL_NAME}")
        print(f"  • Ateeqq: {ATEEQQ_MODEL_NAME}")
        print(f"  • ConvNeXt: {CONVNEXT_MODEL_NAME}")
        print(f"  • Swin: {SWIN_MODEL_NAME}")
