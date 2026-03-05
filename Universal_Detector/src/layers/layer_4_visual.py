"""
Layer 4: Neural Network Ensemble v4.0 (5-Model Consensus)
============================================================
Combines FIVE specialized deep learning models for robust detection:

  4.1 SDXL Detector    → Stable Diffusion XL specific (replaces EfficientNet-B0)
  4.2 ViT Transformer  → Macro-geometry (semantic inconsistencies)
  4.3 SigLIP2          → Global coherence (CLIP-based scene analysis)
  4.4 ConvNeXt         → Modern CNN for AI artifacts
  4.5 Swin Transformer → Hierarchical vision transformer

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
from transformers import AutoImageProcessor, AutoModelForImageClassification
try:
    from transformers import SiglipForImageClassification
    HAS_SIGLIP = True
except ImportError:
    HAS_SIGLIP = False
    print("[Layer 4] Warning: SiglipForImageClassification not available, will skip Layer 4.3")

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
# CONFIGURATION - 5 MODEL ENSEMBLE (v4.0)
# ==========================================
# 1. SDXL Detector (HuggingFace) - Stable Diffusion XL specialist
#    Replaces EfficientNet-B0 which was trained on older GAN data
SDXL_MODEL_NAME = "Organika/sdxl-detector"

# 2. ViT Transformer Settings (HuggingFace) - Deepfake specialist
VIT_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

# 3. SigLIP2 Settings (HuggingFace) - Global coherence detector
SIGLIP_MODEL_NAME = "prithivMLmods/open-deepfake-detection"

# 4. ConvNeXt AI Detector (HuggingFace) - Modern CNN for AI detection
CONVNEXT_MODEL_NAME = "umm-maybe/AI-image-detector"

# 5. Swin Transformer (HuggingFace) - Another deepfake detector
SWIN_MODEL_NAME = "Wvolf/ViT_Deepfake_Detection"  # Alternative Swin-based detector

# 6. Ensemble weights (sum = 1.0) - Spread across 5 models
W_SDXL     = 0.15  # SDXL/Diffusion specialist
W_VIT      = 0.22  # Macro-geometry specialist  
W_SIGLIP   = 0.22  # Global coherence specialist
W_CONVNEXT = 0.22  # Modern CNN specialist
W_SWIN     = 0.19  # Hierarchical transformer

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
_sig_model = None
_sig_processor = None
_sig_fake_idx = 0  # Auto-detected on load
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
    global _sig_model, _sig_processor, _sig_fake_idx
    global _convnext_model, _convnext_processor, _convnext_fake_idx
    global _swin_model, _swin_processor, _swin_fake_idx
    
    if _models_loaded:
        return _device
    
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Layer 4] Initializing 5-Model Ensemble v4.0 on {_device}...")

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

    # --- 4.3: Load SigLIP2 (Global coherence detector) ---
    if HAS_SIGLIP:
        try:
            print(f"[Layer 4.3] Loading SigLIP2: {SIGLIP_MODEL_NAME}...")
            _sig_processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_NAME)
            _sig_model = SiglipForImageClassification.from_pretrained(SIGLIP_MODEL_NAME)
            _sig_model.to(_device).eval()
            _sig_fake_idx = _detect_fake_index(_sig_model, "SigLIP")
            print("[Layer 4.3] SigLIP2 loaded")
        except Exception as e:
            print(f"[Layer 4.3 Error] SigLIP Load Failed: {e}")
            _sig_model = None
            _sig_processor = None
    else:
        print("[Layer 4.3] SigLIP not available (transformers version too old?)")
        _sig_model = None
        _sig_processor = None

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
        _sig_model is not None,
        _convnext_model is not None,
        _swin_model is not None
    ])
    print(f"[Layer 4] Ensemble v4.0 ready: {model_count}/5 models loaded")
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


def _run_siglip_tta(image: Image.Image, device: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Run SigLIP2 with Test-Time Augmentation.
    SigLIP2 excels at global scene coherence and CLIP-style understanding.
    
    Returns: (score, confidence) or (None, None) if model unavailable
    """
    if _sig_model is None or _sig_processor is None:
        return None, None
    
    tta_transforms = _get_tta_transforms()
    all_probs = []
    
    try:
        for aug_fn in tta_transforms:
            img_aug = aug_fn(image)
            inputs = _sig_processor(images=img_aug, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = _sig_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy()[0])
        
        # Average TTA predictions
        avg_probs = np.mean(all_probs, axis=0)
        
        # Use auto-detected fake index
        fake_idx = _sig_fake_idx
        real_idx = 1 - fake_idx
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]
        
        # Calculate score: -50 (fake) to +50 (real)
        score = (prob_real - prob_fake) * 50
        
        # Calculate confidence from entropy
        entropy = _calculate_entropy(avg_probs)
        confidence = 1.0 - (entropy / 0.693)
        
        print(f"   [4.3 SigLIP] P(fake)={prob_fake:.3f} P(real)={prob_real:.3f} "
              f"-> score={score:+.1f} (conf={confidence:.2f})")
        
        return score, confidence
        
    except Exception as e:
        print(f"   [4.3 SigLIP Error] {e}")
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
        s3, c3 = _run_siglip_tta(image, device)        # Scene
        s4, c4 = _run_convnext_tta(image, device)      # Modern CNN
        s5, c5 = _run_swin_tta(image, device)          # Hierarchical ViT
        
        scores = [s1, s2, s3, s4, s5]
        confs = [c1, c2, c3, c4, c5]
        model_names = ['SDXL-Detector', 'ViT', 'SigLIP', 'ConvNeXt', 'Swin']
        
        # Filter out None results
        valid_scores = [(s, c, n) for s, c, n in zip(scores, confs, model_names) if s is not None]
        
        if not valid_scores:
            return 0.0
        
        n_models = len(valid_scores)
        scores_only = [v[0] for v in valid_scores]
        confs_only = [v[1] for v in valid_scores]
        names_only = [v[2] for v in valid_scores]
        
        # 2. ADAPTIVE WEIGHTING
        current_weights = [W_SDXL, W_VIT, W_SIGLIP, W_CONVNEXT, W_SWIN]
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
    
def predict_visuals_detailed(file_path: str) -> Dict:
    """
    Extended version returning full analysis details.
    Useful for debugging and explainability.
    Returns model consensus info for Judge layer.
    """
    try:
        device = _load_models()
        image = Image.open(file_path).convert('RGB')
        
        eff_score, eff_conf = _run_sdxl_tta(image, device)
        vit_score, vit_conf = _run_vit_tta(image, device)
        sig_score, sig_conf = _run_siglip_tta(image, device)
        convnext_score, convnext_conf = _run_convnext_tta(image, device)
        swin_score, swin_conf = _run_swin_tta(image, device)
        
        # Calculate ensemble
        scores, confidences, weights, model_names = [], [], [], []
        
        if eff_score is not None:
            scores.append(eff_score)
            confidences.append(eff_conf)
            weights.append(W_SDXL)
            model_names.append("sdxl")
        if vit_score is not None:
            scores.append(vit_score)
            confidences.append(vit_conf)
            weights.append(W_VIT)
            model_names.append("vit")
        if sig_score is not None:
            scores.append(sig_score)
            confidences.append(sig_conf)
            weights.append(W_SIGLIP)
            model_names.append("siglip")
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
        # 5-MODEL CONSENSUS LOGIC
        # ================================================================
        
        # 1. Confidence-weighted scoring
        conf_weights = []
        for i, (s, c, w) in enumerate(zip(scores, confidences, weights)):
            conf_factor = 0.3 + 0.7 * c
            conf_weights.append(w * conf_factor)
        total_cw = sum(conf_weights)
        conf_weights = [cw / total_cw for cw in conf_weights]
        
        raw_score = sum(s * cw for s, cw in zip(scores, conf_weights))
        overall_conf = sum(c * cw for c, cw in zip(confidences, conf_weights))
        
        # 2. High-confidence override (ONLY specialized models)
        SPECIALIZED_MODELS = ["vit", "siglip", "convnext", "swin", "sdxl"]
        high_conf_override = False
        for i, c in enumerate(confidences):
            if c > 0.85 and model_names[i] in SPECIALIZED_MODELS:
                others_uncertain = all(confidences[j] < 0.35 for j in range(n_models) if j != i)
                if others_uncertain:
                    raw_score = scores[i]
                    overall_conf = confidences[i] * 0.85
                    high_conf_override = True
                    break
        
        # 3. Consensus check
        ai_votes = sum(1 for s in scores if s < -10)
        real_votes = sum(1 for s in scores if s > 10)
        
        # NEW: Model consensus ratio for Judge layer
        model_consensus = max(ai_votes, real_votes) / n_models if n_models > 0 else 0
        consensus_direction = "REAL" if real_votes > ai_votes else "AI" if ai_votes > real_votes else "NEUTRAL"
        
        # 4. Disagreement handling
        model_disagreement = False
        max_gap = max(abs(scores[i] - scores[j]) 
                     for i in range(n_models) for j in range(i+1, n_models)) if n_models >= 2 else 0
        
        if n_models >= 2:
            signs = [s > 0 for s in scores]
            sign_disagree = len(set(signs)) > 1
            if sign_disagree or max_gap > 35:
                model_disagreement = True
                if raw_score < 0:
                    uncertainty_pull = min(0.7, max_gap / 100.0)
                    raw_score = raw_score * (1 - uncertainty_pull)
                disagreement_penalty = min(0.6, max_gap / 80.0)
                overall_conf *= (1.0 - disagreement_penalty)
        
        # 5. AI consensus requirement (need more agreement with 5 models)
        if raw_score < -15:
            if ai_votes < 3 and overall_conf < 0.5:
                raw_score *= 0.5
        
        # 6. REAL consensus boost (4+ models agreeing = strong signal)
        if real_votes >= 4 and overall_conf >= 0.55:
            raw_score = max(raw_score, 20.0)
        
        # 7. Final dampening
        min_confidence = 0.25
        if overall_conf < min_confidence:
            dampening = max(0.15, overall_conf / min_confidence)
        else:
            dampening = 1.0
        
        final_score = raw_score * dampening
        
        # Build findings list
        findings = [
            f"Ensemble confidence: {overall_conf*100:.1f}%",
            f"TTA augmentations: 4 per model",
            f"Models used: {n_models}/5",
            f"AI votes: {ai_votes}/{n_models}, Real votes: {real_votes}/{n_models}",
            f"Model consensus: {model_consensus*100:.0f}% ({consensus_direction})"
        ]
        if model_disagreement:
            findings.append(f"Model disagreement detected (gap={max_gap:.1f})")
        if high_conf_override:
            findings.append("High-confidence override active")
        if real_votes >= 4:
            findings.append("STRONG REAL CONSENSUS (4+ models)")
        if ai_votes >= 4:
            findings.append("STRONG AI CONSENSUS (4+ models)")
        
        result = {
            "impact": round(final_score, 2),
            "raw_score": round(raw_score, 2),
            "confidence": round(overall_conf, 2),
            "dampening": round(dampening, 2),
            "is_uncertain": overall_conf < 0.4 or model_disagreement,
            "model_count": n_models,
            "ai_votes": ai_votes,
            "real_votes": real_votes,
            "model_consensus": round(model_consensus, 2),
            "consensus_direction": consensus_direction,
            "model_disagreement": model_disagreement,
            "high_conf_override": high_conf_override,
            "findings": findings
        }
        
        # Add individual model results
        if eff_score is not None:
            result["sdxl"] = {"score": round(eff_score, 2), "confidence": round(eff_conf, 2)}
        if vit_score is not None:
            result["vit"] = {"score": round(vit_score, 2), "confidence": round(vit_conf, 2)}
        if sig_score is not None:
            result["siglip"] = {"score": round(sig_score, 2), "confidence": round(sig_conf, 2)}
        if convnext_score is not None:
            result["convnext"] = {"score": round(convnext_score, 2), "confidence": round(convnext_conf, 2)}
        if swin_score is not None:
            result["swin"] = {"score": round(swin_score, 2), "confidence": round(swin_conf, 2)}
        
        return result
        
    except Exception as e:
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
        print(f"  • 4.3 SigLIP2 (global coherence) - {W_SIGLIP*100:.0f}%")
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
        if result.get('siglip'):
            print(f"SigLIP:       score={result['siglip']['score']:+.1f}, "
                  f"conf={result['siglip']['confidence']:.2f}")
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
        print("  • 4.3 SigLIP2: Global coherence (CLIP-based scene analysis)")
        print("  • 4.4 ConvNeXt: Modern CNN for AI artifacts")
        print("  • 4.5 Swin Transformer: Hierarchical multi-scale analysis")
        print("\nFeatures:")
        print("  • Test-Time Augmentation (4 augments: original, flip, ±3° rotation)")
        print("  • Entropy-based uncertainty scoring")
        print("  • Confidence dampening for uncertain/disagreeing predictions")
        print("  • 5-model consensus for WhatsApp/Google images")
        print(f"\nWeights: SDXL={W_SDXL:.0%}, ViT={W_VIT:.0%}, SigLIP={W_SIGLIP:.0%}, ConvNeXt={W_CONVNEXT:.0%}, Swin={W_SWIN:.0%}")
        print("\nRequirements:")
        print("  pip install torch torchvision transformers numpy")
        print("\nModels:")
        print(f"  • SDXL Detector: {SDXL_MODEL_NAME}")
        print(f"  • ViT Transformer: {VIT_MODEL_NAME}")
        print(f"  • SigLIP2: {SIGLIP_MODEL_NAME}")
        print(f"  • ConvNeXt: {CONVNEXT_MODEL_NAME}")
        print(f"  • Swin: {SWIN_MODEL_NAME}")
