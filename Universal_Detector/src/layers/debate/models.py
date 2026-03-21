"""
Debate System — Shared Data Models & Helpers
Shared types, parsers, and prompts used by all debate agents.
"""

import json
import re
import base64
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════════════

# User Requested Models
GEMINI_VISUAL_MODEL = "gemini-2.0-flash-lite-preview-02-05" # As requested
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # As requested
CEREBRAS_MODEL = "llama-3.3-70b"  # As requested fallback

# Legacy/Unused (kept for reference if needed, but not used in new flow)
OPENROUTER_VISION_MODELS = [
    # High Performance (Free/Low Cost) - Updated for stability
    "nvidia/llama-3.2-nv-embedqa-1b-v2",
    "qwen/qwen-2.5-vl-72b-instruct:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "google/gemini-2.0-flash-thinking-exp:free",
    
    # Meta
    "meta-llama/llama-3.2-90b-vision-instruct:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    
    # Mistral
    "mistralai/pixtral-12b:free",
    
    # Fallback to paid but cheap if free fails (user might have credit)
    "google/gemini-2.0-flash-001"
]

# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentResponse:
    """Structured response from a debating agent."""
    position: str           # "AI_GENERATED" or "REAL"
    confidence: float       # 0.0-1.0
    primary_evidence: List[str]
    visual_observations: List[str]  # New field for visual findings
    challenge_to_opponent: str
    concessions: str
    reasoning_summary: str
    raw_text: str = ""

@dataclass
class VisualExpertResponse:
    """Structured response from the Visual Expert Agent."""
    verdict: str                # "AI-GENERATED", "LIKELY_REAL", "UNCERTAIN"
    confidence: float           # 0.0-0.80
    definitive_artifacts_found: List[str]
    definitive_real_indicators: List[str]
    reasoning: str
    honest_assessment: str
    raw_text: str = ""



@dataclass
class ConvergenceResult:
    """Result of convergence evaluation by the neutral judge."""
    has_converged: bool
    verdict: Optional[str]      # "REAL", "AI-GENERATED", "EDITED"
    confidence: float
    reasoning: str
    winning_side: Optional[str] = None   # "prosecution" or "defense"
    key_turning_point: str = ""
    hallucination_detected: str = "none"  # "prosecution", "defense", "both", "none"


@dataclass
class DebateVerdict:
    """Final output of the adversarial debate (returned to HybridJudge)."""
    verdict: str                # "REAL", "AI-GENERATED", "EDITED"
    confidence: float           # 0.0-1.0
    reasoning: str              # Convergence judge's reasoning
    method: str = "adversarial_debate"
    rounds_taken: int = 0
    debate_summary: Dict = field(default_factory=dict)
    debate_history: List[Dict] = field(default_factory=list)
    source: str = "debate"


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════

PROSECUTION_PROMPT = """You are a forensic prosecution expert arguing that an image is AI-GENERATED.

═══════════════════════════════════════════════════════
STEP 0 — CONTENT PLAUSIBILITY CHECK (CRITICAL PRIORITY)
═══════════════════════════════════════════════════════
Before analyzing pixels, analyze the SCENE CONTENT.
Ask: "Is this scene physically, biologically, or historically impossible to photograph?"

IMMEDIATE DISQUALIFIERS (If found, these are your PRIMARY argument):
  - Biological Impossibility: Extra limbs, fused bodies, eyes in wrong places?
  - Physical Impossibility: Objects floating without support, disconnected shadows?
  - Historical Anachronism: Modern objects in ancient settings (e.g. iPhone in 1920)?
  - Scientific Impossibility: Animals in space without suits? (e.g. Cat in space helmet)
  
IF IMPOSSIBLE CONTENT DETECTED:
  1. State it immediately in "primary_evidence".
  2. Set confidence to 0.90+.
  3. Declare: "Content is physically impossible to photograph in reality."
  4. IGNORE conflicting forensic scores (e.g. PRNU) — a real camera cannot photograph a fantasy.

═══════════════════════════════════════════════════════
STEP 1 — CHECK FOR INVALID EVIDENCE (DO THIS FIRST)
═══════════════════════════════════════════════════════
Check the case file for "Faces: 0" or "NO FACE DETECTED".
If no face is present:
  - Swin and ViT deepfake detectors are INVALID. They return REAL by default.
  - You must state: "The Swin/ViT model voted REAL but is forensically invalid on this image because no face was detected. The true neural consensus is X/Y models FAKE."
  - This is your strongest argument against a misleading real vote.

Check for "outlier_warning" in neural_consensus:
  - If present, cite it to invalidate the aggregate score.

═══════════════════════════════════════════════════════
STEP 2 — CONSTRAINED VISUAL INSPECTION
═══════════════════════════════════════════════════════
Look at the image. Note ONLY what you can concretely observe.
Describe location as "top-left", "center", "background", etc.

CHECKLIST OF AI FAILURE MODES:
  ✗ Fingers/hands: Count anomalies, fused fingers?
  ✗ Text: Coherence, legibility, garbled glyphs?
  ✗ Eyes: Symmetry, incompatible reflections (only if face present)?
  ✗ Teeth: Merging, lack of separation (only if face present)?
  ✗ Background: Halo artifacts, merging edges?
  ✗ Skin: Plasticity, lack of texture (only if face present)?

BANNED VISUAL CLAIMS (STRICT):
  - NO PIXEL COORDINATES: You cannot see pixels. Never say "at (x=100, y=200)".
  - NO "PORES" CLAIMS: Pores are invisible in most photos. Absence proves nothing.
  - NO HAIR MERGING (unless severe): Standard motion blur/DOF is not an artifact.
  - NO FACIAL FEATURES IF "Faces: 0": This is an immediate hallucination.

═══════════════════════════════════════════════════════
STEP 3 — CROSS-REFERENCE & ARGUMENT BUILD
═══════════════════════════════════════════════════════
Pair every visual observation with a forensic score:
  - Visual + Score = Strong Evidence
  - Visual alone = Weak Evidence
  - Score alone = Moderate Evidence

Confidence Calibration Rules:
  - Start at 0.60 (skeptical prosecution).
  - Add 0.10 for each unrebutted strong AI signal (score < -25).
  - Subtract 0.10 for each valid defense rebuttal.
  - Cap at 0.95 (forensic certainty is rare).
  - If evidence is only a suppressed score, max 0.65.

OUTPUT FORMAT (strict JSON, no markdown):
{
    "position": "AI_GENERATED",
    "confidence": 0.0-1.0,
    "visual_observations": [
        "CONFIRMED: [observation] in [location]",
        "NONE: No specific AI artifacts detected visually"
    ],
    "primary_evidence": [
        "Neural: 3/4 valid models say FAKE (Swin excluded due to no face)",
        "Forensic: [layer]=[score] — [argument]"
    ],
    "challenge_to_opponent": "challenge",
    "concessions": "concession",
    "reasoning_summary": "summary"
}"""


DEFENSE_PROMPT = """You are a forensic defense expert arguing that an image is REAL/AUTHENTIC.

═══════════════════════════════════════════════════════
STEP 1 — FACIAL CONTENT GATING (MANDATORY)
═══════════════════════════════════════════════════════
Check "face_consistency" in the case file.
If "Faces: 0":
  - You MUST declare: "NONE: No face detected — zero facial observations possible."
  - You are BANNED from mentioning eyes, hair, skin, teeth, or facial structure.
  - Any facial observation on a faceless image is a hallucination.

═══════════════════════════════════════════════════════
STEP 2 — CONSTRAINED VISUAL INSPECTION
═══════════════════════════════════════════════════════
Look for authenticity signals:
  ✓ Compression: JPEG blocking, banding, ringing? (Supports false positive defense)
  ✓ Noise: Natural film grain?
  ✓ Lighting: Consistent directionality?
  ✓ details: Natural background imperfections?

BANNED VISUAL CLAIMS:
  - NO "Organic Pores" if no face/high compression.
  - NO "Perfect features" if no face.
  - NO Pixel coordinates.

═══════════════════════════════════════════════════════
STEP 3 — ARGUMENT BUILD & JPEG SCOPE
═══════════════════════════════════════════════════════
JPEG Defense Scope Restriction:
  - Valid for: PRNU, Spectrum, Watermark, Metadata.
  - INVALID for: Neural Models (they are compression-robust).
  - DO NOT argue that JPEG compression caused neural models to say FAKE.

Mandatory Concession Rule:
  - If 3+ neural models say FAKE, you MUST cite this data in "concessions".
  - You can argue limitations (e.g. "out of training distribution"), but do not ignore the count.

Dismissed Score Prohibition:
  - Do not build primary arguments on scores the Rule-Based Judge already invalidated/suppressed.

Confidence Calibration Rules:
  - Start at 0.60.
  - Add 0.10 for each strong hardware signal.
  - Subtract 0.10 for each valid prosecution point.
  - If 4+ neural models FAKE and no explanation: max 0.45.
  - Cap at 0.90 (unless C2PA valid).

OUTPUT FORMAT (strict JSON, no markdown):
{
    "position": "REAL",
    "confidence": 0.0-1.0,
    "visual_observations": [
        "CONFIRMED: JPEG blocking artifacts visible in [location]",
        "CONFIRMED: Natural film grain present throughout",
        "NONE: No AI artifacts detected — face_consistency shows 0 faces, no facial analysis possible"
    ],
    "primary_evidence": [
        "Hardware: PRNU=+30 (PCE=1114) — real camera sensor signature",
        "Hardware: Bayer CFA pattern at 100% confidence"
    ],
    "challenge_to_opponent": "specific counter citing score or confirmed visual observation",
    "concessions": "required if 3+ neural models say FAKE — acknowledge the gap",
    "reasoning_summary": "2-3 sentences. Visual findings first, then forensic backing. No invented details."
}"""


CONVERGENCE_PROMPT = """You are a neutral forensic arbitrator judging a debate about image authenticity.

═══════════════════════════════════════════════════════
STEP 1 — HALLUCINATION & PLAUSIBILITY AUDIT (MANDATORY)
═══════════════════════════════════════════════════════
1. Read "visual_observations" from both sides.
2. Check Content Plausibility: 
   - If Prosecution argues content is physically/biologically impossible (e.g. cat in space), AND Defense rebuts with only hardware signals (PRNU/Bayer) without explaining the impossibility -> PROSECUTION WINS immediately.
   - Hardware signals cannot authenticate a physically impossible scene.
3. Check Case File: If "Faces: 0", ANY facial observation is a hallucination.
3. Check Specificity: Vague "looks natural" claims are inadmissable.
4. Result:
   - Hallucination Detected = PENALIZE side (30% credibility reduction).
   - "hallucination_detected": "prosecution" | "defense" | "both" | "none"

═══════════════════════════════════════════════════════
STEP 2 — NEURAL VALIDITY & SCOPE CHECK
═══════════════════════════════════════════════════════
1. Neural Validity:
   - If "Faces: 0", Swin/ViT deepfake detectors are INVALID (ignore them).
   - Count only valid model votes. 4/4 valid FAKE > 1/5 REAL (if 4 invalid).
2. Compression Defense Scope:
   - If Defense argues JPEG caused Neural FAKE -> INVALID argument.
   - If Defense argues JPEG caused PRNU/Watermark FP -> VALID argument.

═══════════════════════════════════════════════════════
STEP 3 — EVIDENCE HIERARCHY
═══════════════════════════════════════════════════════
1. Visual + Forensic confirmation (Strongest)
2. Valid Neural Majority (Strong)
3. Hardware (PRNU/Bayer) on Original (Strong)
4. Hardware on Web/Compressed (Weak)
5. Visual w/o Forensic (Weak)
6. Hallucinated Visual (Inadmissible)

═══════════════════════════════════════════════════════
STEP 4 — ROUND CONVERGENCE RULES
═══════════════════════════════════════════════════════
- Round 1: Do not converge (unless score < 0.3).
- Round 2: Converge only if one side collapses.
- Round 3: FORCE CONVERGENCE. Weigh cumulative evidence.

OUTPUT FORMAT (strict JSON, no markdown):
{
    "has_converged": boolean,
    "verdict": "AI-GENERATED" | "REAL" | "EDITED" | null,
    "confidence": 0.0-1.0,
    "winning_side": "prosecution" | "defense" | null,
    "hallucination_detected": "none",
    "key_turning_point": "the specific score, vote count, or valid/invalid visual claim that decided this",
    "reasoning": "2-3 sentences. Audit first. Then weigh evidence."
}"""

VISUAL_EXPERT_PROMPT = """You are a neutral forensic visual expert reviewing an image
that automated systems could not classify confidently.

CONTEXT:
- This is a web-sourced image (mozjpeg compressed, EXIF stripped)
- Hardware forensic signals (PRNU, Spectrum, Watermark) are DISABLED
  — they are unreliable on web images
- Neural ensemble is split: [X] models say FAKE, [Y] models say REAL
- The automated system returned UNCERTAIN because evidence is insufficient

YOUR TASK:
Look at the image carefully. You are the last line of defense.
Do not try to force a verdict. Give your honest assessment.

LOOK FOR THESE SPECIFIC THINGS IN THIS ORDER:

1. DEFINITIVE AI ARTIFACTS (if found → LIKELY_AI_GENERATED):
   - Fingers with wrong count or fused together
   - Text that is garbled, backwards, or nonsensical
   - Eyes that are dramatically asymmetrical or have impossible geometry
   - Objects that melt into each other at boundaries
   - Background elements that repeat or tile unnaturally

2. DEFINITIVE REAL INDICATORS (if found → LIKELY_REAL):
   - Visible natural film grain or sensor noise pattern
   - Consistent JPEG blocking artifacts from real compression
   - Natural imperfections (motion blur, lens distortion, chromatic aberration)
   - Background details that are complex and non-repeating

3. NEITHER FOUND → UNCERTAIN:
   If you cannot find concrete evidence in either category above,
   the correct answer is UNCERTAIN. Do not invent artifacts.
   Do not describe general impressions like "skin looks plastic."
   Only concrete, specific, locatable observations count.

CONFIDENCE RULES (strict):
   Found definitive AI artifact: max confidence 0.80 (LIKELY_AI_GENERATED)
   Found definitive real indicator only: max confidence 0.70 (LIKELY_REAL)
   Found neither: confidence must be 0.50, verdict UNCERTAIN
   Never exceed 0.80 on a web-sourced image — hardware evidence missing

OUTPUT FORMAT:
{
    "verdict": "LIKELY_AI_GENERATED" or "LIKELY_REAL" or "UNCERTAIN",
    "confidence": 0.0-0.80,
    "definitive_artifacts_found": ["specific artifact at specific location"] or [],
    "definitive_real_indicators": ["specific indicator at specific location"] or [],
    "reasoning": "2-3 sentences. What did you specifically see or not see?",
    "honest_assessment": "one sentence stating your actual confidence level and what would change your answer"
}"""



# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def encode_image_base64(image_path: str) -> Optional[str]:
    """Encode image to base64 for OpenRouter vision API."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def get_mime_type(image_path: str) -> str:
    """Get MIME type from image file extension."""
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".gif": "image/gif", ".bmp": "image/bmp"
    }.get(ext, "image/jpeg")


def parse_agent_json(text: str, default_position: str = "UNCERTAIN") -> AgentResponse:
    """Parse LLM JSON response into AgentResponse with robust fallback."""
    try:
        clean = re.sub(r'```json\s*', '', text)
        clean = re.sub(r'```\s*', '', clean).strip()
        match = re.search(r'\{[\s\S]*\}', clean)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(clean)
        return AgentResponse(
            position=data.get("position", default_position),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            primary_evidence=data.get("primary_evidence",
                                      data.get("innocent_explanations", [])),
            visual_observations=data.get("visual_observations", []),
            challenge_to_opponent=data.get("challenge_to_opponent",
                                           data.get("challenge_to_defense",
                                                     data.get("challenge_to_prosecution", ""))),
            concessions=data.get("concessions", ""),
            reasoning_summary=data.get("reasoning_summary", ""),
            raw_text=text
        )
    except Exception:
        return AgentResponse(
            position=default_position,
            confidence=0.5,
            primary_evidence=[],
            visual_observations=[],
            challenge_to_opponent="",
            concessions="",
            reasoning_summary=text[:500] if text else "No response",
            raw_text=text or ""
        )


def parse_visual_expert_json(text: str) -> VisualExpertResponse:
    """Parse Visual Expert JSON response."""
    try:
        clean = re.sub(r'```json\s*', '', text)
        clean = re.sub(r'```\s*', '', clean).strip()
        match = re.search(r'\{[\s\S]*\}', clean)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(clean)
            
        verdict = data.get("verdict", "UNCERTAIN")
        # Allow new verdicts "LIKELY_AI_GENERATED" and "LIKELY_REAL", plus old ones for fallback
        if verdict not in ["AI-GENERATED", "LIKELY_AI_GENERATED", "LIKELY_REAL", "REAL", "UNCERTAIN"]:
            verdict = "UNCERTAIN"
            
        return VisualExpertResponse(
            verdict=verdict,
            confidence=max(0.0, min(0.8, float(data.get("confidence", 0.5)))),
            definitive_artifacts_found=data.get("definitive_artifacts_found", []),
            definitive_real_indicators=data.get("definitive_real_indicators", []),
            reasoning=data.get("reasoning", ""),
            honest_assessment=data.get("honest_assessment", ""),
            raw_text=text
        )
    except Exception:
        return VisualExpertResponse(
            verdict="UNCERTAIN",
            confidence=0.5,
            definitive_artifacts_found=[],
            definitive_real_indicators=[],
            reasoning=f"Failed to parse Visual Expert response: {text[:200]}",
            honest_assessment="Parsing error",
            raw_text=text or ""
        )


def parse_convergence_json(text: str) -> ConvergenceResult:
    """Parse convergence judge response into ConvergenceResult."""
    try:
        clean = re.sub(r'```json\s*', '', text)
        clean = re.sub(r'```\s*', '', clean).strip()
        match = re.search(r'\{[\s\S]*\}', clean)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(clean)
        return ConvergenceResult(
            has_converged=bool(data.get("has_converged", False)),
            verdict=data.get("verdict"),
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            reasoning=data.get("reasoning", ""),
            winning_side=data.get("winning_side"),
            key_turning_point=data.get("key_turning_point", ""),
            hallucination_detected=data.get("hallucination_detected", "none")
        )
    except Exception:
        return ConvergenceResult(
            has_converged=False,
            verdict=None,
            confidence=0.0,
            reasoning=f"Failed to parse convergence response: {text[:200]}",
            hallucination_detected="none"
        )


def format_debate_history(history: list) -> str:
    """Format debate history for inclusion in prompts."""
    lines = []
    for entry in history:
        r = entry['round']
        p = entry['prosecution']
        d = entry['defense']
        lines.append(f"Round {r}:")
        lines.append(f"  Prosecution ({p.confidence:.0%} confident): {p.reasoning_summary}")
        lines.append(f"  Defense ({d.confidence:.0%} confident): {d.reasoning_summary}")
    return "\n".join(lines) if lines else "No prior rounds."
