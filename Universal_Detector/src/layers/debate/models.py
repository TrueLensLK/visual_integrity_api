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

# ═══════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentResponse:
    """Structured response from a debating agent."""
    position: str           # "AI_GENERATED" or "REAL"
    confidence: float       # 0.0-1.0
    primary_evidence: List[str]
    visual_observations: List[str]
    challenge_to_opponent: str
    concessions: str
    reasoning_summary: str
    content_assessment: Dict[str, str] = field(default_factory=dict) # NEW: Fix 3
    raw_text: str = ""

@dataclass
class VisualExpertResponse:
    """Structured response from the Visual Expert Agent."""
    verdict: str                # "AI-GENERATED", "LIKELY_REAL", "UNCERTAIN", "EDITED_REAL"
    confidence: float           # 0.0-1.0
    definitive_artifacts_found: List[str]
    definitive_real_indicators: List[str]
    reasoning: str
    honest_assessment: str
    editing_signs: List[str] = field(default_factory=list)
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
    reliability_assessment: str = ""      # NEW: Fix 6


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
STEP 0 — CONTENT ASSESSMENT (MANDATORY START)
═══════════════════════════════════════════════════════
Answer these three questions first. They determine your baseline confidence.

Question 1: What is in this image?
  - Describe subject, setting, and context plainly. No forensic jargon.

Question 2: Is this type of scene commonly photographed by real people?
  - YES or NO with one sentence explanation.
  - If NO (e.g. "Cat in space" or "Anime character in real life"), your confidence floor rises to 0.85 immediately.

Question 3: Does the content itself suggest AI generation?
  - Answer only if CLEARLY YES (impossible physics, non-existent objects, impossible anatomy).
  - If YES, you can claim 0.90+ confidence on content alone.

Output these answers in the "content_assessment" JSON field.

═══════════════════════════════════════════════════════
STEP 1 — CHECK FOR INVALID EVIDENCE (DO THIS FIRST)
═══════════════════════════════════════════════════════
Check the case file for "Faces: 0" or "NO FACE DETECTED".
If no face is present:
  - Swin and ViT deepfake detectors are INVALID. They return REAL by default.
  - You must state: "The Swin/ViT model voted REAL but is forensically invalid on this image because no face was detected. The true neural consensus is X/Y models FAKE."

Check for "outlier_warning" in neural_consensus:
  - If present, cite it to invalidate the aggregate score.

═══════════════════════════════════════════════════════
STEP 2 — CONSTRAINED VISUAL INSPECTION
═══════════════════════════════════════════════════════
Look at the image. Note ONLY what you can concretely observe.
CHECKLIST OF AI FAILURE MODES:
  ✗ Fingers/hands: Count anomalies, fused fingers?
  ✗ Text: Coherence, legibility, garbled glyphs?
  ✗ Eyes: Symmetry, incompatible reflections (only if face present)?
  ✗ Teeth: Merging, lack of separation (only if face present)?
  ✗ Background: Halo artifacts, merging edges?

BANNED VISUAL CLAIMS (STRICT):
  - NO PIXEL COORDINATES.
  - NO "PORES" CLAIMS.
  - NO FACIAL FEATURES IF "Faces: 0".

═══════════════════════════════════════════════════════
STEP 3 — CONFIDENCE CALCULATION (STRICT RULES)
═══════════════════════════════════════════════════════
Start at 0.55.

ADD 0.10 for each (if HIGH reliability):
  + Neural specialist (SDXL) > 0.80
  + PRNU synthetic grid with RELIABILITY: HIGH
  + Watermark "SynthID" or known AI tool signature
  + C2PA confirms AI generation
  + Concrete specific visual AI artifact found (melted fingers, garbled text)
  + Content Assessment = Physically impossible

ADD 0.05 for each:
  + 3 or more valid neural models agree FAKE
  + Spectrum shows AI manipulation (HIGH reliability)
  + Physical continuity violations confirmed

SUBTRACT 0.10 for each:
  - Cited signal is RELIABILITY: LOW in case file
  - Defense provides specific unrebutted explanation
  - Visual inspection found ZERO concrete artifacts
  - Content Assessment = Commonly photographed scene

CAPS:
  - Max 0.85 if CAMERA_ORIGINAL
  - Max 0.75 if LIKELY_WEB_SOURCED
  - Max 0.65 if all primary signals are LOW reliability
  - Minimum 0.40

OUTPUT FORMAT (strict JSON, no markdown):
{
    "position": "AI_GENERATED",
    "confidence": 0.0-1.0,
    "content_assessment": {
        "description": "text",
        "is_common_scene": "Yes/No",
        "suggests_ai_content": "Yes/No"
    },
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
STEP 0 — CONTENT ASSESSMENT (MANDATORY START)
═══════════════════════════════════════════════════════
Answer these three questions first.

Question 1: What is in this image?
  - Describe subject, setting, and context plainly.

Question 2: Is this type of scene commonly photographed by real people?
  - YES or NO with one sentence explanation.
  - If YES (e.g. "Device photo", "Pet photo", "Selfie"), this is a strong defense argument.

Question 3: Does the content itself suggest AI generation?
  - Answer "No definitive content impossibility detected" unless obvious.

Output these answers in the "content_assessment" JSON field.

═══════════════════════════════════════════════════════
STEP 1 — FACIAL CONTENT GATING
═══════════════════════════════════════════════════════
If "face_consistency" says "Faces: 0":
  - You MUST declare: "NONE: No face detected — zero facial observations possible."
  - BANNED from mentioning eyes, hair, skin, teeth, or facial structure.

═══════════════════════════════════════════════════════
STEP 2 — VISUAL REAL-PHOTO INDICATORS (MANDATORY)
═══════════════════════════════════════════════════════
Identify 3 specific visual elements that indicate real photography.
Look for:
  ✓ Natural fabric texture/draping
  ✓ Individual hair/fur strands with natural flow
  ✓ Realistic hand anatomy (wrinkles, knuckles)
  ✓ Recognizable real-world brands/text
  ✓ Random/non-repeating background clutter
  ✓ Natural depth-of-field blur
  ✓ Authentic light reflections

You must find 3 to get the +0.10 confidence bonus.
Cite them in "primary_evidence" as "Visual: [observation]".

═══════════════════════════════════════════════════════
STEP 3 — CONFIDENCE CALCULATION (STRICT RULES)
═══════════════════════════════════════════════════════
Start at 0.55.

ADD 0.10 for each:
  + CAMERA_ORIGINAL classification confirmed
  + PRNU positive with RELIABILITY: HIGH (or Low PCE on Original)
  + Bayer pattern detected
  + Metadata shows genuine EXIF
  + Content Assessment = Commonly photographed scene
  + Found 3+ concrete visual real-photo indicators

ADD 0.05 for each:
  + Spectrum positive or neutral
  + Physical continuity positive
  + 2+ neural models say REAL

SUBTRACT 0.10 for each:
  - Neural specialist (SDXL) > 0.80 FAKE
  - 4+ neural models say FAKE
  - Prosecution finds concrete AI artifact you cannot explain

CAPS:
  - Max 0.90 with C2PA proof
  - Max 0.80 with CAMERA_ORIGINAL + multiple HIGH reliability signals
  - Max 0.70 on LIKELY_WEB_SOURCED
  - Minimum 0.35

OUTPUT FORMAT (strict JSON, no markdown):
{
    "position": "REAL",
    "confidence": 0.0-1.0,
    "content_assessment": {
        "description": "text",
        "is_common_scene": "Yes/No",
        "suggests_ai_content": "Yes/No"
    },
    "visual_observations": [
        "CONFIRMED: Natural texture on router surface",
        "CONFIRMED: Legible text on label"
    ],
    "primary_evidence": [
        "Visual: Realistic hand anatomy with natural wrinkles",
        "Hardware: Bayer CFA pattern detected"
    ],
    "challenge_to_opponent": "specific counter citing score or confirmed visual observation",
    "concessions": "required if 3+ neural models say FAKE — acknowledge the gap",
    "reasoning_summary": "2-3 sentences. V"
}"""


CONVERGENCE_PROMPT = """You are a neutral forensic arbitrator judging a debate about image authenticity.
Your goal is to weigh evidence based on RELIABILITY, not just persuasion.

═══════════════════════════════════════════════════════
STEP 1 — RELIABILITY AUDIT
═══════════════════════════════════════════════════════
Cross-reference every cited forensic signal against the Case File reliability flags.
  - HIGH reliability: Counts fully (100%).
  - MEDIUM reliability: Counts partially (70%).
  - LOW reliability: Counts minimally (30%).

Example: Prosecution cites PRNU=-50. Case File says RELIABILITY: LOW (compression).
Result: This argument is weak despite the high score.

Output a "reliability_assessment" field summarizing who had better quality evidence.

═══════════════════════════════════════════════════════
STEP 2 — CONTENT ASSESSMENT CHECK
═══════════════════════════════════════════════════════
Read "content_assessment" from Round 1.
  - If Prosecution identified impossible content (unrebutted) -> Prosecution Wins.
  - If Defense identified "Common Real Scene" AND Prosecution relies only on LOW reliability forensic signals -> Defense likely wins.

═══════════════════════════════════════════════════════
STEP 3 — VISUAL OBSERVATION AUDIT
═══════════════════════════════════════════════════════
Check "visual_observations" against the image content.
  - Hallucination: Describing faces when Faces: 0.
  - Hallucination: Describing impossible details.
  Result: Penalize credibility by 30% for hallucinations.

"hallucination_detected": "prosecution" | "defense" | "both" | "none"

═══════════════════════════════════════════════════════
STEP 4 — ROUND CONVERGENCE RULES (RELIABILITY WEIGHTED)
═══════════════════════════════════════════════════════
- Round 1: Do not converge (unless score < 0.3).
- Round 2: Converge only if one side collapses.
- Round 3: FORCE CONVERGENCE. Weigh cumulative evidence using Hierarchy:
  1. Visual + HIGH Reliability Forensic (Strongest)
  2. Valid Neural Majority (Strong)
  3. HIGH Reliability Hardware (Strong)
  4. LOW Reliability Hardware (Weak - do not base verdict on this)
  5. Visual alone (Weak)

OUTPUT FORMAT (strict JSON, no markdown):
{
    "has_converged": boolean,
    "verdict": "AI-GENERATED" | "REAL" | "EDITED" | null,
    "confidence": 0.0-1.0,
    "winning_side": "prosecution" | "defense" | null,
    "reliability_assessment": "Prosecution had 1 HIGH, 2 LOW. Defense had 2 HIGH.",
    "hallucination_detected": "none",
    "key_turning_point": "reason",
    "reasoning": "reason"
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

3. HEAVY RETOUCHING (if found → EDITED_REAL):
   - Skin smoothing that removes pores but leaves structure intact
   - Color grading that looks stylized but consistent
   - Objects removed or added cleanly (e.g. in Photoshop)
   - This is common in professional photography (magazines, weddings)
   - Do NOT classify this as AI relative to generative artifacts.
   - Verdict: EDITED_REAL

4. NEITHER FOUND → UNCERTAIN:
   If you cannot find concrete evidence in above categories,
   the correct answer is UNCERTAIN. Do not invent artifacts.

CONFIDENCE RULES (strict):
   Found definitive AI artifact: max confidence 0.95 (LIKELY_AI_GENERATED)
   Found complex retouching: max confidence 0.65 (EDITED_REAL)
   Found definitive real indicator only: max confidence 0.85 (LIKELY_REAL)
   Found neither: confidence must be 0.50, verdict UNCERTAIN

OUTPUT FORMAT:
{
    "verdict": "LIKELY_AI_GENERATED" or "LIKELY_REAL" or "EDITED_REAL" or "UNCERTAIN",
    "confidence": 0.0-1.0,
    "definitive_artifacts_found": ["specific artifact at specific location"] or [],
    "definitive_real_indicators": ["specific indicator at specific location"] or [],
    "editing_signs": ["smooth skin", "color grade"] or [],
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
            content_assessment=data.get("content_assessment", {}),
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
            content_assessment={},
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
        if verdict not in ["AI-GENERATED", "LIKELY_AI_GENERATED", "LIKELY_REAL", "REAL", "UNCERTAIN", "EDITED_REAL"]:
            verdict = "UNCERTAIN"
            
        return VisualExpertResponse(
            verdict=verdict,
            confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
            definitive_artifacts_found=data.get("definitive_artifacts_found", []),
            definitive_real_indicators=data.get("definitive_real_indicators", []),
            editing_signs=data.get("editing_signs", []),
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
            editing_signs=[],
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
            hallucination_detected=data.get("hallucination_detected", "none"),
            reliability_assessment=data.get("reliability_assessment", "")
        )
    except Exception:
        return ConvergenceResult(
            has_converged=False,
            verdict=None,
            confidence=0.0,
            reasoning=f"Failed to parse convergence response: {text[:200]}",
            winning_side=None,
            key_turning_point="",
            hallucination_detected="none",
            reliability_assessment="Parsing error"
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
