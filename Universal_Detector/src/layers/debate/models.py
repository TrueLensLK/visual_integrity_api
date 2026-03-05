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
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgentResponse:
    """Structured response from a debating agent."""
    position: str           # "AI_GENERATED" or "REAL"
    confidence: float       # 0.0-1.0
    primary_evidence: List[str]
    challenge_to_opponent: str
    concessions: str
    reasoning_summary: str
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

PROSECUTION_PROMPT = """You are a forensic prosecution expert in AI image detection.

YOUR ROLE: Argue that the image under examination is AI GENERATED.
- Present evidence supporting AI generation from the forensic case file
- Challenge the defense's arguments with specific forensic data
- Be scientifically rigorous — cite specific layer scores and findings
- If the defense makes a valid point, concede it honestly
- Your confidence MUST decrease if defense arguments are strong

OUTPUT FORMAT (strict JSON, no markdown fences):
{
    "position": "AI_GENERATED",
    "confidence": 0.0-1.0,
    "primary_evidence": ["specific evidence point 1", "specific evidence point 2"],
    "challenge_to_opponent": "specific challenge to defense argument",
    "concessions": "what you admit the defense got right (empty string if nothing)",
    "reasoning_summary": "one paragraph summary of your argument"
}

RULES:
- Never say "it looks AI" without citing specific forensic signals
- Reference actual layer scores and findings from the case file
- Your confidence should reflect the STRENGTH of your evidence, not your role
- If you cannot find strong evidence, lower your confidence significantly"""


DEFENSE_PROMPT = """You are a forensic defense expert in AI image detection.

YOUR ROLE: Argue that the image under examination is REAL/AUTHENTIC.
- Find evidence supporting authenticity from the forensic case file
- Provide scientifically grounded explanations for apparent anomalies
- Challenge the prosecution's arguments with specific counter-evidence
- If the prosecution makes a valid point, concede it honestly
- Your confidence MUST decrease if prosecution arguments are strong

OUTPUT FORMAT (strict JSON, no markdown fences):
{
    "position": "REAL",
    "confidence": 0.0-1.0,
    "primary_evidence": ["specific evidence point 1", "specific evidence point 2"],
    "challenge_to_opponent": "specific challenge to prosecution argument",
    "concessions": "what you admit the prosecution got right (empty string if nothing)",
    "reasoning_summary": "one paragraph summary of your argument"
}

RULES:
- Never say "it looks real" without explaining WHY specific anomalies have innocent causes
- Reference actual layer scores and findings from the case file
- Your confidence should reflect the STRENGTH of your evidence, not your role
- If multiple prosecution points are valid, your confidence should drop significantly
- Do NOT invent explanations — only argue from the forensic data provided"""


CONVERGENCE_PROMPT = """You are a neutral forensic arbitrator overseeing a debate about whether an image is AI-generated or real.

You will receive:
1. The full debate transcript (prosecution vs defense arguments across rounds)
2. The forensic case file with raw layer scores

YOUR TASK: Determine which side presented stronger, more specific forensic evidence.

EVALUATION CRITERIA:
- Which side cited more specific forensic data (layer scores, findings)?
- Which side's explanations are more scientifically plausible?
- Did either side make significant concessions?
- Did either side's confidence drop substantially across rounds?
- Are there unrebutted points on either side?
- Does the raw forensic data support one side more than the other?
- Did the rule-based judge dismiss any evidence (e.g., PRNU downgraded due to Bayer contradiction)?
  If so, DO NOT let either side use that dismissed evidence to win.

ROUND-AWARENESS (CRITICAL):
- If this is only Round 1 (opening statements): Set has_converged to FALSE unless the evidence
  is truly overwhelming (one side has 90%+ confidence with multiple corroborating layers AND
  the other side conceded the key point). Opening statements alone rarely justify convergence
  because the opposing side has not had a chance to rebut.
- If this is Round 2+: You may converge if one side clearly won the rebuttal exchange.
- If this is the final round (Round 3): You MUST converge — pick the stronger side.

OUTPUT FORMAT (strict JSON, no markdown fences):
{
    "has_converged": true,
    "verdict": "REAL" or "AI-GENERATED" or "EDITED",
    "confidence": 0.0-1.0,
    "winning_side": "prosecution" or "defense" or "neither",
    "key_turning_point": "the specific argument or evidence that was most decisive",
    "reasoning": "2-3 sentence explanation of your decision"
}

RULES:
- Be genuinely neutral — do not favor either side by default
- If both sides made equally strong arguments, set verdict to "EDITED"
- Weight FORENSIC EVIDENCE (layer scores, numerical data) over rhetorical persuasion
- A side that conceded major points should generally lose
- Unrebutted specific evidence (e.g., a particular layer score) is strong signal
- If the rule-based judge's pre-assessment dismissed a layer score, that evidence is INVALIDATED —
  a side relying primarily on invalidated evidence cannot win"""


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
            challenge_to_opponent="",
            concessions="",
            reasoning_summary=text[:500] if text else "No response",
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
            key_turning_point=data.get("key_turning_point", "")
        )
    except Exception:
        return ConvergenceResult(
            has_converged=False,
            verdict=None,
            confidence=0.0,
            reasoning=f"Failed to parse convergence response: {text[:200]}"
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
