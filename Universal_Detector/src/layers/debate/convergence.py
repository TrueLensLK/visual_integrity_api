"""
Debate System — Convergence Detector
Neutral LLM judge that reads the debate transcript and delivers a verdict.
"""

import json
from typing import Optional

from .models import (
    ConvergenceResult,
    CONVERGENCE_PROMPT,
    parse_convergence_json,
    GROQ_MODEL,
    CEREBRAS_MODEL,
)


class ConvergenceDetector:
    """
    Neutral 3rd-party LLM that reads the full debate transcript.
    JUDGE: Groq (Primary) with Cerebras fallback.
    """

    def __init__(self, groq_api_key: Optional[str] = None,
                 cerebras_api_key: Optional[str] = None):
        self._groq_client = None
        self._groq_key = groq_api_key
        self._cerebras_client = None
        self._cerebras_key = cerebras_api_key

    def _init_groq(self):
        if self._groq_client or not self._groq_key: return
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self._groq_key)
        except Exception as e:
            print(f"[Debate/Convergence] Groq init failed: {e}")

    def _init_cerebras(self):
        if self._cerebras_client or not self._cerebras_key: return
        try:
            from openai import OpenAI
            self._cerebras_client = OpenAI(
                api_key=self._cerebras_key,
                base_url="https://api.cerebras.ai/v1"
            )
        except Exception as e:
            print(f"[Debate/Convergence] Cerebras init failed: {e}")

    def check(self, debate_history: list, case_string: str) -> ConvergenceResult:
        """
        Evaluate the debate transcript and determine if truth has been found.
        """
        transcript = self._build_transcript(debate_history)

        user_prompt = f"""FORENSIC CASE FILE:
{case_string}

DEBATE TRANSCRIPT:
{transcript}

Evaluate this debate. Which side presented stronger forensic evidence?
Has one side clearly won, or is the debate still genuinely contested?
Deliver your judgment."""

        # 1. Try Groq (Primary)
        response_text = self._call_groq(user_prompt)
        
        # 2. Try Cerebras (Fallback)
        if not response_text:
            print("[Debate/Convergence] Groq failed. Falling back to Cerebras.")
            response_text = self._call_cerebras(user_prompt)

        # Process the result (even if failed, we must evaluate confidence differential)
        if not response_text:
            result = ConvergenceResult(
                has_converged=False, verdict=None, confidence=0.0,
                reasoning="All convergence providers failed"
            )
        else:
            result = parse_convergence_json(response_text)

        # ============================================================
        # HARD OVERRIDE: Code-level confidence differential check
        # Fixes Problem 1: Judge ignoring strong confidence signals
        # ============================================================
        if len(debate_history) >= 2:
            last_round = debate_history[-1]
            p_conf = last_round['prosecution'].confidence
            d_conf = last_round['defense'].confidence
            
            # Trajectory analysis
            p_trajectory = [r['prosecution'].confidence for r in debate_history]
            d_trajectory = [r['defense'].confidence for r in debate_history]
            
            p_increased = p_trajectory[-1] >= p_trajectory[0]
            d_flat_or_dropped = d_trajectory[-1] <= d_trajectory[0]
            
            # Case 1: Prosecution confident win ignored by judge
            if (p_conf > d_conf + 0.05 
                    and p_increased 
                    and d_flat_or_dropped
                    and result.winning_side != "prosecution"):
                
                print(f"[Convergence] OVERRIDE: Prosecution conf={p_conf:.0%} > "
                      f"Defense conf={d_conf:.0%} but judge picked {result.winning_side} — correcting")
                
                result.has_converged = True
                result.winning_side = "prosecution"
                result.verdict = "AI-GENERATED"
                result.confidence = p_conf * 0.95  # Slightly dampen raw agent confidence
                result.reasoning = (
                    f"Override: Prosecution confidence ({p_conf:.0%}) exceeded defense "
                    f"({d_conf:.0%}) and increased across rounds while defense remained flat. "
                    f"Prosecution presented stronger cumulative evidence despite the judge's text."
                )
            
            # Case 2: Defense confident win ignored by judge
            elif (d_conf > p_conf + 0.05
                    and d_trajectory[-1] >= d_trajectory[0]
                    and result.winning_side != "defense"):
                
                print(f"[Convergence] OVERRIDE: Defense conf={d_conf:.0%} > "
                      f"Prosecution conf={p_conf:.0%} but judge picked {result.winning_side} — correcting")
                
                result.has_converged = True
                result.winning_side = "defense"
                result.verdict = "REAL"
                result.confidence = d_conf * 0.95
                result.reasoning = (
                    f"Override: Defense confidence ({d_conf:.0%}) exceeded prosecution "
                    f"({p_conf:.0%}) and maintained stability. Case for authenticity was stronger."
                )

        return result

    def _call_groq(self, user_prompt: str) -> Optional[str]:
        self._init_groq()
        if not self._groq_client: return None
        try:
            response = self._groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": CONVERGENCE_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Debate/Convergence] Groq error: {e}")
            return None

    def _call_cerebras(self, user_prompt: str) -> Optional[str]:
        self._init_cerebras()
        if not self._cerebras_client: return None
        try:
            response = self._cerebras_client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[
                    {"role": "system", "content": CONVERGENCE_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Debate/Convergence] Cerebras error: {e}")
            return None

    def _build_transcript(self, history: list) -> str:
        """Build a detailed transcript for the convergence judge to read."""
        transcript = []
        for round_idx, entry in enumerate(history):
            if "round" in entry:
                transcript.append(f"--- ROUND {entry['round']} ---")
                
            if "prosecution" in entry:
                p = entry["prosecution"]
                transcript.append(f"PROSECUTION: {p.position} ({p.confidence:.0%})")
                
                if hasattr(p, 'content_assessment') and p.content_assessment:
                    transcript.append("Content Assessment:")
                    for k, v in p.content_assessment.items():
                        transcript.append(f"  - {k}: {v}")

                if hasattr(p, 'visual_observations') and p.visual_observations:
                    transcript.append(f"Visual Observations: {', '.join(p.visual_observations)}")
                
                transcript.append(f"Argument: {p.reasoning_summary}")
                transcript.append(f"Key Evidence: {', '.join(p.primary_evidence)}")
                transcript.append("")

            if "defense" in entry:
                d = entry["defense"]
                transcript.append(f"DEFENSE: {d.position} ({d.confidence:.0%})")
                
                if hasattr(d, 'content_assessment') and d.content_assessment:
                    transcript.append("Content Assessment:")
                    for k, v in d.content_assessment.items():
                        transcript.append(f"  - {k}: {v}")
                
                if hasattr(d, 'visual_observations') and d.visual_observations:
                    transcript.append(f"Visual Observations: {', '.join(d.visual_observations)}")

                transcript.append(f"Argument: {d.reasoning_summary}")
                transcript.append(f"Key Evidence: {', '.join(d.primary_evidence)}")
                transcript.append("")
                
        return "\n".join(transcript)
