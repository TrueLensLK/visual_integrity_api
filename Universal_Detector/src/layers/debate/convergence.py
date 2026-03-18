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
)


class ConvergenceDetector:
    """
    Neutral 3rd-party LLM that reads the full debate transcript.
    JUDGE: Gemini 2.5 Flash (primary) with Groq fallback.
    """

    def __init__(self, groq_api_key: Optional[str] = None,
                 gemini_api_key: Optional[str] = None):
        self._groq_client = None
        self._groq_key = groq_api_key
        self._gemini_model = None
        self._gemini_key = gemini_api_key

    def _init_groq(self):
        if self._groq_client: return
        if not self._groq_key: return
        try:
            from groq import Groq
            # Use official SDK directly (no base_url override needed)
            self._groq_client = Groq(api_key=self._groq_key)
        except Exception as e:
            print(f"[Debate/Convergence] Groq init failed: {e}")

    def _init_gemini(self):
        if self._gemini_model: return
        if not self._gemini_key: return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self._gemini_key)
            # Use Gemini 2.5 Flash as requested (Judge)
            self._gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception:
            pass

    def check(self, debate_history: list, case_string: str) -> ConvergenceResult:
        """
        Evaluate the debate transcript and determine if truth has been found.
        
        JUDGE: Gemini 2.5 Flash as main arbitrator (user choice).
        FALLBACK: Groq/Llama-3.3-70b/Versatile.
        """
        transcript = self._build_transcript(debate_history)

        user_prompt = f"""FORENSIC CASE FILE:
{case_string}

DEBATE TRANSCRIPT:
{transcript}

Evaluate this debate. Which side presented stronger forensic evidence?
Has one side clearly won, or is the debate still genuinely contested?
Deliver your judgment."""

        # Try Gemini (Primary Judge) as requested
        response_text = self._call_gemini(user_prompt)

        # Fallback to Groq if Gemini fails
        if not response_text:
            print("[Debate/Convergence] Gemini Judge failed, falling back to Groq Llama-3.3")
            response_text = self._call_groq(user_prompt)

        if response_text:
            return parse_convergence_json(response_text)

        # Total failure — cannot converge
        return ConvergenceResult(
            has_converged=False, verdict=None, confidence=0.0,
            reasoning="All convergence providers failed"
        )

    def _call_groq(self, user_prompt: str) -> Optional[str]:
        self._init_groq()
        if not self._groq_client:
            return None
        try:
            response = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": CONVERGENCE_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Debate/Convergence] Groq call failed: {e}")
            return None

    def _call_gemini(self, user_prompt: str) -> Optional[str]:
        self._init_gemini()
        if not self._gemini_model:
            return None
        
        import time
        for attempt in range(3):
            try:
                full_prompt = f"{CONVERGENCE_PROMPT}\n\n{user_prompt}"
                response = self._gemini_model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg:
                    wait = (attempt + 1) * 2
                    print(f"[Debate/Convergence] Gemini 429 Limit Hit. Retrying in {wait}s...")
                    time.sleep(wait)
                    if attempt == 2:  # Last attempt failed -> Mark dead
                        self._gemini_model = None
                        print("[Debate/Convergence] Gemini quota exhausted. Switching to fallback.")
                        return None
                    continue
                
                # Non-429 error -> Bail immediately
                print(f"[Debate/Convergence] Gemini Judge failed: {e}")
                self._gemini_model = None
                return None
        return None

    def _build_transcript(self, history: list) -> str:
        """Build a detailed transcript for the convergence judge to read."""
        lines = []
        for entry in history:
            r = entry['round']
            p = entry['prosecution']
            d = entry['defense']
            lines.append(f"{'='*40}")
            lines.append(f"ROUND {r}")
            lines.append(f"{'='*40}")
            lines.append(f"PROSECUTION (confidence: {p.confidence:.0%}):")
            lines.append(f"  Evidence: {json.dumps(p.primary_evidence)}")
            lines.append(f"  Challenge to defense: {p.challenge_to_opponent}")
            lines.append(f"  Concessions: {p.concessions}")
            lines.append(f"  Summary: {p.reasoning_summary}")
            lines.append("")
            lines.append(f"DEFENSE (confidence: {d.confidence:.0%}):")
            lines.append(f"  Evidence: {json.dumps(d.primary_evidence)}")
            lines.append(f"  Challenge to prosecution: {d.challenge_to_opponent}")
            lines.append(f"  Concessions: {d.concessions}")
            lines.append(f"  Summary: {d.reasoning_summary}")
            lines.append("")
        return "\n".join(lines)
