"""
Debate System — Orchestrator
Controls the adversarial debate flow: manages rounds, passes arguments between
agents, and produces the final verdict.

"""

import os
from typing import Dict, Any, Optional
from datetime import datetime

from .models import ConvergenceResult, DebateVerdict, AgentResponse
from .prosecution import ProsecutionAgent
from .defense import DefenseAgent
from .convergence import ConvergenceDetector

# Import case file formatter (handles both relative and absolute)
try:
    from forensic_case_builder import case_file_to_prompt_string
except ImportError:
    try:
        from ..forensic_case_builder import case_file_to_prompt_string
    except ImportError:
        from Universal_Detector.src.layers.forensic_case_builder import case_file_to_prompt_string


class DebateOrchestrator:
    """
    Controls the adversarial debate flow.

    Prosecution (Gemini) vs Defense (OpenRouter), judged by Convergence (Groq).
    Max 3 rounds. Image sent only in Round 1.
    """

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        cerebras_api_key: Optional[str] = None
    ):
        import os
        c_key = cerebras_api_key or os.environ.get("CEREBRAS_API_KEY")
        g_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        
        # New Architecture: Groq (Primary) -> Cerebras (Fallback)
        self.prosecution = ProsecutionAgent(groq_api_key=g_key, cerebras_api_key=c_key)
        self.defense = DefenseAgent(groq_api_key=g_key, cerebras_api_key=c_key)
        
        # Judge: Groq -> Cerebras (Fallback)
        self.convergence = ConvergenceDetector(groq_api_key=g_key, cerebras_api_key=c_key)
        
        self.max_rounds = 3

        # Track valid providers
        self._has_prosecution = bool(g_key or c_key)
        self._has_defense = bool(g_key or c_key)
        self._has_convergence = bool(g_key or c_key)

    def run_debate(
        self,
        image_path: str,
        case_file: Dict[str, Any],
        contradiction_context: str = ""
    ) -> DebateVerdict:
        """
        Run the adversarial debate and return a verdict.
        """
        start_time = datetime.now()
        
        # Validate image exists before spending any API calls
        import os
        if not os.path.exists(image_path):
            return DebateVerdict(
                verdict="EDITED", confidence=0.0,
                reasoning=f"Image not found: {image_path}",
                source="invalid_input"
            )

        if not (self._has_prosecution and self._has_defense and self._has_convergence):
            missing = []
            if not self._has_prosecution: missing.append("Gemini/OpenRouter (prosecution)")
            if not self._has_defense:     missing.append("OpenRouter/Gemini (defense)")
            if not self._has_convergence: missing.append("Groq/Gemini (convergence)")
            
            print(f"[Debate] Missing API keys for: {', '.join(missing)}")
            return DebateVerdict(
                verdict="EDITED", confidence=0.0,
                reasoning=f"Debate skipped. Missing API keys for: {', '.join(missing)}",
                source="miss_keys"
            )
        
        case_string = case_file_to_prompt_string(case_file)

        if contradiction_context:
            case_string = (
                f"!!! CONTRADICTION CONTEXT: {contradiction_context} !!!\n\n"
                f"{case_string}"
            )

        debate_history = []

        print(f"[Debate] Starting adversarial debate (max {self.max_rounds} rounds)")
        if contradiction_context:
            print(f"[Debate]    Trigger: {contradiction_context[:120]}...")

        # ── ROUND 1: Opening statements (VISION — both agents see the image) ──
        print("[Debate] Round 1: Opening statements (with image)")

        prosecution_opening = self.prosecution.opening_statement(image_path, case_string)
        defense_opening = self.defense.opening_statement(image_path, case_string)

        debate_history.append({
            'round': 1,
            'prosecution': prosecution_opening,
            'defense': defense_opening
        })

        print(f"[Debate]    Prosecution: {prosecution_opening.confidence:.0%} confident → AI")
        if prosecution_opening.visual_observations:
            print(f"[Debate]       Visual: {prosecution_opening.visual_observations[0][:60]}...")
        print(f"[Debate]    Defense:     {defense_opening.confidence:.0%} confident → REAL")
        if defense_opening.visual_observations:
            print(f"[Debate]       Visual: {defense_opening.visual_observations[0][:60]}...")

        # Abort if both agents failed (no real arguments produced)
        if (not prosecution_opening.primary_evidence and
                not defense_opening.primary_evidence and
                prosecution_opening.reasoning_summary.startswith(("Error:", "Prosecution unavailable")) and
                defense_opening.reasoning_summary.startswith(("Error:", "Defense unavailable"))):
            print("[Debate] Both agents failed — aborting debate")
            return DebateVerdict(
                verdict="EDITED", confidence=0.0,
                reasoning="Both debate agents failed to produce arguments",
                rounds_taken=0, source="debate_error"
            )

        # Abort if debate is one-sided (one agent completely failed)
        if self._is_one_sided(prosecution_opening, defense_opening):
            failed_side = "prosecution" if not prosecution_opening.primary_evidence else "defense"
            working_side = "defense" if failed_side == "prosecution" else "prosecution"
            print(f"[Debate] One-sided debate detected ({failed_side} failed) — aborting")
            print(f"[Debate]    Cannot trust {working_side}-only verdict, returning EDITED")
            return DebateVerdict(
                verdict="EDITED", confidence=0.3,
                reasoning=f"Debate aborted: {failed_side} agent failed to produce arguments. "
                          f"One-sided debate results are unreliable.",
                rounds_taken=1, source="debate_one_sided",
                debate_history=[{
                    'round': 1,
                    'prosecution': prosecution_opening,
                    'defense': defense_opening
                }]
            )

        # ── ROUNDS 2-3: Text-only rebuttals (no image — argue from evidence) ──
        last_defense = defense_opening

        for round_num in range(2, self.max_rounds + 1):
            print(f"[Debate] Round {round_num}: Rebuttals (text-only)")

            prosecution_response = self.prosecution.respond(
                last_defense, debate_history, case_string
            )
            defense_response = self.defense.respond(
                prosecution_response, debate_history, case_string
            )

            debate_history.append({
                'round': round_num,
                'prosecution': prosecution_response,
                'defense': defense_response
            })

            print(f"[Debate]    Prosecution: {prosecution_response.confidence:.0%} confident → AI")
            print(f"[Debate]    Defense:     {defense_response.confidence:.0%} confident → REAL")

            # Only allow early convergence on final rebuttal round (round 3)
            # or if one side completely collapsed (confidence < 0.3)
            one_side_collapsed = (
                prosecution_response.confidence < 0.30 or
                defense_response.confidence < 0.30
            )

            is_final_round = (round_num == self.max_rounds)

            if is_final_round or one_side_collapsed:
                convergence = self.convergence.check(debate_history, case_string)
                if one_side_collapsed:
                    print(f"[Debate] Early collapse detected — checking convergence")
                if convergence.has_converged:
                    print(f"[Debate] Converged after Round {round_num}: "
                          f"{convergence.verdict} ({convergence.confidence:.0%})")
                    return self._build_verdict(convergence, debate_history, start_time)
            else:
                print(f"[Debate] Round {round_num} complete — forcing Round {round_num + 1}")

            last_defense = defense_response

        # ── MAX ROUNDS REACHED — force final verdict ──
        print("[Debate] Max rounds reached — forcing final convergence verdict")
        # Problem 4 Fix: Use explicit "must decide" instruction
        final_convergence = self.convergence.check(
            debate_history, 
            case_string + "\n\nFINAL ROUND: You MUST return has_converged=true now."
        )

        # Fallback if convergence still refuses (rare)
        if not final_convergence.has_converged:
            # Convergence judge still undecided — use confidence differential
            last_p = debate_history[-1]['prosecution']
            last_d = debate_history[-1]['defense']

            if last_p.confidence > last_d.confidence + 0.1:
                final_convergence = ConvergenceResult(
                    has_converged=True,
                    verdict="AI-GENERATED",
                    confidence=last_p.confidence * 0.75,  # Dampened for forced verdict
                    reasoning="Max rounds reached. Prosecution maintained stronger confidence.",
                    winning_side="prosecution",
                    key_turning_point="Confidence differential at final round"
                )
            elif last_d.confidence > last_p.confidence + 0.1:
                final_convergence = ConvergenceResult(
                    has_converged=True,
                    verdict="REAL",
                    confidence=last_d.confidence * 0.75,
                    reasoning="Max rounds reached. Defense maintained stronger confidence.",
                    winning_side="defense",
                    key_turning_point="Confidence differential at final round"
                )
            else:
                final_convergence = ConvergenceResult(
                    has_converged=True,
                    verdict="EDITED",
                    confidence=0.5,
                    reasoning="Max rounds reached. Neither side established clear dominance.",
                    winning_side="neither",
                    key_turning_point="Stalemate after all rounds"
                )

        return self._build_verdict(final_convergence, debate_history, start_time)

    @staticmethod
    def _is_one_sided(prosecution: 'AgentResponse', defense: 'AgentResponse') -> bool:
        """Detect if one agent completely failed (produced no real arguments)."""
        p_failed = (
            not prosecution.primary_evidence and
            (prosecution.reasoning_summary.startswith(("Error:", "Prosecution unavailable", "Prosecution rebuttal"))
             or "all providers failed" in prosecution.reasoning_summary.lower())
        )
        d_failed = (
            not defense.primary_evidence and
            (defense.reasoning_summary.startswith(("Error:", "Defense unavailable", "Defense response"))
             or "all providers failed" in defense.reasoning_summary.lower())
        )
        # One-sided = exactly one failed (both-failed is handled separately)
        return (p_failed and not d_failed) or (d_failed and not p_failed)

    def _build_verdict(self, convergence: ConvergenceResult,
                       history: list, start_time: datetime) -> DebateVerdict:
        """Build the final DebateVerdict from convergence result and debate history."""
        processing_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        last_p = history[-1]['prosecution']
        last_d = history[-1]['defense']

        # Normalize verdict format (AI_GENERATED → AI-GENERATED)
        verdict = convergence.verdict or "EDITED"
        if verdict == "AI_GENERATED":
            verdict = "AI-GENERATED"

        # Build serializable history (AgentResponse → dict)
        serializable_history = []
        for entry in history:
            serializable_history.append({
                'round': entry['round'],
                'prosecution': {
                    'confidence': entry['prosecution'].confidence,
                    'evidence': entry['prosecution'].primary_evidence,
                    'visual_observations': entry['prosecution'].visual_observations,
                    'challenge': entry['prosecution'].challenge_to_opponent,
                    'concessions': entry['prosecution'].concessions,
                    'summary': entry['prosecution'].reasoning_summary
                },
                'defense': {
                    'confidence': entry['defense'].confidence,
                    'evidence': entry['defense'].primary_evidence,
                    'visual_observations': entry['defense'].visual_observations,
                    'challenge': entry['defense'].challenge_to_opponent,
                    'concessions': entry['defense'].concessions,
                    'summary': entry['defense'].reasoning_summary
                }
            })

        return DebateVerdict(
            verdict=verdict,
            confidence=convergence.confidence,
            reasoning=convergence.reasoning,
            method="adversarial_debate",
            rounds_taken=len(history),
            debate_summary={
                "prosecution_final_confidence": last_p.confidence,
                "defense_final_confidence": last_d.confidence,
                "winning_side": convergence.winning_side or "unknown",
                "key_turning_point": convergence.key_turning_point,
                "processing_time_ms": processing_ms
            },
            debate_history=serializable_history,
            source="debate"
        )
