"""
Adversarial Debate System v1.0
Resolves forensic contradictions via structured multi-round LLM debate.

Architecture:
  Prosecution (Gemini Vision)  → argues AI-GENERATED
  Defense (OpenRouter Vision)  → argues REAL
  Convergence (Groq text)      → neutral transcript reader → verdict

Usage:
    from debate import DebateOrchestrator

    orchestrator = DebateOrchestrator(
        gemini_api_key="...",
        openrouter_api_key="...",
        groq_api_key="..."
    )
    result = orchestrator.run_debate(image_path, case_file)
"""

from .orchestrator import DebateOrchestrator
from .models import AgentResponse, ConvergenceResult, DebateVerdict

__all__ = [
    "DebateOrchestrator",
    "AgentResponse",
    "ConvergenceResult",
    "DebateVerdict",
]
