"""
Debate System — Defense Agent
Argues REAL using OpenRouter Vision (different architecture for epistemic diversity).
"""

import json
from typing import Optional, List

from .models import (
    AgentResponse,
    DEFENSE_PROMPT,
    parse_agent_json,
    format_debate_history,
    encode_image_base64,
    get_mime_type,
)


class DefenseAgent:
    """
    Argues REAL. Uses OpenRouter (different architecture from Gemini prosecution).

    Round 1: Vision call (sees the actual image + case file)
    Rounds 2-3: Text-only rebuttals (argues from forensic evidence only)
    """

    def __init__(self, api_key: str, model: str = "nvidia/nemotron-nano-12b-v2-vl:free"):
        self._client = None
        self._api_key = api_key
        self._model_name = model

    def _init(self):
        if self._client or not self._api_key:
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as e:
            print(f"[Debate/Defense] OpenRouter init failed: {e}")

    def _call(self, user_prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        """Make an OpenRouter API call, optionally with an image."""
        self._init()
        if not self._client:
            return None

        messages = [{"role": "system", "content": DEFENSE_PROMPT}]
        user_content = []

        if image_path:
            img_b64 = encode_image_base64(image_path)
            mime = get_mime_type(image_path)
            if img_b64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                })

        user_content.append({"type": "text", "text": user_prompt})
        messages.append({"role": "user", "content": user_content})

        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                extra_headers={"HTTP-Referer": "https://deepfake-detection.local"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Debate/Defense] API call failed: {e}")
            return None

    def opening_statement(self, image_path: str, case_string: str) -> AgentResponse:
        """Round 1: Vision call — sees the actual image + case file."""
        prompt = f"""FORENSIC CASE FILE:
{case_string}

You are opening the defense. Examine the image and the forensic case file.
Present your strongest evidence that this image is REAL/AUTHENTIC.
Explain why any apparent anomalies in the forensic data could have innocent causes.
Be specific — cite actual layer scores and findings."""

        text = self._call(prompt, image_path=image_path)
        if text:
            return parse_agent_json(text, "REAL")
        return AgentResponse("REAL", 0.5, [], "", "",
                             "Defense unavailable (OpenRouter not configured)")

    def respond(self, opponent_arg: AgentResponse, debate_history: list,
                case_string: str) -> AgentResponse:
        """Rounds 2-3: Text-only rebuttal — no image, argues from evidence."""
        history_str = format_debate_history(debate_history)

        prompt = f"""FORENSIC CASE FILE:
{case_string}

PROSECUTION JUST ARGUED:
Position: {opponent_arg.position} (confidence: {opponent_arg.confidence:.0%})
Evidence: {json.dumps(opponent_arg.primary_evidence)}
Challenge to you: {opponent_arg.challenge_to_opponent}
Concessions they made: {opponent_arg.concessions}
Summary: {opponent_arg.reasoning_summary}

DEBATE HISTORY:
{history_str}

Respond to the prosecution's argument.
- Challenge their specific forensic claims with counter-evidence
- Provide scientifically grounded innocent explanations for cited anomalies
- Concede points they got right
- Update your confidence based on the full debate so far"""

        text = self._call(prompt, image_path=None)  # No image for rounds 2+
        if text:
            return parse_agent_json(text, "REAL")
        return AgentResponse("REAL", 0.5, [], "", "", "Defense response failed")
