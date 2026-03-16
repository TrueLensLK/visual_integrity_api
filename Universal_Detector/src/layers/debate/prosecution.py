"""
Debate System — Prosecution Agent
Argues AI-GENERATED using Gemini Vision (primary) with OpenRouter fallback.
Includes retry with backoff for 429 quota errors.
"""

import json
import time
from typing import Optional, List

from .models import (
    AgentResponse,
    PROSECUTION_PROMPT,
    parse_agent_json,
    format_debate_history,
    encode_image_base64,
    get_mime_type,
)


class ProsecutionAgent:
    """
    Argues AI-GENERATED.

    Primary:  Gemini Vision (with retry + backoff for 429 errors)
    Fallback: OpenRouter Vision (different model from defense)

    Round 1: Vision call (sees the actual image + case file)
    Rounds 2-3: Text-only rebuttals (argues from forensic evidence only)
    """

    MAX_RETRIES = 2
    RETRY_DELAYS = [10, 30]  # seconds to wait on 429

    def __init__(self, gemini_api_key: str, openrouter_api_key: str = ""):
        self._model = None
        self._gemini_key = gemini_api_key
        self._openrouter_key = openrouter_api_key
        self._openrouter_client = None
        self._gemini_dead = False  # Set True after repeated 429s to skip retries

    # ── Provider init ──────────────────────────────────────────────

    def _init_gemini(self):
        if self._model or not self._gemini_key or self._gemini_dead:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._gemini_key)
            self._model = genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            print(f"[Debate/Prosecution] Gemini init failed: {e}")

    def _init_openrouter(self):
        if self._openrouter_client or not self._openrouter_key:
            return
        try:
            from openai import OpenAI
            self._openrouter_client = OpenAI(
                api_key=self._openrouter_key,
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as e:
            print(f"[Debate/Prosecution] OpenRouter fallback init failed: {e}")

    # ── Gemini calls (with retry) ──────────────────────────────────

    def _gemini_vision_call(self, prompt: str, image_path: str) -> Optional[str]:
        """Gemini vision call with retry on 429."""
        self._init_gemini()
        if not self._model:
            return None
        try:
            import PIL.Image
            with PIL.Image.open(image_path) as img:
                img.load()
                for attempt in range(1 + self.MAX_RETRIES):
                    try:
                        response = self._model.generate_content([prompt, img])
                        return response.text
                    except Exception as e:
                        if "429" in str(e) and attempt < self.MAX_RETRIES:
                            wait = self.RETRY_DELAYS[attempt]
                            print(f"[Debate/Prosecution] 429 quota hit, retry in {wait}s "
                                  f"(attempt {attempt+1}/{self.MAX_RETRIES})")
                            time.sleep(wait)
                        elif "429" in str(e):
                            print(f"[Debate/Prosecution] Gemini quota exhausted after retries")
                            self._gemini_dead = True
                            return None
                        else:
                            raise
        except Exception as e:
            print(f"[Debate/Prosecution] Gemini vision failed: {e}")
        return None

    def _gemini_text_call(self, prompt: str) -> Optional[str]:
        """Gemini text call with retry on 429."""
        self._init_gemini()
        if not self._model:
            return None
        for attempt in range(1 + self.MAX_RETRIES):
            try:
                response = self._model.generate_content(prompt)
                return response.text
            except Exception as e:
                if "429" in str(e) and attempt < self.MAX_RETRIES:
                    wait = self.RETRY_DELAYS[attempt]
                    print(f"[Debate/Prosecution] 429 quota hit, retry in {wait}s "
                          f"(attempt {attempt+1}/{self.MAX_RETRIES})")
                    time.sleep(wait)
                elif "429" in str(e):
                    print(f"[Debate/Prosecution] Gemini quota exhausted after retries")
                    self._gemini_dead = True
                    return None
                else:
                    print(f"[Debate/Prosecution] Gemini text failed: {e}")
                    return None
        return None

    # ── OpenRouter fallback ────────────────────────────────────────

    def _openrouter_call(self, user_prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        """OpenRouter fallback — uses a DIFFERENT model from defense (Qwen2.5-VL)."""
        self._init_openrouter()
        if not self._openrouter_client:
            return None

        messages = [{"role": "system", "content": PROSECUTION_PROMPT}]
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
            # Use Qwen3-VL (different from defense's Nemotron) for epistemic diversity
            response = self._openrouter_client.chat.completions.create(
                model="qwen/qwen3-vl-30b-a3b-thinking",
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                extra_headers={"HTTP-Referer": "https://deepfake-detection.local"}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Debate/Prosecution] OpenRouter fallback failed: {e}")
            return None

    # ── Public interface ───────────────────────────────────────────

    def opening_statement(self, image_path: str, case_string: str) -> AgentResponse:
        """Round 1: Vision call — sees the actual image + case file."""
        prompt = f"""{PROSECUTION_PROMPT}

FORENSIC CASE FILE:
{case_string}

You are opening the prosecution. Examine the image and the forensic case file.
Present your strongest evidence that this image is AI GENERATED.
Be specific about which forensic layer scores and findings you rely on."""

        # Try Gemini first (with retry)
        text = self._gemini_vision_call(prompt, image_path)

        # Fallback to OpenRouter if Gemini failed
        if not text:
            print("[Debate/Prosecution] Falling back to OpenRouter (Qwen3-VL)")
            text = self._openrouter_call(
                f"FORENSIC CASE FILE:\n{case_string}\n\n"
                "You are opening the prosecution. Examine the image and the forensic case file.\n"
                "Present your strongest evidence that this image is AI GENERATED.\n"
                "Be specific about which forensic layer scores and findings you rely on.",
                image_path=image_path
            )

        if text:
            return parse_agent_json(text, "AI_GENERATED")
        return AgentResponse("AI_GENERATED", 0.5, [], "", "",
                             "Prosecution unavailable (all providers failed)")

    def respond(self, opponent_arg: AgentResponse, debate_history: list,
                case_string: str) -> AgentResponse:
        """Rounds 2-3: Text-only rebuttal — no image, argues from evidence."""
        history_str = format_debate_history(debate_history)

        rebuttal_context = f"""FORENSIC CASE FILE:
{case_string}

DEFENSE JUST ARGUED:
Position: {opponent_arg.position} (confidence: {opponent_arg.confidence:.0%})
Evidence: {json.dumps(opponent_arg.primary_evidence)}
Challenge to you: {opponent_arg.challenge_to_opponent}
Concessions they made: {opponent_arg.concessions}
Summary: {opponent_arg.reasoning_summary}

DEBATE HISTORY:
{history_str}

Respond to the defense's argument.
- Challenge their specific points with forensic data
- Present new evidence if available
- Concede points they got right
- Update your confidence based on the full debate so far"""

        full_prompt = f"{PROSECUTION_PROMPT}\n\n{rebuttal_context}"

        # Try Gemini first
        text = self._gemini_text_call(full_prompt)

        # Fallback to OpenRouter
        if not text:
            print("[Debate/Prosecution] Falling back to OpenRouter for rebuttal")
            text = self._openrouter_call(rebuttal_context)

        if text:
            return parse_agent_json(text, "AI_GENERATED")
        return AgentResponse("AI_GENERATED", 0.5, [], "", "",
                             "Prosecution rebuttal failed (all providers failed)")
