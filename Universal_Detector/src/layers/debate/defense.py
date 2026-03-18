"""
Debate System — Defense Agent
Argues REAL using OpenRouter Vision (different architecture for epistemic diversity).
"""

import json
from typing import Optional, List

from .models import (
    AgentResponse,
    DEFENSE_PROMPT,
    OPENROUTER_VISION_MODELS,  # New import
    parse_agent_json,
    format_debate_history,
    encode_image_base64,
    get_mime_type,
)


class DefenseAgent:
    """
    Argues REAL. Uses OpenRouter (Primary) or Gemini/Groq (Fallback).

    Round 1: Vision call (sees the actual image + case file)
    Rounds 2-3: Text-only rebuttals (argues from forensic evidence only)
    """

    def __init__(self, openrouter_api_key: str = "", gemini_api_key: str = "", groq_api_key: str = ""):
        self._openrouter_key = openrouter_api_key
        self._gemini_key = gemini_api_key
        self._groq_key = groq_api_key
        
        self._openrouter_client = None
        self._gemini_model = None
        self._groq_client = None
        
        # Use user-provided model or default fallback chain
        self._model_chain = OPENROUTER_VISION_MODELS

    def _init_openrouter(self):
        if self._openrouter_client or not self._openrouter_key:
            return
        try:
            from openai import OpenAI
            self._openrouter_client = OpenAI(
                api_key=self._openrouter_key.strip(),
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as e:
            print(f"[Debate/Defense] OpenRouter init failed: {e}")

    def _init_gemini(self):
        if self._gemini_model or not self._gemini_key:
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self._gemini_key)
            self._gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            print(f"[Debate/Defense] Gemini init failed: {e}")

    def call(self, case_file: str, image_path: Optional[str] = None, opponent_points: Optional[List[str]] = None) -> Optional[AgentResponse]:
        """Generate defense argument."""
        user_prompt = f"CASE FILE:\n{case_file}\n"
        if opponent_points:
            user_prompt += f"\nOPPONENT ARGUMENTS:\n{json.dumps(opponent_points)}"
            
        # Call the private method that handles the fallback chain
        raw_response = self._call(user_prompt, image_path)
        if not raw_response:
            return None

        return parse_agent_json(raw_response, default_verdict="REAL")

    def _call(self, user_prompt: str, image_path: Optional[str] = None) -> Optional[str]:
        """Make an API call with fallback chain (OpenRouter -> Gemini -> Groq)."""
        
        # 1. Try OpenRouter (Primary)
        self._init_openrouter()
        if self._openrouter_client:
            messages = [{"role": "system", "content": DEFENSE_PROMPT}]
            user_content = []
            if image_path:
                b64 = encode_image_base64(image_path)
                mime = get_mime_type(image_path)
                if b64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })
            user_content.append({"type": "text", "text": user_prompt})
            messages.append({"role": "user", "content": user_content})

            for model in self._model_chain:
                try:
                    try:
                        # Attempt with JSON mode first
                        response = self._openrouter_client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=0.7,
                            max_tokens=1024,
                            response_format={"type": "json_object"},
                            extra_headers={
                                "HTTP-Referer": "http://localhost:8000",
                                "X-Title": "DeepFake_Detector_Defense"
                            }
                        )
                    except Exception as json_err:
                        if "400" in str(json_err):
                            # Retry without JSON mode if model doesn't support it
                            response = self._openrouter_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=0.7,
                                max_tokens=1024,
                                extra_headers={
                                    "HTTP-Referer": "http://localhost:8000",
                                    "X-Title": "DeepFake_Detector_Defense"
                                }
                            )
                        else:
                            raise json_err

                except Exception:
                    continue
        
        # 2. Try Gemini (Fallback for Vision/Text)
        self._init_gemini()
        if self._gemini_model:
            try:
                full_prompt = f"{DEFENSE_PROMPT}\n\n{user_prompt}"
                if image_path:
                    import PIL.Image
                    with PIL.Image.open(image_path) as img:
                        response = self._gemini_model.generate_content([full_prompt, img])
                else:
                    response = self._gemini_model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                print(f"[Debate/Defense] Gemini fallback failed: {e}")

        # 3. Try Groq (Text-only fallback) - implies skipping image analysis if round 1
        if not image_path and self._groq_key: 
             # Implement Groq fallback here if needed, but Gemini/OpenRouter cover mostly everything
             pass

        print(f"[Debate/Defense] All models failed.")
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
