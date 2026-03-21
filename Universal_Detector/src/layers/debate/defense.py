"""
Debate System — Defense Agent
Argues REAL using Groq (Primary) with Cerebras Fallback.
"""

import json
from typing import Optional, List, Dict, Any

from .models import (
    AgentResponse,
    DEFENSE_PROMPT,
    parse_agent_json,
    format_debate_history,
    GROQ_MODEL,
    CEREBRAS_MODEL,
    encode_image_base64,
    get_mime_type,
)


class DefenseAgent:
    """
    Argues REAL.

    Primary:  Groq (meta-llama/llama-4-scout-17b-16e-instruct)
    Fallback: Cerebras (llama-3.3-70b)

    Round 1: Vision call (sees actual image + formatted case file)
    Rounds 2-3: Text-only rebuttals
    """

    def __init__(self, groq_api_key: str, cerebras_api_key: str):
        self._groq_key = groq_api_key
        self._cerebras_key = cerebras_api_key
        
        self._groq_client = None
        self._cerebras_client = None

    def _init_groq(self):
        if self._groq_client or not self._groq_key: return
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self._groq_key)
        except Exception as e:
            print(f"[Defense] Groq init failed: {e}")

    def _init_cerebras(self):
        if self._cerebras_client or not self._cerebras_key: return
        try:
            from openai import OpenAI
            self._cerebras_client = OpenAI(
                api_key=self._cerebras_key,
                base_url="https://api.cerebras.ai/v1"
            )
        except Exception as e:
            print(f"[Defense] Cerebras init failed: {e}")

    def _call_groq(self, system: str, user: str, image_path: Optional[str]) -> Optional[str]:
        self._init_groq()
        if not self._groq_client: return None

        messages = [{"role": "system", "content": system}]
        content = []
        if image_path:
            b64 = encode_image_base64(image_path)
            if b64:
                mime = get_mime_type(image_path)
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": user})
        messages.append({"role": "user", "content": content})

        try:
            resp = self._groq_client.chat.completions.create(
                messages=messages,
                model=GROQ_MODEL,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Defense] Groq error: {e}")
            return None

    def _call_cerebras(self, system: str, user: str, image_path: Optional[str]) -> Optional[str]:
        self._init_cerebras()
        if not self._cerebras_client: return None

        final_user = user
        if image_path:
            final_user = f"[NOTE: Image analysis skipped in fallback]\n\n{user}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": final_user}
        ]
        
        try:
            resp = self._cerebras_client.chat.completions.create(
                messages=messages,
                model=CEREBRAS_MODEL,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Defense] Cerebras error: {e}")
            return None

    def opening_statement(self, image_path: str, case_string: str) -> AgentResponse:
        prompt = f"FORENSIC CASE FILE:\n{case_string}\n\nArgue REAL/AUTHENTIC based on evidence."
        
        text = self._call_groq(DEFENSE_PROMPT, prompt, image_path)
        if text: return parse_agent_json(text, "REAL")
        
        print("[Defense] Falling back to Cerebras...")
        text = self._call_cerebras(DEFENSE_PROMPT, prompt, image_path)
        if text: return parse_agent_json(text, "REAL")

        return AgentResponse("REAL", 0.0, [], "", "", "All providers failed")

    def respond(self, opponent_arg: AgentResponse, debate_history: list, case_string: str) -> AgentResponse:
        history_str = format_debate_history(debate_history)
        prompt = f"CASE FILE:\n{case_string}\n\nOPPONENT:\n{opponent_arg}\n\nHISTORY:\n{history_str}\n\nRefute based on evidence."
        
        text = self._call_groq(DEFENSE_PROMPT, prompt, None)
        if text: return parse_agent_json(text, "REAL")
        
        print("[Defense] Falling back to Cerebras...")
        text = self._call_cerebras(DEFENSE_PROMPT, prompt, None)
        if text: return parse_agent_json(text, "REAL")
        
        return AgentResponse("REAL", 0.0, [], "", "", "All providers failed")
