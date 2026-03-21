"""
LLM Judge - The "Final Boss" for DeepFake Detection
Uses multiple LLM providers (Gemini, Groq, OpenRouter) with automatic fallback.
Aligned with the Full-Spectrum Case Builder to review ALL forensic layers.
"""

import os
import json
import re
import base64
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Try importing the builder, handle both relative and absolute imports
try:
    from forensic_case_builder import case_file_to_prompt_string
except ImportError:
    from .forensic_case_builder import case_file_to_prompt_string

@dataclass
class LLMVerdict:
    """Structured verdict from the LLM Judge."""
    verdict: str  # "REAL", "AI-GENERATED", "AI-ENHANCED", or "EDITED"
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Detailed reasoning
    key_evidence: list  # List of key evidence points
    contradictions_resolved: list  # How contradictions were resolved
    source: str  # "gemini", "groq", or "fallback"
    processing_time_ms: int
    raw_response: Optional[str] = None

# --- THE "HIERARCHY OF TRUTH" SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are the 'Chief Forensic Arbitrator' for a DeepFake detection system.
You have access to:
1. THE ACTUAL IMAGE (Visual Evidence) - LOOK AT THIS FIRST.
2. A 'Case File' (Forensic Telemetry) - Use this to support your visual findings.

YOUR CORE TASK: Distinguish Valid Forensic Signals from False Positives (Compression/Editing).

=== VISUAL INSPECTION PROTOCOL (Step 1) ===
Look at the image carefully.
- Natural Details: Do strands of hair, skin texture, and fine patterns look organic? (Indicates REAL)
- AI Artifacts: Do you see melted fingers, nonsensical text, asymmetrical eyes, or smooth "plastic" skin? (Indicates AI)
- Compression: Do you see significant jpeg blocking, noise, or banding? These can trigger FALSE POSITIVE forensic alarms (high PRNU/Error Level Analysis).

=== THE HIERARCHY OF EVIDENCE ===
1. VISUAL ANOMALIES & PHYSICS (Primary Inspection):
   - Look for FAKE artifacts FIRST: Melted hands, asymmetrical eyes, weird text, or "plastic" skin = AI GENERATED.
   - If the image is FLAWLESS but forensic tools scream FAKE (PRNU < -40), trust the forensics unless you see clear JPEG blocking.
   - Only trust "Photorealism" if forensic scores are inconclusive (-20 to +20).

2. THE "SOCIAL MEDIA" TRAP (Crucial for False Positives):
   - Images from Google/Facebook/Insta are RESIZED and STRIPPED of metadata.
   - Resizing creates GRID ARTIFACTS that fool PRNU detectors -> FALSE POSITIVE AI ALERT.
   - If the image looks compressed (jpeg blocks) or is low resolution: DISCARD HIGH PRNU SCORES.
   - If C2PA/Metadata is missing + Image is Compressed -> Assume the "Grid" is formatting, not AI.

3. THE HARDWARE TRUTH (Secondary Verification):
   - Presence of Bayer Pattern/CFA artifacts SUGGESTS a real sensor, BUT verify against Visuals. High-end AI can mimic this.
   - Ignore high PRNU/Noise scores ONLY if you visually confirm heavy JPEG compression blocking OR it's a web-sourced image.

4. NEURAL CONSENSUS:
   - Neural Networks are fallible on new models (Flux, MJv6).
   - If Neural = REAL but Forensics = STRONG FAKE (-50), verify it's NOT a compression artifact first.

=== VERDICT LOGIC ===
- AI-GENERATED: Visual artifacts found OR strong forensic evidence (PRNU grid, Spectrum anomalies) without compression cause.
- LIKELY_AI_GENERATED: Strong suspicion of AI origin, but critical evidence is degraded or inconclusive (Score 25-49 range).
- LIKELY_REAL: Natural appearance and valid physics, but some forensic signals are weak or missing (Score 50-75 range).
- REAL: Natural details (pores, hair) + Physics (ISO noise, Bayer) + No AI artifacts.
- EDITED: Real image with some manipulation (color, cropping) leading to mixed signals.

=== OUTPUT FORMAT ===
You MUST respond with ONLY a valid JSON object.
{
    "verdict": "REAL", "LIKELY_REAL", "LIKELY_AI_GENERATED", "AI-GENERATED", or "EDITED",
    "confidence": 0.0 to 1.0,
    "reasoning": "Explain your visual analysis findings first, then how they align/conflict with the case file.",
    "key_evidence": ["Visual: ...", "Forensic: ..."],
    "contradictions_resolved": ["Resolved PRNU alert as compression artifact due to..."]
}
"""

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode an image file to base64 string for LLM vision APIs."""
    try:
        path = Path(image_path)
        if not path.exists():
            return None
        with open(path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        print(f"[LLM Judge] Image encoding failed: {e}")
        return None

def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from image path."""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    return mime_types.get(ext, "image/jpeg")

class ForensicAgent:
    """
    Multi-provider LLM forensic judge with automatic fallback.
    Supports: OpenRouter (Vision), Gemini (Vision), Groq (Text)
    """
    def __init__(
        self, 
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        preferred_provider: str = "gemini",  # Default to Gemini for Vision
        model_name: Optional[str] = None
    ):
        # Load Gemini API key from environment variables or .env file only
        self.gemini_api_key = (
            gemini_api_key
            or os.environ.get("GOOGLE_AI_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.openrouter_api_key = (
            openrouter_api_key
            or os.environ.get("OPENROUTER_API_KEY")
        )
        self.preferred_provider = os.environ.get("LLM_PREFERRED_PROVIDER", preferred_provider)
        self.model_name = model_name
        
        self._gemini_model = None
        self._groq_client = None
        self._openrouter_client = None

    def _init_gemini(self):
        if self._gemini_model or not self.gemini_api_key:
            if not self.gemini_api_key:
                print("[LLM Judge] Gemini API key missing or empty!")
            else:
                print(f"[LLM Judge] Gemini API key present (starts with: {self.gemini_api_key[:6]}...)")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            # Use Gemini 2.0 Flash as requested
            model_name = "gemini-2.0-flash"
            print(f"[LLM Judge] Initializing Gemini model: {model_name}")
            self._gemini_model = genai.GenerativeModel(model_name)
        except Exception as e:
            print(f"[LLM Judge] Gemini init failed: {e}")
            if any(x in str(e).lower() for x in ["quota", "token", "key", "auth", "expired", "limit"]):
                print("[LLM Judge] Gemini API key/token may be invalid, expired, or quota exceeded!")

    def _init_groq(self):
        if self._groq_client or not self.groq_api_key:
            if not self.groq_api_key:
                print("[LLM Judge] Groq API key missing or empty!")
            return
        try:
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_api_key)
        except Exception:
            try:
                from openai import OpenAI
                self._groq_client = OpenAI(api_key=self.groq_api_key, base_url="https://api.groq.com/openai/v1")
            except Exception as e:
                print(f"[LLM Judge] Groq init failed: {e}")
                if any(x in str(e).lower() for x in ["quota", "token", "key", "auth", "expired", "limit"]):
                    print("[LLM Judge] Groq API key/token may be invalid, expired, or quota exceeded!")

    def _init_openrouter(self):
        if self._openrouter_client or not self.openrouter_api_key:
            if not self.openrouter_api_key:
                print("[LLM Judge] OpenRouter API key missing or empty!")
            return
        try:
            from openai import OpenAI
            self._openrouter_client = OpenAI(
                api_key=self.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
        except Exception as e:
            print(f"[LLM Judge] OpenRouter init failed: {e}")
            if any(x in str(e).lower() for x in ["quota", "token", "key", "auth", "expired", "limit"]):
                print("[LLM Judge] OpenRouter API key/token may be invalid, expired, or quota exceeded!")

    def _call_openrouter(self, prompt: str, image_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        self._init_openrouter()
        if not self._openrouter_client: return None, "OpenRouter not configured"
        try:
            # Use a currently supported free vision model on OpenRouter
            # Swap to a currently supported free vision model (Gemma 3 or Qwen 2.5)
            model_name = self.model_name or "nvidia/nemotron-nano-12b-v2-vl:free"
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            user_content = []
            if image_path:
                image_base64 = encode_image_to_base64(image_path)
                mime_type = get_image_mime_type(image_path)
                if image_base64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}
                    })
            user_content.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": user_content})

            response = self._openrouter_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=2048,
                extra_headers={"HTTP-Referer": "https://deepfake-detection.local"}
            )
            return response.choices[0].message.content, None
        except Exception as e:
            return None, f"OpenRouter error: {str(e)}"

    def _call_gemini(self, prompt: str, image_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        self._init_gemini()
        if not self._gemini_model: return None, "Gemini not configured"
        
        import time
        for attempt in range(3):  # Retry up to 3 times
            try:
                full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
                if image_path:
                    import PIL.Image
                    with PIL.Image.open(image_path) as img:
                        img.load() 
                        # Use list wrapping correctly for GenerativeModel
                        response = self._gemini_model.generate_content([full_prompt, img])
                else:
                    response = self._gemini_model.generate_content(full_prompt)
                
                return response.text, None
            
            except Exception as e:
                err_msg = str(e).lower()
                if "429" in err_msg or "quota" in err_msg:
                    wait = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s...
                    print(f"[LLM Judge] Gemini 429 Limit Hit. Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return None, f"Gemini error: {str(e)}"
        
        return None, "Gemini quota exhausted after retries"

    def _call_groq(self, prompt: str, image_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        self._init_groq()
        if not self._groq_client: return None, "Groq not configured"
        try:
            if image_path:
                prompt += "\n\n[NOTE: Image analysis skipped as Groq is text-only. Rely on the Case File data.]"
            model_name = self.model_name or "llama-3.3-70b-versatile"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            response = self._groq_client.chat.completions.create(
                model=model_name, messages=messages, temperature=0.1, max_tokens=2048
            )
            return response.choices[0].message.content, None
        except Exception as e:
            return None, f"Groq error: {str(e)}"

    def make_final_call(self, case_file: Dict[str, Any], image_path: Optional[str] = None) -> LLMVerdict:
        start_time = datetime.now()
        
        # 1. Prepare Prompt
        case_string = case_file_to_prompt_string(case_file)
        special_instruction = case_file.get("special_instruction", "")
        
        # Build the Prompt - Emphasize Visuals
        visual_reminder = "!!! PRIORITY INSTRUCTION: ANALYZE THE IMAGE SCENE FIRST. DO NOT BLINDLY TRUST THE SCORES. !!!"
        
        if special_instruction:
            full_prompt = f"{visual_reminder}\n\n!!! SPECIAL INSTRUCTION: {special_instruction} !!!\n\n=== CASE FILE DATA ===\n{case_string}\n\n=== END CASE FILE ==="
        else:
            full_prompt = f"{visual_reminder}\n\n=== CASE FILE DATA ===\n{case_string}\n\n=== END CASE FILE ==="

        # 2. Select Provider Order
        # STRICT REQUIREMENT: User requested ONLY Gemini for Judge.
        providers = []
        if self.gemini_api_key:
            if image_path:
                providers.append(("Gemini (Vision)", lambda: self._call_gemini(full_prompt, image_path)))
            else:
                providers.append(("Gemini", lambda: self._call_gemini(full_prompt)))
        
        # If no Gemini, we fail (as requested to remove others)
        if not providers:
            print("[LLM Judge] CRITICAL: Gemini API Key missing, and other providers disabled by configuration.")

        # 3. Execution Loop
        last_error = "No providers available"
        for name, call_fn in providers:
            response_text, error = call_fn()
            if response_text:
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                verdict_data = self._parse_llm_response(response_text)
                return LLMVerdict(
                    verdict=verdict_data.get("verdict", "EDITED"),
                    confidence=float(verdict_data.get("confidence", 0.5)),
                    reasoning=verdict_data.get("reasoning", "No reasoning provided"),
                    key_evidence=verdict_data.get("key_evidence", []),
                    contradictions_resolved=verdict_data.get("contradictions_resolved", []),
                    source=name,
                    processing_time_ms=processing_time,
                    raw_response=response_text
                )
            else:
                last_error = error
                print(f"[LLM Judge] {name} failed: {error}")
                if error and any(x in error.lower() for x in ["quota", "token", "key", "auth", "expired", "limit"]):
                    print(f"[LLM Judge] {name} API key/token may be invalid, expired, or quota exceeded!")

        return LLMVerdict(
            verdict="EDITED",
            confidence=0.0,
            reasoning=f"All LLM providers failed. Last error: {last_error}",
            key_evidence=[],
            contradictions_resolved=[],
            source="llm_error",
            processing_time_ms=0
        )

    def _parse_llm_response(self, text: str) -> Dict[str, Any]:
        try:
            clean_text = re.sub(r'```json\s*', '', text)
            clean_text = re.sub(r'```\s*', '', clean_text).strip()
            match = re.search(r'\{[\s\S]*\}', clean_text)
            if match:
                return json.loads(match.group())
            return json.loads(clean_text)
        except Exception:
            verdict = "EDITED"
            if "REAL" in text.upper(): verdict = "REAL"
            elif "AI" in text.upper(): verdict = "AI-GENERATED"
            return {"verdict": verdict, "confidence": 0.5, "reasoning": "JSON Parse Failed. Raw text extraction."}

class HybridJudge:
    """
    Orchestrates Rule-Based, LLM-Based, and Adversarial Debate judging.
    
    Decision flow:
      1. Contradiction detected? → Adversarial Debate (Prosecution vs Defense vs Convergence)
      2. Gray zone / ambiguous?  → Single LLM call (existing behavior)
      3. Clear verdict?          → Rule-based only (no LLM cost)
    """
    def __init__(
        self, 
        gemini_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        enable_llm: bool = True
    ):
        self.enable_llm = enable_llm
        self.agent = None
        self.debate = None
        if enable_llm:
            self.agent = ForensicAgent(
                gemini_api_key=gemini_api_key,
                groq_api_key=groq_api_key,
                openrouter_api_key=openrouter_api_key
            )
            # Initialize Adversarial Debate (uses resolved API keys from ForensicAgent)
            try:
                try:
                    from debate import DebateOrchestrator
                    from debate.visual_expert_agent import VisualExpertAgent
                except ImportError:
                    from .debate import DebateOrchestrator
                    from .debate.visual_expert_agent import VisualExpertAgent
                
                # Initialize Debate Orchestrator
                self.debate = DebateOrchestrator(
                    # No longer passing gemini key for debate as per user request
                    groq_api_key=self.agent.groq_api_key
                )
                
                # Initialize Visual Expert (uses same keys as debate)
                self.visual_expert = VisualExpertAgent(
                    groq_api_key=self.agent.groq_api_key,
                    cerebras_api_key=os.environ.get("CEREBRAS_API_KEY"),
                    gemini_api_key=self.agent.gemini_api_key
                )
                
                print("[HybridJudge] Adversarial Debate system enabled (Groq+Cerebras)")
            except Exception as e:
                print(f"[HybridJudge] Debate/Expert system disabled: {e}")
                self.debate = None
                self.visual_expert = None

    def _should_debate(self, case_file: Dict) -> Tuple[bool, str]:
        """
        Check if forensic contradictions warrant an adversarial debate.
        Uses contradictions already detected by forensic_case_builder.identify_contradictions().
        """
        # 1. Use pre-computed contradictions from the case file
        contradictions = case_file.get("contradictions", [])
        if contradictions:
            context = "; ".join(
                c.get("note", c.get("type", "Unknown contradiction"))
                for c in contradictions
            )
            return True, context

        # 2. Neural civil war (tight model split, 3v2 or 2v3)
        neural = case_file.get("neural_consensus", {})
        real_v = neural.get("real_votes", 0)
        ai_v = neural.get("ai_votes", 0)
        total = real_v + ai_v
        if total >= 3 and abs(real_v - ai_v) <= 1:
            return True, f"Neural civil war: {real_v} Real vs {ai_v} AI votes"

        return False, ""

    def _should_consult_llm(self, case_file: Dict, rule_verdict: str, rule_score: int) -> Tuple[bool, Optional[str]]:
        """
        Smart Gatekeeper: Triggers LLM if Neural Networks & Forensics disagree.
        """
        scores = case_file.get("layer_scores", {})
        
        # 1. Get Key Signals
        neural_score = scores.get("neural_network", 0)     # e.g., +6.4
        prnu_score = scores.get("prnu", 0)                 # e.g., -50.0
        spectrum_score = scores.get("spectrum", 0)         # e.g., +20.0 (or negative)
        
        # 2. "The Tiger Problem" (Forensics say Fake, Neurals say Real)
        # YOUR REQUEST: If neural is POSITIVE (> 0), it contradicts a negative forensic score.
        neural_says_real = neural_score > 0  
        forensics_say_fake = (prnu_score < -20) or (spectrum_score < -20) # Lowered to -20 to catch more conflicts

        if neural_says_real and forensics_say_fake:
            instruction = (
                f"CONFLICT: Forensic tools detected traces (Score: {min(prnu_score, spectrum_score)}), "
                f"but Neural Networks see a REAL image (+{neural_score}). "
                "Check: Is this a high-quality Deepfake that fooled the eyes, or a compressed Real photo that fooled the math?"
            )
            return True, instruction

        # 3. "The Reverse Tiger" (Forensics say Real, Neurals say Fake)
        neural_says_fake = neural_score < 0
        forensics_say_real = (prnu_score > 20) or (spectrum_score > 20)

        if neural_says_fake and forensics_say_real:
            return True, "CONFLICT: Camera sensor noise matches a Real camera, but Neural Networks see visual artifacts. Look for subtle warping."

        # 4. Neural Civil War (Models fighting each other)
        neural_data = case_file.get("neural_analysis", {})
        real_votes = neural_data.get("real_votes", 0)
        ai_votes = neural_data.get("ai_votes", 0)
        
        # If models are split (e.g. 3 vs 2, or 3 vs 1), call the judge
        if (real_votes + ai_votes) > 0 and abs(real_votes - ai_votes) <= 2:
            return True, f"Neural models are split ({real_votes} Real vs {ai_votes} AI). You are the tie-breaker."

        # 5. The "Gray Zone" (Score is Middle of the road)
        if 35 <= rule_score <= 65:
            return True, "The rule-based score is ambiguous (35-65). Review closely."

        return False, None

    def judge(
        self,
        case_file: Dict[str, Any],
        rule_based_verdict: str,
        rule_based_score: int,
        rule_based_description: str,
        image_path: Optional[str] = None
    ) -> Tuple[str, int, str, Optional[LLMVerdict]]:
        
        # ── Phase 0: Visual Expert Check (UNCERTAIN Web Images) ──
        # Fix 15: If Rule-Based says UNCERTAIN on a Web Image, route to Visual Expert
        is_web_sourced = False
        # Check if evidence mentions "web-sourced" (from Fix 5 disable message)
        all_evidence = case_file.get("all_evidence", [])
        for item in all_evidence:
            if "web-sourced" in str(item.get("detail", "")).lower():
                is_web_sourced = True
                break
        
        if is_web_sourced and rule_based_verdict == "UNCERTAIN" and self.visual_expert and image_path:
            print(f"[HybridJudge] UNCERTAIN web image detected → Routing to Visual Expert")
            
            case_string = case_file_to_prompt_string(case_file)
            expert_result = self.visual_expert.analyze(image_path, case_string)
            
            verdict = expert_result.verdict
            final_score = rule_based_score
            description = expert_result.reasoning
            
            # Apply Score Caps (Fix 17)
            if verdict == "AI-GENERATED" or verdict == "LIKELY_AI_GENERATED":
                # Max confidence on web image is capped -> Score 25 (Low end of 'Likely AI' range)
                final_score = 25
                verdict = "LIKELY_AI_GENERATED" # Enforce new user-requested label
                description = f"[Visual Expert] {description} (Score capped at 25 due to missing hardware forensics)"
                
            elif verdict == "LIKELY_REAL":
                # Max confidence on web image is capped -> Score 72 (High end of 'Likely Real' range)
                final_score = 72
                description = f"[Visual Expert] {description} (Score capped at 72 due to missing hardware forensics)"
                
            else: # UNCERTAIN
                final_score = 50
                verdict = "UNCERTAIN"
                description = f"[Visual Expert] {description} (Insufficient evidence for verdict)"

            # Construct LLMVerdict
            llm_verdict = LLMVerdict(
                verdict=verdict,
                confidence=expert_result.confidence,
                reasoning=expert_result.reasoning,
                key_evidence=expert_result.definitive_artifacts_found + expert_result.definitive_real_indicators,
                contradictions_resolved=[],
                source="visual_expert",
                processing_time_ms=0,
                raw_response=expert_result.raw_text
            )
            
            return verdict, final_score, description, llm_verdict

        # ── Phase 1: Adversarial Debate for genuine contradictions ──
        if self.debate and image_path:
            should_debate, debate_context = self._should_debate(case_file)
            if should_debate:
                try:
                    print(f"[HybridJudge] Contradiction detected → launching adversarial debate")
                    debate_result = self.debate.run_debate(
                        image_path=image_path,
                        case_file=case_file,
                        contradiction_context=debate_context
                    )
                    if debate_result.verdict and debate_result.verdict != "EDITED":
                        # ── Guardrail: Debate vs Rule-Based conflict ──
                        # If rule-based judge identified a specific false positive
                        # (e.g., Bayer contradiction) and reached a non-AI verdict,
                        # don't let the debate override based on dismissed evidence.
                        rb_non_ai = rule_based_verdict in (
                            "REAL", "LIKELY_REAL", "EDITED_REAL", "EDITED"
                        )
                        debate_says_ai = debate_result.verdict == "AI-GENERATED"
                        
                        if debate_says_ai and rb_non_ai:
                            # Check if rule-based had strong reasoning (Bayer contradiction, etc.)
                            rb_has_correction = any(kw in rule_based_description.lower() for kw in [
                                "bayer", "contradiction", "texture", "sensor dna",
                                "compressed", "model consensus", "false positive"
                            ])
                            if rb_has_correction and debate_result.confidence < 0.85:
                                print(f"[HybridJudge] DEBATE OVERRIDE BLOCKED")
                                print(f"    Debate: {debate_result.verdict} @ {debate_result.confidence:.0%}")
                                print(f"    Rule-based: {rule_based_verdict} ({rule_based_score}/100)")
                                print(f"    Reason: Rule-based identified false positive; debate confidence too low")
                                print(f"    → Keeping rule-based verdict")
                                return rule_based_verdict, rule_based_score, rule_based_description, None

                        # Debate reached a definitive verdict
                        if debate_result.verdict == "REAL":
                            final_score = 50 + int(debate_result.confidence * 50)
                        elif debate_result.verdict == "AI-GENERATED":
                            final_score = 50 - int(debate_result.confidence * 50)
                        else:
                            final_score = 50
                        final_description = f"[Debate] {debate_result.reasoning}"
                        return debate_result.verdict, final_score, final_description, debate_result
                    else:
                        print("[HybridJudge] Debate inconclusive → falling back to single LLM")
                except Exception as e:
                    print(f"[HybridJudge] Debate failed ({e}) → falling back to single LLM")

        # ── Phase 2: Single LLM call for gray zone / ambiguous cases ──
        should_run, instruction = self._should_consult_llm(case_file, rule_based_verdict, rule_based_score)
        
        # If no LLM needed, return Rule-Based
        if not self.enable_llm or not should_run:
            return rule_based_verdict, rule_based_score, rule_based_description, None

        # Add instructions and run single LLM
        if instruction:
            case_file["special_instruction"] = instruction
            
        llm_result = self.agent.make_final_call(case_file, image_path)
        
        # ── Safety Net: If ALL LLM providers failed, fall back to rule-based ──
        if llm_result.source == "llm_error":
            print(f"[HybridJudge] All LLM providers failed → falling back to rule-based verdict")
            
            # Fix 18: Fallback to SDXL on LLM Error (Uncertain fallback)
            sdxl_data = case_file.get("neural_consensus", {}).get("model_breakdown", {}).get("sdxl")
            if sdxl_data:
                sdxl_score = sdxl_data.get("score", 0)
                if sdxl_score > 0:
                   print(f"[HybridJudge] LIKELY_REAL override via SDXL (score={sdxl_score:+.1f})")
                   return "LIKELY_REAL", 60, f"[LLM FAIL] Fallback to SDXL-Detector (REAL)", None
                else:
                   print(f"[HybridJudge] LIKELY_AI_GENERATED override via SDXL (score={sdxl_score:+.1f})")
                   return "LIKELY_AI_GENERATED", 35, f"[LLM FAIL] Fallback to SDXL-Detector (AI)", None
            
            return rule_based_verdict, rule_based_score, f"[LLM Unavailable] {rule_based_description}", None
        
        # ── Guardrail: LLM vs Rule-Based conflict ──
        # If rule-based judge corrected a false positive (e.g., Bayer contradiction,
        # compression artifacts) and the LLM overrides to AI-GENERATED based on
        # that same dismissed evidence, block the override.
        final_verdict = llm_result.verdict
        rb_non_ai = rule_based_verdict in (
            "REAL", "LIKELY_REAL", "EDITED_REAL", "EDITED"
        )
        llm_says_ai = final_verdict == "AI-GENERATED"
        
        if llm_says_ai and rb_non_ai:
            rb_has_correction = any(kw in rule_based_description.lower() for kw in [
                "bayer", "contradiction", "texture", "sensor dna",
                "compressed", "model consensus", "false positive",
                "downgrad", "dismissed"
            ])
            if rb_has_correction:
                print(f"[HybridJudge] LLM OVERRIDE BLOCKED")
                print(f"    LLM: {final_verdict} @ {llm_result.confidence:.0%}")
                print(f"    Rule-based: {rule_based_verdict} ({rule_based_score}/100)")
                print(f"    Reason: Rule-based identified false positive; LLM used dismissed evidence")
                print(f"    → Keeping rule-based verdict")
                return rule_based_verdict, rule_based_score, rule_based_description, None

        # Calculate Final Score based on LLM Confidence
        final_description = f"[LLM] {llm_result.reasoning}"
        
        if final_verdict == "REAL":
            final_score = 50 + int(llm_result.confidence * 50)
        elif final_verdict == "AI-GENERATED":
            final_score = 50 - int(llm_result.confidence * 50)
        
        # ── Fix 18: Fallback to SDXL on UNCERTAIN ──
        elif final_verdict == "UNCERTAIN" or final_verdict == "EDITED":
            sdxl_data = case_file.get("neural_consensus", {}).get("model_breakdown", {}).get("sdxl")
            if sdxl_data:
                sdxl_score = sdxl_data.get("score", 0)
                if sdxl_score > 0:
                    print(f"[HybridJudge] LIKELY_REAL override via SDXL (score={sdxl_score:+.1f})")
                    return "LIKELY_REAL", 60, f"[Fallback] LLM Uncertain -> SDXL says REAL ({sdxl_score:+.1f})", llm_result
                else:
                    print(f"[HybridJudge] LIKELY_AI_GENERATED override via SDXL (score={sdxl_score:+.1f})")
                    return "LIKELY_AI_GENERATED", 35, f"[Fallback] LLM Uncertain -> SDXL says AI ({sdxl_score:+.1f})", llm_result
            else:
                final_score = 50

        else:
            final_score = 50

        return final_verdict, final_score, final_description, llm_result

def create_judge(enable_llm: bool = True, **kwargs) -> HybridJudge:
    return HybridJudge(enable_llm=enable_llm, **kwargs)