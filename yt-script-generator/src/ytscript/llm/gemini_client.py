import os
from typing import Type, TypeVar, List
from pydantic import BaseModel
from google import genai
from google.genai import types
from ytscript.llm.base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)

class GeminiClient(BaseLLMClient):
    """
    LLM adapter implementation for Google Gemini. Supports automatic key rotation
    if a key encounters quota limits, rate limits, or authorization errors.
    """
    def __init__(self, api_keys: List[str] = None, model: str = "gemini-2.5-flash"):
        if not api_keys:
            env_keys = os.getenv("GEMINI_API_KEYS")
            if env_keys:
                api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            else:
                single_key = os.getenv("GEMINI_API_KEY")
                api_keys = [single_key] if single_key else []

        if not api_keys:
            raise ValueError(
                "No Gemini API keys found. "
                "Set GEMINI_API_KEY or GEMINI_API_KEYS in .env."
            )

        self.api_keys = api_keys
        self.model = model
        self.current_key_idx = 0
        self.clients = [genai.Client(api_key=key) for key in api_keys]

    def _get_client(self) -> genai.Client:
        return self.clients[self.current_key_idx]

    def _rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"\n[warning] Gemini Key rotated to key #{self.current_key_idx + 1}")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        attempts = len(self.api_keys)
        for attempt in range(attempts):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                    )
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e).lower()
                is_safety_or_api = any(
                    x in err_str for x in ["quota", "limit", "exhausted", "429", "500", "502", "503", "504", "api_key", "invalid", "auth", "key", "blocked", "unavailable", "overload", "timeout", "demand", "temporary"]
                )
                if is_safety_or_api and attempt < attempts - 1:
                    print(f"\n[warning] Gemini call failed with key #{self.current_key_idx + 1}: {e}")
                    self._rotate_key()
                    import time
                    time.sleep(1.5)
                    continue
                raise e

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        temperature: float = 0.7
    ) -> T:
        attempts = len(self.api_keys)
        for attempt in range(attempts):
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=response_model,
                        temperature=temperature,
                    )
                )
                text = response.text
                if not text:
                    raise ValueError("Received empty response text from Gemini API.")
                    
                # Clean potential markdown wrapping
                text = text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

                return response_model.model_validate_json(text)
            except Exception as e:
                err_str = str(e).lower()
                is_safety_or_api = any(
                    x in err_str for x in ["quota", "limit", "exhausted", "429", "500", "502", "503", "504", "api_key", "invalid", "auth", "key", "blocked", "unavailable", "overload", "timeout", "demand", "temporary"]
                )
                if is_safety_or_api and attempt < attempts - 1:
                    print(f"\n[warning] Gemini structured call failed with key #{self.current_key_idx + 1}: {e}")
                    self._rotate_key()
                    import time
                    time.sleep(1.5)
                    continue
                raise e
