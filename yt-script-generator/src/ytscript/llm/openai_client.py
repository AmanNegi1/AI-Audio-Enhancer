import os
from typing import Type, TypeVar, List
from pydantic import BaseModel
from openai import OpenAI
from ytscript.llm.base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)

class OpenAIClient(BaseLLMClient):
    """
    LLM adapter implementation for OpenAI. Supports automatic key rotation
    if a key encounters quota limits, rate limits, or authorization errors.
    """
    def __init__(self, api_keys: List[str] = None, model: str = "gpt-4o-mini"):
        if not api_keys:
            env_keys = os.getenv("OPENAI_API_KEYS")
            if env_keys:
                api_keys = [k.strip() for k in env_keys.split(",") if k.strip()]
            else:
                single_key = os.getenv("OPENAI_API_KEY")
                api_keys = [single_key] if single_key else []
                
        if not api_keys:
            raise ValueError(
                "No OpenAI API keys found. "
                "Set OPENAI_API_KEY or OPENAI_API_KEYS in .env."
            )
            
        self.api_keys = api_keys
        self.model = model
        self.current_key_idx = 0
        self.clients = [OpenAI(api_key=key) for key in api_keys]

    def _get_client(self) -> OpenAI:
        return self.clients[self.current_key_idx]

    def _rotate_key(self):
        self.current_key_idx = (self.current_key_idx + 1) % len(self.api_keys)
        print(f"\n[warning] OpenAI Key rotated to key #{self.current_key_idx + 1}")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        attempts = len(self.api_keys)
        for attempt in range(attempts):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e).lower()
                is_safety_or_api = any(
                    x in err_str for x in ["rate_limit", "quota", "limit", "429", "500", "502", "503", "504", "api_key", "auth", "invalid", "blocked", "unavailable", "overload", "timeout", "demand", "temporary"]
                )
                if is_safety_or_api and attempt < attempts - 1:
                    print(f"\n[warning] OpenAI call failed with key #{self.current_key_idx + 1}: {e}")
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
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=response_model,
                    temperature=temperature
                )
                parsed = response.choices[0].message.parsed
                if parsed is None:
                    raise ValueError("Failed to parse response from OpenAI as the requested Pydantic model.")
                return parsed
            except Exception as e:
                err_str = str(e).lower()
                is_safety_or_api = any(
                    x in err_str for x in ["rate_limit", "quota", "limit", "429", "500", "502", "503", "504", "api_key", "auth", "invalid", "blocked", "unavailable", "overload", "timeout", "demand", "temporary"]
                )
                if is_safety_or_api and attempt < attempts - 1:
                    print(f"\n[warning] OpenAI structured call failed with key #{self.current_key_idx + 1}: {e}")
                    self._rotate_key()
                    import time
                    time.sleep(1.5)
                    continue
                raise e
