import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from ytscript.llm import BaseLLMClient, MockLLMClient, OpenAIClient, GeminiClient

# Locate workspace root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variable key configs
load_dotenv(BASE_DIR / ".env")

def load_app_config() -> dict:
    """
    Loads config.yaml guidelines, including default niches, tones,
    and monetization safety policies.
    """
    config_path = BASE_DIR / "config.yaml"
    if not config_path.exists():
        # Fallback dictionary if config.yaml is missing
        return {
            "niches": ["tech", "finance"],
            "tones": ["educational", "energetic"],
            "monetization_policy": {
                "blocked_words": ["kill", "murder", "swear"],
                "sensitive_topics": ["violence", "hate speech"]
            },
            "copyright_policy": {
                "transformative_rules": ["Comment on B-roll"],
                "royalty_free_resources": []
            }
        }
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_llm_client(provider: str, model: str = None) -> BaseLLMClient:
    """
    Factory method returning the requested BaseLLMClient adapter.
    """
    prov_lower = provider.lower()
    if prov_lower == "mock":
        return MockLLMClient()
    elif prov_lower == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    elif prov_lower == "gemini":
        return GeminiClient(model=model or "gemini-2.5-flash")
    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            "Please use 'openai', 'gemini', or 'mock'."
        )
