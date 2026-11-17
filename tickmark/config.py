"""Configuration and the runtime context passed to every graph node."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# Which environment variables hold each provider's key, and the call settings that suit it.
PROVIDERS: dict[str, dict[str, Any]] = {
    "google_genai": {
        "label": "Google Gemini",
        "keys": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "default_model": "gemini-flash-latest",  # Google's alias for its current Flash model
        "package": "langchain-google-genai",
        # Gemini 3+: Google recommends temperature 1.0 (lower values can cause looping), and thinking
        # tokens count toward the output limit, so leave plenty of room.
        "kwargs": {"temperature": 1.0, "max_tokens": 32768, "max_retries": 6, "timeout": 600},
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "keys": ["ANTHROPIC_API_KEY"],
        "default_model": "claude-sonnet-5-5",
        "package": "langchain-anthropic",
        "kwargs": {"max_tokens": 16384, "max_retries": 6, "timeout": 600},
    },
    "openai": {
        "label": "OpenAI",
        "keys": ["OPENAI_API_KEY"],
        "default_model": "gpt-5",
        "package": "langchain-openai",
        "kwargs": {"max_retries": 6, "timeout": 600},
    },
}


def provider_of(model_id: str) -> str:
    return model_id.split(":", 1)[0] if ":" in model_id else "google_genai"


def key_for(provider: str) -> str | None:
    """Name of the first environment variable that holds a key for this provider."""
    for var in PROVIDERS.get(provider, {}).get("keys", []):
        if os.getenv(var):
            return var
    return None


def default_model() -> str:
    """TICKMARK_MODEL wins; otherwise use whichever provider has a key (Gemini first)."""
    if env := os.getenv("TICKMARK_MODEL"):
        return env
    for provider in ("google_genai", "anthropic", "openai"):
        if key_for(provider):
            return f"{provider}:{PROVIDERS[provider]['default_model']}"
    return f"google_genai:{PROVIDERS['google_genai']['default_model']}"


def _env_float(name: str) -> float | None:
    try:
        return float(os.environ[name]) if os.getenv(name) else None
    except ValueError:
        return None


class Accounts(BaseModel):
    bank: str = "Bank - operating account"
    ar: str = "Accounts receivable"
    ap: str = "Accounts payable"
    fees: str = "Bank charges"
    interest: str = "Interest income"
    interest_exp: str = "Interest expense"
    suspense: str = "Suspense"
    credit: str = "Customer credits (unapplied cash)"
    advances: str = "Supplier advances"


class Settings(BaseModel):
    # LLM
    model: str = Field(default_factory=default_model)
    use_llm: bool = True
    max_concurrency: int = 4          # parallel exception-resolver agents
    agent_recursion_limit: int = 16   # max graph steps per resolver agent (~7 tool calls)
    requests_per_minute: float | None = Field(default_factory=lambda: _env_float("TICKMARK_RPM"))  # free tiers need this

    # Matching rules
    date_order: str = "auto"          # auto | dmy | mdy
    date_window: int = 90             # days an invoice may precede the payment
    fee_tolerance: float = 50.0       # shortfalls up to this are booked to bank charges
    name_threshold: float = 0.72      # 0-1 similarity needed to trust a name

    # Approval policy (what may be approved without a person)
    auto_approve_threshold: int = 85  # engine confidence needed for automatic approval
    ai_auto_approve: bool = False     # approve the AI's high-confidence matches automatically
    auto_post_suspense: bool = False  # approve suspense entries automatically
    interactive: bool = False         # pause (LangGraph interrupt) for a person to review the rest

    accounts: Accounts = Field(default_factory=Accounts)
    memory_path: str = ".tickmark/aliases.json"

    @property
    def provider(self) -> str:
        return provider_of(self.model)


def build_model(settings: Settings):
    """Create the LangChain chat model for the configured provider."""
    from langchain.chat_models import init_chat_model
    provider = settings.provider
    spec = PROVIDERS.get(provider, {"kwargs": {"max_retries": 6}})
    kwargs = dict(spec["kwargs"])
    if settings.requests_per_minute:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        rps = settings.requests_per_minute / 60
        kwargs["rate_limiter"] = InMemoryRateLimiter(requests_per_second=rps, check_every_n_seconds=0.1, max_bucket_size=1)
    model_id = settings.model if ":" in settings.model else f"{provider}:{settings.model}"
    return init_chat_model(model_id, **kwargs)


@dataclass
class RunContext:
    """Non-state dependencies, injected through LangGraph's runtime context."""
    settings: Settings = field(default_factory=Settings)
    model: Any = None  # a LangChain BaseChatModel; created lazily when None
    _resolved: bool = False
    unavailable_reason: str = ""

    def get_model(self):
        """Return the chat model, or None when the LLM is disabled or unavailable (reason kept)."""
        if not self.settings.use_llm:
            self.unavailable_reason = "LLM disabled (--no-llm)"
            return None
        if self.model is None and not self._resolved:
            self._resolved = True
            provider = self.settings.provider
            spec = PROVIDERS.get(provider)
            if spec and not key_for(provider):
                self.unavailable_reason = f"no {spec['label']} key: set {' or '.join(spec['keys'])}"
                return None
            try:
                self.model = build_model(self.settings)
            except ImportError:
                pkg = spec["package"] if spec else f"langchain-{provider}"
                self.unavailable_reason = f"the {pkg} package is not installed (pip install {pkg})"
            except Exception as e:
                self.unavailable_reason = f"couldn't create {self.settings.model}: {type(e).__name__}: {e}"
        return self.model

    @property
    def memory_file(self) -> Path:
        return Path(self.settings.memory_path)
