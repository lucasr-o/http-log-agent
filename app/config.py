"""Application configuration, read from the environment or a .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]

# Model used when LLM_MODEL is left empty, per provider.
DEFAULT_MODELS = {"anthropic": "claude-opus-5", "gemini": "gemini-3.5-flash"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- API ---
    api_key: str = Field(default="dev-key-change-me")
    max_batch_size: int = Field(default=5_000)
    sync_timeout_seconds: float = Field(
        default=25.0,
        description="Past this, /analyze returns 202 and the incident continues in the background.",
    )

    # --- models ---
    model_path: Path = Field(default=BASE_DIR / "models" / "detector.joblib")
    novelty_target_fpr: float = Field(
        default=0.002,
        description=(
            "Detector operating point, as a false positive rate. The default is "
            "recall-first: 0.002 is where incident recall saturates at 0.9972. "
            "More permissive points only add triage work — at 0.005 the dossier "
            "count rises 39% without covering one extra incident. The full table "
            "lives in reports/metrics.json."
        ),
    )

    # --- agents ---
    llm_provider: str = Field(
        default="auto",
        description=(
            "anthropic, gemini, or auto. Auto picks whichever key is present, "
            "preferring Anthropic when both are."
        ),
    )
    anthropic_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")
    llm_model: str = Field(
        default="",
        description=(
            "Empty means the provider default: claude-opus-5 for Anthropic, "
            "gemini-2.5-flash for Gemini."
        ),
    )
    llm_effort: str = Field(default="medium")
    llm_max_tokens: int = Field(default=8_000)
    llm_call_budget: int = Field(
        default=12,
        description="Cap on LLM calls per request; the excess falls back to the "
        "deterministic path.",
    )
    llm_max_tool_iterations: int = Field(default=8)

    # --- actuators ---
    block_mode: str = Field(
        default="dry_run",
        description="dry_run writes to the blocklist; enforce runs the firewall command.",
    )
    block_command: str = Field(default="")
    telegram_bot_token: str = Field(default="")
    telegram_chat_id: str = Field(default="")

    # --- persistence ---
    database_path: Path = Field(default=BASE_DIR / "data" / "detector.db")

    @property
    def active_provider(self) -> str:
        """Which provider will actually be used, or an empty string for none."""
        if self.llm_provider == "anthropic":
            return "anthropic" if self.anthropic_api_key else ""
        if self.llm_provider == "gemini":
            return "gemini" if self.gemini_api_key else ""
        if self.anthropic_api_key:
            return "anthropic"
        return "gemini" if self.gemini_api_key else ""

    @property
    def llm_enabled(self) -> bool:
        return bool(self.active_provider)

    @property
    def active_model(self) -> str:
        """The configured model, or the active provider's default."""
        if self.llm_model:
            return self.llm_model
        return DEFAULT_MODELS.get(self.active_provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
