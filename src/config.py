"""Load business policy (policy.yaml) and runtime settings (environment / .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = ROOT / "policy.yaml"
DEFAULT_CACHE_DIR = ROOT / "evals" / "llm_cache"

Role = Literal["extraction", "investigator", "approval", "critic"]
LLMMode = Literal["live", "record", "replay"]


class ApprovalPolicy(BaseModel):
    threshold_usd: Decimal
    near_threshold_pct: Decimal
    critic_required_when: list[str]
    max_critic_rounds: int
    approve_over_threshold_requires_no_unresolved_objections: bool


class PriceTolerance(BaseModel):
    silent_pct: Decimal
    warn_with_reason_pct: Decimal


class RoleConfig(BaseModel):
    model: Literal["fast", "strong"]
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"


class RateLimit(BaseModel):
    tokens_per_minute: int
    max_prompt_tokens: int
    expected_output_tokens: int = 1000
    max_retries: int = 5
    request_timeout_s: float = 60


class LLMPolicy(BaseModel):
    temperature: float
    max_extraction_retries: int
    item_match_min_confidence: float
    roles: dict[Role, RoleConfig]
    rate_limit: RateLimit


class Policy(BaseModel):
    approval: ApprovalPolicy
    price_tolerance: PriceTolerance
    money_tolerance_usd: Decimal
    terms_tolerance_days: int
    fx_rates_to_usd: dict[str, Decimal]
    llm: LLMPolicy


def _floats_to_str(node):
    """YAML floats -> strings so Decimal fields get exact values (0.03, not 0.0299...)."""
    if isinstance(node, dict):
        return {k: _floats_to_str(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_floats_to_str(v) for v in node]
    if isinstance(node, float):
        return repr(node)
    return node


def load_policy(path: Path | str = DEFAULT_POLICY_PATH) -> Policy:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Policy.model_validate(_floats_to_str(raw))


@dataclass(frozen=True)
class Settings:
    base_url: str | None
    api_key: str | None
    model: str | None  # strong tier
    model_fast: str | None  # fast tier; defaults to model
    mode: LLMMode
    cache_dir: Path

    @classmethod
    def from_env(cls, env_file: Path | None = ROOT / ".env") -> Settings:
        if env_file is not None and env_file.exists():
            load_dotenv(env_file, override=False)
        api_key = os.getenv("LLM_API_KEY") or None
        model = os.getenv("LLM_MODEL") or None
        mode = os.getenv("LLM_MODE") or ("record" if api_key else "replay")
        if mode not in ("live", "record", "replay"):
            raise ValueError(f"LLM_MODE must be live, record, or replay (got {mode!r})")
        return cls(
            base_url=os.getenv("LLM_BASE_URL") or None,
            api_key=api_key,
            model=model,
            model_fast=os.getenv("LLM_MODEL_FAST") or model,
            mode=mode,  # type: ignore[arg-type]
            cache_dir=Path(os.getenv("LLM_CACHE_DIR") or DEFAULT_CACHE_DIR),
        )
