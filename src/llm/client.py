"""One LLM client for every agent, on the OpenAI-compatible API (D23).

Provider is chosen by environment variables (Groq for development, xAI Grok
for final runs). Responsibilities:

- Route each agent role to its model tier and reasoning effort (policy.yaml).
- Structured output: strict JSON schema, falling back to JSON mode plus the
  schema in the prompt if the provider rejects strict mode. Pydantic
  validation is the source of truth either way.
- Record/replay cache so unchanged prompts cost nothing and runs work offline.
- Client-side pacing against tokens/min, honoring the provider's stated wait
  on 429s, exponential backoff on 5xx and connection errors.
- Fail closed: any unrecoverable problem raises an LLMError subclass, which
  callers turn into an `extraction_failed` HOLD (D16).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from src.config import LLMPolicy, Role, Settings
from src.llm.cache import LLMCache, request_key
from src.llm.pacer import TokenPacer

T = TypeVar("T", bound=BaseModel)
Message = dict[str, Any]


class LLMError(Exception):
    """Base class: any LLM failure the pipeline must fail closed on."""


class LLMUnavailable(LLMError):
    """No key/model configured, or the provider kept failing after retries."""


class CacheMiss(LLMError):
    """Replay mode and the request isn't in the cache."""


class PromptTooLarge(LLMError):
    """Prompt exceeds max_prompt_tokens; the caller should send less context."""


class ToolCallRejected(LLMError):
    """The provider rejected the model's tool call (e.g. arguments didn't match the
    tool's schema). Retryable: the caller can show the model the error."""


class StructuredOutputError(LLMError):
    """The model's output didn't parse or validate against the schema."""

    def __init__(self, message: str, raw: str | None):
        super().__init__(message)
        self.raw = raw


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    usage: dict[str, int]
    model: str
    role: str
    cache_hit: bool
    latency_ms: int
    waited_s: float = 0.0
    parsed: BaseModel | None = None
    fallbacks: list[str] = field(default_factory=list)


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema adjusted for strict structured-output mode:
    every object closed (additionalProperties false), every property required
    (optional fields are already nullable), and no `default` keywords."""

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        node.pop("default", None)
        props = node.get("properties")
        if isinstance(props, dict):
            node["additionalProperties"] = False
            node["required"] = list(props)
            for sub in props.values():
                walk(sub)
        for key in ("items", "additionalProperties"):
            if isinstance(node.get(key), dict):
                walk(node[key])
        for key in ("anyOf", "allOf", "oneOf"):
            if key in node:
                walk(node[key])
        if isinstance(node.get("$defs"), dict):
            for sub in node["$defs"].values():
                walk(sub)

    schema = model.model_json_schema()
    walk(schema)
    return schema


_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def parse_wait_seconds(headers: Any) -> float | None:
    """Seconds to wait from a 429's headers: `retry-after` (seconds), or Groq's
    `x-ratelimit-reset-tokens` / `-requests` durations like '7.66s' or '2m59.5s'."""
    if headers is None:
        return None
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    waits = []
    for name in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        value = headers.get(name)
        if not value:
            continue
        units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
        total = sum(float(num) * units[unit] for num, unit in _DURATION_PART.findall(value))
        if total:
            waits.append(total)
    return min(waits) if waits else None


def estimate_tokens(payload: Any) -> int:
    """Rough token count (~4 chars per token); used for pacing and size limits only."""
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return len(text) // 4 + 1


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        policy: LLMPolicy,
        *,
        openai_client: Any | None = None,
        cache: LLMCache | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.settings = settings
        self.policy = policy
        self.cache = cache or LLMCache(settings.cache_dir)
        self._client = openai_client
        self._clock = clock
        self._sleep = sleep
        self._on_event = on_event or (lambda event, payload: None)
        self._pacers: dict[str, TokenPacer] = {}
        # Capabilities learned at runtime, per model.
        self._no_reasoning_effort: set[str] = set()
        self._no_strict_schema: set[str] = set()

    # ---------- public API ----------

    def model_for(self, role: Role) -> tuple[str, str | None]:
        cfg = self.policy.roles[role]
        model = self.settings.model_fast if cfg.model == "fast" else self.settings.model
        if not model:
            raise LLMUnavailable(
                "No model configured: set LLM_MODEL (and optionally LLM_MODEL_FAST)"
            )
        return model, cfg.reasoning_effort

    def complete(
        self,
        role: Role,
        messages: list[Message],
        *,
        schema: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One chat completion. If `schema` is given, the result is validated into
        `response.parsed` or StructuredOutputError is raised."""
        model, effort = self.model_for(role)
        fallbacks: list[str] = []
        while True:
            request = self._build_request(
                model, effort, messages, schema, tools, tool_choice, max_tokens
            )
            try:
                response = self._execute(role, request)
            except openai.BadRequestError as e:
                fallback = self._capability_fallback(model, effort, schema, e)
                if fallback is None:
                    text = str(e).lower()
                    if "tool call validation" in text or "tool_use_failed" in text:
                        raise ToolCallRejected(str(e)) from e
                    raise LLMUnavailable(f"{model} rejected the request: {e}") from e
                fallbacks.append(fallback)
                self._on_event("llm_capability_fallback", {"model": model, "fallback": fallback})
                continue
            response.fallbacks = fallbacks
            if schema is not None:
                response.parsed = self._parse(schema, response.content)
            return response

    def structured(
        self, role: Role, messages: list[Message], schema: type[T], **kwargs: Any
    ) -> tuple[T, LLMResponse]:
        response = self.complete(role, messages, schema=schema, **kwargs)
        return response.parsed, response  # type: ignore[return-value]

    # ---------- request building ----------

    def _build_request(
        self,
        model: str,
        effort: str | None,
        messages: list[Message],
        schema: type[BaseModel] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "temperature": self.policy.temperature,
        }
        if effort and model not in self._no_reasoning_effort:
            request["reasoning_effort"] = effort
        if schema is not None:
            if model in self._no_strict_schema:
                schema_text = json.dumps(schema.model_json_schema(), separators=(",", ":"))
                request["messages"].append(
                    {
                        "role": "system",
                        "content": "Respond with a single JSON object that conforms to this "
                        f"JSON schema. No prose, no code fences.\n{schema_text}",
                    }
                )
                request["response_format"] = {"type": "json_object"}
            else:
                request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": strict_json_schema(schema),
                        "strict": True,
                    },
                }
        if tools:
            request["tools"] = tools
            if tool_choice:
                request["tool_choice"] = tool_choice
        if max_tokens:
            request["max_tokens"] = max_tokens
        return request

    def _capability_fallback(
        self,
        model: str,
        effort: str | None,
        schema: type[BaseModel] | None,
        error: openai.BadRequestError,
    ) -> str | None:
        text = str(error).lower()
        if effort and model not in self._no_reasoning_effort and "reasoning" in text:
            self._no_reasoning_effort.add(model)
            return "dropped reasoning_effort"
        if (
            schema is not None
            and model not in self._no_strict_schema
            and ("json_schema" in text or "response_format" in text or "strict" in text)
        ):
            self._no_strict_schema.add(model)
            return "json_object mode instead of strict json_schema"
        return None

    # ---------- execution ----------

    def _execute(self, role: str, request: dict[str, Any]) -> LLMResponse:
        key = request_key(request)
        mode = self.settings.mode
        if mode in ("record", "replay"):
            cached = self.cache.get(key)
            if cached is not None:
                self._on_event(
                    "llm_call", {"role": role, "model": request["model"], "cache_hit": True}
                )
                return self._to_response(
                    cached, request["model"], role, cache_hit=True, latency_ms=0
                )
            if mode == "replay":
                self._on_event("llm_cache_miss", {"role": role, "model": request["model"]})
                raise CacheMiss(
                    f"No cached response for this {role} call ({request['model']}). "
                    "Set LLM_API_KEY to record it, or use the committed cache's model."
                )

        prompt_tokens = estimate_tokens(request["messages"]) + estimate_tokens(
            {k: request[k] for k in ("tools", "response_format") if k in request}
        )
        limit = self.policy.rate_limit
        if prompt_tokens > limit.max_prompt_tokens:
            raise PromptTooLarge(
                f"~{prompt_tokens} prompt tokens exceeds the limit of {limit.max_prompt_tokens}"
            )

        raw, latency_ms, waited = self._call_with_retries(request, prompt_tokens)
        if mode == "record":
            self.cache.put(key, request, raw)
        response = self._to_response(
            raw, request["model"], role, cache_hit=False, latency_ms=latency_ms
        )
        response.waited_s = waited
        self._on_event(
            "llm_call",
            {
                "role": role,
                "model": request["model"],
                "cache_hit": False,
                "latency_ms": latency_ms,
                "waited_s": round(waited, 2),
                **raw["usage"],
            },
        )
        return response

    def _openai(self) -> Any:
        if self._client is None:
            if not self.settings.api_key:
                raise LLMUnavailable(
                    "No LLM_API_KEY set. Set it in .env, or run with LLM_MODE=replay."
                )
            self._client = openai.OpenAI(
                api_key=self.settings.api_key,
                base_url=self.settings.base_url,
                timeout=self.policy.rate_limit.request_timeout_s,
                max_retries=0,  # retries are handled here, with pacing
            )
        return self._client

    def _pacer(self, model: str) -> TokenPacer:
        if model not in self._pacers:
            self._pacers[model] = TokenPacer(
                self.policy.rate_limit.tokens_per_minute, clock=self._clock, sleep=self._sleep
            )
        return self._pacers[model]

    def _call_with_retries(
        self, request: dict[str, Any], prompt_tokens: int
    ) -> tuple[dict[str, Any], int, float]:
        limit = self.policy.rate_limit
        pacer = self._pacer(request["model"])
        client = self._openai()
        waited_total = 0.0
        last_error: Exception | None = None
        for attempt in range(limit.max_retries + 1):
            ticket, waited = pacer.acquire(prompt_tokens + limit.expected_output_tokens)
            waited_total += waited
            start = self._clock()
            try:
                completion = client.chat.completions.create(**request)
            except openai.RateLimitError as e:
                wait = parse_wait_seconds(getattr(e.response, "headers", None)) or 2.0**attempt
                self._on_event("llm_rate_limited", {"model": request["model"], "wait_s": wait})
                pacer.penalize(wait)
                last_error = e
                continue
            except openai.BadRequestError:
                raise  # not retryable; complete() decides on a capability fallback
            except (openai.APIConnectionError, openai.InternalServerError) as e:
                last_error = e
                backoff = min(2.0**attempt, 30.0)
                self._on_event("llm_retry", {"model": request["model"], "error": type(e).__name__})
                self._sleep(backoff)
                waited_total += backoff
                continue
            except openai.APIStatusError as e:
                raise LLMUnavailable(
                    f"{request['model']} returned HTTP {e.status_code}: {e}"
                ) from e
            latency_ms = int((self._clock() - start) * 1000)
            raw = self._serialize(completion)
            pacer.settle(ticket, raw["usage"].get("total_tokens") or ticket.tokens)
            return raw, latency_ms, waited_total
        raise LLMUnavailable(
            f"{request['model']} failed after {limit.max_retries + 1} attempts: {last_error}"
        )

    # ---------- response handling ----------

    @staticmethod
    def _serialize(completion: Any) -> dict[str, Any]:
        message = completion.choices[0].message
        usage = completion.usage
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None) if usage else None
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return {
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in (message.tool_calls or [])
            ],
            "finish_reason": completion.choices[0].finish_reason,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                "cached_tokens": cached,
            },
        }

    @staticmethod
    def _to_response(
        raw: dict[str, Any], model: str, role: str, *, cache_hit: bool, latency_ms: int
    ) -> LLMResponse:
        return LLMResponse(
            content=raw.get("content"),
            tool_calls=raw.get("tool_calls", []),
            usage=raw.get("usage", {}),
            model=model,
            role=role,
            cache_hit=cache_hit,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _parse(schema: type[BaseModel], content: str | None) -> BaseModel:
        if not content:
            raise StructuredOutputError("Empty response where JSON was expected", content)
        text = content.strip()
        if text.startswith("```"):  # tolerate code fences in JSON-mode fallback
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            return schema.model_validate_json(text)
        except ValidationError as e:
            raise StructuredOutputError(
                f"Output failed {schema.__name__} validation: {e}", content
            ) from e
