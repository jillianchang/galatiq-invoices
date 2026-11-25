"""Check that the configured provider supports what the pipeline needs.

Run before building on a provider (Groq feature check, Grok smoke test):

    make check-llm

For each configured model (LLM_MODEL and LLM_MODEL_FAST) it checks:
  1. the model exists on the endpoint
  2. a basic call works with reasoning_effort=low
  3. strict JSON-schema structured output works
  4. tool calling works
and prints the rate-limit headers. Uses a handful of small calls (~2k tokens total).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.config import Settings  # noqa: E402
from src.llm.client import strict_json_schema  # noqa: E402

RATE_LIMIT_HEADERS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
)


class StockAnswer(BaseModel):
    item: str
    quantity: int
    in_catalog: bool
    note: str | None


TOOL = {
    "type": "function",
    "function": {
        "name": "check_stock",
        "description": "Return units in stock for a catalog item.",
        "parameters": {
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
    },
}


def run_check(name: str, fn) -> bool:
    try:
        detail = fn()
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))
        return True
    except Exception as e:  # report every failure, keep going
        print(f"  FAIL  {name}: {type(e).__name__}: {str(e)[:300]}")
        return False


def check_model(client: openai.OpenAI, model: str, available: set[str] | None) -> dict[str, bool]:
    print(f"\n{model}")
    results: dict[str, bool] = {}

    def exists():
        if available is None:
            return "model listing unavailable; skipped"
        if model not in available:
            raise LookupError(f"not in /models ({len(available)} models listed)")
        return None

    def basic():
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            reasoning_effort="low",
            temperature=0,
        )
        completion = raw.parse()
        headers = {h: raw.headers.get(h) for h in RATE_LIMIT_HEADERS if raw.headers.get(h)}
        if headers:
            print("        rate limits: " + ", ".join(f"{k[12:]}={v}" for k, v in headers.items()))
        usage = completion.usage
        return f"reply={completion.choices[0].message.content!r}, tokens={usage.total_tokens}"

    def structured():
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Invoice line: 'Widget A  qty 12  @ $250'. Extract the item as "
                    "'WidgetA', its quantity, in_catalog=true, and note=null.",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "StockAnswer",
                    "schema": strict_json_schema(StockAnswer),
                    "strict": True,
                },
            },
            reasoning_effort="low",
            temperature=0,
        )
        parsed = StockAnswer.model_validate_json(completion.choices[0].message.content)
        if parsed.quantity != 12:
            raise ValueError(f"wrong quantity: {parsed}")
        return parsed.model_dump_json()

    def tools():
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "How many WidgetA are in stock? Use the tool."}],
            tools=[TOOL],
            tool_choice="auto",
            reasoning_effort="low",
            temperature=0,
        )
        calls = completion.choices[0].message.tool_calls or []
        if not calls:
            raise ValueError("model answered without calling the tool")
        return f"{calls[0].function.name}({calls[0].function.arguments})"

    results["exists"] = run_check("model available", exists)
    results["reasoning_effort"] = run_check("basic call with reasoning_effort=low", basic)
    results["strict_schema"] = run_check("strict JSON-schema output", structured)
    results["tools"] = run_check("tool calling", tools)
    return results


def main() -> int:
    settings = Settings.from_env()
    if not settings.api_key:
        print("LLM_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 2
    print(f"Endpoint: {settings.base_url or 'https://api.openai.com/v1 (default)'}")
    client = openai.OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=60)

    try:
        available = {m.id for m in client.models.list()}
    except Exception as e:
        print(f"Could not list models: {e}")
        available = None

    models = list(dict.fromkeys(m for m in (settings.model, settings.model_fast) if m))
    if not models:
        print("LLM_MODEL is not set.")
        return 2
    summary = {model: check_model(client, model, available) for model in models}

    print("\nSummary")
    print(json.dumps(summary, indent=2))
    if not all(summary[m]["reasoning_effort"] for m in models):
        print(
            "- reasoning_effort rejected: the client drops it automatically, but thinking "
            "tokens will then use the provider default."
        )
    if not all(summary[m]["strict_schema"] for m in models):
        print(
            "- strict schema failed: the client falls back to JSON mode + schema in the prompt; "
            "Pydantic validation still applies."
        )
    if not all(summary[m]["tools"] for m in models):
        print("- tool calling failed: the investigator agent needs a different model.")
    return 0 if all(all(r.values()) for r in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
