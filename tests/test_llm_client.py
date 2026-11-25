"""LLM client behavior with a fake OpenAI client: no key, no network."""

from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest
from pydantic import BaseModel

from src.config import Settings, load_policy
from src.llm.cache import LLMCache
from src.llm.client import (
    CacheMiss,
    LLMClient,
    LLMUnavailable,
    PromptTooLarge,
    StructuredOutputError,
    parse_wait_seconds,
    strict_json_schema,
)
from src.llm.pacer import TokenPacer


class Item(BaseModel):
    name: str
    qty: int | None = None


class Nested(BaseModel):
    items: list[Item]
    default: str | None = None  # a property literally named "default" must survive


def completion(content="{}", tool_calls=None, total=100):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(
        prompt_tokens=total - 10,
        completion_tokens=10,
        total_tokens=total,
        prompt_tokens_details=SimpleNamespace(cached_tokens=5),
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage
    )


def http_error(cls, status, headers=None, body="error"):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return cls(body, response=response, body=None)


class FakeOpenAI:
    """Pops scripted results; records every request."""

    def __init__(self, *results):
        self.results = list(results)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def make_client(tmp_path, fake, mode="record", **policy_overrides):
    policy = load_policy().llm
    for k, v in policy_overrides.items():
        setattr(policy.rate_limit, k, v)
    settings = Settings(
        base_url="https://example.test/v1",
        api_key="k",
        model="strong-m",
        model_fast="fast-m",
        mode=mode,
        cache_dir=tmp_path / "cache",
    )
    clock = FakeClock()
    events = []
    client = LLMClient(
        settings,
        policy,
        openai_client=fake,
        cache=LLMCache(tmp_path / "cache"),
        clock=clock,
        sleep=clock.sleep,
        on_event=lambda e, p: events.append((e, p)),
    )
    return client, clock, events


MSGS = [{"role": "user", "content": "extract"}]


def test_roles_route_to_model_tiers(tmp_path):
    fake = FakeOpenAI(completion("hi"), completion("hi"))
    client, _, _ = make_client(tmp_path, fake)
    client.complete("extraction", MSGS)
    client.complete("critic", MSGS)
    assert [r["model"] for r in fake.requests] == ["fast-m", "strong-m"]
    assert fake.requests[0]["reasoning_effort"] == "low"
    assert fake.requests[0]["temperature"] == 0


def test_record_then_replay_costs_no_calls(tmp_path):
    fake = FakeOpenAI(completion('{"name": "WidgetA", "qty": 3}'))
    client, _, _ = make_client(tmp_path, fake)
    first, resp1 = client.structured("extraction", MSGS, Item)
    assert first == Item(name="WidgetA", qty=3) and not resp1.cache_hit
    assert resp1.usage["cached_tokens"] == 5

    second, resp2 = client.structured("extraction", MSGS, Item)  # same request: cache hit
    assert second == first and resp2.cache_hit
    assert len(fake.requests) == 1

    replay, _, _ = make_client(tmp_path, FakeOpenAI(), mode="replay")
    assert replay.structured("extraction", MSGS, Item)[0] == first
    with pytest.raises(CacheMiss):
        replay.complete("extraction", [{"role": "user", "content": "changed prompt"}])


def test_live_mode_always_calls_and_never_writes(tmp_path):
    fake = FakeOpenAI(completion("a"), completion("b"))
    client, _, _ = make_client(tmp_path, fake, mode="live")
    client.complete("approval", MSGS)
    client.complete("approval", MSGS)
    assert len(fake.requests) == 2
    assert not (tmp_path / "cache").exists()


def test_strict_schema_request_and_fallback_to_json_mode(tmp_path):
    rejected = http_error(
        openai.BadRequestError, 400, body="response_format json_schema unsupported"
    )
    fake = FakeOpenAI(rejected, completion('{"name": "X"}'))
    client, _, events = make_client(tmp_path, fake)
    parsed, resp = client.structured("extraction", MSGS, Item)
    assert parsed.name == "X"
    assert fake.requests[0]["response_format"]["type"] == "json_schema"
    assert fake.requests[0]["response_format"]["json_schema"]["strict"] is True
    assert fake.requests[1]["response_format"] == {"type": "json_object"}
    assert "JSON schema" in fake.requests[1]["messages"][-1]["content"]
    assert resp.fallbacks == ["json_object mode instead of strict json_schema"]
    assert events[0][0] == "llm_capability_fallback"


def test_reasoning_effort_dropped_when_rejected(tmp_path):
    rejected = http_error(
        openai.BadRequestError, 400, body="unsupported parameter: reasoning_effort"
    )
    fake = FakeOpenAI(rejected, completion("ok"), completion("ok2"))
    client, _, _ = make_client(tmp_path, fake)
    client.complete("approval", MSGS)
    client.complete("approval", [{"role": "user", "content": "again"}])
    assert "reasoning_effort" in fake.requests[0]
    assert "reasoning_effort" not in fake.requests[1]
    assert "reasoning_effort" not in fake.requests[2]  # remembered per model


def test_unrelated_bad_request_fails_closed(tmp_path):
    fake = FakeOpenAI(http_error(openai.BadRequestError, 400, body="messages must be non-empty"))
    client, _, _ = make_client(tmp_path, fake)
    with pytest.raises(LLMUnavailable):
        client.complete("approval", MSGS)


def test_rate_limit_waits_the_stated_time(tmp_path):
    limited = http_error(openai.RateLimitError, 429, headers={"retry-after": "7"})
    fake = FakeOpenAI(limited, completion("ok"))
    client, clock, events = make_client(tmp_path, fake)
    resp = client.complete("approval", MSGS)
    assert resp.content == "ok"
    assert 7 in clock.sleeps
    assert ("llm_rate_limited", {"model": "strong-m", "wait_s": 7.0}) in events


def test_server_errors_back_off_then_give_up(tmp_path):
    errors = [http_error(openai.InternalServerError, 500) for _ in range(3)]
    client, clock, _ = make_client(tmp_path, FakeOpenAI(*errors), max_retries=2)
    with pytest.raises(LLMUnavailable, match="3 attempts"):
        client.complete("approval", MSGS)
    assert clock.sleeps == [1.0, 2.0, 4.0]


def test_oversized_prompt_is_refused_before_calling(tmp_path):
    fake = FakeOpenAI()
    client, _, _ = make_client(tmp_path, fake, max_prompt_tokens=50)
    with pytest.raises(PromptTooLarge):
        client.complete("critic", [{"role": "user", "content": "x" * 1000}])
    assert fake.requests == []


def test_invalid_structured_output_raises_with_raw_text(tmp_path):
    fake = FakeOpenAI(completion('{"qty": "lots"}'))
    client, _, _ = make_client(tmp_path, fake)
    with pytest.raises(StructuredOutputError) as exc:
        client.structured("extraction", MSGS, Item)
    assert exc.value.raw == '{"qty": "lots"}'


def test_tool_calls_are_returned(tmp_path):
    call = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="check_stock", arguments='{"item": "WidgetA"}')
    )
    fake = FakeOpenAI(completion(None, tool_calls=[call]))
    client, _, _ = make_client(tmp_path, fake)
    resp = client.complete("investigator", MSGS, tools=[{"type": "function"}])
    assert resp.tool_calls == [
        {"id": "c1", "name": "check_stock", "arguments": '{"item": "WidgetA"}'}
    ]


def test_missing_key_fails_closed(tmp_path):
    policy = load_policy().llm
    settings = Settings(None, None, "m", "m", "record", tmp_path)
    client = LLMClient(settings, policy, cache=LLMCache(tmp_path))
    with pytest.raises(LLMUnavailable, match="LLM_API_KEY"):
        client.complete("approval", MSGS)


def test_strict_schema_closes_objects_and_requires_all_fields():
    schema = strict_json_schema(Nested)
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["items", "default"]
    assert "default" in schema["properties"]  # property named "default" kept
    assert "default" not in schema["properties"]["default"]  # default value stripped
    item = schema["$defs"]["Item"]
    assert item["additionalProperties"] is False and item["required"] == ["name", "qty"]


def test_parse_wait_seconds():
    assert parse_wait_seconds({"retry-after": "3"}) == 3.0
    assert parse_wait_seconds({"x-ratelimit-reset-tokens": "7.66s"}) == pytest.approx(7.66)
    assert parse_wait_seconds({"x-ratelimit-reset-tokens": "2m59.5s"}) == pytest.approx(179.5)
    assert parse_wait_seconds({"x-ratelimit-reset-requests": "500ms"}) == pytest.approx(0.5)
    assert parse_wait_seconds({}) is None


def test_pacer_waits_for_window_and_honors_penalty():
    clock = FakeClock()
    pacer = TokenPacer(8000, clock=clock, sleep=clock.sleep)
    pacer.acquire(5000)
    _, waited = pacer.acquire(5000)  # doesn't fit: wait for the first to age out
    assert waited == pytest.approx(60.0)
    pacer.penalize(10)
    _, waited = pacer.acquire(1)
    assert waited >= 10
