"""Fakes shared by tests: a scripted OpenAI client and a manual clock."""

from types import SimpleNamespace

import httpx2 as httpx

from src.config import Settings, load_policy
from src.llm.cache import LLMCache
from src.llm.client import LLMClient


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
