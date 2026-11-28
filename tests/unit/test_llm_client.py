from types import SimpleNamespace

import pytest

from the_loop.config import RunConfig
from the_loop.llm.client import LLMClient, ToolCall


class ServerError(Exception):
    status_code = 500


class RateLimitError(Exception):
    status_code = 429


class ConnectionBoom(Exception):
    pass


class UnauthorizedError(Exception):
    status_code = 401


class BadRequestError(Exception):
    status_code = 400


def fake_response(
    *,
    content="done",
    tool_calls=None,
    prompt_tokens=10,
    completion_tokens=5,
    usage=True,
):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    usage_obj = (
        SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        if usage
        else None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=usage_obj,
    )


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeOpenAI:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)


@pytest.fixture
def config(tmp_path):
    return RunConfig(task="task", workspace=tmp_path, model="gpt-4o-mini")


def make_client(config, outcomes, **kwargs):
    fake = FakeOpenAI(outcomes)
    sleeps = []
    kwargs.setdefault("jitter_fn", lambda: 0.0)
    kwargs.setdefault("base_delay_s", 0.01)
    client = LLMClient(
        config,
        client=fake,
        sleep_fn=sleeps.append,
        **kwargs,
    )
    return client, fake, sleeps


def test_transient_errors_retry_with_exponential_backoff(config):
    client, fake, sleeps = make_client(
        config,
        [ServerError(), RateLimitError(), ConnectionBoom(), fake_response()],
        max_retries=3,
    )

    response = client.complete([{"role": "user", "content": "hi"}])

    assert response.content == "done"
    assert fake.completions.calls == 4
    assert len(sleeps) == 3
    assert sleeps == sorted(sleeps)
    assert sleeps[0] < sleeps[1] < sleeps[2]


def test_non_retryable_4xx_fails_immediately(config):
    for error in (UnauthorizedError(), BadRequestError()):
        client, fake, sleeps = make_client(
            config,
            [error, fake_response()],
            max_retries=3,
        )

        with pytest.raises(type(error)):
            client.complete([{"role": "user", "content": "hi"}])

        assert fake.completions.calls == 1
        assert sleeps == []


def test_retries_are_bounded(config):
    client, fake, sleeps = make_client(
        config,
        [ServerError()],
        max_retries=2,
    )

    with pytest.raises(ServerError):
        client.complete([{"role": "user", "content": "hi"}])

    assert fake.completions.calls == 3
    assert len(sleeps) == 2


def test_missing_usage_defaults_to_zero_tokens(config):
    client, _fake, _sleeps = make_client(config, [fake_response(usage=False)])

    response = client.complete([{"role": "user", "content": "hi"}])

    assert response.prompt_tokens == 0
    assert response.completion_tokens == 0


def test_response_normalizes_tool_calls_and_measures_latency(config, monkeypatch):
    ticks = iter([100.0, 100.123])
    monkeypatch.setattr(
        "the_loop.llm.client.time.monotonic",
        lambda: next(ticks),
    )
    raw_tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="list_dir",
            arguments='{"path": "."}',
        ),
    )
    client, fake, _sleeps = make_client(
        config,
        [fake_response(content=None, tool_calls=[raw_tool_call])],
    )

    response = client.complete(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function"}],
    )

    assert response.content is None
    assert response.tool_calls == [
        ToolCall(id="call_1", name="list_dir", arguments='{"path": "."}')
    ]
    assert response.latency_ms == 123
    assert fake.completions.kwargs[0]["tools"] == [{"type": "function"}]
    assert fake.completions.kwargs[0]["tool_choice"] == "auto"
