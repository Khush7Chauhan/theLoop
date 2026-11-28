import json

import pytest

from the_loop import cli
from the_loop.llm.client import LLMResponse, ToolCall
from the_loop.llm.pricing import estimate_cost_usd


class ScriptedLLM:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def complete(self, messages, tools=None, tool_choice="auto"):
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


def response(*, content=None, tool_calls=None, prompt_tokens=10, completion_tokens=5):
    return LLMResponse(
        content=content,
        tool_calls=tool_calls or [],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=1,
    )


def tool_call(name, arguments, call_id="call_1"):
    return ToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def run_headless(monkeypatch, tmp_path, script, extra_args=()):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    fake = ScriptedLLM(script)

    monkeypatch.setenv("LOOP_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(cli, "LLMClient", lambda config: fake)

    def fail_if_prompted(*_args, **_kwargs):
        raise AssertionError("headless mode must never call input()")

    monkeypatch.setattr("builtins.input", fail_if_prompted)

    code = cli.main(
        [
            "do the task",
            "--workspace",
            str(workspace),
            "--headless",
            "--output-dir",
            str(run_dir),
            *extra_args,
        ]
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    trace_path = run_dir / "trace.jsonl"
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return code, summary, events, workspace, run_dir


def test_headless_run_executes_tool_and_writes_trace_and_summary(monkeypatch, tmp_path):
    code, summary, events, workspace, run_dir = run_headless(
        monkeypatch,
        tmp_path,
        [
            response(
                content=None,
                tool_calls=[tool_call("list_dir", {"path": "."})],
                prompt_tokens=10,
                completion_tokens=5,
            ),
            response(content="Done.", prompt_tokens=20, completion_tokens=7),
        ],
    )

    assert code == 0
    assert summary["status"] == "resolved"
    assert summary["turns"] == 2
    assert summary["tool_calls"] == 1
    assert summary["prompt_tokens"] == 30
    assert summary["completion_tokens"] == 12
    assert summary["estimated_cost_usd"] == pytest.approx(
        estimate_cost_usd("gpt-4o-mini", 30, 12)
    )
    assert summary["wall_clock_s"] >= 0
    assert len(summary["config_hash"]) == 16

    trace_path = run_dir / "trace.jsonl"
    assert summary["trace_path"] == str(trace_path)
    assert trace_path.exists()

    types = [event["type"] for event in events]
    assert types[0] == "run_start"
    assert types[-1] == "run_end"
    assert types.count("llm_turn") == 2
    assert "policy_decision" in types
    assert "tool_result" in types
    assert all(event["ts"].endswith("Z") for event in events)

    policy_event = next(e for e in events if e["type"] == "policy_decision")
    assert policy_event["tool"] == "list_dir"
    assert policy_event["decision"] == "allow"

    tool_event = next(e for e in events if e["type"] == "tool_result")
    assert tool_event["tool"] == "list_dir"
    assert tool_event["ok"] is True

    end_event = events[-1]
    assert end_event["status"] == "resolved"
    assert end_event["turns"] == 2
    assert end_event["tool_calls"] == 1


def test_headless_budget_exhaustion_still_writes_summary(monkeypatch, tmp_path):
    code, summary, events, _workspace, run_dir = run_headless(
        monkeypatch,
        tmp_path,
        [
            response(
                content=None,
                tool_calls=[tool_call("list_dir", {"path": "."})],
            )
        ],
        extra_args=("--max-turns", "1"),
    )

    assert code == 2
    assert summary["status"] == "budget_exceeded"
    assert summary["turns"] == 1
    assert (run_dir / "summary.json").exists()
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "budget_exceeded"


def test_llm_exception_is_summarized_as_error(monkeypatch, tmp_path):
    code, summary, events, _workspace, run_dir = run_headless(
        monkeypatch,
        tmp_path,
        [RuntimeError("model exploded")],
    )

    assert code == 1
    assert summary["status"] == "error"
    assert (run_dir / "summary.json").exists()
    assert any(event["type"] == "error" for event in events)
    assert events[-1]["type"] == "run_end"
    assert events[-1]["status"] == "error"


def test_headless_unresolved_ask_is_denied_without_prompting(monkeypatch, tmp_path):
    code, summary, events, workspace, _run_dir = run_headless(
        monkeypatch,
        tmp_path,
        [
            response(
                content=None,
                tool_calls=[
                    tool_call(
                        "write_file",
                        {"path": "src/main.py", "content": "print('x')"},
                    )
                ],
            ),
            response(content="Stopped."),
        ],
    )

    assert code == 0
    assert summary["status"] == "resolved"
    assert summary["tool_calls"] == 1

    policy_event = next(e for e in events if e["type"] == "policy_decision")
    assert policy_event["tool"] == "write_file"
    assert policy_event["decision"] == "deny"
    assert policy_event["reason"] == "headless_unresolved_ask"

    tool_event = next(e for e in events if e["type"] == "tool_result")
    assert tool_event["tool"] == "write_file"
    assert tool_event["ok"] is False
    assert tool_event["error_type"] == "permission"

    assert not (workspace / "src" / "main.py").exists()
