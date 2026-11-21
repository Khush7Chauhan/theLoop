import json

import the_loop.tools.fs as fs
from the_loop.tools.base import ToolResult


def test_tool_result_defaults():
    result = ToolResult(ok=True)

    assert result.data is None
    assert result.error_type is None
    assert result.retryable is False
    assert result.message == ""
    assert result.exit_code is None
    assert result.duration_ms == 0


def test_tool_result_serializes_to_json():
    result = ToolResult.success(
        {"files": ["a.txt"], "count": 1},
        message="listed",
        duration_ms=7,
    )

    payload = json.loads(result.model_dump_json())

    assert payload == {
        "ok": True,
        "data": {"files": ["a.txt"], "count": 1},
        "error_type": None,
        "retryable": False,
        "message": "listed",
        "exit_code": None,
        "duration_ms": 7,
    }


def test_tool_result_failure_helper():
    result = ToolResult.failure(
        "not_found",
        "missing",
        retryable=True,
        duration_ms=3,
    )

    assert result.ok is False
    assert result.error_type == "not_found"
    assert result.retryable is True
    assert result.message == "missing"
    assert result.duration_ms == 3


def test_tool_result_keeps_exit_code_for_future_shell_tools():
    result = ToolResult(ok=False, exit_code=1)

    assert result.exit_code == 1
    assert json.loads(result.model_dump_json())["exit_code"] == 1


def test_read_file_populates_duration_ms(workspace, monkeypatch):
    (workspace / "notes.txt").write_text("hello", encoding="utf-8")

    ticks = iter([100.0, 100.042])
    monkeypatch.setattr(fs.time, "perf_counter", lambda: next(ticks))

    result = fs.read_file("notes.txt")

    assert result.ok is True
    assert result.data == "hello"
    assert result.duration_ms == 42


def test_list_dir_populates_duration_ms(workspace, monkeypatch):
    (workspace / "b.txt").write_text("b", encoding="utf-8")
    (workspace / "a.txt").write_text("a", encoding="utf-8")

    ticks = iter([10.0, 10.010])
    monkeypatch.setattr(fs.time, "perf_counter", lambda: next(ticks))

    result = fs.list_dir(".")

    assert result.ok is True
    assert result.data == ["a.txt", "b.txt"]
    assert result.duration_ms == 10


def test_write_file_populates_duration_ms(workspace, monkeypatch):
    ticks = iter([0.0, 0.005])
    monkeypatch.setattr(fs.time, "perf_counter", lambda: next(ticks))

    result = fs.write_file("new.txt", "content")

    assert result.ok is True
    assert result.data == "Wrote new.txt"
    assert result.duration_ms == 5
    assert (workspace / "new.txt").read_text(encoding="utf-8") == "content"


def test_edit_file_populates_duration_ms(workspace, monkeypatch):
    (workspace / "notes.txt").write_text("old", encoding="utf-8")

    ticks = iter([1.0, 1.250])
    monkeypatch.setattr(fs.time, "perf_counter", lambda: next(ticks))

    result = fs.edit_file("notes.txt", "old", "new")

    assert result.ok is True
    assert result.data == "Edited notes.txt"
    assert result.duration_ms == 250


def test_failed_tool_result_also_reports_duration(workspace, monkeypatch):
    ticks = iter([5.0, 5.500])
    monkeypatch.setattr(fs.time, "perf_counter", lambda: next(ticks))

    result = fs.read_file("missing.txt")

    assert result.ok is False
    assert result.error_type == "not_found"
    assert result.retryable is True
    assert result.duration_ms == 500
