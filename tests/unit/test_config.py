import json
from dataclasses import replace

import pytest

from the_loop.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    RunConfig,
    config_hash,
    load_config,
    normalize_features,
    normalize_policy,
)
from the_loop.errors import ConfigError


def test_defaults_when_no_flags_or_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOOP_BASE_URL", raising=False)
    monkeypatch.delenv("LOOP_API_KEY", raising=False)
    monkeypatch.delenv("LOOP_MODEL", raising=False)

    config = load_config(["summarize the repo"])

    assert config.task == "summarize the repo"
    assert config.workspace == tmp_path.resolve()
    assert config.headless is False
    assert config.policy_file is None
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL
    assert config.max_turns == DEFAULT_MAX_TURNS
    assert config.max_tool_calls == DEFAULT_MAX_TOOL_CALLS
    assert config.max_tokens is None
    assert config.max_wall_clock_s is None
    assert config.features == frozenset()


def test_cli_flags_override_env_and_defaults(monkeypatch, tmp_path):
    workspace = tmp_path / "chosen"
    workspace.mkdir()
    monkeypatch.setenv("LOOP_BASE_URL", "http://env.example/v1")
    monkeypatch.setenv("LOOP_API_KEY", "env-key")
    monkeypatch.setenv("LOOP_MODEL", "env-model")

    config = load_config(
        [
            "task",
            "--workspace",
            str(workspace),
            "--headless",
            "--max-turns",
            "3",
            "--max-tool-calls",
            "5",
            "--max-tokens",
            "900",
            "--max-wall-clock-s",
            "2.5",
            "--features",
            "b,a",
        ]
    )

    assert config.workspace == workspace.resolve()
    assert config.headless is True
    # No CLI flags exist for these, so environment values win over defaults.
    assert config.base_url == "http://env.example/v1"
    assert config.api_key == "env-key"
    assert config.model == "env-model"
    assert config.max_turns == 3
    assert config.max_tool_calls == 5
    assert config.max_tokens == 900
    assert config.max_wall_clock_s == 2.5
    assert config.features == frozenset({"a", "b"})


def test_features_are_normalized_and_sorted():
    assert normalize_features(" B, a,a, ,b ") == frozenset({"a", "b"})
    assert normalize_features(None) == frozenset()
    assert normalize_features("") == frozenset()


def test_policy_file_is_loaded_and_normalized(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "allow": {"tools": ["run_shell"], "paths": ["src/*"]},
                "deny": {"tools": ["run_tests"]},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(["task", "--policy-file", str(policy_path)])

    assert config.policy_file == policy_path
    assert config.policy == {
        "allow": {"tools": ["run_shell"], "paths": ["src/*"]},
        "deny": {"tools": ["run_tests"], "paths": []},
    }


def test_missing_policy_file_fails_fast(tmp_path):
    missing = tmp_path / "does-not-exist.json"

    with pytest.raises(ConfigError, match="Policy file not found"):
        load_config(["task", "--policy-file", str(missing)])


@pytest.mark.parametrize(
    "contents, message",
    [
        ("{not json", "not valid JSON"),
        (json.dumps({"allow": {"tool": ["run_shell"]}}), "unknown keys"),
        (json.dumps({"allow": {"tools": "run_shell"}}), "list of strings"),
        (json.dumps({"unexpected": {}}), "unknown top-level keys"),
    ],
)
def test_malformed_policy_file_fails_fast(tmp_path, contents, message):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(["task", "--policy-file", str(policy_path)])


def test_config_hash_is_stable_and_excludes_secrets_and_workspace(tmp_path):
    base = RunConfig(
        task="task",
        workspace=tmp_path,
        model="model-a",
        max_turns=4,
        features=frozenset({"b", "a"}),
        policy=normalize_policy({"allow": {"tools": ["run_shell"]}}),
    )

    digest = config_hash(base)

    assert digest == base.config_hash()
    assert len(digest) == 16
    int(digest, 16)

    # Feature order, credentials, and the raw workspace do not matter.
    assert config_hash(replace(base, features=frozenset({"a", "b"}))) == digest
    assert config_hash(replace(base, api_key="rotated-secret")) == digest
    assert config_hash(replace(base, base_url="http://other-host/v1")) == digest
    assert config_hash(replace(base, workspace=tmp_path / "elsewhere")) == digest
    assert config_hash(replace(base, task="a different task")) == digest

    # Behavioral changes do matter.
    assert config_hash(replace(base, model="model-b")) != digest
    assert config_hash(replace(base, headless=True)) != digest
    assert config_hash(
        replace(base, policy=normalize_policy({"allow": {"tools": ["run_tests"]}}))
    ) != digest

    # Rule ordering inside a policy file does not affect the hash.
    reversed_policy = {
        "allow": {"tools": ["run_tests", "run_shell"], "paths": []},
        "deny": {"tools": [], "paths": []},
    }
    ordered_policy = {
        "allow": {"tools": ["run_shell", "run_tests"], "paths": []},
        "deny": {"tools": [], "paths": []},
    }
    assert config_hash(replace(base, policy=reversed_policy)) == config_hash(
        replace(base, policy=ordered_policy)
    )


def test_invalid_budget_is_a_config_error(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="--max-turns"):
        load_config(["task", "--max-turns", "0"])
