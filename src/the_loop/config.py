"""Typed runtime configuration for the agent loop.

Precedence is always CLI flags > environment variables > defaults.  The
resulting :class:`RunConfig` is immutable and can be hashed for experiment
bookkeeping without leaking credentials or machine-specific paths.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from the_loop.errors import ConfigError

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"
DEFAULT_MODEL = "qwen/qwen2.5-vl-7b"
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_TOOL_CALLS = 12

_POLICY_SECTIONS = ("allow", "deny")
_POLICY_KEYS = ("tools", "paths")


def empty_policy() -> dict:
    """Return the canonical empty policy-override structure."""
    return {section: {key: [] for key in _POLICY_KEYS} for section in _POLICY_SECTIONS}


def normalize_features(value: str | None) -> frozenset[str]:
    """Parse a comma-separated feature list into a sorted, normalized set."""
    if not value:
        return frozenset()

    features: set[str] = set()
    for raw in value.split(","):
        feature = raw.strip().lower()
        if feature:
            features.add(feature)
    return frozenset(sorted(features))


def normalize_policy(raw: object, *, source: str = "policy file") -> dict:
    """Validate and canonicalize policy overrides.

    The accepted shape is::

        {
          "allow": {"tools": ["run_shell"], "paths": ["src/*"]},
          "deny":  {"tools": ["run_tests"], "paths": ["docs/*"]}
        }

    Both sections and both keys are optional; missing values become empty
    lists.  Unknown keys are rejected so a typo cannot silently weaken a
    policy.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"{source} must contain a JSON object.")

    unknown_sections = sorted(set(raw) - set(_POLICY_SECTIONS))
    if unknown_sections:
        raise ConfigError(
            f"{source} has unknown top-level keys: {', '.join(unknown_sections)}"
        )

    policy = empty_policy()
    for section in _POLICY_SECTIONS:
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ConfigError(f"{source}: '{section}' must be a JSON object.")
        unknown_keys = sorted(set(values) - set(_POLICY_KEYS))
        if unknown_keys:
            raise ConfigError(
                f"{source}: '{section}' has unknown keys: "
                f"{', '.join(unknown_keys)}"
            )
        for key in _POLICY_KEYS:
            entries = values.get(key, [])
            if not isinstance(entries, list) or not all(
                isinstance(entry, str) for entry in entries
            ):
                raise ConfigError(
                    f"{source}: '{section}.{key}' must be a list of strings."
                )
            policy[section][key] = sorted(
                {entry.strip().replace("\\", "/") for entry in entries if entry.strip()}
            )
    return policy


def load_policy_file(path: str | Path) -> dict:
    """Load a policy-override JSON file or fail fast with a clear error."""
    policy_path = Path(path).expanduser()
    if not policy_path.exists():
        raise ConfigError(f"Policy file not found: {policy_path}")
    if not policy_path.is_file():
        raise ConfigError(f"Policy file is not a regular file: {policy_path}")

    try:
        raw = json.loads(policy_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Could not read policy file {policy_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"Policy file is not valid JSON: {policy_path} "
            f"(line {error.lineno}, column {error.colno}): {error.msg}"
        ) from error

    return normalize_policy(raw, source=f"Policy file {policy_path}")


@dataclass(frozen=True)
class RunConfig:
    """Everything the runner needs for one invocation."""

    task: str
    workspace: Path
    headless: bool = False
    policy_file: Path | None = None
    policy: dict = field(default_factory=empty_policy)
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    max_turns: int = DEFAULT_MAX_TURNS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_tokens: int | None = None
    max_wall_clock_s: float | None = None
    features: frozenset[str] = field(default_factory=frozenset)
    output_dir: Path | None = None

    def config_hash(self) -> str:
        return config_hash(self)


def _positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{name} must be a positive number.")
    return float(value)


def load_config(argv: list[str] | None = None) -> RunConfig:
    """Build a :class:`RunConfig` from CLI flags, environment, and defaults."""
    # Imported lazily to keep config.py usable without importing the CLI module
    # (and to avoid a cycle when cli.py imports this module).
    from the_loop.cli import build_parser

    args = build_parser().parse_args(argv)

    workspace = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else Path.cwd().resolve()
    )
    if not workspace.exists():
        raise ConfigError(f"Workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise ConfigError(f"Workspace is not a directory: {workspace}")

    policy_file = Path(args.policy_file).expanduser() if args.policy_file else None
    policy = load_policy_file(policy_file) if policy_file is not None else empty_policy()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    )

    max_turns = _positive_int(args.max_turns, "--max-turns")
    max_tool_calls = _positive_int(args.max_tool_calls, "--max-tool-calls")
    max_tokens = _positive_int(args.max_tokens, "--max-tokens")
    max_wall_clock_s = _positive_float(args.max_wall_clock_s, "--max-wall-clock-s")

    return RunConfig(
        task=args.task,
        workspace=workspace,
        headless=bool(args.headless),
        policy_file=policy_file,
        policy=policy,
        base_url=os.getenv("LOOP_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("LOOP_API_KEY", DEFAULT_API_KEY),
        model=os.getenv("LOOP_MODEL", DEFAULT_MODEL),
        max_turns=max_turns if max_turns is not None else DEFAULT_MAX_TURNS,
        max_tool_calls=(
            max_tool_calls if max_tool_calls is not None else DEFAULT_MAX_TOOL_CALLS
        ),
        max_tokens=max_tokens,
        max_wall_clock_s=max_wall_clock_s,
        features=normalize_features(args.features),
        output_dir=output_dir,
    )


def _canonical_policy(policy: object) -> dict:
    """Order-insensitive policy representation for hashing."""
    canonical = {}
    source = policy if isinstance(policy, dict) else {}
    for section in _POLICY_SECTIONS:
        values = source.get(section, {})
        if not isinstance(values, dict):
            values = {}
        canonical[section] = {}
        for key in _POLICY_KEYS:
            entries = values.get(key, [])
            if not isinstance(entries, list):
                entries = []
            canonical[section][key] = sorted(str(entry) for entry in entries)
    return canonical


def config_hash(config: RunConfig) -> str:
    """Hash behavioral settings only.

    Credentials (``api_key``) and the raw workspace path are deliberately
    excluded so the hash is stable across machines and secret rotation.  The
    policy file contributes its normalized rules rather than its path.
    """
    payload = {
        "features": sorted(config.features),
        "headless": config.headless,
        "max_tokens": config.max_tokens,
        "max_tool_calls": config.max_tool_calls,
        "max_turns": config.max_turns,
        "max_wall_clock_s": config.max_wall_clock_s,
        "model": config.model,
        "policy": _canonical_policy(config.policy),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
