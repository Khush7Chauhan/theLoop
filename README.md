# the-loop

A tiny coding-agent CLI. It sends a task to an OpenAI-compatible chat model and
lets the model inspect and edit the workspace through four filesystem tools
(`list_dir`, `read_file`, `write_file`, `edit_file`) plus two sandbox-only tools
(`run_shell`, `run_tests`).

Every tool call goes through a policy engine before it runs:

| Action | Rule |
| --- | --- |
| Read `.env` or `secrets/` | denied |
| Write or edit `.env` or `secrets/` | denied |
| Write or edit `.loop/` | denied |
| Write or edit `src/` or `tests/` | asks for approval (once or for the session) |
| `run_shell` | asks for approval (once or for the session) |
| `run_tests` | allowed; pytest runs in the sandbox |
| Unrecognized actions or paths | denied |
| Everything else | allowed |

Paths are resolved against the workspace, and anything that escapes it
(`../`, absolute paths outside the workspace, symlinks pointing outside) is
rejected.

`edit_file` replaces exactly one occurrence of `old_text` with `new_text`. If
the text is missing or appears more than once it returns a retryable error
(`not_found` / `not_unique`) and leaves the file untouched, so the model has to
supply more context instead of guessing.

The policy engine, filesystem tools, shell tools, and Docker sandbox live in
`src/the_loop/`; `loop.py` is the CLI entrypoint and re-exports the old symbols
for compatibility.

## Docker sandbox

`run_shell` and `run_tests` never execute agent-chosen commands on the host.
They call `run_in_sandbox` in `src/the_loop/sandbox/docker.py`, which is the
only place that invokes the Docker CLI. Each call starts a fresh container with:

- no network (`--network=none`)
- one CPU, 512 MB of memory, and a 64-process limit
- all capabilities dropped and `no-new-privileges`
- a read-only root filesystem, with `/workspace` bind-mounted and `/tmp` as a
  writable tmpfs
- a non-root user (`sandbox`, UID 1000)

`run_shell` runs `sh -c <command>` inside that container. `run_tests` runs
pytest strictly via argv, never through a shell. Timeouts forcibly remove the
container, and stdout/stderr are bounded with a `[TRUNCATED]` marker.

Build the image once:

```bash
docker build -f Dockerfile.sandbox -t the-loop-sandbox:latest .
```

`agent.py` lives on as `experiments/mcp_agent.py`, an unfinished MCP + Anthropic
experiment. It is not part of the CLI and is not maintained.

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv/Scripts/activate
pip install -e ".[dev]"
```

## Configuration

`loop.py` reads these environment variables (defaults shown):

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOOP_BASE_URL` | `http://localhost:1234/v1` | OpenAI-compatible endpoint (e.g. LM Studio) |
| `LOOP_API_KEY` | `lm-studio` | API key for that endpoint |
| `LOOP_MODEL` | `qwen/qwen2.5-vl-7b` | Model name |

To use a different endpoint:

```bash
export LOOP_BASE_URL="https://api.openai.com/v1"
export LOOP_API_KEY="sk-..."
export LOOP_MODEL="gpt-4o-mini"
```

## Run

```bash
python loop.py "list files"
python loop.py --workspace ../some-project "summarize this repo"
```

The workspace defaults to the current directory. Agent events are printed as
JSON lines; the run exits `0` on a final answer, `2` when a turn or tool-call
budget is exhausted, and `3` when the workspace does not exist.

## Tests

```bash
pytest tests/unit          # fast; no Docker or model required
pytest -m integration      # requires Docker and the sandbox image
pytest                     # both suites; integration tests skip without Docker
```

The unit tests cover `safe_path` (including symlink escapes), the policy rules,
`edit_file`, `ToolResult`, the Docker runner's hardening/error handling, and the
shell tools. The integration tests verify the edit-then-pytest flow, no-network
enforcement, timeout cleanup, the non-root user, and output truncation. They do
not call a model.
