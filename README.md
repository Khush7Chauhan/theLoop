# the-loop

A tiny coding-agent CLI. It sends a task to an OpenAI-compatible chat model and
lets the model inspect and edit the workspace through three tools
(`list_dir`, `read_file`, `write_file`).

Every tool call goes through a policy engine before it runs:

| Action | Rule |
| --- | --- |
| Read `.env` or `secrets/` | denied |
| Write `.env` or `secrets/` | denied |
| Write `.loop/` | denied |
| Write `src/` or `tests/` | asks for approval (once or for the session) |
| Everything else | allowed |

Paths are resolved against the workspace, and anything that escapes it
(`../`, absolute paths outside the workspace) is rejected.

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
pytest
```

The tests cover `safe_path` and the policy rules only; they do not call a model.
