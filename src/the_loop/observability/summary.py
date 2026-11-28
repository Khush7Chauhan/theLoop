"""Run-level summary written next to the JSONL trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["resolved", "gave_up", "budget_exceeded", "error"]
    turns: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    wall_clock_s: float
    config_hash: str
    trace_path: str


def write_summary(summary: RunSummary, path: str | Path) -> Path:
    """Write formatted JSON and return the path that was written."""
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path
