from typing import Any

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Uniform envelope returned by every tool.

    The agent loop serializes this model into the tool message it sends back
    to the model, so the shape must stay stable and JSON-friendly.
    """

    ok: bool
    data: Any = None
    error_type: str | None = None
    retryable: bool = False
    message: str = ""
    exit_code: int | None = None
    duration_ms: int = 0

    @classmethod
    def success(
        cls,
        data: Any = None,
        message: str = "",
        duration_ms: int = 0,
    ) -> "ToolResult":
        return cls(ok=True, data=data, message=message, duration_ms=duration_ms)

    @classmethod
    def failure(
        cls,
        error_type: str,
        message: str,
        retryable: bool = False,
        duration_ms: int = 0,
    ) -> "ToolResult":
        return cls(
            ok=False,
            error_type=error_type,
            message=message,
            retryable=retryable,
            duration_ms=duration_ms,
        )
