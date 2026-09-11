import json
import os
from pathlib import Path

from openai import OpenAI


BASE_URL = os.getenv("LOOP_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LOOP_API_KEY", "lm-studio")
MODEL = os.getenv("LOOP_MODEL", "qwen/qwen2.5-vl-7b")
WORKSPACE = Path.cwd().resolve()
MAX_TURNS = 8
MAX_TOOL_CALLS = 12

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

IGNORED_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
}


class ToolError(Exception):
    def __init__(self, error_type: str, message: str, retryable: bool):
        self.error_type = error_type
        self.message = message
        self.retryable = retryable


def safe_path(path: str) -> Path:
    resolved = (WORKSPACE / path).resolve()
    if not resolved.is_relative_to(WORKSPACE):
        raise ToolError(
            "permission",
            "Path escapes the workspace.",
            retryable=False,
        )
    return resolved


def list_dir(path: str) -> list[str]:
    target = safe_path(path)
    return sorted(
        p.name
        for p in target.iterdir()
        if p.name not in IGNORED_NAMES
    )


def read_file(path: str) -> str:
    target = safe_path(path)
    return target.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    target = safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {path}"


TOOL_FUNCTIONS = {
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
}

REQUIRED_ARGUMENTS = {
    "list_dir": ["path"],
    "read_file": ["path"],
    "write_file": ["path", "content"],
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List visible files and folders inside the workspace. "
                "Use this to inspect project structure. "
                "Path must be relative to the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file from the workspace. "
                "Use this when you need exact file contents. "
                "Path must be relative to the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a UTF-8 text file inside the workspace. "
                "Use this only when the task requires changing or creating a file. "
                "Path must be relative to the workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


def parse_arguments(raw_arguments: str) -> dict:
    try:
        return json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as error:
        raise ToolError(
            "invalid_tool_arguments",
            f"Arguments were not valid JSON: {error}",
            retryable=True,
        )


def require(arguments: dict, *names: str) -> None:
    missing = [name for name in names if name not in arguments]
    if missing:
        raise ToolError(
            "missing_arguments",
            f"Missing required arguments: {', '.join(missing)}",
            retryable=True,
        )


def validate_tool_call(name: str, arguments: dict) -> None:
    if name not in REQUIRED_ARGUMENTS:
        raise ToolError("unknown_tool", f"Unknown tool: {name}", False)
    require(arguments, *REQUIRED_ARGUMENTS[name])


def check_policy(name: str, arguments: dict) -> dict:
    if name == "write_file":
        path = arguments.get("path", "")
        if path.startswith(".loop/"):
            return {
                "ok": False,
                "tool": name,
                "error_type": "permission",
                "message": "The agent may not write into .loop internal state.",
                "retryable": False,
            }
    return {"ok": True}


def normalize_result(name: str, result) -> dict:
    return {"ok": True, "tool": name, "result": result}


def error_result(name: str, error: ToolError) -> dict:
    return {
        "ok": False,
        "tool": name,
        "error_type": error.error_type,
        "message": error.message,
        "retryable": error.retryable,
    }


def execute_tool(name: str, arguments: dict):
    return TOOL_FUNCTIONS[name](**arguments)


def handle_tool_call(tool_call) -> dict:
    name = tool_call.function.name
    try:
        arguments = parse_arguments(tool_call.function.arguments)
        validate_tool_call(name, arguments)
        policy = check_policy(name, arguments)
        if not policy["ok"]:
            return policy
        result = execute_tool(name, arguments)
        return normalize_result(name, result)
    except ToolError as error:
        return error_result(name, error)


def log_event(event: dict) -> None:
    print(json.dumps(event))


def call_model(messages: list[dict]):
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    return response.choices[0].message


def run_agent(messages: list[dict]) -> dict:
    tool_call_count = 0
    stop_reason = "max_turns"

    for turn in range(MAX_TURNS):
        log_event({"type": "model_call", "turn": turn})
        message = call_model(messages)
        messages.append(message.model_dump(exclude_none=True))

        if message.content:
            print("\nassistant:")
            print(message.content)

        if not message.tool_calls:
            stop_reason = "final_answer"
            break

        for tool_call in message.tool_calls:
            tool_call_count += 1
            if tool_call_count > MAX_TOOL_CALLS:
                stop_reason = "tool_budget_exceeded"
                break

            result = handle_tool_call(tool_call)
            log_event(
                {
                    "type": "tool_result",
                    "turn": turn,
                    "tool": result.get("tool"),
                    "ok": result["ok"],
                    "error_type": result.get("error_type"),
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        if stop_reason != "max_turns":
            break

    return {
        "messages": messages,
        "stop_reason": stop_reason,
        "tool_call_count": tool_call_count,
    }


messages = [
    {
        "role": "system",
        "content": (
            "You are a tiny coding agent. Use tools when you need to inspect "
            "or change the workspace. Paths must stay inside the workspace. "
            "When you have enough information, answer clearly."
        ),
    },
    {
        "role": "user",
        "content": (
            "Inspect notes.txt, then write a short summary to summary.txt. "
            "After writing it, read summary.txt back to confirm what you wrote."
        ),
    },
]


result = run_agent(messages)
print(f"Stopped because: {result['stop_reason']}")