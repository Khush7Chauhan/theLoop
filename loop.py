import json
import os
from pathlib import Path

from openai import OpenAI


BASE_URL = os.getenv("LOOP_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LOOP_API_KEY", "lm-studio")
MODEL = os.getenv("LOOP_MODEL", "qwen/qwen3-coder-30b")
WORKSPACE = Path.cwd()

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


tools = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "List files in a workspace directory. "
                "Use this when you need to discover file names. "
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
                "For listing filenames, use list_dir instead. "
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
                "Use this when you need to create or replace a whole file. "
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


def safe_path(path: str) -> Path:
    resolved = (WORKSPACE / path).resolve()
    if not resolved.is_relative_to(WORKSPACE):
        raise ValueError("Path escapes the workspace")
    return resolved


def list_dir(path: str) -> str:
    target = safe_path(path)
    return "\n".join(sorted(p.name for p in target.iterdir()))


def read_file(path: str) -> str:
    return safe_path(path).read_text()


def write_file(path: str, content: str) -> str:
    target = safe_path(path)
    target.write_text(content)
    return f"Wrote {len(content)} characters to {path}"


def run_tool(name: str, arguments: dict) -> str:
    try:
        if name == "list_dir":
            return list_dir(arguments["path"])
        if name == "read_file":
            return read_file(arguments["path"])
        if name == "write_file":
            return write_file(arguments["path"], arguments["content"])
        return f"Unknown tool: {name}"
    except Exception as error:
        return json.dumps(
            {
                "ok": False,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )


messages = [
    {
        "role": "system",
        "content": (
            "You are a tiny coding agent. Use tools when you need to inspect "
            "or change the workspace. When you have enough information, "
            "answer clearly."
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


while True:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    message = response.choices[0].message
    messages.append(message.model_dump(exclude_none=True))

    if message.content:
        print("\nassistant:")
        print(message.content)

    if not message.tool_calls:
        break

    for tool_call in message.tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments or "{}")

        print(f"\ntool call: {name}({arguments})")
        result = run_tool(name, arguments)
        print(f"tool result:\n{result}")

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
        )