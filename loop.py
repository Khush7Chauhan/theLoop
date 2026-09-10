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
            "description": "List files in the current workspace.",
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
            "description": "Read a text file from the current workspace.",
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


def run_tool(name: str, arguments: dict) -> str:
    if name == "list_dir":
        return list_dir(arguments["path"])
    if name == "read_file":
        return read_file(arguments["path"])
    return f"Unknown tool: {name}"


messages = [
    {
        "role": "system",
        "content": (
            "You are a tiny coding agent. Use tools when you need to inspect "
            "the workspace. When you have enough information, answer clearly."
        ),
    },
    {
        "role": "user",
        "content": "What files are here, and what does notes.txt say?",
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