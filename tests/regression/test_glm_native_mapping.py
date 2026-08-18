import sys
sys.path.insert(0, ".")

from glm_rev.config import parse_tool_calls

# opencode's real tool catalog (OpenAI shape) — the client toolset the server
# validates against. GLM-5.2 emits native <tool_call> with hallucinated/generic
# names; the parser must map them onto these real tools.
CLIENT_TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command on the user's machine",
        "parameters": {"type": "object",
                        "properties": {"command": {"type": "string"},
                                        "timeout": {"type": "integer"},
                                        "workdir": {"type": "string"}},
                        "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Read a file from the local filesystem",
        "parameters": {"type": "object",
                        "properties": {"filePath": {"type": "string"},
                                        "offset": {"type": "integer"},
                                        "limit": {"type": "integer"}},
                        "required": ["filePath"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Write a file to the local filesystem",
        "parameters": {"type": "object",
                        "properties": {"content": {"type": "string"},
                                        "filePath": {"type": "string"}},
                        "required": ["content", "filePath"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Edit a file by replacing text",
        "parameters": {"type": "object",
                        "properties": {"filePath": {"type": "string"},
                                        "oldString": {"type": "string"},
                                        "newString": {"type": "string"},
                                        "replaceAll": {"type": "boolean"}},
                        "required": ["filePath", "oldString", "newString"]}}},
    {"type": "function", "function": {
        "name": "glob", "description": "Fast file pattern matching tool",
        "parameters": {"type": "object",
                        "properties": {"pattern": {"type": "string"},
                                        "path": {"type": "string"}},
                        "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Fast content search tool",
        "parameters": {"type": "object",
                        "properties": {"pattern": {"type": "string"},
                                        "path": {"type": "string"},
                                        "include": {"type": "string"}},
                        "required": ["pattern"]}}},
]


def check(text, expected_name, expected_args_substr=None):
    calls = parse_tool_calls(text, CLIENT_TOOLS)
    assert calls, f"no calls parsed from: {text!r}"
    name, args, _ = calls[0]
    assert name == expected_name, f"{text!r} -> name {name!r} != {expected_name!r}"
    if expected_args_substr is not None:
        for k, v in expected_args_substr.items():
            assert k in args and args[k] == v, \
                f"{text!r} -> args {args!r} missing {k}={v!r}"
    print(f"[PASS] {text[:60]!r} -> ({name}, {args})")


# GLM native <tool_call> with hallucinated names must map to real opencode tools
check('<tool_call>\n{"name": "list_files", "arguments": {"path": "."}}\n</tool_call>',
      "glob", {"path": "."})
# glob must get a default pattern
calls = parse_tool_calls(
    '<tool_call>\n{"name": "list_files", "arguments": {"path": "."}}\n</tool_call>',
    CLIENT_TOOLS)
assert calls[0][1].get("pattern") == "*", f"glob missing default pattern: {calls[0][1]}"

check('<tool_call>\n{"name": "read_file", "arguments": {"path": "/etc/hostname"}}\n</tool_call>',
      "read", {"filePath": "/etc/hostname"})
check('<tool_call>\n{"name": "run_command", "arguments": {"command": "ls -la"}}\n</tool_call>',
      "bash", {"command": "ls -la"})
check('<tool_call>\n{"name": "write_file", "arguments": {"path": "a.txt", "content": "hi"}}\n</tool_call>',
      "write", {"filePath": "a.txt", "content": "hi"})

# Already-correct names pass through unchanged
check('<tool_call>\n{"name": "bash", "arguments": {"command": "pwd"}}\n</tool_call>',
      "bash", {"command": "pwd"})
check('TOOL:bash({"command": "pwd"})', "bash", {"command": "pwd"})

# Plain text must not produce any tool call (prefer missing over misfire)
assert parse_tool_calls("just chat, no tool", CLIENT_TOOLS) == [], "plain text fired a tool"
# Unknown name with no intent match must be dropped
assert parse_tool_calls('<tool_call>\n{"name": "frobnicate", "arguments": {}}\n</tool_call>',
                        CLIENT_TOOLS) == [], "unknown name should be dropped"

print("=== GLM native -> opencode tool mapping tests passed ===")
