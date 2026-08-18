"""Regression: the opencode contract is curated + GLM-friendly (repl-style).

Mirrors the --repl's reliable behavior: only a small set of GLM-friendly core
tool names (run_command, list_dir, read_file, write_file, grep, edit) is shown
to GLM, and the parser maps those names back to opencode's real tool names +
arg keys. Specialized opencode tools (chrome_*, frida_*) are intentionally NOT
surfaced (they dilute / confuse GLM).
"""
import sys

from glm_rev.config import parse_tool_calls
from server import _tools_contract, _tool_hint_client, _contracted_intents

OPENCODE_TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Execute a shell command",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "glob", "description": "Find files by glob pattern",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a file",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}, "required": ["filePath"]}}},
    {"type": "function", "function": {"name": "write", "description": "Write a file",
     "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["filePath", "content"]}}},
    {"type": "function", "function": {"name": "edit", "description": "Edit a file by replacing text",
     "parameters": {"type": "object", "properties": {"old_string": {"type": "string"}, "new_string": {"type": "string"}},
                    "required": ["old_string", "new_string"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search file contents",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "chrome_navigate", "description": "Navigate browser",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "frida_spawn", "description": "Spawn a process",
     "parameters": {"type": "object", "properties": {"program": {"type": "string"}}}}},
]

results = []


def check(label, cond, detail=""):
    results.append((label, bool(cond), detail))
    print(("PASS" if cond else "FAIL"), label, ("" if cond else f"-> {detail}"))


# 1. Contract shows only the curated GLM-friendly core set.
contract = _tools_contract(OPENCODE_TOOLS)
check("contract shows run_command", "run_command" in contract, contract[:0])
check("contract shows list_dir", "list_dir" in contract)
check("contract shows read_file", "read_file" in contract)
check("contract shows write_file", "write_file" in contract)
check("contract shows grep", "grep" in contract)
check("contract shows edit", "edit" in contract)
check("contract omits chrome_navigate", "chrome_navigate" not in contract)
check("contract omits frida_spawn", "frida_spawn" not in contract)
check("contract omits raw bash name", "- bash:" not in contract and "TOOL:bash(" not in contract)

# 2. GLM-friendly calls map back to opencode's real tool names + args.
c = parse_tool_calls('TOOL:run_command({"cmd": "ls -la"})', known_tools=OPENCODE_TOOLS)
check("run_command -> bash", c and c[0][0] == "bash", c)
check("run_command args -> command", c and c[0][1] == {"command": "ls -la"}, c[0][1] if c else None)

c = parse_tool_calls('TOOL:list_dir({"path": "."})', known_tools=OPENCODE_TOOLS)
check("list_dir -> glob", c and c[0][0] == "glob", c)
check("list_dir path -> pattern *", c and c[0][1] == {"pattern": "*"}, c[0][1] if c else None)

c = parse_tool_calls('TOOL:read_file({"path": "/x"})', known_tools=OPENCODE_TOOLS)
check("read_file -> read", c and c[0][0] == "read", c)

c = parse_tool_calls('TOOL:write_file({"path": "/x", "content": "y"})', known_tools=OPENCODE_TOOLS)
check("write_file -> write", c and c[0][0] == "write", c)

c = parse_tool_calls('TOOL:grep({"pattern": "foo"})', known_tools=OPENCODE_TOOLS)
check("grep -> grep", c and c[0][0] == "grep", c)
check("grep args pass through", c and c[0][1] == {"pattern": "foo"}, c[0][1] if c else None)

c = parse_tool_calls('TOOL:edit({"old_string": "a", "new_string": "b"})', known_tools=OPENCODE_TOOLS)
check("edit -> edit", c and c[0][0] == "edit", c)
check("edit args pass through", c and c[0][1] == {"old_string": "a", "new_string": "b"}, c[0][1] if c else None)

# 3. Per-prompt TOOL_HINT mirrors the contracted set (repl parity, no KeyError
#    on edit which has no _TOOL_DISPLAY entry).
intents = _contracted_intents(OPENCODE_TOOLS)
check("contracted intents cover all 6 core", intents == {"exec", "list", "read", "write", "search", "edit"}, intents)
hint = _tool_hint_client(intents)
check("hint lists run_command", "TOOL:run_command(" in hint)
check("hint lists list_dir", "TOOL:list_dir(" in hint)
check("hint lists edit", "TOOL:edit(" in hint)
check("hint omits web_fetch (not contracted)", "web_fetch" not in hint)

failed = [r for r in results if not r[1]]
print("\n== RESULT ==")
if failed:
    for label, _, detail in failed:
        print("  FAIL:", label, "->", detail)
    sys.exit(1)
print("ALL PASS")
sys.exit(0)
