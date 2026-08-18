"""Regression: GLM-5.2 tool calls are parsed into OpenAI tool_calls that fire.

Covers the critical preamble-detection bug (prose before a TOOL: call must not
downgrade the call to content), the native <tool_call>+```bash fallback, name
normalization / arg remap against known_tools, skipping unmatched names, a
2-turn tool round-trip, and parallel tool calls.

Requires the production change where parse_tool_calls(text, known_tools=None)
resolves names/args against the provided tool list and server.py BUFFERS the
whole answer before deciding tool-vs-content at end-of-stream. Both are
expected to be landed in glm_rev/config.py and server.py.

Runs standalone:
    python tests/regression/test_tool_detection.py
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/home/bld/glm-rev")

os.environ["GLM_CLIENT_TOOLS"] = "1"
os.environ.pop("GLM_SERVER_TOOLS", None)

# ----------------------------------------------------------------------------
# Load server.py as a module (no repo edits).
# ----------------------------------------------------------------------------
_SPEC = importlib.util.spec_from_file_location("server", "/home/bld/glm-rev/server.py")
server = importlib.util.module_from_spec(_SPEC)
sys.modules["server"] = server
_SPEC.loader.exec_module(server)

server.refresh_token = lambda t: t

async def _fake_acquire_captcha(token):
    return ("fake_captcha", None)

server.acquire_captcha = _fake_acquire_captcha
server.create_chat = lambda *a, **k: ("chat_1", "msg_1")
server.sign = lambda *a, **k: ("sig", "", 123)
server.user_id_from_token = lambda token: "user_123"
server.load_mcp_config = lambda: None


class _FakeSSEResponse:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def _install_fake_upstream(sses):
    server.requests.post = lambda *a, **k: _FakeSSEResponse(sses)


def _parse_sse(body):
    out = []
    for raw in body.decode("utf-8").split("\n"):
        raw = raw.strip()
        if not raw.startswith("data:"):
            continue
        d = raw[5:].strip()
        if d == "[DONE]":
            out.append("__DONE__")
        else:
            try:
                out.append(json.loads(d))
            except Exception:
                pass
    return out


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {extra}" if extra else ""))
    return cond


RUN_CMD = [{"type": "function", "function": {
    "name": "run_command", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}}]
EXEC_CMD = [{"type": "function", "function": {
    "name": "execute_command", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
}}]
READ_FILE = [{"type": "function", "function": {
    "name": "read_file", "description": "read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}}]


def _tool_req(stream, tools, content="list the directory"):
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
        "tools": tools,
        "stream": stream,
    }


def _tool_calls_from_chunks(chunks):
    """Return all (delta_index, name, args_str) triples across tool_calls deltas."""
    out = []
    for c in chunks:
        if c == "__DONE__":
            continue
        delta = c.get("choices", [{}])[0].get("delta", {})
        for tc in delta.get("tool_calls") or []:
            fn = tc.get("function", {})
            out.append((tc.get("index"), fn.get("name"), fn.get("arguments")))
    return out


def _finish_reason(chunks):
    fr = [c for c in chunks if c != "__DONE__" and c.get("choices", [{}])[0].get("finish_reason")]
    return fr[-1]["choices"][0]["finish_reason"] if fr else None


from fastapi.testclient import TestClient

client = TestClient(server.app)  # NO `with` -> lifespan does NOT run

fails = []

# ============================================================================
# A. Preamble detection (the critical bug).
# ============================================================================
PREAMBLE_SSE = [
    'data: {"data": {"phase": "thinking", "delta_content": "Let", "id": "t0"}}',
    'data: {"data": {"phase": "thinking", "delta_content": " me", "id": "t1"}}',
    'data: {"data": {"phase": "thinking", "delta_content": " plan.", "id": "t2"}}',
    'data: {"data": {"phase": "answer", "delta_content": "Let me check that for you. ", "id": "x1"}}',
    'data: {"data": {"phase": "answer", "delta_content": "TOOL:run_command({\\"cmd\\": \\"ls -la\\"})", "id": "x2"}}',
    "data: [DONE]",
]
_install_fake_upstream(PREAMBLE_SSE)
r = client.post("/v1/chat/completions", json=_tool_req(True, RUN_CMD))
chunks = _parse_sse(r.content)
tc = _tool_calls_from_chunks(chunks)
fr = _finish_reason(chunks)
ok_a = (
    r.status_code == 200
    and len(tc) == 1
    and tc[0][1] == "run_command"
    and fr == "tool_calls"
)
try:
    a_args = json.loads(tc[0][2]) if tc else {}
except Exception:
    a_args = {}
ok_a = ok_a and ("cmd" in a_args or "command" in a_args)
fails.append(("A", ok_a))
check("A preamble does not downgrade TOOL call to content", ok_a,
      f"n_tc={len(tc)} name={tc[0][1] if tc else None} finish={fr} args={a_args} http={r.status_code}")

# ============================================================================
# B. Native <tool_call> + ```bash fallback (the drop-bug regression).
# ============================================================================
BASH_SSE = [
    'data: {"data": {"phase": "answer", "delta_content": "<tool_call>\\n```bash\\nls -la /tmp\\n```\\n</tool_call>", "id": "y1"}}',
    "data: [DONE]",
]
_install_fake_upstream(BASH_SSE)
r = client.post("/v1/chat/completions", json=_tool_req(True, EXEC_CMD))
chunks = _parse_sse(r.content)
tc = _tool_calls_from_chunks(chunks)
fr = _finish_reason(chunks)
ok_b = (
    r.status_code == 200
    and len(tc) == 1
    and tc[0][1] == "execute_command"
    and fr == "tool_calls"
)
try:
    b_args = json.loads(tc[0][2]) if tc else {}
except Exception:
    b_args = {}
ok_b = ok_b and "command" in b_args and b_args["command"] == "ls -la /tmp"
fails.append(("B", ok_b))
check("B native <tool_call>+```bash maps to execute_command", ok_b,
      f"name={tc[0][1] if tc else None} args={b_args} finish={fr} http={r.status_code}")

# ============================================================================
# C. Name normalization (case-insensitive + synonym resolution).
# ============================================================================
from glm_rev.config import parse_tool_calls

c1 = parse_tool_calls('TOOL:ReadFile({"path": "/etc/hosts"})', known_tools=READ_FILE)
ok_c1 = len(c1) == 1 and c1[0][0] == "read_file"
fails.append(("C1", ok_c1))
check("C1 case-insensitive name resolution -> read_file", ok_c1, repr(c1))

c2 = parse_tool_calls('TOOL:cat("/etc/hosts")', known_tools=READ_FILE)
ok_c2 = len(c2) == 1 and c2[0][0] == "read_file"
fails.append(("C2", ok_c2))
check("C2 synonym resolution cat -> read_file", ok_c2, repr(c2))

# ============================================================================
# D. Argument remap (cmd -> command).
# ============================================================================
d = parse_tool_calls('TOOL:bash({"cmd": "whoami"})', known_tools=EXEC_CMD)
ok_d = (len(d) == 1 and d[0][0] == "execute_command" and d[0][1] == {"command": "whoami"})
fails.append(("D", ok_d))
check("D arg remap cmd->command for execute_command", ok_d, repr(d))

# ============================================================================
# E. Skip unmatched names (no phantom tool_calls).
# ============================================================================
e = parse_tool_calls('TOOL:totally_unknown({"a": 1})', known_tools=READ_FILE)
ok_e = e == []
fails.append(("E", ok_e))
check("E unmatched tool name skipped -> empty list", ok_e, repr(e))

# ============================================================================
# F. 2-turn round-trip: tool result reaches the final answer.
# ============================================================================
# Turn 1: upstream emits a tool call -> server replies with a tool_calls chunk.
_install_fake_upstream(PREAMBLE_SSE)
r1 = client.post("/v1/chat/completions", json={
    "model": "gpt-4o",
    "tools": RUN_CMD,
    "stream": False,
    "messages": [{"role": "user", "content": "how many files?"}],
})
j1 = r1.json()
tc1 = j1.get("choices", [{}])[0].get("message", {}).get("tool_calls") or []
tool_id = tc1[0]["id"] if tc1 else None
ok_f1 = r1.status_code == 200 and len(tc1) == 1 and tc1[0]["function"]["name"] == "run_command"
fails.append(("F1", ok_f1))
check("F1 turn-1 emits a tool_calls entry", ok_f1,
      f"n={len(tc1)} name={tc1[0]['function']['name'] if tc1 else None} http={r1.status_code}")

# Turn 2: role:tool result comes back; fake upstream returns the final answer.
FINAL_SSE = [
    'data: {"data": {"phase": "thinking", "delta_content": "let", "id": "z0"}}',
    'data: {"data": {"phase": "answer", "delta_content": "The command returned: 5 files", "id": "z1"}}',
    "data: [DONE]",
]
_install_fake_upstream(FINAL_SSE)
r2 = client.post("/v1/chat/completions", json={
    "model": "gpt-4o",
    "tools": RUN_CMD,
    "stream": True,
    "messages": [
        {"role": "user", "content": "how many files?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": tool_id, "type": "function",
             "function": {"name": "run_command", "arguments": "{\"cmd\": \"ls\"}"}}]},
        {"role": "tool", "tool_call_id": tool_id, "name": "run_command", "content": "5 files"},
    ],
})
chunks2 = _parse_sse(r2.content)
content2 = "".join(
    c["choices"][0]["delta"].get("content") or ""
    for c in chunks2 if c != "__DONE__" and c.get("choices", [{}])[0].get("delta", {}).get("content")
)
ok_f2 = (r2.status_code == 200 and "5 files" in content2)
fails.append(("F2", ok_f2))
check("F2 turn-2 tool result reaches final answer", ok_f2,
      f"content={content2!r} http={r2.status_code}")

# ============================================================================
# G. Parallel tool calls parsed as multiple tuples.
# ============================================================================
TOOLS_PAR = READ_FILE + [{"type": "function", "function": {
    "name": "list_dir", "description": "list a dir",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}}]
g = parse_tool_calls(
    'TOOL:read_file({"path": "/a"})\nTOOL:list_dir({"path": "/b"})',
    known_tools=TOOLS_PAR,
)
names_g = [n for n, _, _ in g]
paths_g = sorted(a.get("path") for n, a, _ in g)
ok_g = len(g) == 2 and names_g == ["read_file", "list_dir"] and paths_g == ["/a", "/b"]
fails.append(("G", ok_g))
check("G two parallel TOOL lines -> 2 tuples", ok_g, repr(g))

# ============================================================================
# H. Greeting with tools present -> plain content, NO tool_calls, NO doc dump.
# Regression for GLM hallucinating dummy Assistant-API docs on "hi" when the
# contract forced "MUST invoke a tool" every turn.
# ============================================================================
GREET_SSE = [
    'data: {"data": {"phase": "answer", "delta_content": "Hi there! How can I help you today?", "id": "h1"}}',
    "data: [DONE]",
]
_install_fake_upstream(GREET_SSE)
r = client.post("/v1/chat/completions", json=_tool_req(True, READ_FILE + RUN_CMD, content="hi"))
chunks = _parse_sse(r.content)
tc_h = _tool_calls_from_chunks(chunks)
fr_h = _finish_reason(chunks)
content_h = "".join(
    c["choices"][0]["delta"].get("content") or ""
    for c in chunks if c != "__DONE__" and c.get("choices", [{}])[0].get("delta", {}).get("content")
)
ok_h = (
    r.status_code == 200
    and len(tc_h) == 0
    and fr_h == "stop"
    and "hi" in content_h.lower()
    and "api.example.com" not in content_h
    and "TOOL:" not in content_h
)
fails.append(("H", ok_h))
check("H greeting with tools -> content, no tool_calls/no doc dump", ok_h,
      f"n_tc={len(tc_h)} finish={fr_h} content={content_h!r} http={r.status_code}")

# ============================================================================
print()
bad = [k for k, ok in fails if not ok]
if bad:
    print(f"FAILED checks: {bad}")
    sys.exit(1)
print("ALL PASS")
