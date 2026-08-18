"""test_parallel_tools.py — New behaviors for the opencode harness:
  1. glm_rev.config.parse_tool_calls accepts markdown ```json blocks and XML
     <tool_call> tags (F1) and multi-line prompt args (F7), not just TOOL: lines.
  2. Client-side tool relay (F3/F4) emits MULTIPLE parallel tool_calls with
     UNIQUE UUID call_ ids and attaches upstream usage.

  Part B runs against a FastAPI TestClient WITHOUT lifespan (no warm solver);
  all upstream/captcha/network is monkeypatched out.
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/home/bld/glm-rev")

# ---- Part A: real parser accepts md / XML / multi-line args ----
from glm_rev.config import parse_tool_calls

fails = []
def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" -- {extra}" if extra else ""))
    if not cond:
        fails.append(name)
    return cond

# A1: markdown fenced JSON block with name+arguments
md_text = (
    'Thought: I need to read a file.\n'
    '```json\n'
    '{"name": "read_file", "arguments": {"path": "/tmp/a.py"}}\n'
    '```\n'
)
a1 = parse_tool_calls(md_text)
check("A1 markdown json block", any(n == "read_file" and a.get("path") == "/tmp/a.py" for n, a, _ in a1), repr(a1))

# A2: XML <tool_call name=...> with inner JSON args
xml_text = (
    '<tool_call name="task">\n'
    '  {"description": "summarize", "prompt": "Read server.py"}\n'
    '</tool_call>\n'
)
a2 = parse_tool_calls(xml_text)
check("A2 xml tool_call tag", any(n == "task" and a.get("description") == "summarize" for n, a, _ in a2), repr(a2))

# A3: XML empty self-closing tag
a3 = parse_tool_calls('<tool name="list_dir"/>')
check("A3 xml empty tool tag", any(n == "list_dir" for n, a, _ in a3), repr(a3))

# A4: multi-line prompt arg (unescaped newlines inside a JSON string) parses intact
ml = 'TOOL:task({"description": "spawn", "prompt": "Line one\\nLine two\\nLine three", "subagent_type": "general"})'
a4 = parse_tool_calls(ml)
ok4 = False
for n, a, _ in a4:
    if n == "task":
        ok4 = a.get("prompt") == "Line one\nLine two\nLine three" and a.get("description") == "spawn"
check("A4 multiline prompt arg", ok4, repr(a4))

# A5: two parallel TOOL: calls on one line -> BOTH returned (multi-call loop)
a5 = parse_tool_calls('TOOL:read_file({"path": "/a"}) TOOL:read_file({"path": "/b"})')
names5 = [n for n, _, _ in a5]
paths5 = sorted(a.get("path") for n, a, _ in a5 if n == "read_file")
check("A5 two parallel calls", names5 == ["read_file", "read_file"] and paths5 == ["/a", "/b"], repr(a5))

# ---- Part B: client-side relay, multiple tool_calls + unique ids + usage ----
os.environ["GLM_CLIENT_TOOLS"] = "1"
os.environ.pop("GLM_SERVER_TOOLS", None)

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

TWO_CALL_SSE = [
    'data: {"data": {"phase": "thinking", "delta_content": "spawning", "id": "m0"}}',
    'data: {"data": {"phase": "answer", "delta_content": "TOOL:read_file({\\"path\\": \\"/a\\"}) TOOL:read_file({\\"path\\": \\"/b\\"})", "id": "m1"}}',
    'data: {"data": {"usage": {"prompt_tokens": 3, "completion_tokens": 5}, "id": "m2"}}',
    "data: [DONE]",
]
server.requests.post = lambda *a, **k: _FakeSSEResponse(TWO_CALL_SSE)

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
}}]

from fastapi.testclient import TestClient
client = TestClient(server.app)

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

# B1: non-stream response -> multiple tool_calls, unique UUID ids, usage present
r = client.post("/v1/chat/completions", json={
    "model": "gpt-4o",
    "stream": False,
    "tools": TOOLS,
    "messages": [{"role": "user", "content": "read a and b"}],
})
j = r.json()
msg = j.get("choices", [{}])[0].get("message", {})
tc = msg.get("tool_calls") or []
b1 = (
    r.status_code == 200
    and len(tc) == 2
    and all(t["id"].startswith("call_") for t in tc)
    and len({t["id"] for t in tc}) == 2
    and tc[0]["id"] != tc[1]["id"]
    and j["choices"][0]["finish_reason"] == "tool_calls"
)
check("B1 non-stream 2 calls unique ids", b1, f"ids={[t['id'] for t in tc]} http={r.status_code}")
b1u = bool(j.get("usage")) and j["usage"].get("total_tokens") == 8
check("B1b usage present+normalized", b1u, f"usage={j.get('usage')}")
b1r = bool(msg.get("reasoning_content"))
check("B1c reasoning_content preserved", b1r, repr(msg.get("reasoning_content"))[:40])

# B2: stream -> tool_calls delta has unique ids; finish has usage
r = client.post("/v1/chat/completions", json={
    "model": "gpt-4o",
    "stream": True,
    "tools": TOOLS,
    "messages": [{"role": "user", "content": "read a and b"}],
})
chunks = _parse_sse(r.content)
tc_chunks = [c for c in chunks if c != "__DONE__" and
             c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
tc_all = [tc for c in tc_chunks for tc in c["choices"][0]["delta"]["tool_calls"]]
b2 = (r.status_code == 200 and len(tc_all) == 2
      and all(t["id"].startswith("call_") for t in tc_all)
      and len({t["id"] for t in tc_all}) == 2)
check("B2 stream 2 calls unique ids", b2, f"ids={[t['id'] for t in tc_all]} http={r.status_code}")
usage_chunks = [c for c in chunks if c != "__DONE__" and c.get("usage")]
b2u = bool(usage_chunks) and any(u.get("usage", {}).get("total_tokens") == 8 for u in usage_chunks)
check("B2b stream usage present", b2u, f"n_usage_chunks={len(usage_chunks)}")

print()
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL PASS")
