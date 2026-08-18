"""Contract test for env-gated client-side tool calling in server.py.

Runs against a FastAPI TestClient WITHOUT entering the lifespan context (so the
Playwright warm solver never starts). All real upstream/captcha/network is
monkeypatched out of the server module.
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/home/bld/glm-rev")
from types import SimpleNamespace

# ----------------------------------------------------------------------------
# 0. Env gate ON for the main test process.
# ----------------------------------------------------------------------------
os.environ["GLM_CLIENT_TOOLS"] = "1"

# ----------------------------------------------------------------------------
# Load server.py as a module (no repo edits).
# ----------------------------------------------------------------------------
_SPEC = importlib.util.spec_from_file_location("server", "/home/bld/glm-rev/server.py")
server = importlib.util.module_from_spec(_SPEC)
sys.modules["server"] = server
_SPEC.loader.exec_module(server)

# ----------------------------------------------------------------------------
# 1. Monkeypatch module-level names to avoid real captcha / upstream.
# ----------------------------------------------------------------------------
server.refresh_token = lambda t: t

async def _fake_acquire_captcha(token):
    return ("fake_captcha", None)

server.acquire_captcha = _fake_acquire_captcha
server.create_chat = lambda *a, **k: ("chat_1", "msg_1")
server.sign = lambda *a, **k: ("sig", "", 123)
server.user_id_from_token = lambda token: "user_123"
server.load_mcp_config = lambda: None  # never connect real MCP servers

async def _fake_sync_graph(*a, **k):
    return (None, None)

server._sync_graph = _fake_sync_graph


class _FakeSSEResponse:
    def __init__(self, lines):
        self._lines = lines
        self.status_code = 200

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def _install_fake_upstream(sses):
    server.requests.post = lambda *a, **k: _FakeSSEResponse(sses)


TOOL_SSE = [
    'data: {"data": {"phase": "answer", "delta_content": "TOOL:run_command({\\"cmd\\": \\"ls\\"})", "id": "m1"}}',
    "data: [DONE]",
]
FINAL_SSE = [
    'data: {"data": {"phase": "thinking", "delta_content": "thinking...", "id": "m1"}}',
    'data: {"data": {"phase": "answer", "delta_content": "Here is the result", "id": "m2"}}',
    "data: [DONE]",
]

TOOLS = [{"type": "function", "function": {
    "name": "run_command",
    "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}}]


def _tool_req(stream):
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "list the directory"}],
        "tools": TOOLS,
        "stream": stream,
    }


def _final_req(stream):
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": stream,
    }


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


from fastapi.testclient import TestClient

client = TestClient(server.app)  # NO `with` -> lifespan does NOT run

results = []

# 1. Tool-call stream True
_install_fake_upstream(TOOL_SSE)
r = client.post("/v1/chat/completions", json=_tool_req(True))
chunks = _parse_sse(r.content)
tool_chunks = [c for c in chunks if c != "__DONE__" and
               c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
ok_tc = (
    r.status_code == 200
    and len(tool_chunks) >= 1
    and tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    and str(tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["id"]).startswith("call_")
    and tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["type"] == "function"
    and tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "run_command"
)
args = json.loads(tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"])
ok_args = isinstance(args, dict) and "cmd" in args
results.append(check("1. stream tool-call chunk shape+args", ok_tc and ok_args,
                      f"http={r.status_code}"))

# 2. finish_reason tool_calls + single [DONE]
fr = [c for c in chunks if c != "__DONE__" and c.get("choices", [{}])[0].get("finish_reason")]
final = fr[-1]["choices"][0]["finish_reason"] if fr else None
results.append(check("2. stream finish_reason=tool_calls + one DONE",
                      final == "tool_calls" and chunks.count("__DONE__") == 1,
                      f"final={final} done={chunks.count('__DONE__')}"))

# 3. Tool-call stream False
_install_fake_upstream(TOOL_SSE)
r = client.post("/v1/chat/completions", json=_tool_req(False))
j = r.json()
msg = j["choices"][0]["message"]
tc = msg.get("tool_calls")
ok = (r.status_code == 200 and tc and str(tc[0]["id"]).startswith("call_")
      and tc[0]["function"]["name"] == "run_command"
      and j["choices"][0]["finish_reason"] == "tool_calls")
results.append(check("3. non-stream tool_calls response", ok,
                      f"http={r.status_code}"))

# 4. Final-answer stream False
_install_fake_upstream(FINAL_SSE)
r = client.post("/v1/chat/completions", json=_final_req(False))
j = r.json()
m = j["choices"][0]["message"]
ok = (r.status_code == 200 and m.get("content") == "Here is the result"
      and m.get("reasoning_content") == "thinking..."
      and j["choices"][0]["finish_reason"] == "stop")
results.append(check("4. non-stream final answer+reasoning", ok,
                      f"http={r.status_code}"))

# 5. Final-answer stream True
_install_fake_upstream(FINAL_SSE)
r = client.post("/v1/chat/completions", json=_final_req(True))
chunks = _parse_sse(r.content)
content_chunks = [c for c in chunks if c != "__DONE__" and
                  c.get("choices", [{}])[0].get("delta", {}).get("content") == "Here is the result"]
reason_chunks = [c for c in chunks if c != "__DONE__" and
                 c.get("choices", [{}])[0].get("delta", {}).get("reasoning_content") == "thinking..."]
fr = [c for c in chunks if c != "__DONE__" and c.get("choices", [{}])[0].get("finish_reason")]
final = fr[-1]["choices"][0]["finish_reason"] if fr else None
ok = (r.status_code == 200 and len(content_chunks) >= 1 and len(reason_chunks) >= 1
      and final == "stop" and chunks.count("__DONE__") == 1)
results.append(check("5. stream final content+reasoning+stop", ok,
                      f"http={r.status_code} final={final} done={chunks.count('__DONE__')}"))

# 6. role:tool round-trip (tool-call SSE again, must not error)
_install_fake_upstream(TOOL_SSE)
rt_req = {
    "model": "gpt-4o",
    "tools": TOOLS,
    "stream": False,
    "messages": [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_0", "type": "function", "function": {"name": "run_command", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "name": "run_command", "content": "ok"},
    ],
}
r = client.post("/v1/chat/completions", json=rt_req)
j = r.json()
ok = (r.status_code == 200 and "finish_reason" in j.get("choices", [{}])[0]
      and not (400 <= r.status_code <= 599))
results.append(check("6. role:tool round-trip returns 200", ok,
                      f"http={r.status_code}"))

# 7. Default routing: with no GLM_SERVER_TOOLS env, tools request routes TO client tools.
server._run_client_tools_called = False
_orig_run_client_tools = server._run_client_tools

def _record(*a, **k):
    server._run_client_tools_called = True
    return _orig_run_client_tools(*a, **k)

server._run_client_tools = _record
saved_env = os.environ.pop("GLM_CLIENT_TOOLS", None)
saved_server_env = os.environ.pop("GLM_SERVER_TOOLS", None)

try:
    r = client.post("/v1/chat/completions", json=_tool_req(False))
    results.append(check("7. env-unset defaults TO client tools",
                          server._run_client_tools_called is True,
                          f"client_called={server._run_client_tools_called} http={r.status_code}"))
finally:
    if saved_env is not None:
        os.environ["GLM_CLIENT_TOOLS"] = saved_env
    if saved_server_env is not None:
        os.environ["GLM_SERVER_TOOLS"] = saved_server_env
    server._run_client_tools = _orig_run_client_tools
    del server.send_with_tools

# ----------------------------------------------------------------------------
fails = [n for n, ok in enumerate(results, 1) if not ok]
print(f"\nRESULT: {len(results) - len(fails)}/{len(results)} passed")
if fails:
    print(f"FAILED checks: {fails}")
    sys.exit(1)
print("ALL PASS")