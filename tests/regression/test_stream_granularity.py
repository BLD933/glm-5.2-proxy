"""Streaming granularity test for _run_client_tools.

Verifies the streaming refactor emits reasoning as MULTIPLE incremental chunks
(not one monolithic blob), streams the final answer content after all reasoning
(the server BUFFERS the answer to decide tool-vs-content at end-of-stream, so
content is not emitted live chunk-by-chunk), and still emits a single
structured tool_calls chunk for tool output.

Runs against a FastAPI TestClient WITHOUT entering the lifespan context.
All real upstream/captcha/network is monkeypatched out.
"""
import importlib.util
import json
import os
import sys

sys.path.insert(0, "/home/bld/glm-rev")

os.environ["GLM_CLIENT_TOOLS"] = "1"

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


# Multi-delta thinking then multi-delta answer (normal text, no tool call)
MULTI_SSE = [
    'data: {"data": {"phase": "thinking", "delta_content": "Let", "id": "t1"}}',
    'data: {"data": {"phase": "thinking", "delta_content": " me", "id": "t2"}}',
    'data: {"data": {"phase": "thinking", "delta_content": " think...", "id": "t3"}}',
    'data: {"data": {"phase": "answer", "delta_content": "The res", "id": "a1"}}',
    'data: {"data": {"phase": "answer", "delta_content": "ult is 4", "id": "a2"}}',
    'data: {"data": {"phase": "answer", "delta_content": "2.", "id": "a3"}}',
    "data: [DONE]",
]

# Tool call arrives as TOOL: line (single answer delta)
TOOL_SSE = [
    'data: {"data": {"phase": "answer", "delta_content": "TOOL:run_command({\\"cmd\\": \\"ls\\"})", "id": "m1"}}',
    "data: [DONE]",
]

TOOLS = [{"type": "function", "function": {
    "name": "run_command", "description": "run a shell command",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
}}]


def _tool_req(stream):
    return {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "list the directory"}],
        "tools": TOOLS,
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

# 1. Reasoning streamed as multiple incremental chunks (not one monolithic blob)
_install_fake_upstream(MULTI_SSE)
r = client.post("/v1/chat/completions", json=_tool_req(True))
chunks = _parse_sse(r.content)
reason_chunks = [c for c in chunks if c != "__DONE__" and
                 c.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")]
ok1 = (r.status_code == 200 and len(reason_chunks) == 3
       and "".join(c["choices"][0]["delta"]["reasoning_content"] for c in reason_chunks) == "Let me think...")
results.append(check("1. reasoning streamed as 3 incremental chunks", ok1,
                      f"n={len(reason_chunks)} http={r.status_code}"))

# 2. Content arrives after all reasoning (buffered at end-of-stream) and the
#    joined content equals the expected answer. The server now BUFFERS the whole
#    answer and emits it once it has decided tool-vs-content, so content may
#    arrive as one or more chunks — not necessarily 3 live incremental ones.
content_chunks = [c for c in chunks if c != "__DONE__" and
                  c.get("choices", [{}])[0].get("delta", {}).get("content")]
full_content = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)
ok2 = (r.status_code == 200 and len(content_chunks) >= 1 and full_content == "The result is 42.")
results.append(check("2. content arrives buffered, joined equals full answer", ok2,
                      f"n={len(content_chunks)} full={full_content!r}"))

# 3. Correct phase ordering: reasoning deltas all before content deltas
reason_idxs = [chunks.index(c) for c in reason_chunks]
content_idxs = [chunks.index(c) for c in content_chunks]
ok3 = max(reason_idxs) < min(content_idxs)
results.append(check("3. reasoning emitted before content (phase order)", ok3))

# 4. Tool call: exactly ONE tool_calls chunk + finish_reason tool_calls + single DONE
_install_fake_upstream(TOOL_SSE)
r = client.post("/v1/chat/completions", json=_tool_req(True))
chunks = _parse_sse(r.content)
tool_chunks = [c for c in chunks if c != "__DONE__" and
               c.get("choices", [{}])[0].get("delta", {}).get("tool_calls")]
fr = [c for c in chunks if c != "__DONE__" and c.get("choices", [{}])[0].get("finish_reason")]
final = fr[-1]["choices"][0]["finish_reason"] if fr else None
ok4 = (
    r.status_code == 200
    and len(tool_chunks) == 1
    and tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "run_command"
    and final == "tool_calls"
    and chunks.count("__DONE__") == 1
)
results.append(check("4. tool call = single tool_calls chunk + finish_reason", ok4,
                      f"n_tc={len(tool_chunks)} final={final}"))

fails = [n for n, ok in enumerate(results, 1) if not ok]
print(f"\nRESULT: {len(results) - len(fails)}/{len(results)} passed")
if fails:
    print(f"FAILED checks: {fails}")
    sys.exit(1)
print("ALL PASS")
