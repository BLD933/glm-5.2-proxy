import sys
import json

sys.path.insert(0, "/home/bld/glm-rev")
import server
from fastapi.testclient import TestClient

# ---- monkeypatches (no browser, no network, no captcha solver) ----
server.refresh_token = lambda token: token


async def _fake_acquire_captcha(token):
    return ("fake_captcha", None)


server.acquire_captcha = _fake_acquire_captcha
server.create_chat = lambda *a, **k: ("chat_1", "msg_1")
server.sign = lambda *a, **k: ("sig", "", 123)

SSE_LINES = [
    'data: {"data": {"phase": "thinking", "delta_content": "Let me think", "id": "m1"}}',
    'data: {"data": {"phase": "answer", "delta_content": "Hello", "id": "m2"}}',
    'data: {"data": {"phase": "answer", "delta_content": " world", "id": "m3", "usage": {"in": 12, "out": 5}}}',
    "data: [DONE]",
]


class FakeResponse:
    status_code = 200
    text = ""

    def iter_lines(self, decode_unicode=True):
        return iter(SSE_LINES)


class FakeRequests:
    def post(self, url, *args, **kwargs):
        return FakeResponse()


server.requests = FakeRequests()

# Plain construction -> lifespan (Playwright warm-up) does NOT run.
client = TestClient(server.app)


def parse_sse(resp):
    events = []
    done_count = 0
    for line in resp.text.split("\n"):
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done_count += 1
            events.append("__DONE__")
        else:
            try:
                events.append(json.loads(payload))
            except Exception:
                pass
    return events, done_count


def sse_choice(ev):
    return ev["choices"][0]


results = []


def check(name, ok, actual=None, expected=None):
    results.append(ok)
    status = "PASS" if ok else "FAIL"
    print(f"{status}: {name}")
    if not ok:
        print(f"      actual:   {actual!r}")
        print(f"      expected: {expected!r}")


# ---- 1. Streaming basics ----
r = client.post("/v1/chat/completions", json={
    "model": "glm-5.2", "stream": True,
    "messages": [{"role": "user", "content": "hi"}]})
events, done = parse_sse(r)
first = events[0]
check("1. streaming: HTTP 200", r.status_code == 200, r.status_code, 200)
check("1. streaming: first delta.role == 'assistant', content == ''",
      sse_choice(first).get("delta", {}).get("role") == "assistant"
      and sse_choice(first).get("delta", {}).get("content") == "",
      sse_choice(first).get("delta"), {"role": "assistant", "content": ""})

# ---- 2. reasoning_content + content concat ----
reason = [sse_choice(e).get("delta", {}).get("reasoning_content")
          for e in events if isinstance(e, dict)
          and "reasoning_content" in sse_choice(e).get("delta", {})]
content = "".join(sse_choice(e).get("delta", {}).get("content", "")
                  for e in events if isinstance(e, dict)
                  and sse_choice(e).get("delta", {}).get("content"))
check("2. streaming: reasoning_content == 'Let me think'",
      "Let me think" in reason, reason, ["Let me think"])
check("2. streaming: content concatenates to 'Hello world'",
      content == "Hello world", content, "Hello world")

# ---- 3. finish chunk before [DONE] ----
dpos = events.index("__DONE__") if "__DONE__" in events else -1
finish = events[dpos - 1] if dpos > 0 else {}
fc = sse_choice(finish) if isinstance(finish, dict) else {}
usage = finish.get("usage") if isinstance(finish, dict) else None
check("3. streaming: chunk before [DONE] finish_reason == 'stop'",
      fc.get("finish_reason") == "stop", fc.get("finish_reason"), "stop")
check("3. streaming: usage with prompt/completion/total tokens",
      isinstance(usage, dict)
      and all(k in usage for k in ("prompt_tokens", "completion_tokens", "total_tokens")),
      usage, {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17})

# ---- 4. exactly one [DONE] ----
check("4. exactly one 'data: [DONE]' line", done == 1, done, 1)

# ---- 5. non-streaming ----
r = client.post("/v1/chat/completions", json={
    "model": "glm-5.2", "stream": False,
    "messages": [{"role": "user", "content": "hi"}]})
j = r.json()
msg = j["choices"][0]["message"]
check("5. non-streaming: message.content == 'Hello world'",
      msg.get("content") == "Hello world", msg.get("content"), "Hello world")
check("5. non-streaming: message.reasoning_content == 'Let me think'",
      msg.get("reasoning_content") == "Let me think", msg.get("reasoning_content"), "Let me think")
check("5. non-streaming: finish_reason == 'stop'",
      j["choices"][0]["finish_reason"] == "stop", j["choices"][0]["finish_reason"], "stop")
check("5. non-streaming: usage present",
      isinstance(j.get("usage"), dict)
      and all(k in j["usage"] for k in ("prompt_tokens", "completion_tokens", "total_tokens")),
      j.get("usage"), {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17})

# ---- 6. model alias ----
r = client.post("/v1/chat/completions", json={
    "model": "gpt-4o", "stream": False,
    "messages": [{"role": "user", "content": "hi"}]})
check("6. model='gpt-4o' -> response model == 'glm-5.2'",
      r.json().get("model") == "glm-5.2", r.json().get("model"), "glm-5.2")

# ---- 7. /v1/models ----
r = client.get("/v1/models")
ids = {m["id"] for m in r.json()["data"]}
check("7. /v1/models contains 'glm-5.2'", "glm-5.2" in ids,
      sorted(ids & {"glm-5.2", "gpt-4o"}), ["glm-5.2", "gpt-4o"])
check("7. /v1/models contains alias 'gpt-4o'", "gpt-4o" in ids,
      sorted(ids & {"glm-5.2", "gpt-4o"}), ["glm-5.2", "gpt-4o"])

# ---- 8. extra request fields don't 422 ----
r = client.post("/v1/chat/completions", json={
    "model": "glm-5.2", "stream": False,
    "frequency_penalty": 1.0, "user": "abc",
    "stream_options": {"include_usage": True},
    "response_format": {"type": "text"}, "presence_penalty": 0.5,
    "messages": [{"role": "user", "content": "hi"}]})
check("8. extra params (frequency_penalty/user/stream_options/response_format/presence_penalty) not 422",
      r.status_code == 200, r.status_code, 200)

# ---- 9. tool_calls / tool_call_id / name messages don't 422 ----
r = client.post("/v1/chat/completions", json={
    "model": "glm-5.2", "stream": False,
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "x", "content": "42"}]})
check("9. tool_calls/tool_call_id/name messages not 422",
      r.status_code == 200, r.status_code, 200)

passed = sum(results)
print(f"\nTOTAL: {passed}/{len(results)} assertions passed")
sys.exit(0 if passed == len(results) else 1)