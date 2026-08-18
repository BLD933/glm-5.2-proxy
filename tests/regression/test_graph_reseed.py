"""QA verification for the multi-turn threading fix in send_with_tools (tools.py).

Root cause (verified live 2026-08-16, 3-arm probe on chat.z.ai): the
completions backend recalls prior context ONLY when a follow-up request is
parented at a server-assigned assistant node id — i.e. a node the backend
streamed/created itself. The SSE stream usually omits that stored reply id.
POSTing a rewritten node tree to /api/v1/chats/<id> does NOT restore context
(those fresh uuid ids come back DEAF) and in fact poisons recall, so the fix
must NOT POST. Instead, after each streamed turn the code GETs the chat graph
and resolves the assistant child of the user node it just sent
(client.fetch_reply_node), then parents the NEXT request beneath that id.

Tests:
  D1 fetch_reply_node resolves the assistant child of after_user_id over GET,
     and falls back to the deepest assistant on the root chain
  D2 follow-up tool iteration anchors at the fetched server reply node
     (fresh user uuid, parent == fetched assistant id), same chat_id
  D3 committed history drops tool/contract artifacts
  D4 fetch failure (None) keeps legacy anchor and still streams
"""
import contextlib
import io
import sys

sys.path.insert(0, "/home/bld/glm-rev")

import glm_rev.client as client
from glm_rev import tools

PASS = 0
FAIL = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"  FAIL  {name}" + (f"  [{detail}]" if detail else ""))


class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _graph(nodes_by_id):
    msgs = {}
    for mid, parent, role, children in nodes_by_id:
        msgs[mid] = {"id": mid, "parentId": parent, "role": role,
                     "childrenIds": children}
    return {"chat": {"history": {"messages": msgs}}}


# --- TEST D1 — fetch_reply_node resolution ------------------------------

def test_d1():
    print("TEST D1 — fetch_reply_node resolves assistant child via GET")
    real = client._HTTP

    class F:
        def __init__(self, payload, code=200):
            self._p = payload
            self._c = code

        def get(self, url, headers=None, **kw):
            self._u = url
            r = _FakeResp(self._p)
            r.status_code = self._c
            return r

    # direct child: user u1 -> assistant a1
    f = F(_graph([("r", None, "user", ["u1"]),
                   ("u1", "r", "user", ["a1"]),
                   ("a1", "u1", "assistant", [])]))
    client._HTTP = f
    try:
        aid, par = client.fetch_reply_node("tok", "chat_x", after_user_id="u1")
        check("D1a resolves direct assistant child of after_user_id",
              (aid, par) == ("a1", "r"), repr((aid, par)))
        check("D1b GET targets /api/v1/chats/chat_x",
              getattr(f, "_u", "").endswith("/api/v1/chats/chat_x"),
              getattr(f, "_u", ""))
    finally:
        client._HTTP = real

    # no after_user_id -> deepest assistant on root chain
    f = F(_graph([("r", None, "user", ["a0"]),
                   ("a0", "r", "assistant", ["u2"]),
                   ("u2", "a0", "user", ["a1"]),
                   ("a1", "a1", "assistant", [])]))
    client._HTTP = f
    try:
        aid, par = client.fetch_reply_node("tok", "chat_x")
        check("D1c falls back to deepest assistant on chain", aid == "a1", repr(aid))
    finally:
        client._HTTP = real

    # error / empty -> (None, None)
    client._HTTP = F(_graph([]), code=500)
    try:
        check("D1d HTTP error -> (None, None)",
              client.fetch_reply_node("tok", "chat_x") == (None, None))
    finally:
        client._HTTP = real
    client._HTTP = F({"chat": {"history": {"messages": {}}}})
    try:
        check("D1e empty graph -> (None, None)",
              client.fetch_reply_node("tok", "chat_x") == (None, None))
    finally:
        client._HTTP = real


# --- TEST D2/D3 — send_with_tools anchors at fetched reply node ---------

def _run_tools(fetch_results, stream_answers, start_chat_id="chat_1", stream_ids=None):
    """Runs one send_with_tools invocation with fake transport; returns
    (state, fetch_calls, stream_calls, ok, err)."""
    fetch_calls = []
    stream_calls = []

    def fake_fetch(token, chat_id, after_user_id=None, cookie=None):
        fetch_calls.append({"chat_id": chat_id, "after_user_id": after_user_id})
        if fetch_results == "fail":
            return None, None
        i = len(fetch_calls) - 1
        return (f"server_reply_{i}", None)

    def fake_stream(**kw):
        stream_calls.append(dict(kw))
        ans = stream_answers[min(len(stream_calls) - 1, len(stream_answers) - 1)]
        sid = None
        if stream_ids:
            idx = len(stream_calls) - 1
            if idx < len(stream_ids):
                sid = stream_ids[idx]
        return {"id": sid, "answer": ans, "usage": {}}

    saved = {}
    for name, val in [
        ("refresh_token", lambda t: t),
        ("create_chat", lambda *a, **k: ("chat_1", "seed_m", "seed_p")),
        ("sign", lambda *a, **k: ("sig", "", 123)),
        ("build_features", lambda *a, **k: {}),
        ("stream_turn", fake_stream),
        ("fetch_reply_node", fake_fetch),
        ("dispatch_tool", lambda name, args, mcp=None: (True, "mock")),
        ("approve_tool", lambda *a, **k: True),
        ("approve_tool_auto", lambda *a, **k: True),
    ]:
        saved[name] = getattr(tools, name)
        setattr(tools, name, val)

    state = {
        "token": "tok", "cookie": None, "model": "glm-5.2",
        "enable_thinking": False, "reasoning_effort": "low", "web_search": False,
        "temperature": 0.7, "max_tokens": 100,
        "chat_id": start_chat_id, "last_assistant_id": "prior_ast",
        "last_assistant_parent_id": None,
        "history": [{"role": "user", "content": "earlier?"},
                     {"role": "assistant", "content": "earlier answer"}],
        "usage": {"prompts": 0, "in": 0, "out": 0},
        "solver": None,
    }
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ok, err = tools.send_with_tools(
                state, "list the cwd", mcp=None, auto_approve=True,
                captcha_fn=lambda: ("cap", None), writer=io.StringIO())
    finally:
        for name, val in saved.items():
            setattr(tools, name, val)
    return state, fetch_calls, stream_calls, ok, err


def test_d2():
    print("TEST D2 — follow-up threads parent at fetched server reply id")
    state, fetch_calls, stream_calls, ok, err = _run_tools(
        "ok", ['TOOL:run_command({"cmd": "pwd"})', "final answer"])
    check("D2a send_with_tools ok", ok is True, repr(err))
    # turn 0 is a follow-up (chat_id pre-set): anchored under prior anchor,
    # then fetch called to learn the reply node for turn 1.
    s0 = stream_calls[0]
    check("D2b stream #0 is a follow-up with fresh user uuid",
          s0["is_first"] is False and s0["current_user_message_id"] != "seed_m",
          repr((s0["is_first"], s0["current_user_message_id"])))
    check("D2c fetch called with the user node just sent",
          len(fetch_calls) >= 1
          and fetch_calls[0]["after_user_id"] == s0["current_user_message_id"],
          repr([f["after_user_id"] for f in fetch_calls]))
    # turn 1 (tool iteration) must parent beneath fetched server_reply_0
    if len(stream_calls) > 1:
        s1 = stream_calls[1]
        check("D2d stream #1 parents under fetched server reply id",
              s1["current_user_message_parent_id"] == "server_reply_0",
              repr(s1["current_user_message_parent_id"]))
        check("D2e stream #1 uses a fresh user uuid (not seed / not sync leaf)",
              s1["current_user_message_id"] not in ("seed_m", None)
              and not s1["current_user_message_id"].startswith("sync"),
              repr(s1["current_user_message_id"]))
    check("D2f every fetch targets the SAME chat_id",
          all(c["chat_id"] == "chat_1" for c in fetch_calls),
          repr([c["chat_id"] for c in fetch_calls]))
    # committed history must drop the TOOL loop artifacts
    check("D2g committed history ends with clean user+assistant pair",
          state["history"][-2:] == [
              {"role": "user", "content": "list the cwd"},
              {"role": "assistant", "content": "final answer"},
          ], repr(state["history"][-2:]))
    check("D2h state stores fetched reply id as last_assistant_id",
          state["last_assistant_id"] == "server_reply_1"
          or state["last_assistant_id"] == "server_reply_0",
          repr(state["last_assistant_id"]))


def test_d3():
    print("TEST D3 — SSE-provided id used when fetch yields nothing")
    state, fetch_calls, stream_calls, ok, err = _run_tools(
        "fail", ["plain answer"], stream_ids=["sse_id_0"])
    check("D3a ok", ok is True, repr(err))
    check("D3b last_assistant_id falls back to SSE id",
          state["last_assistant_id"] == "sse_id_0",
          repr(state["last_assistant_id"]))


def test_d4():
    print("TEST D4 — fetch failure keeps legacy anchor, still streams")
    state, fetch_calls, stream_calls, ok, err = _run_tools("fail", ["plain answer"])
    check("D4a send_with_tools ok despite fetch outage", ok is True, repr(err))
    check("D4b first stream anchored at legacy prior assistant id",
          stream_calls[0]["current_user_message_parent_id"] == "prior_ast",
          repr(stream_calls[0]["current_user_message_parent_id"]))
    check("D4c final answer delivered",
          state["history"][-1] == {"role": "assistant", "content": "plain answer"},
          repr(state["history"][-1]))


def main():
    for fn in (test_d1, test_d2, test_d3, test_d4):
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} raised exception", False, f"{type(e).__name__}: {e}")
    print("=" * 44)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAILURES))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
