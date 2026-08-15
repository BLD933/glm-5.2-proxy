"""QA verification for NEW multi-turn memory behavior in glm-rev.

Covers:
  TEST A  glm_rev.client.create_chat multi-node linked-list seeding
  TEST B  server._tool_messages / _tool_state / _tool_prompt sanitization
  TEST C  glm_rev.tools.send_with_tools carrying full prior context
"""
import contextlib
import io
import sys

sys.path.insert(0, "/home/bld/glm-rev")

import glm_rev.client as client
from glm_rev.client import create_chat
import server
from server import Message, ChatCompletionRequest, _tool_messages, _tool_state, _tool_prompt
from glm_rev.config import TOOL_CONTRACT, TOOL_HINT

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


# --- fakes ---------------------------------------------------------------

class _FakeResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeHTTP:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, headers=None, **kw):
        self.calls.append({"url": url, "body": json, "headers": headers})
        return _FakeResp({"id": "chat_abc"})


class _Buffer:
    def __init__(self):
        self._parts = []

    def write(self, text):
        self._parts.append(text)

    @property
    def text(self):
        return "".join(self._parts)


# --- TEST A --------------------------------------------------------------

def test_a():
    print("TEST A — create_chat multi-node graph")
    real = client._HTTP
    fake = _FakeHTTP()
    client._HTTP = fake
    try:
        msgs = [
            {"role": "user", "content": "m1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "m2"},
            {"role": "user", "content": "current prompt"},
        ]
        chat_id, last_id = create_chat("tok", "current prompt", messages=msgs)
        body = fake.calls[0]["body"]
        hist = body["chat"]["history"]
        nodes = hist["messages"]

        chain = []
        cur = next(n["id"] for n in nodes.values() if n["parentId"] is None)
        while cur:
            n = nodes[cur]
            chain.append(n)
            cur = n["childrenIds"][0] if n["childrenIds"] else None

        check("A1 chat_id=='chat_abc' and last_id==4th node id",
              chat_id == "chat_abc" and last_id == chain[3]["id"],
              repr((chat_id, last_id, chain[3]["id"])))
        check("A2 exactly 4 nodes", len(nodes) == 4, repr(len(nodes)))
        check("A3 chain integrity (parentId/childrenIds/root/leaf)",
              (chain[0]["parentId"] is None
               and all(chain[i]["parentId"] == chain[i - 1]["id"] for i in range(1, 4))
               and all(chain[i]["childrenIds"] == [chain[i + 1]["id"]] for i in range(3))
               and chain[3]["childrenIds"] == []))
        check("A4 roles/contents match input dicts",
              [n["role"] for n in chain] == [m["role"] for m in msgs]
              and [n["content"] for n in chain] == [m["content"] for m in msgs])
        check("A5 history.currentId == last node id", hist["currentId"] == chain[3]["id"])
        ts = [n["timestamp"] for n in chain]
        check("A6 timestamps strictly increasing", all(ts[i] < ts[i + 1] for i in range(3)), repr(ts))

        _, last_id2 = create_chat("tok", "single prompt")
        body2 = fake.calls[1]["body"]
        hist2 = body2["chat"]["history"]
        nodes2 = list(hist2["messages"].values())
        check("A7 backward compat: 1 node, role user, content, currentId==node, returns node id",
              (len(nodes2) == 1
               and nodes2[0]["role"] == "user"
               and nodes2[0]["content"] == "single prompt"
               and hist2["currentId"] == nodes2[0]["id"]
               and last_id2 == nodes2[0]["id"]),
              repr((len(nodes2), nodes2[0]["role"], nodes2[0]["content"], hist2["currentId"], last_id2)))
    finally:
        client._HTTP = real


# --- TEST B --------------------------------------------------------------

def test_b():
    print("TEST B — _tool_messages/_tool_state/_tool_prompt")
    req = ChatCompletionRequest(model="glm-5.2", messages=[
        Message(role="system", content="sys"),
        Message(role="user", content="first?"),
        Message(role="assistant", content="first answer"),
        Message(role="tool", tool_call_id="call_9", content="42"),
        Message(role="user", content="what did i say first?"),
    ])
    expected = [
        {"role": "user", "content": "first?"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "[Tool result for call_9]: 42"},
        {"role": "user", "content": "what did i say first?"},
    ]
    sanitized = _tool_messages(req)
    check("B1 _tool_messages drops system, sanitizes roles/contents", sanitized == expected, repr(sanitized))
    check("B2 _tool_state history == messages before last user",
          _tool_state(req, "tok")["history"] == expected[:3],
          repr(_tool_state(req, "tok")["history"]))
    check("B3 _tool_prompt == last user message content",
          _tool_prompt(req) == "what did i say first?", repr(_tool_prompt(req)))


# --- TEST C --------------------------------------------------------------

def test_c():
    print("TEST C — send_with_tools full context via state history")
    from glm_rev import tools

    stream_calls = []
    def fake_stream_turn(**kw):
        stream_calls.append(dict(kw))
        if len(stream_calls) == 1:
            return {"id": "a1", "answer": 'TOOL:run_command({"cmd":"pwd"})', "usage": {}}
        return {"id": "a2", "answer": "The cwd is /home/bld.", "usage": {}}

    create_calls = []
    def fake_create_chat(token, prompt, **kw):
        create_calls.append({"token": token, "prompt": prompt, "kwargs": kw})
        return ("chat_1", "seed_msg_1")

    saved = {}
    for name, val in [
        ("refresh_token", lambda t: t),
        ("create_chat", fake_create_chat),
        ("sign", lambda *a, **k: ("sig", "", 123)),
        ("build_features", lambda *a, **k: {}),
        ("stream_turn", fake_stream_turn),
        ("dispatch_tool", lambda name, args, mcp=None: (True, "mock")),
        ("approve_tool", lambda *a, **k: True),
        ("approve_tool_auto", lambda *a, **k: True),
        ("approve_mcp_auto", lambda *a, **k: True),
    ]:
        saved[name] = getattr(tools, name)
        setattr(tools, name, val)

    history = [
        {"role": "user", "content": "first?"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "[Tool result for call_9]: 42"},
    ]
    state = {
        "token": "tok", "cookie": None, "model": "glm-5.2",
        "enable_thinking": False, "reasoning_effort": "low", "web_search": False,
        "temperature": 0.7, "max_tokens": 100,
        "chat_id": None, "last_assistant_id": None, "last_assistant_parent_id": None,
        "history": list(history),
        "usage": {"prompts": 0, "in": 0, "out": 0},
        "solver": None,
    }
    writer = _Buffer()
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            ok, errmsg = tools.send_with_tools(
                state, "what did i say first?", mcp=None, auto_approve=True,
                captcha_fn=lambda: ("cap", None), md=None, writer=writer)
    finally:
        for name, val in saved.items():
            setattr(tools, name, val)

    check("C1 send_with_tools returned ok", ok is True, repr(errmsg))
    check("C2 stream_turn called exactly twice", len(stream_calls) == 2, repr(len(stream_calls)))

    contract_content = TOOL_CONTRACT + "\n\nUser: what did i say first?" + TOOL_HINT
    call1_msgs = stream_calls[0]["messages"]
    check("C3 call1 messages == history + contract user (prior turns present, len 4)",
          (len(call1_msgs) == 4
           and call1_msgs[0]["content"] == "first?"
           and call1_msgs == history + [{"role": "user", "content": contract_content}]))
    check("C4 call1 ids: current_user_message_id=='seed_msg_1', parent None",
          (stream_calls[0]["current_user_message_id"] == "seed_msg_1"
           and stream_calls[0]["current_user_message_parent_id"] is None),
          repr((stream_calls[0]["current_user_message_id"],
                stream_calls[0]["current_user_message_parent_id"])))
    check("C5 create_chat seeded with history + prompt",
          (len(create_calls) == 1
           and create_calls[0]["kwargs"].get("messages")
               == history + [{"role": "user", "content": "what did i say first?"}]),
          repr(create_calls[0]["kwargs"].get("messages") if create_calls else None))

    tool_result = "[Tool result for run_command]:\nmock"
    expected_call2 = history + [
        {"role": "user", "content": contract_content},
        {"role": "assistant", "content": 'TOOL:run_command({"cmd":"pwd"})'},
        {"role": "user", "content": tool_result + TOOL_HINT},
    ]
    call2_msgs = stream_calls[1]["messages"]
    check("C6 call2 messages == history + 3 turn artifacts (len %d)" % len(expected_call2),
          call2_msgs == expected_call2 and len(call2_msgs) == len(expected_call2))
    check("C7 call2 threading: fresh uuid (not seed), parent == last assistant id a1",
          (stream_calls[1]["current_user_message_id"] != "seed_msg_1"
           and bool(stream_calls[1]["current_user_message_id"])
           and stream_calls[1]["current_user_message_parent_id"] == "a1"),
          repr((stream_calls[1]["current_user_message_id"],
                stream_calls[1]["current_user_message_parent_id"])))
    check("C8 writer final answer clean (no TOOL lines)",
          writer.text == "The cwd is /home/bld.", repr(writer.text))
    check("C9 committed state history ends with clean (user, assistant) pair",
          state["history"][-2:] == [
              {"role": "user", "content": "what did i say first?"},
              {"role": "assistant", "content": "The cwd is /home/bld."},
          ]
          and state["history"][:3] == history,
          repr(state["history"][-2:]))


def main():
    for fn in (test_a, test_b, test_c):
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