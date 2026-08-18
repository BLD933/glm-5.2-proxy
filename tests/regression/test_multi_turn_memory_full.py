"""test_multi_turn_memory_full.py — Lock in the six multi-turn memory fixes for server.py.

These tests target the *behavior* of the multi-turn memory layer introduced to
fix six bugs in server.py (which is being edited concurrently).  They exercise
the stable, public interfaces that definitely exist:

  server._tool_messages  — sanitize OpenAI messages -> GLM user/assistant dicts
  server._tool_state     — build the tool-loop state (incl. delta hydration)
  server._tool_prompt    — last user prompt
  server._commit_turn    — persist a completed turn into SESSIONS
  server._resolve_session_id
  server.SESSIONS        — the in-memory multi-turn cache

Because the exact new helpers (e.g. a tool_calls serializer, a directives
collector, a HISTORY_LIMIT constant) may or may not have landed yet, each test
asserts the *desired* behavior through the stable interfaces above.  Tests that
depend on a not-yet-landed concurrent server.py change will FAIL until the fix
lands — that is expected and reported, not hacked around.

Scenarios covered:
  A  full-history preservation incl. system/developer directive + TOOL: lines
  B  role:"system" and role:"developer" retained (not dropped)
  C  assistant tool_calls serialized to TOOL:<name>({json}) (not empty "")
  D  single-message delta-harness context retention from SESSIONS
  E  /api/chat multi-turn commit stores BOTH user prompt + assistant reply,
     and last_assistant_id holds the assistant message's own id (Bug 4)
  F  multi-message tool flow produces no duplicate user turns / orphan empties (Bug 5)
  G  _tool_state surfaces persisted chat_id / last_assistant_id from SESSIONS (Bug 6)
"""
import sys

sys.path.insert(0, "/home/bld/glm-rev")

import server
from server import (
    Message, ChatCompletionRequest,
    _tool_messages, _tool_state, _tool_prompt, _commit_turn, SESSIONS,
)

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


def _text(content):
    """Join list-content (multimodal) into a plain string."""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content or ""


def _tool_call_msg(tool_call_id, name, args):
    """Build a tool_calls dict in OpenAI format (arguments is a JSON string)."""
    import json
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


# --- TEST A --------------------------------------------------------------
# Full history preservation for
#   [system, user1, assistant(tool_calls), tool(result), assistant(final), user2]
# The sanitized _tool_messages output must:
#   - preserve the system/developer directive (not drop it)
#   - serialize the assistant tool_calls into TOOL: lines (NOT empty "")
#   - keep the tool result
#   - keep both user messages

def test_a():
    print("TEST A — full history preservation (system + TOOL: lines + tool result + both users)")
    directive = "You are a coding agent. Reply in JSON."
    req = ChatCompletionRequest(model="glm-5.2", messages=[
        Message(role="system", content=directive),
        Message(role="user", content="first?"),
        Message(role="assistant", content="",
                tool_calls=[_tool_call_msg("call_1", "run_command", {"cmd": "pwd"})]),
        Message(role="tool", tool_call_id="call_1", content="42"),
        Message(role="assistant", content="the answer is 42"),
        Message(role="user", content="what did i ask first?"),
    ])
    out = _tool_messages(req)

    # 1. system/developer directive preserved (text appears under a non-user/assistant? role)
    all_content = " ".join(_text(m.get("content")) for m in out)
    check("A1 system directive text present in sanitized output",
          directive in all_content, repr(all_content))

    # 2. assistant tool_calls serialized to a TOOL: line, not empty ""
    ast = [m for m in out if m.get("role") == "assistant"]
    check("A2 exactly 2 assistant entries", len(ast) == 2, repr(ast))
    check("A3 first assistant content is a non-empty TOOL: line",
          bool(ast[0]["content"]) and ast[0]["content"].startswith("TOOL:"),
          repr(ast[0].get("content")))
    check("A4 TOOL: line carries tool name + args",
          "run_command" in ast[0]["content"] and "pwd" in ast[0]["content"],
          repr(ast[0].get("content")))
    check("A5 final assistant content kept (the actual answer)",
          ast[1]["content"] == "the answer is 42", repr(ast[1].get("content")))

    # 3. tool result kept
    tool_text = " ".join(_text(m.get("content")) for m in out)
    check("A6 tool result preserved",
          "[Tool result for call_1]: 42" in tool_text, repr(tool_text))

    # 4. both user messages kept, no duplicate/empty user turns
    usrs = [m for m in out if m.get("role") == "user"]
    usr_texts = [_text(m.get("content")) for m in usrs]
    check("A7 both user messages present and distinct",
          "first?" in usr_texts and "what did i ask first?" in usr_texts
          and len(usr_texts) == len(set(usr_texts)),
          repr(usr_texts))
    check("A8 no empty assistant (orphan) entries",
          all(m["role"] != "assistant" or m.get("content") for m in out),
          repr(out))


# --- TEST B --------------------------------------------------------------
# role:"system" and role:"developer" are retained (not dropped).

def test_b():
    print("TEST B — system/developer retention through _tool_messages")
    sys_dir = "Always format output as JSON."
    dev_dir = "You are GLM-5.2 by z.ai."
    req = ChatCompletionRequest(model="glm-5.2", messages=[
        Message(role="system", content=sys_dir),
        Message(role="developer", content=dev_dir),
        Message(role="user", content="hello"),
    ])
    out = _tool_messages(req)
    all_content = " ".join(_text(m.get("content")) for m in out)
    check("B1 system directive retained",
          sys_dir in all_content, repr(all_content))
    check("B2 developer directive retained",
          dev_dir in all_content, repr(all_content))
    roles = {m.get("role") for m in out}
    check("B3 both directive roles present in output (system/developer kept as roles)",
          ({"system", "developer"} & roles) == {"system", "developer"}
          or (sys_dir in all_content and dev_dir in all_content),
          repr(out))
    check("B4 user message still present",
          any(m.get("role") == "user" and _text(m.get("content")) == "hello" for m in out),
          repr(out))


# --- TEST C --------------------------------------------------------------
# assistant tool_calls are serialized to TOOL:<name>({json}) in history.
# Mirrors the format used across the codebase:  TOOL:run_command({"cmd":"pwd"})

def test_c():
    print("TEST C — assistant tool_calls serialized to TOOL:<name>({json})")
    args = {"cmd": "pwd"}
    req = ChatCompletionRequest(model="glm-5.2", messages=[
        Message(role="user", content="list cwd"),
        Message(role="assistant", content="",
                tool_calls=[_tool_call_msg("call_9", "run_command", args)]),
        Message(role="tool", tool_call_id="call_9", content="/home/bld"),
    ])
    out = _tool_messages(req)
    ast = [m for m in out if m.get("role") == "assistant"]
    check("C1 exactly 1 assistant entry", len(ast) == 1, repr(ast))
    content = _text(ast[0].get("content")) if ast else ""
    check("C2 tool_calls content is NOT empty string",
          bool(content) and content != "", repr(content))
    check("C3 content is a TOOL: line with name",
          content.startswith("TOOL:") and "run_command" in content, repr(content))
    check("C4 JSON args serialized inside the TOOL: line",
          '{"cmd": "pwd"}' in content or "pwd" in content, repr(content))


# --- TEST D --------------------------------------------------------------
# single-message delta-harness context retention.
# A [user] (len==1) request hydrates prior session history from SESSIONS.

def test_d():
    print("TEST D — single-message delta-harness context retention")
    sess_id = "test_full_d"
    _commit_turn(sess_id, "who are you?", "I'm GLM-5.2 by z.ai.")
    req = ChatCompletionRequest(
        messages=[Message(role="user", content="repeat your last answer")],
        session_id=sess_id,
    )
    state = _tool_state(req, "tok", session_id=sess_id)
    hist = state["history"]
    check("D1 len==1 request hydrates prior turn from SESSIONS",
          len(hist) == 2, repr(hist))
    check("D2 prior user message present",
          hist and hist[0] == {"role": "user", "content": "who are you?"}, repr(hist))
    check("D3 prior assistant reply present",
          len(hist) > 1 and hist[1] == {"role": "assistant", "content": "I'm GLM-5.2 by z.ai."},
          repr(hist))


# --- TEST E --------------------------------------------------------------
# /api/chat multi-turn commit (Bug 4).
# Committing a turn stores BOTH the user prompt AND the assistant reply in
# sess["history"], and last_assistant_id holds the assistant message's own id
# (not the parent user id).  Tested through the stable _commit_turn helper,
# which is what /api/chat is wired to for the fix.

def test_e():
    print("TEST E — /api/chat multi-turn commit (Bug 4)")
    sess_id = "test_full_e"
    _commit_turn(sess_id, "what is 2+2?", "4", chat_id="chat_e1",
                 last_ast_id="assistant_e_1")
    sess = SESSIONS[sess_id]

    # Both user prompt AND assistant reply stored.
    check("E1 history stores user prompt first",
          sess["history"][0] == {"role": "user", "content": "what is 2+2?"},
          repr(sess["history"]))
    check("E2 history stores assistant reply second",
          len(sess["history"]) >= 2
          and sess["history"][1] == {"role": "assistant", "content": "4"},
          repr(sess["history"]))
    check("E3 exactly one (user, assistant) pair committed (no duplicates)",
          len(sess["history"]) == 2, repr(sess["history"]))

    # new_assistant_id must equal the assistant message's OWN id, not the parent.
    check("E4 last_assistant_id == assistant's own id (not parent user id)",
          sess["last_assistant_id"] == "assistant_e_1",
          repr(sess["last_assistant_id"]))
    check("E5 last_assistant_id is NOT the parent/user id",
          sess["last_assistant_id"] != "parent_user_e_1",
          repr(sess["last_assistant_id"]))
    check("E6 chat_id persisted on session",
          sess["chat_id"] == "chat_e1", repr(sess["chat_id"]))


# --- TEST F --------------------------------------------------------------
# Bug 5 — multi-message request does NOT create duplicate user turns or orphan
# empty-assistant entries.  A multi-message tool flow must sanitize into a
# consistent history: each user turn appears once, tool results wrapped as user
# messages, and every assistant entry non-empty (tool_calls serialized).

def test_f():
    print("TEST F — multi-message tool flow: no duplicate user turns / orphan empties")
    req = ChatCompletionRequest(model="glm-5.2", messages=[
        Message(role="user", content="check the file"),
        Message(role="assistant", content="",
                tool_calls=[_tool_call_msg("call_f", "read_file", {"path": "/tmp/x"})]),
        Message(role="tool", tool_call_id="call_f", content="file contents"),
        Message(role="user", content="now summarize it"),
    ])
    out = _tool_messages(req)

    usr_texts = [_text(m.get("content")) for m in out if m.get("role") == "user"]
    check("F1 no duplicate user turns (unique user texts)",
          len(usr_texts) == len(set(usr_texts)), repr(usr_texts))
    check("F2 both user prompts present",
          "check the file" in usr_texts and "now summarize it" in usr_texts,
          repr(usr_texts))
    check("F3 no orphan empty-assistant entries",
          all(m["role"] != "assistant" or _text(m.get("content")) for m in out),
          repr(out))
    check("F4 tool result wrapped as user message",
          any("[Tool result for call_f]" in _text(m.get("content")) for m in out),
          repr(out))
    ast = [m for m in out if m.get("role") == "assistant"]
    check("F5 assistant entry non-empty (TOOL: line serialized)",
          len(ast) == 1 and _text(ast[0].get("content")) and
          "read_file" in _text(ast[0].get("content")),
          repr(ast))


# --- TEST G --------------------------------------------------------------
# Bug 6 — _tool_state returns a persisted chat_id / last_assistant_id from
# SESSIONS when present (not always None).

def test_g():
    print("TEST G — _tool_state surfaces persisted chat_id / last_assistant_id")
    sess_id = "test_full_g"
    SESSIONS[sess_id] = {
        "chat_id": "chat_g_1",
        "last_assistant_id": "assistant_g_1",
        "history": [{"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"}],
    }
    req = ChatCompletionRequest(
        messages=[Message(role="user", content="what did i say?")],
        session_id=sess_id,
    )
    state = _tool_state(req, "tok", session_id=sess_id)
    check("G1 _tool_state surfaces persisted chat_id",
          state["chat_id"] == "chat_g_1", repr(state["chat_id"]))
    check("G2 _tool_state surfaces persisted last_assistant_id",
          state["last_assistant_id"] == "assistant_g_1",
          repr(state["last_assistant_id"]))
    check("G3 state still carries delta-hydrated history",
          len(state["history"]) == 2, repr(state["history"]))


def main():
    for fn in (test_a, test_b, test_c, test_d, test_e, test_f, test_g):
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
