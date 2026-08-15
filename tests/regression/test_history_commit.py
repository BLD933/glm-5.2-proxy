"""Stub-driver test for send_with_tools' clean-history commit (tools.py:224 _commit_history).

Verifies: only the clean (user prompt, assistant final answer) pair is persisted to
state["history"]; in-flight loop artifacts (TOOL-only replies, tool results, hints)
reach the model mid-turn but never land in history; turn 2 (is_first=False) reuses
the existing history with the plain prompt; empty TOOL-only replies leave no ghost.
"""
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/home/bld/glm-rev")

import glm_rev.tools as tools
from glm_rev.tools import send_with_tools

# ---- stub everything at module level, BEFORE calling ----
tools.refresh_token = lambda token: token
tools.create_chat = lambda *a, **k: ("chat_1", "msg_1")
tools.sign = lambda *a, **k: ("sig", "", 123)
tools.build_features = lambda *a, **k: {}
tools.dispatch_tool = lambda *a, **k: (True, "mock output")
tools.approve_tool = lambda *a, **k: True
tools.approve_tool_auto = lambda *a, **k: True

RESPONSES = [
    {"id": "a1", "answer": 'TOOL:run_command({"cmd": "ls"})',
     "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    {"id": "a2", "answer": 'TOOL:read_file({"path": "/etc/hostname"})',
     "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    {"id": "a3", "answer": "The hostname is bld.",
     "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    {"id": "b1", "answer": 'TOOL:read_file({"path": "/x"})',
     "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    {"id": "b2", "answer": "Done.",
     "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
    {"id": "c1", "answer": "\n",
     "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
]
_stream = {"n": 0, "messages": []}


def fake_stream_turn(**kwargs):
    i = _stream["n"]
    _stream["n"] += 1
    _stream["messages"].append(list(kwargs["messages"]))
    return dict(RESPONSES[i])


tools.stream_turn = fake_stream_turn


def make_state():
    return {
        "token": "tok", "chat_id": None, "last_assistant_id": None,
        "last_assistant_parent_id": None, "history": [],
        "usage": {"prompts": 0, "in": 0, "out": 0},
        "model": "glm-5.2", "cookie": None, "enable_thinking": False,
        "reasoning_effort": "low", "max_tokens": 100, "temperature": 0.7,
        "web_search": False,
    }


def run(prompt, state):
    with contextlib.redirect_stderr(io.StringIO()):
        return send_with_tools(state, prompt, mcp=None, auto_approve=True,
                               md=None, writer=None,
                               captcha_fn=lambda: ("cap", None))


fails = 0
def check(cond, label):
    global fails
    if cond:
        print(f"PASS  {label}")
    else:
        print(f"FAIL  {label}")
        fails += 1

# ---- turn 1: is_first=True, two tool iterations then a final answer ----
state = make_state()
ok, err = run("list the files", state)
hist = state["history"]
c0 = hist[0]["content"]

check(ok and err is None, "turn1: send_with_tools returned (True, None)")
check(len(hist) == 2, f"turn1: history len == 2 (got {len(hist)})")
check(hist[0]["role"] == "user", "turn1: history[0].role == user")
check(c0.startswith("You are a coding agent running on the user's real Linux machine"),
      "turn1: user content starts with the TOOL_CONTRACT")
check(c0.endswith("list the files"), "turn1: user content ends with the prompt")
check(hist[1]["role"] == "assistant" and hist[1]["content"] == "The hostname is bld.",
      "turn1: history[1] is the clean final answer")
j = json.dumps(hist)
for bad in ("[Tool result for ", "DECLINED by policy", "OPERATOR STOP",
            "(Reminder:", "Immediate instruction from the operator", "TOOL_NUDGE"):
    check(bad not in j, f"turn1: history has no artifact '{bad}'")
check(j.count("[Tool result") == 1,
      "turn1: only '[Tool result]' is the contract's docs mention (count==1)")
check(all(str(e.get("content", "")).strip() for e in hist),
      "turn1: no empty/whitespace content entries")

# ---- in-flight hist carried tool results (call 2 of turn 1) ----
msgs2 = _stream["messages"][1]
hit = [i for i, m in enumerate(msgs2)
       if "[Tool result for run_command]" in str(m.get("content", ""))]
check(bool(hit), f"in-flight: call-2 messages carried '[Tool result for run_command]' at idx {hit}")
check(len(_stream["messages"][0]) == 1, "in-flight: call-1 messages = [contract+prompt+hint] only")

# ---- turn 2: is_first=False, plain prompt committed, no tool artifacts ----
ok2, err2 = run("show the file", state)
hist = state["history"]
check(ok2 and err2 is None, "turn2: send_with_tools returned (True, None)")
check(len(hist) == 4, f"turn2: history len == 4 (got {len(hist)})")
check(hist[2]["role"] == "user" and hist[2]["content"] == "show the file",
      "turn2: user entry is exactly 'show the file' (no contract re-prepended)")
check(hist[3]["role"] == "assistant" and hist[3]["content"] == "Done.",
      "turn2: assistant entry is exactly 'Done.'")
check(hist[0]["content"] == c0 and hist[1]["content"] == "The hostname is bld.",
      "turn2: turn-1 pair preserved verbatim")
j2 = json.dumps(hist)
for bad in ("[Tool result for ", "DECLINED by policy", "OPERATOR STOP",
            "(Reminder:", "Immediate instruction from the operator", "TOOL_NUDGE"):
    check(bad not in j2, f"turn2: history has no artifact '{bad}'")
check(j2.count("[Tool result") == 1,
      "turn2: only '[Tool result]' is the contract's docs mention (count==1)")

# ---- empty-answer guard: pure TOOL-only reply leaves no assistant ghost ----
state3 = make_state()
ok3, err3 = run("ghost check", state3)
hist3 = state3["history"]
check(ok3 and err3 is None, "scenario3: send_with_tools returned (True, None)")
check(len(hist3) == 1, f"scenario3: history grew by exactly 1 user entry (got {len(hist3)})")
check(hist3[0]["role"] == "user", "scenario3: the single entry is user")
check(all(e["role"] != "assistant" for e in hist3), "scenario3: no assistant ghost")

print()
print(f"REPORT: {'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'} "
      f"(stream_turn calls: {_stream['n']}, usage: {state['usage']})")
sys.exit(1 if fails else 0)