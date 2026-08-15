import sys

sys.path.insert(0, "/home/bld/glm-rev")

import glm_rev.tools as tools

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS: {name}")
    else:
        FAIL += 1
        print(f"FAIL: {name}")


def base_state():
    return {
        "token": "tok-123",
        "cookie": None,
        "chat_id": None,
        "history": [],
        "last_assistant_id": None,
        "model": "glm-5.2",
        "enable_thinking": True,
        "reasoning_effort": "max",
        "web_search": False,
        "max_tokens": 1024,
        "temperature": 0.7,
        "usage": {"prompts": 0, "in": 0, "out": 0},
    }


def patch(calls, answers, solved=True):
    calls["create_chat"] = 0
    calls["solve_fresh"] = 0
    calls["stream_turn"] = 0

    def fake_refresh(token):
        return token

    def fake_sign(content, user_id, token, current_url=""):
        return "sig", "urlparams", 123

    def fake_user_id(token):
        return "u1"

    def fake_create_chat(token, prompt, **kw):
        calls["create_chat"] += 1
        return ("cid123", None)

    def fake_solve_fresh(solver, state):
        calls["solve_fresh"] += 1
        return ("cap1", "ck1") if solved else None

    def fake_stream_turn(**kw):
        calls["stream_turn"] += 1
        idx = calls["stream_turn"] - 1
        if isinstance(answers, list):
            return answers[min(idx, len(answers) - 1)]
        return answers

    tools.refresh_token = fake_refresh
    tools.sign = fake_sign
    tools.user_id_from_token = fake_user_id
    tools.create_chat = fake_create_chat
    tools.solve_fresh = fake_solve_fresh
    tools.stream_turn = fake_stream_turn


print("== BUG 1: no pre-create solve ==")
calls = {}
patch(calls, {"answer": "ok", "id": "a1"})
state = base_state()
ok, err = tools.send_with_tools(state, "hello")
check("first-message flow succeeds", ok is True and err is None)
check("create_chat called exactly once", calls["create_chat"] == 1)
check("chat_id stored in state", state["chat_id"] == "cid123")
check("solve_fresh called EXACTLY once (no pre-create solve)", calls["solve_fresh"] == 1)
check("stream_turn called exactly once", calls["stream_turn"] == 1)

print("== BUG 2a: stream_error does NOT retry ==")
calls = {}
patch(calls, {"answer": "", "stream_error": {"code": "X"}})
state = base_state()
ok, err = tools.send_with_tools(state, "hello")
check("stream_error returns failure", ok is False)
check("stream_error returns formatted error", isinstance(err, str) and "GLM error" in err)
check("stream_error: solve_fresh called only once", calls["solve_fresh"] == 1)
check("stream_error: stream_turn called only once", calls["stream_turn"] == 1)

print("== BUG 2b: plain empty does NOT retry ==")
calls = {}
patch(calls, {"answer": ""})
state = base_state()
ok, err = tools.send_with_tools(state, "hello")
check("empty returns failure", ok is False)
check("empty returns formatted error", isinstance(err, str) and "empty response" in err)
check("empty: solve_fresh called only once", calls["solve_fresh"] == 1)
check("empty: stream_turn called only once", calls["stream_turn"] == 1)

print("== BUG 2c: captcha_error DOES retry with fresh captcha ==")
calls = {}
patch(calls, [{"answer": "", "captcha_error": {"code": "F018"}}, {"answer": "ok", "id": "a1"}])
state = base_state()
ok, err = tools.send_with_tools(state, "hello")
check("captcha retry succeeds", ok is True and err is None)
check("captcha retry: solve_fresh called twice", calls["solve_fresh"] == 2)
check("captcha retry: stream_turn called twice", calls["stream_turn"] == 2)

print("== BUG 2d: captcha_fn still honored for turn solve ==")
calls = {}
callbacks = {"fn": 0}
answers = [{"answer": "", "captcha_error": {"code": "F018"}}, {"answer": "ok", "id": "a1"}]

def fake_refresh2(token):
    return token

def fake_sign2(content, user_id, token, current_url=""):
    return "sig", "urlparams", 123

def fake_user_id2(token):
    return "u1"

def fake_create_chat2(token, prompt, **kw):
    calls["create_chat"] = 0
    return ("cid123", None)

def fake_captcha_fn():
    callbacks["fn"] += 1
    return ("capX", "ckX")

def fake_stream_turn2(**kw):
    idx = calls.get("stream_turn", 0)
    calls["stream_turn"] = idx + 1
    return answers[min(idx, len(answers) - 1)]

tools.refresh_token = fake_refresh2
tools.sign = fake_sign2
tools.user_id_from_token = fake_user_id2
tools.create_chat = fake_create_chat2

def fake_solve_fresh2(solver, state):
    calls["solve_fresh"] = calls.get("solve_fresh", 0) + 1
    return None

tools.solve_fresh = fake_solve_fresh2
tools.stream_turn = fake_stream_turn2
state = base_state()
ok, err = tools.send_with_tools(state, "hello", captcha_fn=fake_captcha_fn)
check("captcha_fn path succeeds", ok is True and err is None)
check("captcha_fn called twice (turn + retry), not for create", callbacks["fn"] == 2)

print(f"\n{FAIL} FAILED, {PASS} PASSED")
sys.exit(1 if FAIL else 0)
