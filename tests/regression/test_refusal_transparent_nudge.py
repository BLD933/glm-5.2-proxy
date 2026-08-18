"""Transparent refusal-nudge test for the client tool path (_client_nudge) and
contract authorization framing (_tools_contract). Network-free: a fake upstream
callback stands in for stream_turn."""

import asyncio
import importlib
from types import SimpleNamespace

import sys
import os

sys.path.insert(0, "/home/bld/glm-rev")
os.environ["GLM_CLIENT_TOOLS"] = "1"

_SPEC = importlib.util.spec_from_file_location("server", "/home/bld/glm-rev/server.py")
server = importlib.util.module_from_spec(_SPEC)
sys.modules["server"] = server
_SPEC.loader.exec_module(server)

from glm_rev.config import TOOL_NUDGE_CLIENT

TOOLS = [
    {"type": "function", "function": {"name": "glob", "description": "list files",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "bash", "description": "run command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

REFUSAL = ("I don't have access to tools or the file system in this conversation, "
           "so I'm unable to list the files of the current working directory.")


class FakeReq:
    tools = TOOLS


async def test_refusal_triggers_nudge_and_returns_tool_call():
    seen = {}
    async def fake_upstream(messages, nudge_text):
        assert messages[-1]["role"] == "user"
        assert TOOL_NUDGE_CLIENT in messages[-1]["content"]
        assert any(m["role"] == "assistant" and "don't have access" in m["content"] for m in messages)
        seen["called"] = True
        return 'TOOL:glob({"pattern": "*"})'
    calls, final = await server._client_nudge(
        REFUSAL, [{"role": "user", "content": "list the files of cwd"}],
        token="t", cookie=None, chat_id="c", model="glm-5.2", msg_id="m",
        features={}, params={}, captcha=None, req=FakeReq(), call_upstream=fake_upstream)
    assert seen.get("called"), "nudge upstream was never called"
    assert calls and calls[0][0] == "glob", calls
    assert final.strip().startswith("TOOL:glob")


async def test_no_refusal_no_nudge():
    async def fake_upstream(messages, nudge_text):
        raise AssertionError("should not nudge on a normal answer")
    calls, final = await server._client_nudge(
        "Here are the files: a.py b.py", [{"role": "user", "content": "x"}],
        token="t", cookie=None, chat_id="c", model="glm-5.2", msg_id="m",
        features={}, params={}, captcha=None, req=FakeReq(), call_upstream=fake_upstream)
    assert calls == []
    assert final == "Here are the files: a.py b.py"


async def test_existing_call_no_nudge():
    async def fake_upstream(messages, nudge_text):
        raise AssertionError("should not nudge when a call is already present")
    calls, final = await server._client_nudge(
        'TOOL:bash({"command": "ls"})', [{"role": "user", "content": "x"}],
        token="t", cookie=None, chat_id="c", model="glm-5.2", msg_id="m",
        features={}, params={}, captcha=None, req=FakeReq(), call_upstream=fake_upstream)
    assert calls and calls[0][0] == "bash"


def test_contract_has_auth_and_core_first():
    contract = server._tools_contract(TOOLS + [
        {"type": "function", "function": {"name": "chrome_navigate", "description": "nav",
            "parameters": {"type": "object", "properties": {}}}}])
    # (1) authorization framing is present
    assert "AUTHORIZATION & WORKSPACE CAPABILITY" in contract
    # (2) curated core tool is shown under its GLM-friendly name ...
    assert "run_command" in contract
    assert "list_dir" in contract
    # (3) ... and specialized tools are intentionally omitted so they don't
    #     dilute / confuse GLM (the repl-style reliability trade-off).
    assert "- chrome_navigate" not in contract
    assert "chrome_navigate" not in contract


if __name__ == "__main__":
    failures = []
    for fn in (test_refusal_triggers_nudge_and_returns_tool_call,
               test_no_refusal_no_nudge, test_existing_call_no_nudge):
        try:
            asyncio.run(fn())
            print("PASS", fn.__name__)
        except Exception as e:
            failures.append((fn.__name__, e))
            print("FAIL", fn.__name__, "->", e)
    try:
        test_contract_has_auth_and_core_first()
        print("PASS test_contract_has_auth_and_core_first")
    except Exception as e:
        failures.append(("test_contract_has_auth_and_core_first", e))
        print("FAIL test_contract_has_auth_and_core_first ->", e)
    if failures:
        print("=== FAILURES ===")
        for n, e in failures:
            print(n, e)
        sys.exit(1)
    print("=== ALL NUDGE TESTS PASSED ===")
