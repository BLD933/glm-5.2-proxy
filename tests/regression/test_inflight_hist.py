"""Verify NEW in-flight history behavior in glm_rev/tools.py send_with_tools.

In-flight `hist` stores RAW assistant text (TOOL: lines intact); user-visible
output (writer) and committed state["history"] stay stripped.
"""
import json
import re
import sys

sys.path.insert(0, "/home/bld/glm-rev")

from glm_rev import tools
from glm_rev.config import TOOL_CONTRACT, TOOL_HINT

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


class _Buffer:
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s


PROMPT = "inspect cwd"
EXPECT_U0 = TOOL_CONTRACT + "\n\nUser: inspect cwd" + TOOL_HINT
EXPECT_TOOL_RUN = 'TOOL:run_command({"cmd": "pwd"})'
EXPECT_TOOL_LIST = 'TOOL:list_dir({"path": "."})'
EXPECT_DECLINED = "[Tool result for run_command]: DECLINED by policy"

sent_messages = []


def fake_stream_turn(**kw):
    sent_messages.append(kw["messages"])
    canned = {
        1: {"id": "a1", "answer": EXPECT_TOOL_RUN,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        2: {"id": "a2", "answer": EXPECT_TOOL_LIST,
            "usage": {"prompt_tokens": 8, "completion_tokens": 4}},
        3: {"id": "a3", "answer": "The directory contains the files.",
            "usage": {"prompt_tokens": 6, "completion_tokens": 9}},
    }
    return canned[len(sent_messages)]


tools.refresh_token = lambda token: token
tools.create_chat = lambda *a, **k: ("chat_1", "msg_1")
tools.sign = lambda *a, **k: ("sig", "", 123)
tools.build_features = lambda *a, **k: {}
tools.stream_turn = fake_stream_turn
tools.fetch_reply_node = lambda *a, **k: (None, None)
tools.dispatch_tool = lambda name, args, mcp=None: (True, "mock listing")
tools.approve_tool = lambda name, args: name == "list_dir"

state = {"token": "tok", "chat_id": None, "last_assistant_id": None,
         "last_assistant_parent_id": None, "history": [],
         "usage": {"prompts": 0, "in": 0, "out": 0}, "model": "glm-5.2",
         "cookie": None, "enable_thinking": False, "reasoning_effort": "low",
         "max_tokens": 100, "temperature": 0.7, "web_search": False}

writer = _Buffer()
ok, err = tools.send_with_tools(state, PROMPT, md=None, mcp=None,
                                auto_approve=False, writer=writer,
                                captcha_fn=lambda: ("cap", None))
check("send_with_tools returns ok", ok is True and err is None)

# 1. first stream_turn sees only user0 (contract + prompt + hint)
m1 = sent_messages[0]
check("call1 len==1", len(m1) == 1, f"len={len(m1)}")
c1 = m1[0]["content"]
check("call1 content starts with contract", c1.startswith(TOOL_CONTRACT))
check("call1 content contains 'inspect cwd'", "inspect cwd" in c1)
check("call1 content == contract+prompt+hint", c1 == EXPECT_U0)

# 2. second call: user0 + RAW assistant TOOL:run_command + DECLINED result
m2 = sent_messages[1]
check("call2 len==3", len(m2) == 3, f"len={len(m2)}")
check("call2[0] == user0", m2[0] == m1[0])
a2 = m2[1]
check("call2[1] role assistant", a2["role"] == "assistant")
check("call2[1] content raw TOOL:run_command (no strip)",
      a2["content"] == EXPECT_TOOL_RUN, repr(a2["content"]))
check("call2[2] user DECLINED result", m2[2]["role"] == "user"
      and "DECLINED by policy" in m2[2]["content"])

# 3. third call sees full chain: user0, ast0, user1, ast1(list), user2
m3 = sent_messages[2]
check("call3 len==5", len(m3) == 5, f"len={len(m3)}")
check("call3[0:3] == call2 messages", m3[:3] == m2)
a3 = m3[3]
check("call3[3] raw TOOL:list_dir (no strip)",
      a3["role"] == "assistant" and a3["content"] == EXPECT_TOOL_LIST,
      repr(a3["content"]))
check("call3[4] user has list_dir result",
      m3[4]["role"] == "user" and "[Tool result for list_dir]" in m3[4]["content"])
chain = [m["content"] for m in m3]
check("call3 model sees full chain", all(t in "".join(chain) for t in
      (EXPECT_TOOL_RUN, EXPECT_DECLINED, EXPECT_TOOL_LIST, "[Tool result for list_dir]")))

# 4. committed history is clean: exactly [user(no hint), assistant(final)]
hist = state["history"]
check("history len==2", len(hist) == 2, f"len={len(hist)}")
check("history[0] user == contract+prompt (no hint)",
      hist[0]["role"] == "user" and hist[0]["content"] ==
      TOOL_CONTRACT + "\n\nUser: inspect cwd")
check("history[1] assistant == final answer",
      hist[1]["role"] == "assistant" and
      hist[1]["content"] == "The directory contains the files.")
asst_clean = all(re.search(r"TOOL:(run_command|list_dir)", m["content"]) is None
                 for m in hist if m["role"] == "assistant")
check("no TOOL:(run_command|list_dir) in assistant entries", asst_clean)
hj = json.dumps(hist)
no_standalone = ("TOOL:run_command({\"cmd\": \"pwd\"})" not in hj
                 and "TOOL:list_dir({\"path\": \".\"})" not in hj)
check("no standalone TOOL: call line in history JSON", no_standalone)

# 5. writer output fully stripped
check("writer.text == final answer only",
      writer.text == "The directory contains the files.", repr(writer.text))
check("writer.text has no TOOL:",
      re.search(r"TOOL:", writer.text) is None)

print()
if FAILS:
    print(f"{len(FAILS)} FAILURE(S): {FAILS}")
    sys.exit(1)
print("ALL CHECKS PASSED")