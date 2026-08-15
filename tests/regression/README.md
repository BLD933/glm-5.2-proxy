# glm-rev regression suite (offline)

These are standalone, offline regression tests for the `glm-rev` project
(GLM-5.2 chat.z.ai CLI/REPL/server client). They stub all networking and
Aliyun captcha internals, so they run in seconds with no credentials.

They were originally developed in `/tmp/opencode` during bug-fix work and
are preserved here so a fresh checkout (or a future project) can re-verify
behavior after any change.

## Run

```bash
bash tests/regression/run_all.sh
```

Each test is also runnable directly:

```bash
cd /home/bld/glm-rev
PYTHONPATH=. python3 tests/regression/test_contract.py
GLM_CLIENT_TOOLS=1 PYTHONPATH=. python3 tests/regression/test_client_tools.py
PYTHONPATH=. python3 tests/regression/test_memory.py
```

The runner compiles every source file, then runs each test and reports a
PASS/FAIL table (exit non-zero on any failure).

## What each test covers

| File | Assertions | Coverage |
|------|-----------|----------|
| `test_fix1.py` | token consumption | consumed tokens never resurrect; fresh tokens still load |
| `test_fix2.py` | captcha pool | pool fill/drain, LIFO order, stale-TTL pop |
| `test_fix3.py` | tools | tool-call parsing, refusal detection, writer output |
| `test_fix4.py` | parser | multi-turn tool loop, dispatch, commits |
| `test_fix5.py` | solver/captcha | 7 scenarios incl. F001 device flag behavior |
| `test_fix6.py` | parser | 5 edge cases (nested JSON, prose strips) |
| `test_parser_edge_cases.py` | parser | prose stripping, tool-syntax detection |
| `test_contract.py` | tools/server | 16 assertions: contract build, no double-contract, prompt |
| `test_client_tools.py` | tools/server | 7 assertions: client-side tool loop (env-gated) |
| `test_refusal.py` | tools | 21 assertions: refusal short-circuit |
| `test_history_commit.py` | tools | 32 assertions: committed history is clean (user, assistant) pairs |
| `test_inflight_hist.py` | tools | 22 assertions: model sees its own raw TOOL calls in-loop |
| `test_memory.py` | client/server/tools | 19 assertions: multi-turn memory — multi-node graph seeding, `_tool_messages` sanitization, history threading, seeded msg_id |

## Key stub pattern

Tests monkeypatch `glm_rev.client._HTTP` and the module-level functions
(`create_chat`, `stream_turn`, `sign`, `build_features`, `refresh_token`,
`dispatch_tool`, `approve_tool`, `approve_mcp_auto`). They rely on
`fake_create_chat(token, prompt, **kw)` / `lambda *a, **k` stubs, so new
keyword args (e.g. `messages=`) must remain absorbed — a good reason to keep
new `create_chat` params keyword-compatible.

## Porting to a future project

The tests import the real `glm_rev` and `server` modules. To reuse them on
a fork/rename, update the `PYTHONPATH` and the `server.Message` /
`server.ChatCompletionRequest` construction in `test_memory.py` and
`test_contract.py` to match the new schema. The invariants they encode are
the valuable part — especially:

- committed history must be clean `(user, assistant)` pairs, never loop
  artifacts or contract text
- the in-flight `hist` must carry raw assistant `TOOL:` lines so the model
  can see its own calls
- upstream chat seeding must thread the full conversation, not just the
  last message