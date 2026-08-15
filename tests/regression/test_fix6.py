"""Verification that solve()/steal_captcha serve ONLY in-memory payloads.

The DOM token (a browser's own intercepted captcha_verify_param) must never be
returned — chat.z.ai rejects it as consumed. These tests mock the browser layer
and assert the solve paths route through compute_final over harvested device
tokens, not through any DOM token.
"""
import asyncio
import sys
import threading
import time
from concurrent.futures import Future

sys.path.insert(0, "/home/bld/glm-rev")

from glm_rev import captcha_aliyun as ca
import glm_rev.solver as solver_mod


def main():
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
        if not cond:
            failures += 1

    old_steal = solver_mod.steal_captcha
    old_oneshot = solver_mod.collect_device_tokens_oneshot
    old_compute = ca.compute_final
    old_solve = ca.solve
    old_dev_tokens = ca.device_tokens

    # ---- Case A: steal_captcha harvests -> compute_final, never DOM.
    harvested = []
    calls = {"oneshot": 0, "compute": 0}

    async def fake_oneshot(token, count=200, timeout=180):
        calls["oneshot"] += 1
        return ["dev_token_1", "dev_token_2"]

    def fake_compute(token):
        calls["compute"] += 1
        return "INMEM_PAYLOAD" if token.startswith("dev_token") else None

    class _Tok:
        def load(self):
            pass
        def pop(self):
            return "dev_token_1"
        def add_many(self, t):
            pass

    solver_mod.collect_device_tokens_oneshot = fake_oneshot
    ca.compute_final = fake_compute
    ca.device_tokens = _Tok()

    out = asyncio.run(solver_mod.steal_captcha("t"))
    check("A steal_captcha returns in-memory payload",
          out == ("INMEM_PAYLOAD", None), f"out={out!r}")
    check("A steal_captcha used harvest + compute_final",
          calls["oneshot"] == 1 and calls["compute"] >= 1,
          f"calls={calls}")
    assert out == ("INMEM_PAYLOAD", None), "Case A regression"

    # ---- Case B: steal_captcha returns (None, None) when harvest empty.
    async def fake_oneshot_empty(token, count=200, timeout=180):
        return []

    solver_mod.collect_device_tokens_oneshot = fake_oneshot_empty
    out = asyncio.run(solver_mod.steal_captcha("t"))
    check("B steal_captcha (None,None) on empty harvest", out == (None, None),
          f"out={out!r}")
    assert out == (None, None), "Case B regression"

    # ---- Case C: solver.solve() waits on pending _start_fut.
    class FakeSolver(solver_mod.CaptchaSolver):
        def __init__(self, start_fut=None):
            super().__init__()
            self._start_fut = start_fut
            self._thread = threading.Thread(
                target=lambda: time.sleep(3600), daemon=True)
            self._thread.start()

        def collect_tokens(self, token, count=150):
            return ["dev_token_1"]

    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()

    s = FakeSolver(start_fut=Future())

    def _resolve():
        time.sleep(0.3)
        asyncio.run_coroutine_threadsafe(
            _set_result(s._start_fut, ("WARM_PAYLOAD", None)), loop)

    async def _set_result(fut, val):
        fut.set_result(val)

    threading.Thread(target=_resolve, daemon=True).start()
    t0 = time.time()
    out = s.solve("tok", timeout=10)
    dt = time.time() - t0
    check("C solve() waits on pending _start_fut then serves warm payload",
          out == ("WARM_PAYLOAD", None) and dt >= 0.2, f"out={out!r} dt={dt:.2f}")
    assert out == ("WARM_PAYLOAD", None), "Case C regression"

    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=2)

    # ---- Case D: solve() with done _start_fut + dry pool harvests.
    done_fut = loop.create_future()
    done_fut.set_result((None, None))  # warm browser open yielded nothing
    s2 = FakeSolver(start_fut=done_fut)
    s2._start_fut = done_fut

    # Pool dry, harvest via collect_tokens -> compute_final.
    ca.solve = lambda: None  # pool has no payload
    ca.device_tokens = _Tok()
    out = s2.solve("tok", timeout=5)
    check("D solve() harvest path returns in-memory payload",
          out == ("INMEM_PAYLOAD", None), f"out={out!r}")
    assert out == ("INMEM_PAYLOAD", None), "Case D regression"

    # ---- Case E: solve() returns (None,None) when harvest yields nothing.
    class DrySolver(FakeSolver):
        def collect_tokens(self, token, count=150):
            return []

    ca.device_tokens = _Tok()  # pop returns dev_token_1 -> compute returns None? no
    # Force compute_final to fail:
    def fail_compute(token):
        return None

    ca.compute_final = fail_compute
    s3 = DrySolver(start_fut=done_fut)
    s3._start_fut = done_fut
    out = s3.solve("tok", timeout=5)
    check("E solve() (None,None) when nothing computable", out == (None, None),
          f"out={out!r}")
    assert out == (None, None), "Case E regression"

    # Restore.
    solver_mod.steal_captcha = old_steal
    solver_mod.collect_device_tokens_oneshot = old_oneshot
    ca.compute_final = old_compute
    ca.solve = old_solve
    ca.device_tokens = old_dev_tokens

    print(f"\n{5 - failures}/5 cases passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()