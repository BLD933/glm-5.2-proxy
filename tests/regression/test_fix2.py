"""Lightweight verification for the non-blocking captcha startup fix."""
import sys
import time

sys.path.insert(0, "/home/bld/glm-rev")

import glm_rev.solver as solver_mod
from glm_rev import captcha_aliyun as ca


def _fake_steal(token):
    async def run():
        return None, None
    return run()


async def _fake_oneshot(token, count=200, timeout=180):
    return []


def main():
    # Never launch a real browser during the test.
    solver_mod.steal_captcha = _fake_steal
    solver_mod.collect_device_tokens_oneshot = _fake_oneshot

    # 1) register_solver(None, token) is fast, no browser, installs token.
    t0 = time.time()
    out = solver_mod.register_solver(None, "testtoken")
    dt = time.time() - t0
    ok = out is None and solver_mod._registered_token == "testtoken"
    print(f"{'PASS' if ok else 'FAIL'} register_solver(None, 'testtoken') "
          f"-> {out!r} token={solver_mod._registered_token!r} in {dt:.2f}s")
    assert ok, "register_solver(None, token) failed"

    # 2) solve_fresh with empty pool / no warm solver returns fast, no browser.
    t0 = time.time()
    fresh = solver_mod.solve_fresh(None, {"token": "t"})
    dt = time.time() - t0
    ok = fresh is None or isinstance(fresh, tuple)
    fast = dt < 10.0
    print(f"{'PASS' if ok and fast else 'FAIL'} solve_fresh(None, state) "
          f"-> {str(fresh)[:40]!r} in {dt:.2f}s")
    assert ok, f"solve_fresh returned unexpected {fresh!r}"
    assert fast, f"solve_fresh took {dt:.2f}s (>=10s)"

    # 3) start_background exists, spawns thread+loop, callback re-registers,
    #    and close() cleans up without blocking. Stub _open so no browser runs.
    async def _fake_open(self, token):
        self._token = token
        return None, None

    orig_open = solver_mod.CaptchaSolver._open
    solver_mod.CaptchaSolver._open = _fake_open
    regs = []

    def _fake_register(solver, token=""):
        regs.append((solver, token))

    solver_mod.register_solver = _fake_register
    try:
        s = solver_mod.CaptchaSolver()
        assert hasattr(s, "start_background"), "start_background missing"
        assert callable(s.start_background), "start_background not callable"
        s.start_background("testtoken")
        deadline = time.time() + 5.0
        while not regs and time.time() < deadline:
            time.sleep(0.05)
        ok = s._thread is not None and s._thread.is_alive()
        ok = ok and regs and regs[0][1] == "testtoken"
        print(f"{'PASS' if ok else 'FAIL'} start_background spawns thread + "
              f"callback re-registers: {regs!r}")
        assert ok, "start_background did not re-register via _on_started"
        t0 = time.time()
        s.close()
        print(f"PASS start_background close() clean in {time.time() - t0:.2f}s")
    finally:
        solver_mod.CaptchaSolver._open = orig_open

    print("ALL PASS")


if __name__ == "__main__":
    main()
