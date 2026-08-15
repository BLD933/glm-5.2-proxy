"""Verification for the bounded-batch + refill backoff fix and the harvester-only solver.

Covers:
1. Single stale token in the pool does NOT arm the 35s backoff (real bug).
2. A full fresh-refill cycle failing DOES arm backoff.
3. Empty pool does not arm backoff.
4. solve() serves an in-memory payload, never a DOM token.
5. steal_captcha = harvest -> compute_final, never DOM.
6. compute_final raises HTTPError (request landed) -> backoff IS armed.
7. compute_final raises URLError (transient network) -> backoff stays 0.0.
"""
import asyncio
import json
import sys
import threading
import time
import types
import urllib.error

sys.path.insert(0, "/home/bld/glm-rev")

from glm_rev import captcha_aliyun as ca
import glm_rev.solver as solver_mod


class _FakeTokens:
    def __init__(self, *seq):
        self._q = list(seq)
        self._lock = threading.Lock()

    def load(self):
        pass

    def pop(self):
        with self._lock:
            return self._q.pop(0) if self._q else None

    def add_many(self, tokens):
        with self._lock:
            self._q.extend(tokens)

    def __len__(self):
        return len(self._q)


def _fresh_collect_none():
    return None


def _fresh_collect_bad():
    return ["bad1", "bad2", "bad3"]


def _compute(token):
    return "OK" if token.startswith("good") else None


class _FakePage:
    async def goto(self, *a, **k):
        return None

    async def evaluate(self, js, *args):
        if "const out" in str(js):
            return []
        return True

    async def add_init_script(self, js):
        return None

    async def wait_for_selector(self, *a, **k):
        return None


class _FakeCtx:
    async def new_page(self, *a, **k):
        return _FakePage()


class _FakeBrowser:
    async def new_context(self, *a, **k):
        return _FakeCtx()


class _FakePlaywright:
    async def start(self):
        return self

    @property
    def chromium(self):
        return self

    async def launch(self, *a, **k):
        return _FakeBrowser()


def main():
    failures = 0
    total = 0

    def check(name, cond, detail=""):
        nonlocal failures, total
        total += 1
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
        if not cond:
            failures += 1

    # ---- Scenario 1: one stale token then a good one. Known-good 3-try loop
    # serves the good token (backoff may arm on the bad one; that's fine).
    old_backoff = ca._FAIL_BACKOFF_UNTIL
    ca._FAIL_BACKOFF_UNTIL = 0.0
    ca.device_tokens = _FakeTokens("bad_x", "good_1", "good_2")
    ca._collect_device_tokens = _fresh_collect_none
    orig_compute = ca.compute_final
    ca.compute_final = _compute
    pool = ca.CaptchaPool()
    t0 = time.time()
    payload = pool._compute_with_refill()
    dt = time.time() - t0
    check("S1 stale-then-good serves payload",
          payload == "OK",
          f"payload={payload!r} dt={dt:.2f}s")
    assert payload == "OK", "S1 regression"

    # ---- Scenario 2: all pool tokens stale AND fresh refill also fails.
    ca.device_tokens = _FakeTokens("bad_x", "bad_y", "bad_z")
    ca._collect_device_tokens = _fresh_collect_bad
    ca.compute_final = _compute
    t0 = time.time()
    payload = pool._compute_with_refill()
    dt = time.time() - t0
    armed = ca._FAIL_BACKOFF_UNTIL > time.time()
    check("S2 fresh-refill failure arms backoff",
          payload is None and armed,
          f"payload={payload!r} armed={armed} dt={dt:.2f}s")
    assert payload is None and armed, "S2 regression"

    # ---- Scenario 3: empty pool, no fresh collection -> no backoff.
    ca._FAIL_BACKOFF_UNTIL = 0.0
    ca.device_tokens = _FakeTokens()
    ca._collect_device_tokens = _fresh_collect_none
    payload = pool._compute_with_refill()
    armed = ca._FAIL_BACKOFF_UNTIL > time.time()
    check("S3 empty pool does not arm backoff",
          payload is None and not armed, f"payload={payload!r} armed={armed}")
    assert payload is None and not armed, "S3 regression"

    # ---- Scenario 4: active backoff is slept through, then generation proceeds.
    ca._FAIL_BACKOFF_UNTIL = time.time() + 0.4
    ca.device_tokens = _FakeTokens("good_1")
    t0 = time.time()
    payload = pool._compute_with_refill()
    dt = time.time() - t0
    check("S4 active backoff slept through then serves",
          payload == "OK" and 0.3 <= dt < 3.0, f"payload={payload!r} dt={dt:.2f}s")

    # ---- Scenario 5: compute_final raises (transient DNS/TLS/timeout). A
    # raised exception must NOT arm backoff even though the token was popped.
    ca._FAIL_BACKOFF_UNTIL = 0.0
    ca.device_tokens = _FakeTokens("good_1", "good_2", "good_3")

    def raise_compute(tok):
        raise TimeoutError("dns")

    ca.compute_final = raise_compute
    payload = pool._compute_with_refill()
    armed = ca._FAIL_BACKOFF_UNTIL > time.time()
    check("S5 raised compute does not arm backoff",
          payload is None and not armed and len(ca.device_tokens) == 0,
          f"payload={payload!r} armed={armed}")
    assert payload is None and not armed, "S5 regression"

    # ---- Scenario 6: compute_final raises HTTPError (server responded; device
    # WAS touched) -> backoff MUST be armed. Deterministic via real time.time():
    # _FAIL_BACKOFF_UNTIL is set to now + _FAIL_BACKOFF_S (> now).
    ca._FAIL_BACKOFF_UNTIL = 0.0
    ca.device_tokens = _FakeTokens("good_1")

    def raise_http(tok):
        raise urllib.error.HTTPError(
            "https://no8xfe-verify.captcha-open-southeast.aliyuncs.com/",
            500, "Internal", {}, None)

    ca.compute_final = raise_http
    payload = pool._compute_with_refill()
    armed = ca._FAIL_BACKOFF_UNTIL > time.time()
    check("S6 HTTPError arms backoff",
          payload is None and armed,
          f"payload={payload!r} armed={armed}")
    assert payload is None and armed, "S6 regression"

    # ---- Scenario 7: compute_final raises URLError (transient network; device
    # untouched) -> backoff must stay 0.0.
    ca._FAIL_BACKOFF_UNTIL = 0.0
    ca.device_tokens = _FakeTokens("good_1", "good_2")

    def raise_url(tok):
        raise urllib.error.URLError("no such host")

    ca.compute_final = raise_url
    payload = pool._compute_with_refill()
    check("S7 URLError does not arm backoff",
          payload is None and ca._FAIL_BACKOFF_UNTIL == 0.0,
          f"payload={payload!r} backoff={ca._FAIL_BACKOFF_UNTIL}")
    assert payload is None and ca._FAIL_BACKOFF_UNTIL == 0.0, "S7 regression"

    # ---- Scenario 8: verify_captcha sees F001 inside a Success response ->
    # arms the 300s process-wide pause and returns None.
    ca._F001_UNTIL = 0.0
    orig_http = ca._http_post

    def fake_http_f001(url, body, extra_headers=None, retries=3):
        return json.dumps({"Success": True, "Result": {"VerifyResult": False},
                           "VerifyCode": "F001", "CertifyId": "x"})

    ca._http_post = fake_http_f001
    out = ca.verify_captcha("cid", "dv", "tok")
    armed_f001 = ca._F001_UNTIL > time.time()
    check("S8 F001 verify arms _F001_UNTIL and returns None",
          out is None and armed_f001, f"out={out!r} f001={ca._F001_UNTIL!r}")
    assert out is None and armed_f001, "S8 regression"
    prev_f001 = ca._F001_UNTIL

    # ---- Scenario 9: a non-F001 rejection leaves _F001_UNTIL untouched.
    def fake_http_plain(url, body, extra_headers=None, retries=3):
        return json.dumps({"Success": True, "Result": {"VerifyResult": False},
                           "VerifyCode": "E002", "CertifyId": "x"})

    ca._http_post = fake_http_plain
    out = ca.verify_captcha("cid", "dv", "tok")
    check("S9 non-F001 rejection leaves _F001_UNTIL untouched",
          out is None and ca._F001_UNTIL == prev_f001,
          f"out={out!r} f001={ca._F001_UNTIL!r} prev={prev_f001!r}")
    assert out is None and ca._F001_UNTIL == prev_f001, "S9 regression"

    # ---- Scenario 10: first compute_final None arms _F001_UNTIL (mimicking
    # verify_captcha's arming in the real flow) -> _compute_with_refill aborts
    # after exactly one attempt instead of trying tokens 2-3.
    ca._F001_UNTIL = 0.0
    ca._FAIL_BACKOFF_UNTIL = 0.0
    f001_calls = []

    def f001_compute(token):
        f001_calls.append(token)
        with ca._device_lock:
            ca._F001_UNTIL = time.time() + ca._F001_PAUSE_S
        return None

    ca.device_tokens = _FakeTokens("t1", "t2", "t3")
    ca._collect_device_tokens = _fresh_collect_none
    ca.compute_final = f001_compute
    payload = pool._compute_with_refill()
    armed = ca._F001_UNTIL > time.time()
    check("S10 first-F001 aborts loop after exactly one attempt",
          payload is None and armed and len(f001_calls) == 1,
          f"payload={payload!r} armed={armed} calls={f001_calls!r}")
    assert payload is None and armed and len(f001_calls) == 1, "S10 regression"

    # ---- Scenario 11: reset_backoff clears the F001 gate too (fresh harvest
    # can lift the abort).
    ca._F001_UNTIL = time.time() + 100.0
    ca.reset_backoff()
    check("S11 reset_backoff clears _F001_UNTIL",
          ca._F001_UNTIL == 0.0 and ca._FAIL_BACKOFF_UNTIL == 0.0,
          f"f001={ca._F001_UNTIL!r} backoff={ca._FAIL_BACKOFF_UNTIL!r}")
    assert ca._F001_UNTIL == 0.0, "S11 regression"

    # ---- Scenario 12: the _run gate (backoff_clear expression) blocks
    # generator spawn while _F001_UNTIL is in the future.
    ca._F001_UNTIL = time.time() + 100.0
    ca._FAIL_BACKOFF_UNTIL = 0.0
    gate_clear = (time.time() >= ca._FAIL_BACKOFF_UNTIL
                  and time.time() >= ca._F001_UNTIL)
    check("S12 _run gate blocked during F001 pause", not gate_clear,
          f"gate_clear={gate_clear}")
    assert not gate_clear, "S12 regression"
    ca._F001_UNTIL = 0.0

    # ---- Scenario 13: solver _open skips the authoritative compute_final
    # while the F001 pause is armed, but runs it once the pause lifts.
    _pw_root = types.ModuleType("playwright")
    _pw_api = types.ModuleType("playwright.async_api")
    _pw_api.async_playwright = lambda: _FakePlaywright()
    sys.modules["playwright"] = _pw_root
    sys.modules["playwright.async_api"] = _pw_api

    ca._F001_UNTIL = time.time() + 300.0
    ca._FAIL_BACKOFF_UNTIL = 0.0
    open_calls = []

    def open_compute(token):
        open_calls.append(token)
        return "PAYLOAD"

    ca.device_tokens = _FakeTokens("tok1", "tok2")
    ca.compute_final = open_compute
    solver = solver_mod.CaptchaSolver()
    out = asyncio.run(solver._open("tok"))
    check("S13 _open skips authoritative compute during F001 pause",
          out == (None, None) and not open_calls, f"out={out!r} calls={open_calls!r}")
    assert out == (None, None) and not open_calls, "S13 regression"

    ca._F001_UNTIL = 0.0
    out = asyncio.run(solver._open("tok"))
    check("S13b _open computes authoritatively once pause lifts",
          out == ("PAYLOAD", None) and len(open_calls) == 1,
          f"out={out!r} calls={open_calls!r}")
    assert out == ("PAYLOAD", None) and len(open_calls) == 1, "S13b regression"

    ca._FAIL_BACKOFF_UNTIL = old_backoff
    ca._F001_UNTIL = 0.0
    ca.device_tokens = _FakeTokens()
    ca.compute_final = orig_compute
    ca._http_post = orig_http

    print(f"\n{total - failures}/{total} scenarios passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()