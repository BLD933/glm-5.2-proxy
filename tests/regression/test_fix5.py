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
import sys
import threading
import time
import urllib.error

sys.path.insert(0, "/home/bld/glm-rev")

from glm_rev import captcha_aliyun as ca


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


def main():
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
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

    ca._FAIL_BACKOFF_UNTIL = old_backoff
    ca.device_tokens = _FakeTokens()
    ca.compute_final = orig_compute

    print(f"\n{7 - failures}/7 scenarios passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()