"""Offline sanity test for glm_rev/captcha_aliyun.py bug fixes.

Verifies BUG 1 (token resurrection): consumed single-use device tokens must
NOT come back after load()/add_many(). Runs fully offline: compute_final and
the device-token collector are monkeypatched, no browser/network is touched.
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "/home/bld/glm-rev")
import glm_rev.captcha_aliyun as ca

orig_compute_final = ca.compute_final
orig_device_tokens = ca.device_tokens
orig_collect = ca._collect_device_tokens
orig_backoff = ca._FAIL_BACKOFF_UNTIL

ca.compute_final = lambda tok: None
ca._collect_device_tokens = lambda: []

tmpdir = Path(tempfile.mkdtemp(prefix="fix1-"))
try:
    pool = ca.DeviceTokenPool(path=tmpdir / "device_tokens.txt")
    ca.device_tokens = pool

    added = pool.add_many(["a", "b", "c"])
    assert added == 3, f"add_many added {added}"
    assert len(pool) == 3, f"len after add_many = {len(pool)}"

    # Pool path: pops all 3, compute_final -> None (F001-style), all consumed.
    ca.captcha_pool._compute_with_refill()
    assert len(pool) == 0, f"len after failed compute = {len(pool)}"

    pool.load()
    assert len(pool) == 0, f"consumed tokens resurrected by load(): {pool._tokens}"

    pool.load()
    assert len(pool) == 0, "consumed tokens resurrected by second load()"

    pool.add_many(["a", "d"])
    assert len(pool) == 1, f"add_many re-added consumed token: {pool._tokens}"

    pool.load()
    assert len(pool) == 1, f"fresh non-consumed token lost by load(): {pool._tokens}"

    print("PASS: consumed tokens never resurrect; fresh tokens still load.")
finally:
    ca.compute_final = orig_compute_final
    ca.device_tokens = orig_device_tokens
    ca._collect_device_tokens = orig_collect
    ca._FAIL_BACKOFF_UNTIL = orig_backoff
    shutil.rmtree(tmpdir, ignore_errors=True)
