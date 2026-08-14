"""Browserless AliyunCaptcha generation — Python port of GLM-Free-API (Go).

chat.z.ai requires a freshly solved ``captcha_verify_param`` on every
``chat/completions`` request. This module reproduces the whole AliyunCaptchaV3
flow in-process (no browser at request time):

    InitCaptchaV3  -> certifyId
    generate_arg   -> RC4-like KSA+PRGA stream cipher over a 64-byte table
    Track JSON + ali_hash + zlib + base64 + encrypt
    VerifyCaptchaV3 (needs a single-use deviceToken) -> final base64 payload

The only externally collected input is the one-time-use ``deviceToken``,
which is bulk-harvested from a real browser via ``window.z_um.getToken()``
(see solver.collect_device_tokens). A background pool keeps 2 payloads warm.
"""
import base64
import hashlib
import hmac
import json
import os
import random
import threading
import time
import uuid
import zlib
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# --- Aliyun API credentials (from GLM-Free-API; chat.z.ai scene) ------------

ACCESS_KEY = "REDACTED"
SECRET_KEY = "REDACTED"
SCENE_ID = "didk33e0"
INIT_URL = "https://no8xfe.captcha-open-southeast.aliyuncs.com/"
VERIFY_URL = "https://no8xfe-verify.captcha-open-southeast.aliyuncs.com/"

ARG_CONSTANT = "REDACTED"
ENCRYPT_KEY = "REDACTED"

ARG_PERM_TABLE = [
    32, 50, 10, 51, 6, 44, 37, 16, 46, 11, 62, 19, 43, 25, 23, 30,
    60, 33, 53, 34, 7, 26, 12, 48, 5, 2, 20, 4, 61, 13, 47, 49,
    18, 29, 27, 22, 1, 17, 39, 56, 41, 38, 55, 31, 15, 58, 52, 40,
    8, 57, 45, 35, 59, 36, 42, 54, 63, 3, 24, 28, 14, 9, 0, 21,
]

HEX_UPPER = "0123456789ABCDEF"
HEX_LOWER = "0123456789abcdef"

_SAFE = frozenset(b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_.~")

_CACHE_TTL = 75.0
_MAX_PARAMS = 2
_IDLE_PAUSE_S = 180.0

# Minimum spacing between background verify requests. Aliyun risk-control
# re-flags the device (F001) after ~3 rapid verify calls, so pace generation
# instead of firing bursts the instant a param is consumed.
_MIN_GEN_INTERVAL_S = 6.0

# Shared cooldown gate: when a device token is rejected (F001 risk-control
# flag), all generator threads pause ~35s before trying again instead of
# hammering the flagged device with parallel verify bursts.
_FAIL_BACKOFF_S = 35.0
_FAIL_BACKOFF_UNTIL = 0.0

# _compute_with_refill tries at most 3 pool tokens (range(3)) before arming
# the shared backoff, so a flagged device is never hammered with rapid
# verify bursts (each compute_final = 1 verify; ~2-3 re-flag the device).

# Default device-token collection hook (overridden by solver/server).
_DEVICE_COLLECTOR = None
_device_lock = threading.Lock()


def set_device_collector(cb):
    """Set a callable returning a list of fresh device tokens (lazy refill)."""
    global _DEVICE_COLLECTOR
    with _device_lock:
        _DEVICE_COLLECTOR = cb


def _collect_device_tokens():
    with _device_lock:
        cb = _DEVICE_COLLECTOR
    if cb is None:
        return []
    try:
        out = cb()
        return [t for t in out if t] if isinstance(out, (list, tuple)) else []
    except Exception:
        return []


# --- Low-level helpers ------------------------------------------------------


def url_encode(s: str) -> str:
    """Percent-encode a string, safe set = alnum + -_.~ (matches Go)."""
    out = []
    for b in s.encode("utf-8"):
        if b in _SAFE:
            out.append(chr(b))
        else:
            out.append("%" + HEX_UPPER[b >> 4] + HEX_UPPER[b & 0xF])
    return "".join(out)


def hmac_sha1_b64(key: bytes, msg: bytes) -> str:
    return base64.b64encode(hmac.new(key, msg, hashlib.sha1).digest()).decode()


def aliyun_signature(params: dict, sec_key: str) -> str:
    keys = sorted(params)
    canonical = "&".join(f"{url_encode(k)}={url_encode(params[k])}" for k in keys)
    string_to_sign = "POST&" + url_encode("/") + "&" + url_encode(canonical)
    return hmac_sha1_b64((sec_key + "&").encode(), string_to_sign.encode())


def build_query_string(params: dict) -> str:
    keys = sorted(params)
    return "&".join(f"{url_encode(k)}={url_encode(params[k])}" for k in keys)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uuid() -> str:
    return str(uuid.uuid4())


def _http_post(url: str, body: str, extra_headers: dict | None = None, retries: int = 3):
    """POST with DNS-flakiness retries (resolving this host is unreliable).

    Every attempt is hard-capped by a worker-thread deadline so a stuck TLS
    handshake or DNS stall (common on this box) can never block a pool
    generator (and therefore the whole pool) for minutes."""
    import threading
    import time as _time
    import urllib.error
    import urllib.request

    def _do() -> str:
        req = urllib.request.Request(url, data=body.encode("utf-8"), method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
        req.add_header("User-Agent", "okhttp/4.9.0")
        for k, v in (extra_headers or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=12) as resp:
            return resp.read().decode("utf-8")

    last_exc = None
    for attempt in range(retries):
        result = {}

        def _work():
            try:
                result["out"] = _do()
            except Exception as e:
                result["err"] = e

        th = threading.Thread(target=_work, daemon=True)
        th.start()
        th.join(timeout=20)
        if th.is_alive():
            last_exc = TimeoutError(f"POST {url} exceeded 20s deadline")
        elif "err" in result:
            last_exc = result["err"]
        else:
            return result["out"]
        if attempt < retries - 1:
            _time.sleep(0.8 * (attempt + 1))
    raise last_exc


# --- Cipher primitives ------------------------------------------------------


def _ksa_and_prga(data: bytes, key: str, perm_table: list) -> bytes:
    """KSA then PRGA of the RC4-like cipher shared by arg generation and encrypt."""
    r = list(perm_table)
    rlen = 64
    n = key

    i, j = 0, 0
    while i < rlen:
        j = (((i + j + r[i] + r[j]) >> 1) + ord(n[i % len(n)])) & (rlen - 1)
        if i != j:
            r[i], r[j] = r[j], r[i]
        i += 1

    t = bytearray()
    e, a = 0, 0
    for idx in range(len(data)):
        a = ((e ^ a) + (r[e] ^ r[a])) & (rlen - 1)
        if e != a:
            r[e], r[a] = r[a], r[e]
        m = data[idx] + e + r[e] - a - r[a]
        m ^= r[e] + r[a]
        m ^= r[(r[e] + r[a]) & (rlen - 1)]
        m &= 255
        t.append(m)
        e = (e + 1) & (rlen - 1)
    return bytes(t)


def generate_arg(certify_id: str) -> str:
    o = certify_id.encode("utf-8")
    return base64.b64encode(_ksa_and_prga(o, ARG_CONSTANT, ARG_PERM_TABLE)).decode()


def encrypt(plaintext: bytes) -> str:
    return base64.b64encode(_ksa_and_prga(plaintext, ENCRYPT_KEY, ARG_PERM_TABLE)).decode()


def ali_hash(input_str: str, salt_str: str) -> str:
    o = input_str
    r = salt_str
    a_len = len(o)
    m = len(r)

    e = [((i << 4) + (i % 16)) for i in range(16)]
    f = 16

    i, j = 0, 0
    while i < f:
        j = (((i + j + e[i] + e[j]) >> 1) + ord(r[i % m])) & (f - 1)
        e[i], e[j] = e[j], e[i]
        i += 1

    idx, p, q = 0, 0, 0
    while idx < a_len:
        q = ((p ^ q) + (e[p] ^ e[q])) & (f - 1)
        e[p], e[q] = e[q], e[p]
        c = (ord(o[idx]) + p + q) ^ e[p] ^ e[q]
        c &= 255
        e[p] = c
        p = (p + 1) & (f - 1)
        idx += 1

    for step in range(2 * f):
        pos = step % f
        if pos != 0:
            e[pos] ^= e[pos - 1]
        else:
            e[0] ^= e[f - 1]

    return "".join(HEX_LOWER[(b >> 4) & 0xF] + HEX_LOWER[b & 0xF] for b in e)


# --- API calls --------------------------------------------------------------


def init_captcha() -> str:
    params = {
        "AccessKeyId": ACCESS_KEY,
        "Action": "InitCaptchaV3",
        "Format": "JSON",
        "Language": "en",
        "Mode": "popup",
        "SceneId": SCENE_ID,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": _uuid(),
        "SignatureVersion": "1.0",
        "Timestamp": _utc_timestamp(),
        "UpLang": "true",
        "Version": "2023-03-05",
    }
    params["Signature"] = aliyun_signature(params, SECRET_KEY)
    resp = _http_post(INIT_URL, build_query_string(params))
    data = json.loads(resp)
    return data["CertifyId"]


def verify_captcha(certify_id: str, data_value: str, device_token: str) -> str | None:
    cvp = json.dumps({
        "certifyId": certify_id,
        "data": data_value,
        "deviceToken": device_token,
        "sceneId": SCENE_ID,
    }, separators=(",", ":"), ensure_ascii=False)
    params = {
        "AccessKeyId": ACCESS_KEY,
        "Action": "VerifyCaptchaV3",
        "Format": "JSON",
        "SignatureMethod": "HMAC-SHA1",
        "SignatureVersion": "1.0",
        "Timestamp": _utc_timestamp(),
        "Version": "2023-03-05",
        "SceneId": SCENE_ID,
        "CertifyId": certify_id,
        "CaptchaVerifyParam": cvp,
        "SignatureNonce": _uuid(),
    }
    params["Signature"] = aliyun_signature(params, SECRET_KEY)
    resp = _http_post(VERIFY_URL, build_query_string(params), {"Referer": ""})
    data = json.loads(resp)
    if not data.get("Success"):
        return None
    result = data.get("Result") or {}
    if not result.get("VerifyResult"):
        return None
    security_token = result.get("securityToken")
    result_certify = result.get("certifyId")
    if not security_token or not result_certify:
        return None
    final = json.dumps({
        "certifyId": result_certify,
        "isSign": True,
        "sceneId": SCENE_ID,
        "securityToken": security_token,
    }, separators=(",", ":"), ensure_ascii=False)
    return base64.b64encode(final.encode("utf-8")).decode()


def compute_final(device_token: str) -> str | None:
    """Full in-memory flow for one device token. Returns final payload or None."""
    certify_id = init_captcha()
    arg_value = generate_arg(certify_id)
    ct = int(time.time() * 1000)
    track = json.dumps({
        "TrackList": {
            "fi": "", "ks": "", "mc": "", "mp": "", "mu": "",
            "startTime": ct, "tc": "", "te": "", "tmv": "",
        },
        "TrackStartTime": ct,
        "VerifyTime": ct + 300,
        "arg": arg_value,
    }, separators=(",", ":"), ensure_ascii=False)
    h = ali_hash(track, "0000")
    combined = (h + track).encode("utf-8")
    compressed = zlib.compress(combined)
    fb64 = base64.b64encode(compressed).decode()
    final_val = encrypt(fb64.encode("utf-8"))
    return verify_captcha(certify_id, final_val, device_token)


# --- Device token pool (file-backed, one-time-use) --------------------------

# Device tokens are single-use AND short-lived (fresh ones verify ~100%,
# tokens ~10 min old are all rejected with F001). Keep a conservative TTL.
_DEVICE_TOKEN_TTL_S = 240
# Bounded recall of handed-out single-use tokens so load() can't resurrect them.
_CONSUMED_CAP = 500


class DeviceTokenPool:
    """Thread-safe FIFO of short-lived single-use device tokens.

    Tokens are persisted as ``epoch_ms <token>`` lines so stale ones can be
    pruned after a restart. Tokens older than ``_DEVICE_TOKEN_TTL_S`` are
    dropped on load/pop and trigger a lazy refill upstream.
    """

    def __init__(self, path: Path | None = None):
        self._path = path or Path(__file__).resolve().parent.parent / "zai" / "device_tokens.txt"
        self._lock = threading.Lock()
        self._tokens: list[tuple[float, str]] = []  # (added_epoch, token)
        self._consumed = set()  # single-use tokens handed out; never resurrect
        self._consumed_order = deque()

    def load(self) -> int:
        with self._lock:
            now = time.time()
            fresh = []
            try:
                if self._path.exists():
                    for line in self._path.read_text().splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            try:
                                added = float(parts[0])
                            except ValueError:
                                continue
                            if now - added < _DEVICE_TOKEN_TTL_S and parts[1] not in self._consumed:
                                fresh.append((added, parts[1]))
                        # legacy token-only lines are treated as stale
            except Exception:
                fresh = []
            self._tokens = fresh
            return len(self._tokens)

    def pop(self) -> str | None:
        with self._lock:
            now = time.time()
            while self._tokens:
                added, tok = self._tokens.pop(0)
                if now - added < _DEVICE_TOKEN_TTL_S:
                    self._consumed.add(tok)
                    self._consumed_order.append(tok)
                    if len(self._consumed_order) > _CONSUMED_CAP:
                        self._consumed.discard(self._consumed_order.popleft())
                    return tok
            return None

    def add_many(self, tokens: list[str]) -> int:
        with self._lock:
            now = time.time()
            seen = {tok for _, tok in self._tokens}
            added = 0
            for t in tokens:
                t = t.strip()
                if t and t not in seen and t not in self._consumed:
                    self._tokens.append((now, t))
                    seen.add(t)
                    added += 1
            if added:
                self._persist()
            return added

    def _persist(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"{added} {tok}" for added, tok in self._tokens]
            self._path.write_text("\n".join(lines) + "\n")
        except Exception:
            pass

    def clear(self):
        with self._lock:
            self._tokens = []
            try:
                if self._path.exists():
                    self._path.write_text("")
            except Exception:
                pass

    def __len__(self):
        with self._lock:
            return len(self._tokens)


device_tokens = DeviceTokenPool()


# --- Background payload cache -----------------------------------------------


class CaptchaPool:
    """Keeps 2 freshly generated captcha payloads warm (75s TTL).

    Runs a background daemon thread. When device tokens run dry, lazily calls
    the configured collector to refill the pool, then keeps generating.
    """

    def __init__(self, max_params: int = _MAX_PARAMS, ttl: float = _CACHE_TTL):
        self.max_params = max_params
        self.ttl = ttl
        self._lock = threading.Lock()
        self._params: list[tuple[float, str]] = []
        self._generating = 0
        self._last_active = time.time()
        self._last_gen = 0.0
        self._active = True
        self._thread = None
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="captcha-pool")
        self._thread.start()

    def get(self) -> str | None:
        """Return a cached payload (fast), else None (caller falls back)."""
        with self._lock:
            self._last_active = time.time()
            now = time.time()
            valid = [p for p in self._params if now - p[0] < self.ttl]
            self._params = valid
            if valid:
                _, val = valid.pop(0)
                self._params = valid
                return val
            return None

    def _run(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                now = time.time()
                self._params = [p for p in self._params if now - p[0] < self.ttl]
                idle = time.time() - self._last_active > _IDLE_PAUSE_S
                if idle and not self._params:
                    self._active = False
                else:
                    self._active = True
                needed = self.max_params - len(self._params) - self._generating
                # Strictly one generator at a time: concurrent compute_final
                # bursts against a flagged device amplify F001.
                paced = (time.time() - self._last_gen) >= (
                    _MIN_GEN_INTERVAL_S + random.uniform(-1.5, 1.5))
                backoff_clear = time.time() >= _FAIL_BACKOFF_UNTIL
                if needed > 0 and self._generating == 0 and paced and backoff_clear:
                    self._generating += 1
                    self._last_gen = time.time()
                    targets = 1
                else:
                    targets = 0
            for _ in range(targets):
                threading.Thread(target=self._generate_one, daemon=True).start()

    def _generate_one(self):
        try:
            payload = self._compute_with_refill()
            if payload:
                with self._lock:
                    self._params.append((time.time(), payload))
        except Exception:
            pass
        finally:
            with self._lock:
                self._generating -= 1

    def _compute_with_refill(self) -> str | None:
        global _FAIL_BACKOFF_UNTIL
        # Respect the shared F001 cooldown. A single rejected token arms the
        # backoff immediately so a flagged device is never hammered with
        # concurrent verify bursts (each compute_final = 1 verify; ~2-3 rapid
        # ones re-flag the device). Never reharvest the browser here.
        with _device_lock:
            until = _FAIL_BACKOFF_UNTIL
        if time.time() < until:
            time.sleep(until - time.time())
        for _ in range(3):
            token = device_tokens.pop()
            if token is None:
                collected = _collect_device_tokens()
                if collected:
                    device_tokens.add_many(collected)
                    token = device_tokens.pop()
                if token is None:
                    return None
            try:
                payload = compute_final(token)
            except Exception:
                continue
            if payload:
                return payload
            with _device_lock:
                _FAIL_BACKOFF_UNTIL = time.time() + _FAIL_BACKOFF_S
        return None


captcha_pool = CaptchaPool()


def ensure_started():
    device_tokens.load()
    captcha_pool.start()


def warm(timeout: float = 15.0) -> bool:
    """Block until the pool has at least one valid payload ready without popping it."""
    if not enabled():
        return False
    ensure_started()
    deadline = time.time() + timeout
    while True:
        with captcha_pool._lock:
            now = time.time()
            if any(now - p[0] < captcha_pool.ttl for p in captcha_pool._params):
                return True
        if time.time() >= deadline:
            return False
        time.sleep(0.25)


def enabled() -> bool:
    """In-memory solver is on by default; disable with ZAI_CAPTCHA_INMEMORY=0."""
    return os.environ.get("ZAI_CAPTCHA_INMEMORY", "1") not in ("0", "false", "no")


_SOLVE_TIMEOUT = 6.0


def solve(timeout: float = _SOLVE_TIMEOUT) -> str | None:
    """Fast path: in-memory payload, BLOCKING up to `timeout` for the pool.

    The pool needs a few seconds to generate a payload from pre-harvested
    device tokens. Returning None instantly forced every request onto the
    slow browser challenge path. Poll the pool briefly before giving up.
    """
    if not enabled():
        return None
    ensure_started()
    deadline = time.time() + timeout
    while True:
        payload = captcha_pool.get()
        if payload:
            return payload
        if time.time() >= deadline:
            return None
        time.sleep(0.25)