"""Crypto signing, device fingerprinting, and DNS pinning."""
import base64
import hashlib
import hmac
import time
import uuid
import urllib3.util.connection as _uc
from urllib3.util.connection import create_connection as _cc
from urllib.parse import urlencode

from .config import PINNED_IP, SECRET, UA, BASE, DEVICE_ID, FE_VERSION, REGION


def _patched_create_connection(address, *args, **kwargs):
    host, port = address
    if host == "chat.z.ai":
        host = PINNED_IP
    return _cc((host, port), *args, **kwargs)


# Apply DNS patch globally
_uc.create_connection = _patched_create_connection


def user_id_from_token(token: str) -> str:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        import json
        return json.loads(base64.urlsafe_b64decode(payload)).get("id", "")
    except Exception:
        return ""


def device_fingerprint(user_id: str, token: str, current_url: str = "https://chat.z.ai/") -> tuple[dict, dict]:
    ts = str(int(time.time() * 1000))
    o = {"timestamp": ts, "requestId": str(uuid.uuid4()), "user_id": user_id}
    now_utc = time.gmtime()
    l = {
        "version": "0.0.1",
        "platform": "web",
        "token": token,
        "user_agent": UA,
        "language": "en-US",
        "languages": "en-US",
        "timezone": "UTC",
        "cookie_enabled": "true",
        "screen_width": "1920",
        "screen_height": "1080",
        "screen_resolution": "1920x1080",
        "viewport_height": "793",
        "viewport_width": "1720",
        "viewport_size": "1720x793",
        "color_depth": "24",
        "pixel_ratio": "1",
        "current_url": current_url,
        "pathname": current_url.replace(BASE, "") or "/",
        "search": "",
        "hash": "",
        "host": "chat.z.ai",
        "hostname": "chat.z.ai",
        "protocol": "https:",
        "referrer": "",
        "title": "",
        "timezone_offset": "0",
        "local_time": time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "utc_time": time.strftime("%a, %d %b %Y %H:%M:%S GMT", now_utc),
        "is_mobile": "false",
        "is_touch": "false",
        "max_touch_points": "0",
        "browser_name": "Chrome",
        "os_name": "Linux",
    }
    return o, l


def sign(prompt: str, user_id: str, token: str, current_url: str = "https://chat.z.ai/") -> tuple[str, str, str]:
    o, l = device_fingerprint(user_id, token, current_url)
    c = {**o, **l}
    url_params = urlencode(c)
    sorted_payload = ",".join(f"{k},{c[k]}" for k in sorted(o))
    ts = o["timestamp"]

    p = base64.b64encode(prompt.encode("utf-8")).decode()
    h = f"{sorted_payload}|{p}|{ts}"
    m = str(int(int(ts) / (5 * 60 * 1000)))
    inner = hmac.new(SECRET.encode(), m.encode(), hashlib.sha256).hexdigest()
    sig = hmac.new(inner.encode(), h.encode(), hashlib.sha256).hexdigest()
    return sig, url_params, ts


def headers(token: str, sig: str = "x", cookie: str = None) -> dict:
    h = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept-Language": "en-US",
        "X-FE-Version": FE_VERSION,
        "X-Signature": sig,
        "X-Device-ID": DEVICE_ID,
        "X-Region": REGION,
        "Origin": BASE,
        "Referer": f"{BASE}/",
        "User-Agent": UA,
    }
    if cookie:
        h["Cookie"] = cookie
    return h
