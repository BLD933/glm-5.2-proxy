"""Playwright-based browser solver for acquiring AliyunCaptcha device tokens.

AliyunCaptcha verify tokens are SINGLE-USE: chat.z.ai requires a freshly solved
token on every `chat/completions` request. The browser is only a HARVESTER of
device tokens (never a server of DOM captcha params — chat.z.ai rejects those
as already consumed). The authoritative path is the in-memory pool:
`compute_final(device_token)` ~0.3-1s. `CaptchaSolver` keeps one browser alive
on a background asyncio loop for cheap re-harvesting; `steal_captcha` is the
one-shot fallback.
"""
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import requests

from .config import BASE, UA
from .api import user_id_from_token, headers
from . import captcha_aliyun as ca

PATCH_JS = r"""
(function() {
  if (window.__fetch_patched) return 'already';
  window.__fetch_patched = true;
  window.__last_captcha_token = null;
  window.AliyunCaptchaConfig = { region: 'sgp', prefix: 'no8xfe' };
  const originalFetch = window.fetch;
  window.fetch = async function(...args) {
    const url = args[0];
    if (typeof url === 'string' && url.includes('chat/completions')) {
      try {
        const body = JSON.parse(args[1].body);
        if (body.captcha_verify_param) window.__last_captcha_token = body.captcha_verify_param;
      } catch (e) {}
      const mockStream = new ReadableStream({
        start(controller) { controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n')); controller.close(); }
      });
      return new Response(mockStream, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    }
    return originalFetch.apply(this, args);
  };
  return 'ok';
})()
"""

WARM_JS = r"""
async () => {
  if (typeof window.z_um !== 'undefined' && typeof window.z_um.getToken === 'function') {
    return true;
  }
  window.AliyunCaptchaConfig = { region: 'sgp', prefix: 'no8xfe' };
  if (!window.initAliyunCaptcha) {
    await new Promise((resolve, reject) => {
      const existing = document.querySelector('script[src*="AliyunCaptcha.js"]');
      if (existing) {
        existing.addEventListener('load', () => resolve());
        existing.addEventListener('error', () => reject(new Error('script load failed')));
        return;
      }
      const s = document.createElement('script');
      s.src = 'https://o.alicdn.com/captcha-frontend/aliyunCaptcha/AliyunCaptcha.js';
      s.onload = () => resolve();
      s.onerror = () => reject(new Error('script load failed'));
      document.head.appendChild(s);
    });
  }
  let el = document.getElementById('chat-captcha-element');
  if (!el) {
    el = document.createElement('div');
    el.id = 'chat-captcha-element';
    el.style.cssText = 'position:absolute;left:-99999px;top:-99999px;width:0;height:0;overflow:hidden;pointer-events:none;';
    document.body.appendChild(el);
  }
  let btn = document.getElementById('chat-captcha-trigger');
  if (!btn) {
    btn = document.createElement('button');
    btn.id = 'chat-captcha-trigger';
    btn.type = 'button';
    btn.style.cssText = 'position:absolute;left:-99999px;top:-99999px;width:1px;height:1px;opacity:0;';
    document.body.appendChild(btn);
  }
  await new Promise((resolve) => {
    window.initAliyunCaptcha({
      SceneId: 'didk33e0',
      mode: 'popup',
      element: '#chat-captcha-element',
      button: '#chat-captcha-trigger',
      timeout: 10000,
      getInstance: (instance) => {
        window.__captcha_instance = instance;
        resolve();
      }
    });
  });
  for (let i = 0; i < 50; i++) {
    if (typeof window.z_um !== 'undefined' && typeof window.z_um.getToken === 'function') {
      return true;
    }
    await new Promise(r => setTimeout(r, 100));
  }
  return false;
}
"""

CHROME = "/usr/bin/google-chrome"
LAUNCH_ARGS = [
    "--no-sandbox", "--disable-blink-features=AutomationControlled",
    "--host-resolver-rules=MAP chat.z.ai 146.19.236.205",
    "--disable-web-security", "--ignore-certificate-errors",
    "--disable-dev-shm-usage", "--disable-gpu",
]
COOKIE_NAMES = ("_c_WBKFRo", "_nb_ioWEgULi")

# Collect device tokens in bulk from the live app (Aliyun SDK must be warm).
COLLECT_JS = r"""
async (total) => {
  const out = [];
  for (let i = 0; i < total; i++) {
    if (typeof window.z_um === 'undefined' || !window.z_um.getToken) return out;
    const tok = window.z_um.getToken();
    out.push((tok && typeof tok.then === 'function') ? await tok : tok);
    if (i % 50 === 0) await new Promise(r => setTimeout(r, 0));
  }
  return out;
}
"""


def solve_fresh(solver, state, one_shot_attempts=1):
    """Acquire a FRESH single-use captcha + cookie pair.

    AliyunCaptcha tokens are single-use: re-sending an already-consumed token
    to chat.completions triggers FRONTEND_CAPTCHA_REQUIRED (F018). This never
    returns a stale token — it tries the in-memory pool, then the warm solver,
    then up to `one_shot_attempts` headless solves. Returns None only if all
    fail. Updates state["captcha"]/["cookie"] in place on success."""
    print("\x1b[2m[*] solving fresh captcha...\x1b[0m", file=sys.stderr)
    if ca.enabled():
        param = ca.solve()
        if param:
            state["captcha"], state["cookie"] = param, state.get("cookie")
            print("\x1b[2m[*] fresh captcha from in-memory pool\x1b[0m", file=sys.stderr)
            return param, state.get("cookie")
    if solver is not None:
        try:
            captcha, cookie = solver.solve(state["token"])
        except Exception:
            captcha, cookie = None, None
        if captcha:
            state["captcha"], state["cookie"] = captcha, cookie or state.get("cookie")
            print("\x1b[2m[*] fresh captcha from warm solver\x1b[0m", file=sys.stderr)
            return captcha, cookie or state.get("cookie")
    for i in range(one_shot_attempts):
        try:
            captcha, cookie = asyncio.run(steal_captcha(state["token"]))
        except Exception:
            captcha, cookie = None, None
        if captcha:
            state["captcha"], state["cookie"] = captcha, cookie or state.get("cookie")
            print(f"\x1b[2m[*] fresh captcha from one-shot solve ({i + 1})\x1b[0m", file=sys.stderr)
            return captcha, cookie or state.get("cookie")
        print(f"\x1b[33m[!] one-shot captcha solve attempt {i + 1} failed\x1b[0m", file=sys.stderr)
    return None


def _default_device_collector() -> list[str]:
    """Lazy device-token refill: warm solver first, else a throwaway browser."""
    global _registered_solver, _registered_token
    s = _registered_solver
    tk = _registered_token or (getattr(s, "_token", "") if s else "")
    if s is not None and s._thread and s._thread.is_alive():
        tokens = s.collect_tokens(tk, count=150)
        if tokens:
            return tokens
    if _registered_token:
        try:
            return asyncio.run(collect_device_tokens_oneshot(_registered_token, 150))
        except Exception:
            return []
    return []


_registered_solver = None
_registered_token = None


def register_solver(solver, token: str = "") -> None:
    """Point the lazy device-token collector at the warm browser (once)."""
    global _registered_solver, _registered_token
    _registered_solver = solver
    _registered_token = token or getattr(solver, "_token", "") or _registered_token
    if solver is not None and hasattr(solver, "_token") and not solver._token:
        solver._token = _registered_token
    ca.set_device_collector(_default_device_collector)
    ca.ensure_started()
    if ca.enabled() and solver is not None:
        # Block briefly so the pool has a payload ready for the first request.
        ca.warm(timeout=15.0)


def _cookie_header(cookies) -> str:
    return "; ".join(
        f"{c['name']}={c['value']}" for c in cookies
        if c.get("domain", "").endswith("z.ai") and c["name"] in COOKIE_NAMES)


async def steal_captcha(token: str) -> tuple[str | None, str | None]:
    """One-shot fallback: launch a throwaway browser, warm the Aliyun SDK,
    and bulk-harvest device tokens, then serve an in-memory computed captcha.

    No DOM token is ever returned (chat.z.ai rejects those as consumed).
    Returns (None, None) only if the harvest or the in-memory compute fails."""
    try:
        tokens = await collect_device_tokens_oneshot(token, count=150)
        if tokens:
            ca.device_tokens.add_many(tokens)
            device_token = ca.device_tokens.pop()
            if device_token:
                payload = ca.compute_final(device_token)
                if payload:
                    return payload, None
    except Exception:
        pass
    return None, None


async def collect_device_tokens_oneshot(token: str, count: int = 200,
                                        timeout: float = 60) -> list[str]:
    """Launch a throwaway browser, warm the Aliyun SDK, and bulk-harvest tokens.

    Hard-bounded so a slow/failed one-shot can never block the caller for
    minutes; the whole launch+challenge+harvest is capped by `timeout`."""
    try:
        return await asyncio.wait_for(
            _oneshot_impl(token, count), timeout=timeout)
    except Exception:
        return []


async def _oneshot_impl(token: str, count: int) -> list[str]:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, executable_path=CHROME, args=LAUNCH_ARGS)
        try:
            ctx = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=UA, locale="en-US")
            page = await ctx.new_page()
            await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=40000)
            await _wait_for(page, "body", 15000)
            await page.evaluate(
                f"localStorage.setItem('token', {json.dumps(token)});"
                "localStorage.setItem('_arms_uid', 'REDACTED');"
                "localStorage.setItem('last_mode', 'chat');")
            await page.add_init_script(PATCH_JS)
            await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=40000)
            await _wait_for(page, "#chat-input", 20000)
            await page.evaluate(PATCH_JS)
            has_zum = await page.evaluate(
                "typeof window.z_um !== 'undefined' && !!window.z_um.getToken")
            if not has_zum:
                await _drive_challenge(page, ctx, token, "hello test", None)
                has_zum = await page.evaluate(
                    "typeof window.z_um !== 'undefined' && !!window.z_um.getToken")
            if not has_zum:
                return []
            result = await page.evaluate(COLLECT_JS, count)
            return [str(v) for v in result if v] if result else []
        finally:
            try:
                await browser.close()
            except Exception:
                pass


async def _wait_for(page, selector: str, timeout: float = 8000):
    try:
        await page.wait_for_selector(selector, timeout=timeout)
    except Exception:
        pass


async def _drive_challenge(page, ctx, token, probe_text, last_captcha):
    """Directly initialize the Aliyun Captcha SDK so window.z_um is warm."""
    try:
        warmed = await page.evaluate(WARM_JS)
        if warmed:
            cookies = await ctx.cookies()
            return "warmed", _cookie_header(cookies)
    except Exception:
        pass
    return None, None


class CaptchaSolver:
    """Persistent Playwright browser used as a device-token HARVESTER.

    The browser lives on a dedicated daemon thread running its own asyncio loop,
    so Playwright objects survive across turns (objects created under
    ``asyncio.run`` die with that loop). Main-thread calls are synchronous.
    Captcha params are served from the in-memory Aliyun pool, never from DOM.
    """

    def __init__(self):
        self._loop = None
        self._thread = None
        self._p = None
        self._browser = None
        self._ctx = None
        self._page = None
        self._last_captcha = None
        self._page_lock = None  # asyncio.Lock, created loop-side
        self._token = ""
        self._start_fut = None

    # --- main-thread API -------------------------------------------------

    def start(self, token: str, timeout: float = 150) -> tuple[str | None, str | None]:
        """Launch the warm browser and return the first captcha + cookie."""
        self._token = token
        if self._thread and self._thread.is_alive():
            return self.solve(token, timeout=timeout)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="captcha-solver")
        self._thread.start()
        try:
            fut = asyncio.run_coroutine_threadsafe(self._open(token), self._loop)
            return fut.result(timeout=timeout)
        except Exception:
            self.close()
            raise

    def start_background(self, token: str) -> None:
        """Launch the warm browser on the background thread without blocking.

        `_on_started` re-registers the live solver (and warms the pool) on the
        solver thread once the browser is ready, so the REPL is never blocked."""
        self._token = token
        if self._thread and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="captcha-solver")
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._open(token), self._loop)
        self._start_fut = fut
        fut.add_done_callback(self._on_started)

    def _on_started(self, fut) -> None:
        """Runs on the solver thread once `_open` completes; never on the REPL thread."""
        try:
            fut.result()
        except Exception as e:
            print(f"\x1b[33m[!] warm captcha solver failed: {e}\x1b[0m",
                  file=sys.stderr)
            self._page = None
            return
        print("\x1b[2m[*] warm captcha solver ready\x1b[0m", file=sys.stderr)
        try:
            register_solver(self, self._token)
        except Exception:
            pass

    def solve(self, token: str, timeout: float = 150) -> tuple[str | None, str | None]:
        """Get a fresh captcha + cookie. Authoritative path is the in-memory
        Aliyun pool (compute_final over harvested device tokens); the browser
        is only used to (re)harvest device tokens. Returns (None, None) if no
        in-memory payload is available and no harvest was possible."""
        if self._start_fut is not None and not self._start_fut.done():
            # Background warm-up still running: wait for it instead of falling
            # through to a redundant one-shot browser launch (~50s).
            try:
                warm = self._start_fut.result(timeout=min(timeout, 25.0))
            except Exception:
                warm = (None, None)
            if warm and warm[0]:
                return warm
            # Browser may be up even though its open-challenge yielded nothing;
            # fall through to the pool/harvest path on the warm page.
        if not self._thread or not self._thread.is_alive():
            return None, None
        if ca.enabled():
            payload = ca.captcha_pool.get()
            if payload:
                return payload, None
            # Pool dry: prefer an already-harvested device token (in-memory
            # compute ~0.3-1s) over a browser harvest. Skip the direct verify
            # while an F001 backoff is armed so a flagged device is not handed
            # another re-arming verify during its cooldown.
            try:
                device_token = None
                if time.time() >= ca._FAIL_BACKOFF_UNTIL:
                    device_token = ca.device_tokens.pop()
                if not device_token and self._page is not None:
                    tokens = self.collect_tokens(token, count=150)
                    if tokens:
                        ca.device_tokens.add_many(tokens)
                        device_token = ca.device_tokens.pop()
                if device_token:
                    payload = ca.compute_final(device_token)
                    if payload:
                        return payload, None
            except Exception:
                pass
        return None, None

    def collect_tokens(self, token: str, count: int = 200,
                       timeout: float = 60) -> list[str]:
        """Bulk-harvest device tokens from the warm page (window.z_um.getToken).

        Hard-bounded: a stuck page (hung challenge / stale window.z_um) must
        fail fast rather than block the caller (the pool generator or a
        request) for minutes."""
        if not self._thread or not self._thread.is_alive() or self._page is None:
            return []
        fut = asyncio.run_coroutine_threadsafe(
            self._collect_coro(token, count), self._loop)
        try:
            return fut.result(timeout=timeout) or []
        except Exception:
            return []

    def close(self) -> None:
        if self._loop and self._thread and self._thread.is_alive():
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(timeout=10)
            except Exception:
                pass
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join(timeout=10)
            except Exception:
                pass

    # --- loop-side -------------------------------------------------------

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _open(self, token) -> tuple[str | None, str | None]:
        from playwright.async_api import async_playwright
        self._p = await async_playwright().start()
        self._browser = await self._p.chromium.launch(
            headless=True, executable_path=CHROME, args=LAUNCH_ARGS)
        self._ctx = await self._browser.new_context(
            viewport={"width": 1920, "height": 1080}, user_agent=UA, locale="en-US")
        self._page_lock = asyncio.Lock()
        self._token = token
        page = await self._ctx.new_page()
        self._page = page
        await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=40000)
        await _wait_for(page, "body", 15000)
        await page.evaluate(
            f"localStorage.setItem('token', {json.dumps(token)});"
            "localStorage.setItem('_arms_uid', 'REDACTED');"
            "localStorage.setItem('show_coding_plan_guide', 'false');"
            "localStorage.setItem('last_mode', 'chat');")
        await page.add_init_script(PATCH_JS)
        await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=40000)
        await _wait_for(page, "#chat-input", 20000)
        await page.evaluate(PATCH_JS)
        await page.evaluate(WARM_JS)
        # Bulk-harvest device tokens so the in-memory pool can serve captcha
        # params without any further challenge round-trip. This warms the
        # Aliyun SDK (window.z_um) internally; the DOM token it produces is
        # DISCARDED (chat.z.ai rejects it as already consumed).
        harvested = []
        try:
            harvested = await self._collect_coro(token, 150)
            if harvested:
                ca.device_tokens.add_many(harvested)
                print(f"[solver] _open harvested {len(harvested)} tokens, pool={len(ca.device_tokens)}", file=sys.stderr)
                # The pool is warmed by register_solver AFTER _open returns
                # (never start it here: its generator would run compute_final
                # concurrently with the authoritative compute below, and
                # concurrent verify bursts amplify the F001 device flag).
            else:
                print("[solver] _open harvest returned 0 tokens", file=sys.stderr)
        except Exception as e:
            print(f"[solver] _open harvest raised {type(e).__name__}: {e}", file=sys.stderr)
        # Authoritative in-memory path: compute a fresh captcha from a
        # harvested device token instead of serving the DOM token.
        try:
            device_token = ca.device_tokens.pop()
            if device_token:
                payload = ca.compute_final(device_token)
                if payload:
                    return payload, None
                print("[!] _open: compute_final returned None (verify rejected)", file=sys.stderr)
        except Exception as e:
            print(f"[!] _open: compute_final raised {type(e).__name__}: {e}", file=sys.stderr)
        return None, None

    async def _collect_coro(self, token: str, count: int) -> list[str]:
        if self._page is None:
            return []
        # Hard bound on the whole harvest so a stuck page releases
        # _page_lock and lets other callers proceed.
        try:
            return await asyncio.wait_for(
                self._collect_coro_unlocked(token, count), timeout=45.0)
        except Exception:
            return []

    async def _collect_coro_unlocked(self, token: str, count: int) -> list[str]:
        async with self._page_lock:
            try:
                # The Aliyun SDK (window.z_um) is only warm after a challenge.
                has_zum = await self._page.evaluate(
                    "typeof window.z_um !== 'undefined' && !!window.z_um.getToken")
                if not has_zum:
                    await _drive_challenge(self._page, self._ctx, token, "x",
                                           self._last_captcha)
                result = await self._page.evaluate(COLLECT_JS, count)
                if not result:
                    return []
                return [str(v) for v in result if v]
            except Exception:
                return []

    async def _shutdown(self):
        try:
            if self._page:
                await self._page.close()
        except Exception:
            pass
        try:
            if self._ctx:
                await self._ctx.close()
        except Exception:
            pass
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._p:
                await self._p.stop()
        except Exception:
            pass