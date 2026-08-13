"""Playwright-based browser solver for acquiring AliyunCaptcha tokens and session cookies.

AliyunCaptcha verify tokens are SINGLE-USE: chat.z.ai requires a freshly solved
token on every `chat/completions` request. `steal_captcha` is a one-shot solver;
`CaptchaSolver` keeps one browser alive on a background asyncio loop so each turn
only pays a warm challenge instead of a full browser launch.
"""
import asyncio
import json
import threading
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import requests

from .config import BASE, UA
from .api import user_id_from_token, headers

PATCH_JS = r"""
(function() {
  if (window.__fetch_patched) return 'already';
  window.__fetch_patched = true;
  window.__last_captcha_token = null;
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

CHROME = "/usr/bin/google-chrome"
LAUNCH_ARGS = [
    "--no-sandbox", "--disable-blink-features=AutomationControlled",
    "--host-resolver-rules=MAP chat.z.ai 146.19.236.205",
    "--disable-web-security", "--ignore-certificate-errors",
    "--disable-dev-shm-usage", "--disable-gpu",
]
COOKIE_NAMES = ("_c_WBKFRo", "_nb_ioWEgULi")


def _cookie_header(cookies) -> str:
    return "; ".join(
        f"{c['name']}={c['value']}" for c in cookies
        if c.get("domain", "").endswith("z.ai") and c["name"] in COOKIE_NAMES)


async def steal_captcha(token: str) -> tuple[str | None, str | None]:
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
            await page.wait_for_timeout(3000)
            await page.evaluate(
                f"localStorage.setItem('token', {json.dumps(token)});"
                "localStorage.setItem('_arms_uid', 'REDACTED');"
                "localStorage.setItem('show_coding_plan_guide', 'false');"
                "localStorage.setItem('last_mode', 'chat');")
            await page.add_init_script(PATCH_JS)
            await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=40000)
            await page.wait_for_timeout(5000)
            await page.evaluate(PATCH_JS)
            tk, cookie = await _drive_challenge(page, ctx, token, "hello test", None)
            return tk, cookie
        finally:
            try:
                await browser.close()
            except Exception:
                pass


async def _drive_challenge(page, ctx, token, probe_text, last_captcha):
    """Click through to a fresh chat, send a probe, and wait for a new captcha token."""
    await page.evaluate("""() => {
      const b = [...document.querySelectorAll('button')];
      const t = b.find(x => (x.innerText||'').trim() === 'Chat');
      if (t) t.click();
    }""")
    await page.wait_for_timeout(1000)
    await page.evaluate("""() => {
      const b = [...document.querySelectorAll('button')];
      const n = b.find(x => /^new chat$/i.test((x.innerText||'').trim()));
      if (n) n.click();
    }""")
    await page.wait_for_timeout(1500)
    await page.evaluate("""(probe) => {
      const input = document.querySelector('#chat-input') || document.querySelector('textarea');
      if (!input) return;
      input.focus();
      const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
      setter.call(input, probe);
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }""", probe_text)
    await page.wait_for_timeout(300)
    await page.evaluate("""() => {
      const btn = document.getElementById('send-message-button');
      if (btn) btn.click();
    }""")
    for _ in range(400):
        tk = await page.evaluate("window.__last_captcha_token || null")
        if tk and tk != last_captcha:
            cookies = await ctx.cookies()
            return tk, _cookie_header(cookies)
        await asyncio.sleep(0.1)
    return None, None


class CaptchaSolver:
    """Persistent Playwright browser issuing a fresh captcha token per request.

    The browser lives on a dedicated daemon thread running its own asyncio loop,
    so Playwright objects survive across turns (objects created under
    ``asyncio.run`` die with that loop). Main-thread calls are synchronous.
    """

    def __init__(self):
        self._loop = None
        self._thread = None
        self._p = None
        self._browser = None
        self._ctx = None
        self._page = None
        self._last_captcha = None

    # --- main-thread API -------------------------------------------------

    def start(self, token: str, timeout: float = 150) -> tuple[str | None, str | None]:
        """Launch the warm browser and return the first captcha + cookie."""
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

    def solve(self, token: str, timeout: float = 150) -> tuple[str | None, str | None]:
        """Get a fresh captcha + cookie from the warm page. Returns (None, None) on timeout."""
        if not self._thread or not self._thread.is_alive():
            return None, None
        fut = asyncio.run_coroutine_threadsafe(self._challenge(token), self._loop)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            return None, None

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
        page = await self._ctx.new_page()
        self._page = page
        await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(3000)
        await page.evaluate(
            f"localStorage.setItem('token', {json.dumps(token)});"
            "localStorage.setItem('_arms_uid', 'REDACTED');"
            "localStorage.setItem('show_coding_plan_guide', 'false');"
            "localStorage.setItem('last_mode', 'chat');")
        await page.add_init_script(PATCH_JS)
        await page.goto(f"{BASE}/", wait_until="domcontentloaded", timeout=40000)
        await page.wait_for_timeout(5000)
        await page.evaluate(PATCH_JS)
        tk, cookie = await _drive_challenge(page, self._ctx, token, "hello test",
                                            self._last_captcha)
        if tk:
            self._last_captcha = tk
        return tk, cookie

    async def _challenge(self, token) -> tuple[str | None, str | None]:
        if self._page is None:
            return None, None
        tk, cookie = await _drive_challenge(self._page, self._ctx, token, "x",
                                            self._last_captcha)
        if tk:
            self._last_captcha = tk
        return tk, cookie

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