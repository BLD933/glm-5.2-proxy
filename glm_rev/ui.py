"""GLM REPL — interactive terminal UI with full option toggles.

Inspired by claude-rev's ui.py: streaming markdown-lite ANSI rendering,
slash commands, persistent conversation state, and transcript export.
"""
import asyncio
import json
import os
import re
import sys
import time
import uuid
import requests

from .config import BASE, ENDPOINT
from .api import sign, headers, user_id_from_token
from .client import refresh_token, create_chat, build_features
from .solver import CaptchaSolver, steal_captcha

HERE = os.path.dirname(os.path.abspath(__file__))

SLASH_COMMANDS = ["clear", "effort", "export", "help", "history", "max",
                  "model", "models", "new", "pretty", "search", "status",
                  "temp", "think", "usage"]

ANSI = {
    "reset": "\x1b[0m", "bold": "\x1b[1m", "dim": "\x1b[2m", "underline": "\x1b[4m",
    "green": "\x1b[32m", "yellow": "\x1b[33m", "blue": "\x1b[34m", "cyan": "\x1b[36m",
}

MODEL_FALLBACK = [
    "glm-5.2", "GLM-5.1", "GLM-5-Turbo", "GLM-5v-Turbo", "glm-4.7",
    "glm-4.6v", "0727-106B-API", "0727-360B-API",
    "GLM-4.1V-Thinking-FlashX", "deep-research", "zero",
    "glm-4-flash", "0808-360B-DR", "glm-4-air-250414",
]


def model_catalog():
    """Read live model ids from zai/models.json when present."""
    try:
        data = json.load(open(os.path.join(HERE, "zai", "models.json")))

        def walk(o):
            if isinstance(o, dict):
                if o.get("openai") and o.get("id"):
                    yield o
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)

        out, seen = [], set()
        for m in walk(data):
            mid = m.get("id")
            if mid in seen:
                continue
            seen.add(mid)
            out.append((mid, m.get("name", mid)))
        if out:
            return out
    except Exception:
        pass
    return [(m, m) for m in MODEL_FALLBACK]


class MDPrinter:
    """Streaming markdown-lite ANSI renderer — fence-aware across chunk boundaries.
    Falls back to raw passthrough when stdout is not a tty."""

    def __init__(self, enabled=True):
        self.enabled = enabled and sys.stdout.isatty()
        self._buf = ""
        self._in_code = False

    def write(self, chunk):
        if not self.enabled:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            return
        self._buf += chunk
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            print(self._render(line))
        sys.stdout.flush()

    def finish(self):
        if self._buf:
            print(self._render(self._buf))
            self._buf = ""
        sys.stdout.flush()

    def _render(self, line):
        fence = line.strip()
        if not self._in_code and fence.startswith(("```", "~~~")):
            self._in_code = True
            return f"{ANSI['dim']}── {fence[3:].strip() or 'code'} ──{ANSI['reset']}"
        if self._in_code and fence.startswith(("```", "~~~")):
            self._in_code = False
            return f"{ANSI['dim']}{'─' * 12}{ANSI['reset']}"
        if self._in_code:
            return f"{ANSI['cyan']}{line}{ANSI['reset']}"
        if fence.startswith(">"):
            return f"{ANSI['dim']}{line}{ANSI['reset']}"
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            return f"{ANSI['bold']}{ANSI['underline']}{m.group(2)}{ANSI['reset']}"
        line = re.sub(r"^\s*[-*+]\s+", f"{ANSI['green']}•{ANSI['reset']} ", line, count=1)
        line = re.sub(r"\*\*(.+?)\*\*", f"{ANSI['bold']}\\1{ANSI['reset']}", line)
        line = re.sub(r"`([^`]+)`", f"{ANSI['yellow']}\\1{ANSI['reset']}", line)
        return line


# --- captcha: single-use tokens require a fresh solve per completion request ---


_DETAILS_RE = re.compile(r'<details type="reasoning"[^>]*>(.*?)</details>', re.S)


def split_reasoning(text):
    """Return (reasoning_or_None, answer_without_reasoning)."""
    m = _DETAILS_RE.search(text)
    if not m:
        return None, text
    reasoning = m.group(1).strip()
    answer = (text[:m.start()] + text[m.end():]).strip()
    return reasoning, answer


def solve_fresh(solver, state, one_shot_attempts=3):
    """Acquire a FRESH single-use captcha + cookie pair.

    AliyunCaptcha tokens are single-use: re-sending an already-consumed token
    to chat.completions triggers FRONTEND_CAPTCHA_REQUIRED (F018). This never
    returns a stale token — it tries the warm solver, then up to
    `one_shot_attempts` headless solves, and returns None only if all fail.
    Updates state["captcha"]/["cookie"] in place on success."""
    if solver is not None:
        try:
            captcha, cookie = solver.solve(state["token"])
        except Exception:
            captcha, cookie = None, None
        if captcha:
            state["captcha"], state["cookie"] = captcha, cookie
            print("[*] fresh captcha from warm solver", file=sys.stderr)
            return captcha, cookie
    for i in range(one_shot_attempts):
        try:
            captcha, cookie = asyncio.run(steal_captcha(state["token"]))
        except Exception:
            captcha, cookie = None, None
        if captcha:
            state["captcha"], state["cookie"] = captcha, cookie
            print(f"[*] fresh captcha from one-shot solve ({i + 1})", file=sys.stderr)
            return captcha, cookie
        print(f"[!] one-shot captcha solve attempt {i + 1} failed", file=sys.stderr)
    return None


def print_help():
    print("""
  Commands:
    /model <name>      switch model (resets conversation)
    /models            list available models
    /new               start a fresh conversation
    /think on|off      toggle Deep Think (default: on)
    /effort high|max   reasoning effort (used when Deep Think is on)
    /search on|off     web search
    /temp <0..2>       temperature
    /max <n>           max_tokens (default 8192)
    /status            show all options + conversation state
    /usage             show session token/usage stats
    /history [n]       show last n exchanges
    /export [file]     write transcript to markdown
    /pretty on|off     toggle colored markdown rendering (default: on, tty)
    /clear             clear on-screen transcript
    /help              show this help
    exit|quit|q        quit
""")


def print_models(current):
    print("\n  Available models:")
    for mid, name in model_catalog():
        marker = " <--" if mid == current else ""
        shown = f"{mid} ({name})" if name != mid else mid
        print(f"    {shown}{marker}")
    print()


def fmt_status(state):
    s = [f"  model: {state['model']}"]
    s.append(f"  deep think: {'ON' if state['enable_thinking'] else 'off'}"
             + (f" (effort {state['reasoning_effort']})" if state["enable_thinking"] else ""))
    s.append(f"  web search: {'ON' if state['web_search'] else 'off'}")
    s.append(f"  temperature: {state['temperature']}  max_tokens: {state['max_tokens']}")
    s.append(f"  chat: {state['chat_id'] or '(none — created on first prompt)'}")
    return "\n".join(s)


def send_message(state, prompt, md, debug_sse=False):
    """Send one prompt in the current conversation, streaming to stdout.
    Returns (ok, error). Updates state: chat_id, last_assistant_id,
    last_assistant_parent_id, history, usage.

    Multi-turn threading follows the app's own request builder: for a follow-up
    the new user node attaches under the last assistant node, so the body sends
        current_user_message_id = last assistant id
        current_user_message_parent_id = that assistant's parent id
    while `id` is a fresh uuid for the new user message."""
    token = state["token"]
    token = refresh_token(token)
    state["token"] = token

    if state["chat_id"] is None:
        print("[*] creating conversation...", file=sys.stderr)
        chat_id, _ = create_chat(token, prompt, model=state["model"], cookie=state["cookie"],
                                 enable_thinking=state["enable_thinking"],
                                 reasoning_effort=state["reasoning_effort"])
        state["chat_id"] = chat_id
    else:
        chat_id = state["chat_id"]

    is_first = not state["history"]
    if is_first:
        current_user_msg_id = str(uuid.uuid4())
        current_user_parent_id = None
    else:
        current_user_msg_id = state["last_assistant_id"]
        current_user_parent_id = state["last_assistant_parent_id"]

    sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                               current_url=f"{BASE}/c/{chat_id}")
    url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"

    body_messages = list(state["history"]) + [{"role": "user", "content": prompt}]

    fresh = solve_fresh(state.get("solver"), state)
    if not fresh:
        return False, ("could not acquire a fresh captcha token "
                       "(warm + one-shot solve failed); please retry")
    captcha, _ = fresh

    def attempt(captcha):
        body = {
            "chat_id": chat_id,
            "current_user_message_id": current_user_msg_id,
            "current_user_message_parent_id": current_user_parent_id,
            "extra": {},
            "features": build_features(state["enable_thinking"], state["reasoning_effort"],
                                       state["web_search"]),
            "id": str(uuid.uuid4()),
            "messages": body_messages,
            "model": state["model"],
            "params": {"max_tokens": state["max_tokens"],
                       "temperature": state["temperature"], "top_p": 0.95},
            "signature_prompt": prompt,
            "stream": True,
            "variables": {},
            "captcha_verify_param": captcha,
        }
        if is_first:
            body["background_tasks"] = {"title_generation": True, "tags_generation": True}
        try:
            resp = requests.post(url, headers=headers(token, sig, state["cookie"]),
                                 json=body, stream=True, timeout=180)
        except requests.exceptions.RequestException as e:
            return None, f"upstream error: {e}"
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:500]}"

        collected = []
        new_id = state["last_assistant_id"]
        new_parent = state["last_assistant_parent_id"]
        usage = None
        stream_error = None
        captcha_error = None
        stream_status = None
        buf = []
        thinking_len = 0

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if debug_sse:
                print(f"[sse] {data}", file=sys.stderr)
            try:
                j = json.loads(data)
            except Exception:
                continue
            inner = j.get("data")
            if not isinstance(inner, dict):
                continue
            phase = inner.get("phase")
            delta = inner.get("delta_content")
            edit_index = inner.get("edit_index")
            edit_content = inner.get("edit_content")
            if inner.get("id"):
                new_id = inner["id"]
            if inner.get("parent_message_id"):
                new_parent = inner["parent_message_id"]
            if inner.get("usage"):
                usage = inner["usage"]
            if inner.get("error"):
                e = inner["error"]
                code = e.get("code") if isinstance(e, dict) else None
                if code == "FRONTEND_CAPTCHA_REQUIRED":
                    captcha_error = e
                elif stream_error is None:
                    stream_error = e
            if inner.get("status"):
                stream_status = inner["status"]
            if delta:
                buf.append(delta)
                if phase == "thinking":
                    thinking_len += len(delta)
            elif edit_index is not None and edit_content is not None:
                idx = int(edit_index)
                cur = "".join(buf)
                buf = [cur[:idx] + edit_content]
                if idx < thinking_len:
                    thinking_len = idx

        full = "".join(buf)
        reasoning, text = split_reasoning(full)
        if reasoning is None and thinking_len:
            reasoning = full[:thinking_len].strip() or None
            text = full[thinking_len:]
        if reasoning:
            print(f"{ANSI['dim']}── reasoning ──{ANSI['reset']}", file=sys.stderr)
            for ln in reasoning.splitlines():
                print(f"{ANSI['dim']}{ln}{ANSI['reset']}", file=sys.stderr)
            print(f"{ANSI['dim']}{'─' * 14}{ANSI['reset']}", file=sys.stderr)
        text = text.strip()
        if text:
            md.write(text)
        return (text, new_id, new_parent, usage, stream_error, stream_status,
                captcha_error), None

    detail, err = attempt(captcha)
    if err:
        return False, err
    text, new_id, new_parent, usage, stream_error, stream_status, captcha_error = detail
    if not text and captcha_error:
        print("[*] captcha rejected by server; re-solving...", file=sys.stderr)
        fresh = solve_fresh(state.get("solver"), state)
        if fresh:
            detail, err = attempt(fresh[0])
            if err:
                return False, err
            text, new_id, new_parent, usage, stream_error, stream_status, captcha_error = detail

    if not text:
        show_err = captcha_error or stream_error
        if show_err:
            if isinstance(show_err, dict):
                msg = ("GLM error: "
                       + (show_err.get("detail") or show_err.get("code")
                          or json.dumps(show_err)))
            else:
                msg = f"GLM error: {show_err}"
        elif stream_status and stream_status not in (None, "done", "ok", 200, "completed"):
            msg = f"GLM status: {stream_status}"
        else:
            msg = "empty response from GLM"
        if debug_sse:
            msg += " (see [sse] lines above)"
        return False, msg
    state["history"].append({"role": "user", "content": prompt})
    state["history"].append({"role": "assistant", "content": text})
    if len(state["history"]) > 120:
        state["history"] = state["history"][-120:]
    state["last_assistant_id"] = new_id
    state["last_assistant_parent_id"] = new_parent
    if usage:
        state["usage"]["prompts"] += 1
        state["usage"]["in"] += usage.get("prompt_tokens", 0)
        state["usage"]["out"] += usage.get("completion_tokens", 0)
    return True, None


def run_repl(token, start_model="glm-5.2", no_pretty=False, start_think=True,
             debug_sse=False):
    """Interactive REPL loop."""
    state = {
        "token": token,
        "model": start_model,
        "enable_thinking": start_think,
        "reasoning_effort": "max",
        "web_search": False,
        "temperature": 1.0,
        "max_tokens": 8192,
        "chat_id": None,
        "last_assistant_id": None,
        "last_assistant_parent_id": None,
        "history": [],
        "transcript": [],
        "usage": {"prompts": 0, "in": 0, "out": 0},
        "captcha": None,
        "cookie": None,
        "solver": None,
    }

    print()
    print("  ╔════════════════════════════════════════════════════╗")
    print("  ║  GLM interactive client (chat.z.ai)                 ║")
    print("  ╚════════════════════════════════════════════════════╝")

    print("[*] warming captcha solver...", file=sys.stderr)
    solver = CaptchaSolver()
    captcha, cookie = None, None
    try:
        captcha, cookie = solver.start(token)
    except Exception as e:
        print(f"[!] warm solver unavailable ({e}); one-shot solve instead", file=sys.stderr)
        try:
            solver.close()
        except Exception:
            pass
        solver = None
        try:
            captcha, cookie = asyncio.run(steal_captcha(token))
        except Exception:
            captcha, cookie = None, None
    if not captcha:
        print("[!] failed to acquire captcha token — /new may retry", file=sys.stderr)
    state["captcha"], state["cookie"] = captcha, cookie
    state["solver"] = solver

    pretty = not no_pretty
    md = MDPrinter(pretty)
    print("[*] Type /help for commands\n")

    try:
        import readline
        HIST = os.path.expanduser("~/.glm-rev-history")
        if os.path.exists(HIST):
            readline.read_history_file(HIST)
        readline.set_history_length(1000)
        readline.set_completer(
            lambda text, st: [f"/{c}" for c in SLASH_COMMANDS
                              if text.startswith("/") and c.startswith(text[1:])][st]
            if text.startswith("/") else None)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    while True:
        try:
            inp = input("  You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not inp:
            continue
        if inp.lower() in ("exit", "quit", "q"):
            break

        if inp.startswith("/"):
            parts = inp.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/help":
                print_help()
            elif cmd == "/models":
                print_models(state["model"])
            elif cmd == "/model":
                if not arg:
                    print(f"  Current: {state['model']}\n  Usage: /model <name> (/models)")
                elif arg not in [m for m, _ in model_catalog()]:
                    print(f"  [!] unknown model: {arg} (/models)")
                elif arg != state["model"]:
                    state["model"] = arg
                    state["chat_id"] = None
                    state["last_assistant_id"] = None
                    state["last_assistant_parent_id"] = None
                    state["history"] = []
                    print(f"  [+] Model -> {arg} (fresh conversation)")
                else:
                    print(f"  [+] Already on {arg}")
            elif cmd == "/new":
                state["chat_id"] = None
                state["last_assistant_id"] = None
                state["last_assistant_parent_id"] = None
                state["history"] = []
                state["transcript"] = []
                print("  [+] Fresh conversation (memory reset)")
            elif cmd == "/think":
                if arg.lower() in ("on", "1", "true"):
                    state["enable_thinking"] = True
                    print("  [+] Deep Think ON")
                elif arg.lower() in ("off", "0", "false"):
                    state["enable_thinking"] = False
                    print("  [+] Deep Think OFF")
                else:
                    print(f"  Deep Think: {'ON' if state['enable_thinking'] else 'off'}"
                          f"  Usage: /think on|off")
            elif cmd == "/effort":
                if arg in ("high", "max"):
                    state["reasoning_effort"] = arg
                    print(f"  [+] Reasoning effort -> {arg}")
                else:
                    print(f"  Effort: {state['reasoning_effort']}  Usage: /effort high|max")
            elif cmd == "/search":
                if arg.lower() in ("on", "1", "true"):
                    state["web_search"] = True
                    print("  [+] Web search ON")
                elif arg.lower() in ("off", "0", "false"):
                    state["web_search"] = False
                    print("  [+] Web search OFF")
                else:
                    print(f"  Web search: {'ON' if state['web_search'] else 'off'}"
                          f"  Usage: /search on|off")
            elif cmd == "/temp":
                try:
                    v = float(arg)
                    if 0 <= v <= 2:
                        state["temperature"] = v
                        print(f"  [+] Temperature -> {v}")
                    else:
                        print("  [!] temperature must be in [0, 2]")
                except ValueError:
                    print(f"  Temperature: {state['temperature']}  Usage: /temp <0..2>")
            elif cmd == "/max":
                try:
                    v = int(arg)
                    if v >= 256:
                        state["max_tokens"] = v
                        print(f"  [+] max_tokens -> {v}")
                    else:
                        print("  [!] max_tokens must be >= 256")
                except ValueError:
                    print(f"  max_tokens: {state['max_tokens']}  Usage: /max <n>")
            elif cmd == "/status":
                print(fmt_status(state))
            elif cmd == "/usage":
                u = state["usage"]
                print(f"  prompts: {u['prompts']}")
                print(f"  tokens:  {u['in']} in / {u['out']} out (from usage field)")
            elif cmd == "/history":
                n = int(arg) if arg.isdigit() else 10
                for m in state["transcript"][-2 * n:]:
                    who = "You" if m["role"] == "user" else "GLM"
                    t = m["text"].replace("\n", " ")
                    print(f"  {who}: {t[:300]}{'...' if len(t) > 300 else ''}")
            elif cmd == "/export":
                if not state["transcript"]:
                    print("  (nothing to export yet)")
                else:
                    path = arg or f"glm-export-{time.strftime('%Y%m%d-%H%M%S')}.md"
                    with open(path, "w") as f:
                        f.write("# GLM export\n\n")
                        f.write(f"- model: {state['model']}\n"
                                f"- chat: {state['chat_id'] or '(none)'}\n"
                                f"- exported: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
                        for m in state["transcript"]:
                            who = "You" if m["role"] == "user" else "GLM"
                            f.write(f"## {who}\n\n{m['text']}\n\n")
                    print(f"  [+] exported {len(state['transcript'])} messages -> {path}")
            elif cmd == "/pretty":
                if arg in ("on", "1", "true"):
                    pretty = True
                    md.enabled = pretty and sys.stdout.isatty()
                    print("  [+] pretty output ON")
                elif arg in ("off", "0", "false"):
                    pretty = False
                    md.enabled = False
                    print("  [+] pretty output OFF (raw)")
                else:
                    print(f"  pretty output: {'ON' if md.enabled else 'off'}")
            elif cmd == "/clear":
                state["transcript"] = []
                print("  [+] on-screen transcript cleared (memory kept)")
            else:
                print(f"  Unknown command: {cmd}  (/help)")
            continue

        try:
            ok, err = send_message(state, inp, md, debug_sse=debug_sse)
            if not ok:
                print(f"\n[!] {err}", file=sys.stderr)
            else:
                state["transcript"].append({"role": "user", "text": inp})
                state["transcript"].append({"role": "assistant",
                                            "text": state["history"][-1]["content"]})
                if pretty:
                    md.finish()
                print()
        except KeyboardInterrupt:
            print("\n[!] interrupted", file=sys.stderr)

    u = state["usage"]
    print(f"[session] {u['prompts']} prompts, {u['in']} in / {u['out']} out tokens",
          file=sys.stderr)
    solver = state.get("solver")
    if solver:
        try:
            solver.close()
        except Exception:
            pass
    try:
        import readline
        readline.write_history_file(os.path.expanduser("~/.glm-rev-history"))
    except Exception:
        pass