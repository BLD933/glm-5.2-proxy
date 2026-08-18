"""GLM REPL — interactive terminal UI with full option toggles.

Inspired by claude-rev's ui.py: streaming markdown-lite ANSI rendering,
slash commands, persistent conversation state, and transcript export.
"""
import json
import os
import re
import sys
import threading
import time
import uuid

from .config import BASE, ENDPOINT, HISTORY_LIMIT, LOCAL_TOOL_NAMES, load_allowlist, save_allowlist
from .api import sign, user_id_from_token
from .client import refresh_token, create_chat, build_features, stream_turn, fetch_reply_node
from .solver import CaptchaSolver, solve_fresh, register_solver
from .tools import send_with_tools
from .mcp import MCPManager, load_mcp_config
from .render import ReasoningStream

HERE = os.path.dirname(os.path.abspath(__file__))

SLASH_COMMANDS = ["allow", "clear", "effort", "export", "help", "history",
                  "max", "mcp", "model", "models", "new", "pretty", "search",
                  "status", "temp", "think", "tools", "usage"]

ANSI = {
    "reset": "\x1b[0m", "bold": "\x1b[1m", "dim": "\x1b[2m", "underline": "\x1b[4m",
    "green": "\x1b[32m", "yellow": "\x1b[33m", "blue": "\x1b[34m", "cyan": "\x1b[36m",
    "red": "\x1b[31m", "magenta": "\x1b[35m",
}


def _color(enabled, key, text):
    if not enabled:
        return text
    return f"{ANSI[key]}{text}{ANSI['reset']}"


def st_note(msg, enabled=True):
    """Status/progress line -> stderr, dim."""
    print(_color(enabled, "dim", msg), file=sys.stderr)


def st_ok(msg, enabled=True):
    """Success line -> stderr, green."""
    print(_color(enabled, "green", msg), file=sys.stderr)


def st_warn(msg, enabled=True):
    """Non-fatal warning line -> stderr, yellow."""
    print(_color(enabled, "yellow", msg), file=sys.stderr)


def st_err(msg, enabled=True):
    """Error line -> stderr, red."""
    print(_color(enabled, "red", msg), file=sys.stderr)


class working:
    """Progress beacon: prints an elapsed-time line to stderr every `every` seconds
    while a long blocking operation runs, so the UI never looks frozen."""

    def __init__(self, msg, every=10, enabled=True):
        self._msg = msg
        self._every = every
        self._enabled = enabled
        self._stop = threading.Event()
        self._thr = None

    def __enter__(self):
        self._stop.clear()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)

    def stop(self):
        """Stop the beacon immediately (e.g. once the first token arrives)."""
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)

    def _run(self):
        t0 = time.time()
        while not self._stop.wait(self._every):
            st_note(f"[*] {self._msg} — {int(time.time() - t0)}s...")

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

    def __init__(self, enabled=True, header=None):
        self.enabled = enabled and sys.stdout.isatty()
        self._buf = ""
        self._in_code = False
        self._header = header
        self._started = False

    def write(self, chunk):
        if not self.enabled:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            return
        if not self._started and chunk.strip():
            self._started = True
            if self._header:
                print(self._header)
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
# (solve_fresh now lives in solver.py; split_reasoning/stream_turn in client.py)


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
    /tools on|off|list toggle local + MCP tool execution
    /allow <path>|url  add a path or domain to the tool allowlist
    /mcp status|connect|disconnect|reload|allow|deny|tools   manage MCP servers
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
    s.append(f"  tools: {'ON' if state.get('tools_on') else 'off'}"
             + (f" ({state['mcp'].connected_count()} MCP server(s), "
                f"{state['mcp'].tool_count()} tool(s))" if state.get("mcp") else ""))
    s.append(f"  chat: {state['chat_id'] or '(none — created on first prompt)'}")
    return "\n".join(s)


def send_message(state, prompt, md, debug_sse=False, on_token=None):
    """Send one prompt in the current conversation, streaming to stdout.
    Returns (ok, error). Updates state: chat_id, last_assistant_id,
    last_assistant_parent_id, history, usage.

    `on_token` (optional) is invoked once when the first thinking or answer
    delta arrives, so callers can stop a progress beacon mid-stream.

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
        chat_id, seed_msg_id, *_ = create_chat(token, prompt, model=state["model"],
                                               cookie=state["cookie"],
                                               messages=list(state["history"]) + [{"role": "user", "content": prompt}],
                                               enable_thinking=state["enable_thinking"],
                                               reasoning_effort=state["reasoning_effort"])
        state["chat_id"] = chat_id
        state["seed_msg_id"] = seed_msg_id
    else:
        chat_id = state["chat_id"]

    is_first = not state["history"]
    current_user_msg_id = (state.get("seed_msg_id") if is_first and state.get("seed_msg_id")
                           else str(uuid.uuid4()))
    current_user_parent_id = None if is_first else state.get("last_assistant_id")

    sig, url_params, ts = sign(prompt, user_id_from_token(token), token,
                               current_url=f"{BASE}/c/{chat_id}")
    url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"

    body_messages = list(state["history"]) + [{"role": "user", "content": prompt}]

    fresh = solve_fresh(state.get("solver"), state)
    if not fresh:
        return False, ("could not acquire a fresh captcha token "
                       "(warm + one-shot solve failed); please retry")

    def attempt(pair):
        captcha, cookie = pair
        rstream = ReasoningStream()
        fired = {"token": False}
        streamed = {"answer": False}

        def _first_token(_d):
            if not fired["token"]:
                fired["token"] = True
                if on_token is not None:
                    on_token()

        def _on_answer(delta):
            _first_token(delta)
            if delta:
                streamed["answer"] = True
                md.write(delta)

        res = stream_turn(token=token, cookie=cookie, sig=sig, url=url,
                          chat_id=chat_id, model=state["model"],
                          messages=body_messages,
                          current_user_message_id=current_user_msg_id,
                          current_user_message_parent_id=current_user_parent_id,
                          is_first=is_first,
                          features=build_features(state["enable_thinking"],
                                                  state["reasoning_effort"],
                                                  state["web_search"]),
                          params={"max_tokens": state["max_tokens"],
                                  "temperature": state["temperature"], "top_p": 0.95},
                          captcha=captcha, debug_sse=debug_sse,
                          on_thinking=lambda d: (rstream.write(d), _first_token(d)),
                          on_answer=_on_answer)
        if "error" in res:
            return None, res["error"]
        reasoning = res.get("reasoning")
        if rstream.started:
            rstream.finish()
        elif reasoning:
            print(f"{ANSI['dim']}── reasoning ──{ANSI['reset']}", file=sys.stderr)
            for ln in reasoning.splitlines():
                print(f"{ANSI['dim']}{ln}{ANSI['reset']}", file=sys.stderr)
            print(f"{ANSI['dim']}{'─' * 14}{ANSI['reset']}", file=sys.stderr)
        text = (res.get("answer") or "").strip()
        if text and not streamed["answer"]:
            md.write(text)
        return (text, res.get("id"), res.get("parent"), res.get("usage"),
                res.get("stream_error"), res.get("stream_status"),
                res.get("captcha_error")), None

    detail, err = attempt(fresh)
    if err:
        return False, err
    text, new_id, new_parent, usage, stream_error, stream_status, captcha_error = detail
    if not text and captcha_error:
        print("[*] captcha rejected by server; re-solving...", file=sys.stderr)
        fresh = solve_fresh(state.get("solver"), state)
        if fresh:
            detail, err = attempt(fresh)
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
    if len(state["history"]) > HISTORY_LIMIT:
        state["history"] = state["history"][-HISTORY_LIMIT:]
    state["last_assistant_id"] = new_id
    state["last_assistant_parent_id"] = new_parent
    # The SSE stream usually omits the stored assistant node id for this turn.
    # Follow-ups MUST be parented at the server-assigned reply node (POSTed
    # /rewritten ids are DEAF — verified live 2026-08-16), so learn it via GET.
    aid, _ = fetch_reply_node(token, chat_id, after_user_id=current_user_msg_id,
                              cookie=state["cookie"])
    if aid:
        state["last_assistant_id"] = aid
    if usage:
        state["usage"]["prompts"] += 1
        state["usage"]["in"] += usage.get("prompt_tokens", 0)
        state["usage"]["out"] += usage.get("completion_tokens", 0)
    return True, None


def run_repl(token, start_model="glm-5.2", no_pretty=False, start_think=True,
             debug_sse=False, tools=False, no_mcp=False):
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
        "tools_on": tools,
        "mcp": None,
    }

    print()
    print("  ╔════════════════════════════════════════════════════╗")
    print("  ║  GLM interactive client (chat.z.ai)                 ║")
    print("  ╚════════════════════════════════════════════════════╝")

    st_note("[*] warming captcha solver in background (playwright)")
    try:
        register_solver(None, token)
    except Exception:
        pass
    solver = CaptchaSolver()
    solver.start_background(token)
    state["solver"] = solver

    mcp = None
    if not no_mcp:
        try:
            cfg = load_mcp_config()
            if cfg:
                st_note(f"[*] connecting {len(cfg)} MCP server(s)...")
                mcp = MCPManager(cfg)
                mcp.start()
                if mcp.connected_count():
                    st_ok(f"[+] MCP: {mcp.connected_count()} server(s), "
                       f"{mcp.tool_count()} tool(s)")
                else:
                    st_warn("[!] MCP: no servers connected")
            else:
                st_note("[*] no MCP servers configured (~/.mcp.json)")
        except Exception as e:
            st_warn(f"[!] MCP unavailable ({e})")
            mcp = None
    state["mcp"] = mcp

    pretty = not no_pretty
    md = MDPrinter(pretty,
                   header=(f"\n{ANSI['dim']}── GLM ──{ANSI['reset']}" if pretty else None))
    tools_on = state.get("tools_on", False)
    mcp_str = f"{mcp.connected_count()} server(s), {mcp.tool_count()} tool(s)" if mcp else "disabled"
    st_ok(f"[+] model: {state['model']}   tools: {'ON' if tools_on else 'off'}   "
       f"MCP: {mcp_str}   pretty: {'ON' if pretty else 'off'}")
    st_note("[*] Type /help for commands")
    print()

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
            you = _color(pretty, "cyan", "You")
            inp = input(f"  {you} > ").strip()
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
            elif cmd == "/tools":
                if arg.lower() in ("on", "1", "true"):
                    state["tools_on"] = True
                    print("  [+] Tools ON (local + MCP)")
                elif arg.lower() in ("off", "0", "false"):
                    state["tools_on"] = False
                    print("  [+] Tools OFF")
                else:
                    print(f"  tools: {'ON' if state['tools_on'] else 'off'}")
                    print(f"  local tools: {len(LOCAL_TOOL_NAMES)} "
                          f"({', '.join(LOCAL_TOOL_NAMES)})")
                    print(f"  MCP tools: {state['mcp'].tool_count() if state['mcp'] else 0}")
                    print("  Usage: /tools on|off|list")
            elif cmd == "/allow":
                if not arg:
                    al = load_allowlist()
                    print("  allowlist paths:")
                    for p in al["paths"]:
                        print(f"    {p}")
                    print("  allowlist domains:")
                    for d in al["domains"]:
                        print(f"    {d}")
                    print("  allowlist servers:")
                    for s in al.get("servers", []):
                        print(f"    {s}")
                    print("  Usage: /allow <path> | <url>")
                else:
                    from urllib.parse import urlparse
                    u = urlparse(arg)
                    if u.scheme in ("http", "https") and u.hostname:
                        al = load_allowlist()
                        if u.hostname not in al["domains"]:
                            al["domains"].append(u.hostname)
                            save_allowlist(al)
                        print(f"  [+] allowlisted domain: {u.hostname}")
                    else:
                        p = os.path.abspath(os.path.expanduser(arg))
                        al = load_allowlist()
                        if p not in al["paths"]:
                            al["paths"].append(p)
                            save_allowlist(al)
                        print(f"  [+] allowlisted path: {p}")
            elif cmd == "/mcp":
                if arg in ("", "status", "list"):
                    if state["mcp"] is None:
                        print("  (MCP disabled at startup)")
                    else:
                        for ln in state["mcp"].status_lines():
                            print(ln)
                        print(f"  registry: {state['mcp'].tool_count()} tool(s)")
                elif arg in ("connect", "reload"):
                    if state["mcp"] is None:
                        print("  [!] MCP was disabled at startup (--no-mcp)")
                    else:
                        print("  [*] (re)connecting MCP servers...")
                        state["mcp"].stop()
                        state["mcp"].start()
                        print(f"  [+] {state['mcp'].connected_count()} server(s), "
                              f"{state['mcp'].tool_count()} tool(s)")
                elif arg == "disconnect":
                    if state["mcp"] is not None:
                        state["mcp"].stop()
                        print("  [+] MCP disconnected")
                elif arg.startswith("allow "):
                    name = arg.split(None, 1)[1].strip()
                    if state["mcp"] is not None and name in state["mcp"].config:
                        al = load_allowlist()
                        al.setdefault("servers", [])
                        if name not in al["servers"]:
                            al["servers"].append(name)
                            save_allowlist(al)
                        print(f"  [+] MCP server allowlisted: {name}")
                    else:
                        print(f"  [!] unknown MCP server: {name}")
                elif arg.startswith("deny "):
                    name = arg.split(None, 1)[1].strip()
                    al = load_allowlist()
                    if name in al.get("servers", []):
                        al["servers"].remove(name)
                        save_allowlist(al)
                        print(f"  [+] MCP server removed from allowlist: {name}")
                    else:
                        print(f"  {name}: not in allowlist")
                elif arg == "tools":
                    if state["mcp"] is not None:
                        for t in state["mcp"].tool_names():
                            print(f"    {t}")
                    else:
                        print("  (MCP disabled at startup)")
                else:
                    print("  Usage: /mcp status|connect|disconnect|reload|"
                          "allow <name>|deny <name>|tools")
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
            with working("waiting for GLM") as beacon:
                if state["tools_on"]:
                    ok, err = send_with_tools(state, inp, md=md, mcp=state.get("mcp"),
                                              solver=state.get("solver"),
                                              debug_sse=debug_sse,
                                              on_token=beacon.stop)
                else:
                    ok, err = send_message(state, inp, md, debug_sse=debug_sse,
                                           on_token=beacon.stop)
            if not ok:
                print(f"\n[!] {err}", file=sys.stderr)
            else:
                state["transcript"].append({"role": "user", "text": inp})
                if state["history"] and state["history"][-1].get("role") == "assistant":
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
    mcp = state.get("mcp")
    if mcp:
        try:
            mcp.stop()
        except Exception:
            pass
    try:
        import readline
        readline.write_history_file(os.path.expanduser("~/.glm-rev-history"))
    except Exception:
        pass