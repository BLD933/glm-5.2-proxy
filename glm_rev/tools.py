"""Local + MCP tool execution: run TOOL: calls from GLM's replies on this machine.

Port of claude-rev's tools.py, adapted for glm-rev's single-use captcha model
(fresh token per loop iteration) and id-based multi-turn threading.

execute_tool          — the actual local operations (read/write/list/run/web_fetch).
approve_tool          — REPL approval gate: allowlist auto-approves reads, writes/shell ask.
approve_tool_auto     — non-interactive allowlist-only gate for the headless server.
approve_mcp/auto      — MCP equivalents (server allowlist auto-approves).
send_with_tools       — prompt + tool loop: send, detect TOOL: lines, execute, re-send.
"""
import os
import subprocess
import sys
import uuid

from .api import sign, headers, user_id_from_token
from .client import refresh_token, create_chat, build_features, stream_turn
from .config import (BASE, ENDPOINT, LOCAL_TOOL_NAMES, MAX_TOOL_ITERS, REFUSAL_RE,
                     TOOL_CONTRACT, TOOL_HINT, TOOL_NUDGE, UA, domain_allowed,
                     load_allowlist, path_allowed, safe_json, parse_tool_calls,
                     save_allowlist, strip_tool_lines)
from .solver import solve_fresh
from .render import ReasoningStream

ANSI = {
    "reset": "\x1b[0m", "bold": "\x1b[1m", "dim": "\x1b[2m", "underline": "\x1b[4m",
    "green": "\x1b[32m", "yellow": "\x1b[33m", "blue": "\x1b[34m", "cyan": "\x1b[36m",
    "red": "\x1b[31m", "magenta": "\x1b[35m",
}


def execute_tool(name, args):
    """Execute a tool call locally. Returns (ok, text_result)."""
    try:
        if name == "read_file":
            raw_p = str(args.get("path", "")).strip("\"'")
            if raw_p in ("path:", ""): raw_p = "."
            p = os.path.expanduser(raw_p)
            with open(p, "rb") as f:
                data = f.read(256 * 1024 + 1)
            text = data.decode("utf-8", "replace")
            if len(data) > 256 * 1024:
                text = "(truncated to 256KB)\n" + text[:256 * 1024]
            numbered = "\n".join(f"{i + 1:6d}| {ln}" for i, ln in enumerate(text.split("\n")))
            return True, numbered
        if name == "list_dir":
            raw_p = str(args.get("path", "")).strip("\"'")
            if raw_p in ("path:", ""): raw_p = "."
            p = os.path.expanduser(raw_p)
            entries = sorted(os.listdir(p))
            return True, "\n".join(entries[:500]) if entries else "(empty)"
        if name == "write_file":
            raw_p = str(args.get("path", "")).strip("\"'")
            p = os.path.expanduser(raw_p)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            with open(p, "w") as f:
                f.write(args.get("content", ""))
            return True, f"wrote {len(args.get('content', ''))} bytes to {p}"
        if name == "edit_lines":
            raw_p = str(args.get("path", "")).strip("\"'")
            p = os.path.expanduser(raw_p)
            start = int(args.get("start", 1))
            delete = int(args.get("delete", 0))
            if start < 1:
                return False, "start must be >= 1 (1-based line number)"
            with open(p, encoding="utf-8", errors="replace") as f:
                lines = f.read().split("\n")
            before = len(lines)
            begin = min(start - 1, len(lines))
            end = min(begin + delete, len(lines))
            content = args.get("content", "")
            del lines[begin:end]
            lines[begin:begin] = content.split("\n")
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return True, (f"edited {p}: replaced {end - begin} line(s) at line {start} "
                          f"with {len(content.split(chr(10)))} new line(s) "
                          f"({before} -> {len(lines)} lines total)")
        if name == "run_command":
            cmd = str(args.get("cmd") or args.get("command") or "").strip()
            if not cmd:
                return False, "No command provided"
            r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)
            out = (r.stdout or "") + (("\n[stderr] " + r.stderr) if r.stderr else "")
            return r.returncode == 0, (out[:50000] or "(no output)") + f" [exit {r.returncode}]"
        if name == "web_fetch":
            from urllib.parse import urlparse
            raw_url = str(args.get("url", "")).strip("\"'")
            if not raw_url.startswith(("http://", "https://")):
                raw_url = "https://" + raw_url
            import urllib.request
            u = urlparse(raw_url)
            if u.scheme not in ("http", "https"):
                return False, "only http/https allowed"
            req = urllib.request.Request(raw_url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read(5 * 1024 * 1024 + 1)
            text = data.decode("utf-8", "replace")
            if len(data) > 5 * 1024 * 1024:
                text = text[:5 * 1024 * 1024] + "\n(truncated to 5MB)"
            return True, text[:30000]
        return False, f"unknown tool: {name}"
    except Exception as e:
        return False, f"ERROR: {e}"


def approve_tool(name, args):
    """REPL approval gate: allowlist auto-approves reads/edits/web; writes/shell always ask."""
    from urllib.parse import urlparse

    al = load_allowlist()
    if name in ("read_file", "list_dir", "edit_lines") and path_allowed(args.get("path", ""), al):
        return True
    if name == "web_fetch" and domain_allowed(args.get("url", ""), al):
        return True
    while True:
        try:
            a = input(f"  approve {name}({safe_json(args)[:160]})? [y/N/a=always] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if a in ("y", "yes"):
            return True
        if a in ("a", "always"):
            if name in ("read_file", "list_dir") and args.get("path"):
                al["paths"].append(os.path.abspath(os.path.expanduser(args["path"])))
            elif name == "web_fetch" and args.get("url"):
                host = urlparse(args["url"]).hostname
                if host:
                    al["domains"].append(host)
            save_allowlist(al)
            return True
        if a in ("n", "no", ""):
            return False


def approve_tool_auto(name, args):
    """Server-side non-interactive approval policy.

    Reads and web fetch approve if allowlisted. Writes/edits/shell commands only
    run when GLM_TOOL_AUTORUN=1 (writes/edits also need their path allowlisted),
    so an OpenAI client hitting the server cannot silently mutate files or run
    commands by default."""
    al = load_allowlist()
    if name in ("read_file", "list_dir") and path_allowed(args.get("path", ""), al):
        return True
    if name == "web_fetch" and domain_allowed(args.get("url", ""), al):
        return True
    if os.environ.get("GLM_TOOL_AUTORUN") == "1":
        if name in ("write_file", "edit_lines") and path_allowed(args.get("path", ""), al):
            return True
        if name == "run_command":
            return True
    return False


def approve_mcp(server_name, name, args):
    """REPL approval for MCP tools: server allowlist auto-approves, else ask."""
    al = load_allowlist()
    if server_name in al.get("servers", []):
        return True
    while True:
        try:
            a = input(f"  approve {server_name}:{name}({safe_json(args)[:160]})? "
                      f"[y/N/a=always] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if a in ("y", "yes"):
            return True
        if a in ("a", "always"):
            al.setdefault("servers", [])
            if server_name not in al["servers"]:
                al["servers"].append(server_name)
            save_allowlist(al)
            return True
        if a in ("n", "no", ""):
            return False


def approve_mcp_auto(server_name, name, args):
    al = load_allowlist()
    return server_name in al.get("servers", [])


def build_contract(mcp=None):
    """TOOL_CONTRACT plus the MCP tool appendix, when a manager is connected."""
    block = TOOL_CONTRACT
    if mcp is not None:
        appendix = mcp.contract_block()
        if appendix.strip():
            block += "\n" + appendix
    return block


def dispatch_tool(name, args, mcp=None):
    """Route a TOOL: call to a local executor or an MCP server. Returns (ok, text)."""
    if name in LOCAL_TOOL_NAMES:
        return execute_tool(name, args)
    if mcp is not None and name in mcp.registry:
        return mcp.call_tool(name, args)
    return False, f"unknown tool: {name}"


def _fmt_error(res, debug_sse):
    show_err = res.get("captcha_error") or res.get("stream_error")
    if show_err:
        if isinstance(show_err, dict):
            msg = ("GLM error: "
                   + (show_err.get("detail") or show_err.get("code")
                      or safe_json(show_err)))
        else:
            msg = f"GLM error: {show_err}"
    elif res.get("stream_status") and res.get("stream_status") not in (None, "done", "ok", 200, "completed"):
        msg = f"GLM status: {res.get('stream_status')}"
    else:
        msg = "empty response from GLM"
    if debug_sse:
        msg += " (see [sse] lines above)"
    return msg


def _commit_history(prior, is_first, mcp, prompt, text):
    """Persist only the clean (user: prompt, assistant: final_answer) pair for a
    completed tool turn, dropping in-flight loop artifacts ([Tool result],
    DECLINED, TOOL_NUDGE, TOOL_HINT, empty TOOL-only replies) from history so
    later turns don't replay operator directives to the model."""
    clean = list(prior)
    if is_first and not clean:
        clean.append({"role": "user",
                      "content": build_contract(mcp) + "\n\nUser: " + prompt})
    else:
        clean.append({"role": "user", "content": prompt})
    answer = strip_tool_lines(text).strip()
    if answer:
        clean.append({"role": "assistant", "content": answer})
    return clean[-120:]


def send_with_tools(state, prompt, md=None, mcp=None, solver=None, debug_sse=False,
                    auto_approve=False, writer=None, captcha_fn=None, on_token=None):
    """Prompt + tool loop adapted to glm-rev.

    Each tool-loop iteration requires a FRESH single-use captcha (solve_fresh,
    or `captcha_fn` when supplied), and multi-turn threading follows the app's
    request builder:
      iteration 0 -> current_user_message_id = fresh uuid, parent = None
      iteration k -> current_user_message_id = last assistant id, parent = its parent
    The full conversation accumulates in state["history"] so later REPL turns
    continue the same thread. Reasoning and tool activity print to stderr; the
    final answer streams through `writer` (server) or `md` (REPL).
    Returns (ok, error) and updates state: chat_id, last_assistant_id,
    last_assistant_parent_id, history, usage, cookie."""
    token = refresh_token(state["token"])
    state["token"] = token
    solver = solver if solver is not None else state.get("solver")

    chat_id = state["chat_id"]
    is_first = chat_id is None
    seed_msg_id = None
    if is_first:
        print(f"{ANSI['dim']}[*] creating conversation...{ANSI['reset']}", file=sys.stderr)
        seed_msgs = list(state["history"]) + [{"role": "user", "content": prompt}]
        chat_id, seed_msg_id = create_chat(token, prompt, model=state["model"],
                                           cookie=state["cookie"],
                                           messages=seed_msgs,
                                           enable_thinking=state["enable_thinking"],
                                           reasoning_effort=state["reasoning_effort"])
        state["chat_id"] = chat_id

    hist = list(state["history"])
    prior = list(state["history"])
    last_ast_id = state.get("last_assistant_id")

    def send(content, first):
        nonlocal last_ast_id
        sig, url_params, ts = sign(content, user_id_from_token(token), token,
                                   current_url=f"{BASE}/c/{chat_id}")
        url = f"{BASE}{ENDPOINT}?{url_params}&signature_timestamp={ts}"
        messages = hist + [{"role": "user", "content": content}]
        params = {"max_tokens": state["max_tokens"],
                  "temperature": state["temperature"], "top_p": 0.95}
        features = build_features(state["enable_thinking"], state["reasoning_effort"],
                                  state["web_search"])

        fresh = captcha_fn() if captcha_fn is not None else solve_fresh(solver, state)
        if not fresh:
            return False, ("could not acquire a fresh captcha token "
                           "(warm + one-shot solve failed); please retry")
        captcha, cookie = fresh
        state["cookie"] = cookie

        new_user_msg_id = seed_msg_id if (first and seed_msg_id) else str(uuid.uuid4())
        parent_msg_id = None if first else last_ast_id

        rstream = ReasoningStream()
        fired = {"token": False}

        def _first_token(_d):
            if not fired["token"]:
                fired["token"] = True
                if on_token is not None:
                    on_token()

        res = stream_turn(token=token, cookie=cookie, sig=sig, url=url, chat_id=chat_id,
                          model=state["model"], messages=messages,
                          current_user_message_id=new_user_msg_id,
                          current_user_message_parent_id=parent_msg_id,
                          is_first=first, features=features, params=params,
                          captcha=captcha, debug_sse=debug_sse,
                          on_thinking=lambda d: (rstream.write(d), _first_token(d)),
                          on_answer=_first_token)
        if "error" in res:
            return False, res["error"]
        if not res.get("answer"):
            if res.get("captcha_error"):
                print(f"{ANSI['yellow']}[*] captcha rejected; retrying once with a fresh captcha...{ANSI['reset']}",
                      file=sys.stderr)
                fresh2 = captcha_fn() if captcha_fn is not None else solve_fresh(solver, state)
                if fresh2:
                    res = stream_turn(token=token, cookie=fresh2[1] or state["cookie"],
                                      sig=sig, url=url, chat_id=chat_id,
                                      model=state["model"], messages=messages,
                                      current_user_message_id=new_user_msg_id,
                                      current_user_message_parent_id=parent_msg_id,
                                      is_first=first, features=features, params=params,
                                      captcha=fresh2[0], debug_sse=debug_sse,
                                      on_thinking=lambda d: (rstream.write(d), _first_token(d)),
                                      on_answer=_first_token)
                    if "error" in res:
                        return False, res["error"]
            if not res.get("answer"):
                return False, _fmt_error(res, debug_sse)

        if res.get("id"):
            last_ast_id = res["id"]

        reasoning = res.get("reasoning")
        if rstream.started:
            rstream.finish()
        elif reasoning:
            print(f"{ANSI['dim']}── reasoning ──{ANSI['reset']}", file=sys.stderr)
            for ln in reasoning.splitlines():
                print(f"{ANSI['dim']}{ln}{ANSI['reset']}", file=sys.stderr)
            print(f"{ANSI['dim']}{'─' * 14}{ANSI['reset']}", file=sys.stderr)

        return True, (res, (res.get("answer") or "").strip())

    tool_results = None
    nudged = False
    it = 0
    seen = {}
    stuck = 0
    while True:
        if is_first and it == 0:
            content = build_contract(mcp) + "\n\nUser: " + prompt + TOOL_HINT
        elif it == 0:
            content = prompt + TOOL_HINT
        else:
            content = tool_results + TOOL_HINT

        ok, err = send(content, first=(is_first and it == 0))
        if not ok:
            return False, err
        res, text = err
        it += 1

        hist.append({"role": "user", "content": content})
        hist.append({"role": "assistant", "content": text})
        state["last_assistant_id"] = last_ast_id
        if res.get("parent"):
            state["last_assistant_parent_id"] = res["parent"]
        if res.get("usage"):
            state["usage"]["prompts"] += 1
            state["usage"]["in"] += res["usage"].get("prompt_tokens", 0)
            state["usage"]["out"] += res["usage"].get("completion_tokens", 0)

        calls = parse_tool_calls(text)
        if not calls:
            if not nudged and REFUSAL_RE.search(text):
                print(f"{ANSI['dim']}[*] model declined tools — nudging once{ANSI['reset']}", file=sys.stderr)
                nudged = True
                tool_results = TOOL_NUDGE
                continue
            out = strip_tool_lines(text)
            if writer is not None:
                writer.write(out)
            elif md is not None:
                md.write(out)
            state["history"] = _commit_history(prior, is_first, mcp, prompt, text)
            return True, None

        if it >= MAX_TOOL_ITERS:
            content = ("You have reached the maximum allowed number of tool calls. "
                       "Stop calling tools. The tool results already returned above "
                       "contain what you need. Give your final answer to the user's "
                       "original request now, based on that output.")
            ok, err = send(content, first=False)
            if not ok:
                return False, err
            res, text = err
            hist.append({"role": "user", "content": content})
            hist.append({"role": "assistant", "content": text})
            out = strip_tool_lines(text)
            if writer is not None:
                writer.write(out)
            elif md is not None:
                md.write(out)
            state["history"] = _commit_history(prior, is_first, mcp, prompt, text)
            return True, None

        seen_before = set(seen.keys())
        results = []
        for name, args, line in calls:
            key = (name, safe_json(args))
            if key in seen_before:
                print(f"{ANSI['yellow']}[tool] repeat {name}({safe_json(args)[:200]}) — already executed{ANSI['reset']}",
                      file=sys.stderr)
                results.append(
                    f"OPERATOR STOP: you already received the result for this exact call "
                    f"above ({seen[key][:400]}). Do NOT emit TOOL: lines again. "
                    "Write your final answer now.")
                continue
            print(f"\n{ANSI['cyan']}[tool] {name}({safe_json(args)[:200]}){ANSI['reset']}", file=sys.stderr)
            srv = mcp.server_of(name) if mcp is not None else None
            if srv:
                approved = (approve_mcp_auto(srv, name, args) if auto_approve
                            else approve_mcp(srv, name, args))
            else:
                approved = (approve_tool_auto(name, args) if auto_approve
                            else approve_tool(name, args))
            if not approved:
                print(f"{ANSI['yellow']}[tool] declined by policy{ANSI['reset']}", file=sys.stderr)
                results.append(f"[Tool result for {name}]: DECLINED by policy - answer without it.")
                continue
            ok, result = dispatch_tool(name, args, mcp=mcp)
            preview = result[:400].replace("\n", " | ")
            color = ANSI['green'] if ok else ANSI['red']
            print(f"{color}[tool result] {'OK' if ok else 'ERR'}: {preview}{ANSI['reset']}", file=sys.stderr)
            seen[key] = result
            results.append(f"[Tool result for {name}]:\n{result}")
        tool_results = "\n\n".join(results)
        if calls and all((n, safe_json(a)) in seen_before for n, a, _ in calls):
            stuck += 1
        else:
            stuck = 0
        if stuck >= 2:
            content = ("Stop calling tools. The complete tool results you already "
                       "received are:\n\n" + tool_results +
                       "\n\nNow give your final answer to the user's original request, "
                       "directly summarizing this data. Do not emit any TOOL: line.")
            ok, err = send(content, first=False)
            if not ok:
                return False, err
            res, text = err
            hist.append({"role": "user", "content": content})
            hist.append({"role": "assistant", "content": text})
            out = strip_tool_lines(text)
            if writer is not None:
                writer.write(out)
            elif md is not None:
                md.write(out)
            state["history"] = _commit_history(prior, is_first, mcp, prompt, text)
            return True, None