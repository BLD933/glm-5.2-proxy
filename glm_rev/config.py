"""GLM reverse-engineered configuration and constants."""
import ast
import json
import os
import re
import time

BASE = "https://chat.z.ai"
SECRET = "REDACTED"
FE_VERSION = "prod-fe-1.1.83"
ENDPOINT = "/api/v2/chat/completions"
DEVICE_ID = "REDACTED"
REGION = "overseas"
PINNED_IP = "146.19.236.205"

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"

# --- local tool execution (claude-rev style text contract) -------------------

ALLOWLIST_FILE = os.path.expanduser("~/.config/glm-rev-tools.json")
MAX_TOOL_ITERS = 8

LOCAL_TOOL_NAMES = ("read_file", "list_dir", "write_file", "edit_lines",
                    "run_command", "web_fetch")

TOOL_SYNONYMS = {
    "list_files": "list_dir", "ls": "list_dir", "dir": "list_dir",
    "bash": "run_command", "shell": "run_command", "run": "run_command",
    "exec": "run_command",
    "cat": "read_file", "view": "read_file", "read": "read_file",
    "open": "read_file",
    "write": "write_file", "save": "write_file", "create_file": "write_file",
    "fetch": "web_fetch", "http_get": "web_fetch", "get_url": "web_fetch",
    "edit": "edit_lines", "edit_file": "edit_lines", "replace_lines": "edit_lines",
}

_POSITIONAL_KEYS = {
    "read_file": ("path",),
    "list_dir": ("path",),
    "edit_lines": ("path", "content"),
    "run_command": ("cmd",),
    "write_file": ("path", "content"),
    "web_fetch": ("url",),
}

TOOL_CONTRACT = (
    "You are a coding agent running on the user's real Linux machine. You can get "
    "real data by emitting TOOL: lines — one per call, nothing else on that line:\n"
    "  TOOL:list_dir({\"path\": \"/home/bld\"})       list directory entries\n"
    "  TOOL:read_file({\"path\": \"/etc/hostname\"})  read a text file (256KB cap)\n"
    "  TOOL:write_file({\"path\": \"...\", \"content\": \"...\"})  write a file\n"
    "  TOOL:edit_lines({\"path\": \"...\", \"start\": N, \"delete\": M, \"content\": \"...\"})\n"
    "        replace M consecutive lines at 1-based line N with content (delete=0 inserts)\n"
    "  TOOL:run_command({\"cmd\": \"...\"})     run a shell command (30s timeout)\n"
    "  TOOL:web_fetch({\"url\": \"...\"})       GET a web page (15s timeout)\n\n"
    "How this works:\n"
    "1. When your answer needs data you do NOT yet have in this conversation, emit the "
    "needed TOOL: line(s) and nothing else.\n"
    "2. The operator executes them and returns the real output as a [Tool result] "
    "message right after your call.\n"
    "3. If your task requires multiple steps (e.g. listing directory entries and then "
    "reading files, or inspecting code and testing), continue emitting TOOL: lines for each "
    "subsequent step until you have gathered all the information needed to complete the user's request. "
    "Only when the task is fully complete, provide your comprehensive final answer. "
    "Never re-issue a tool call with the exact same arguments you already received.\n\n"
    "You have no other tools or APIs — no bash_tool, create_file, str_replace, view, "
    "present_files, or any built-in/container tools. The TOOL: line is the ONLY way to "
    "read files, run commands, or fetch pages, and it executes on the user's real "
    "computer. Use it whenever file, directory, command, or web data is needed.\n"
)

TOOL_HINT = (
    "\n(Reminder: you may emit TOOL: lines across iterations for multi-step tasks. "
    "Do NOT repeat calls with identical arguments; when you have all needed information, "
    "write your final answer.)"
)

REFUSAL_RE = re.compile(
    r"can'?t (access|reach|read)|cannot (access|reach|read)|upload (them|it|files)|"
    r"paste (the |its )?content|not your (machine|pc|computer)|"
    r"only (work|run|reach) (within|inside|in) (my|their|a) (sandbox|container)|"
    r"no (direct )?path (to|between)|there's no (way|tunnel)|"
    r"i (don'?t|cannot) have.*(access|visibility)", re.IGNORECASE)

TOOL_NUDGE = (
    "(Immediate instruction from the operator: stop describing tool limitations. "
    "The TOOL: line from your instructions is executed locally on the user's real "
    "machine and it works. Reply with exactly one TOOL: line, e.g. "
    "TOOL:list_dir({\"path\": \".\"}), then I will send you the real output.)"
)


def safe_json(v):
    """json.dumps that tolerates non-JSON types (e.g. sets from literal_eval)."""
    try:
        return json.dumps(v)
    except (TypeError, ValueError):
        return repr(v)


def load_allowlist():
    try:
        return json.load(open(ALLOWLIST_FILE))
    except Exception:
        return {"paths": [], "domains": [], "servers": []}


def save_allowlist(al):
    os.makedirs(os.path.dirname(ALLOWLIST_FILE) or ".", exist_ok=True)
    json.dump(al, open(ALLOWLIST_FILE, "w"), indent=2)


def path_allowed(path, al):
    p = os.path.abspath(os.path.expanduser(path))
    return any(p == a or p.startswith(a.rstrip("/") + "/") for a in al["paths"])


def domain_allowed(url, al):
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    return any(host == d or host.endswith("." + d) for d in al["domains"])


_TOOL_NAME_RE = re.compile(r"TOOL:[ \t]*([A-Za-z_][A-Za-z0-9_-]*)")


def _balanced_span(text, start, open_ch, close_ch):
    """Index one past the close_ch matching the bracket at text[start],
    ignoring brackets inside string literals; -1 if unbalanced."""
    depth = 1
    in_str = False
    str_ch = None
    k = start + 1
    while k < len(text):
        ch = text[k]
        if in_str:
            if ch == "\\":
                k += 1
            elif ch == str_ch:
                in_str = False
                str_ch = None
        elif ch in ('"', "'"):
            in_str = True
            str_ch = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if not depth:
                return k + 1
        k += 1
    return -1


def _map_positional(name, value):
    """Map a non-dict parsed arg onto the tool's primary parameters."""
    if isinstance(value, dict):
        return value
    keys = _POSITIONAL_KEYS.get(name)
    if not keys:
        return {}
    values = list(value) if isinstance(value, (list, tuple)) else [value]
    return dict(zip(keys, values))


def _parse_bare_args(canonical, raw_str):
    raw_str = raw_str.strip()
    if not raw_str:
        return {"path": "."} if canonical == "list_dir" else {}
    
    # 0. Invalid dict literal (e.g. {path="/tmp/x"})
    if raw_str.startswith("{") and raw_str.endswith("}"):
        return {}
    
    # 1. If run_command: capture the whole command string
    if canonical == "run_command":
        m = re.match(r'^(?:cmd|command)\s*[:=]\s*(.*)$', raw_str, re.IGNORECASE)
        val = m.group(1).strip() if m else raw_str
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        return {"cmd": val}
    
    # 2. Single-arg tools (read_file, list_dir, web_fetch)
    if canonical in ("read_file", "list_dir", "web_fetch"):
        m = re.match(r'^(?:path|file|dir|url)\s*[:=]\s*(.*)$', raw_str, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            else:
                val = val.split(None, 1)[0].strip("\"'")
        else:
            if (raw_str.startswith('"') and raw_str.endswith('"')) or (raw_str.startswith("'") and raw_str.endswith("'")):
                val = raw_str[1:-1]
            else:
                val = raw_str.split(None, 1)[0].strip("\"'")
        if canonical == "web_fetch":
            return {"url": val}
        return {"path": val}
    
    # 3. Multi-arg tools (write_file, edit_lines)
    kv_pattern = r'([A-Za-z_][A-Za-z0-9_-]*)\s*[:=]\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))'
    matches = re.findall(kv_pattern, raw_str)
    if matches:
        res = {}
        for k, v1, v2, v3 in matches:
            val = v1 if v1 != "" else (v2 if v2 != "" else v3)
            try:
                val = int(val)
            except ValueError:
                pass
            res[k] = val
        return res
    
    # 4. Fallback positional token
    token = raw_str.split(None, 1)[0].strip("\"'")
    return _map_positional(canonical, token)


def parse_tool_calls(text):
    """Return [(name, args_dict, matched_text), ...] for ALL TOOL: calls.
    Accepts bare names (TOOL: list_files), bare JSON args, and positional
    or keyword args in parens. Balanced scans tolerate calls glued on one
    line and parens/braces inside string values."""
    calls = []
    i = 0
    while True:
        m = _TOOL_NAME_RE.search(text, i)
        if not m:
            break
        name = m.group(1)
        canonical = name if name in LOCAL_TOOL_NAMES else TOOL_SYNONYMS.get(name, name)
        j = m.end()
        k = j
        while k < len(text) and text[k] in " \t":
            k += 1
        if k < len(text) and text[k] in "([{":
            close_ch = {"(": ")", "[": "]", "{": "}"}[text[k]]
            end = _balanced_span(text, k, text[k], close_ch)
            if end < 0:
                eol = text.find("\n", k)
                matched_end = eol if eol >= 0 else len(text)
                i = matched_end
                continue
            raw = text[k + 1:end - 1] if text[k] == "(" else text[k:end]
            matched_end = end
            args = None
            if raw.strip():
                try:
                    args = json.loads(raw)
                except Exception:
                    try:
                        args = ast.literal_eval(raw)
                    except Exception:
                        pass
            if args is None:
                args = _parse_bare_args(canonical, raw)
            elif not isinstance(args, dict):
                args = _map_positional(canonical, args)
        else:
            eol = text.find("\n", k)
            if eol < 0:
                eol = len(text)
            next_tool = text.find("TOOL:", k)
            if next_tool >= 0 and next_tool < eol:
                eol = next_tool
            raw_str = text[k:eol]
            args = _parse_bare_args(canonical, raw_str)
            has_kv = bool(re.search(r'^[A-Za-z_][A-Za-z0-9_-]*\s*[:=]', raw_str.strip()))
            matched_end = eol if has_kv else j
        if canonical == "list_dir" and "path" not in args:
            args = dict(args)
            args["path"] = "."
        calls.append((canonical, args, text[m.start():matched_end]))
        i = matched_end
    return calls


def parse_tool_line(text):
    """Return (name, args_dict, line) for the first TOOL: call, else None."""
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def strip_tool_lines(text):
    """Remove every TOOL: call (incl. several glued on one line)."""
    out = text
    for name, args, matched in parse_tool_calls(text):
        out = out.replace(matched, "", 1)
    return out

