"""GLM reverse-engineered configuration and constants."""
import ast
import json
import os
import re
import time

try:
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

BASE = "https://chat.z.ai"
FE_VERSION = "prod-fe-1.1.83"
ENDPOINT = "/api/v2/chat/completions"
REGION = "overseas"
PINNED_IP = "146.19.236.205"


def _require(name: str) -> str:
    val = os.environ.get(name, "")
    if val:
        return val
    raise RuntimeError(
        f"Missing env var {name}: copy .env.example to .env and fill it in."
    )


# HMAC signing key, device fingerprint id (see .env / .env.example).
SECRET = _require("ZAI_SIGNING_SECRET")
DEVICE_ID = _require("ZAI_DEVICE_ID")

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"

# --- local tool execution (claude-rev style text contract) -------------------

ALLOWLIST_FILE = os.path.expanduser("~/.config/glm-rev-tools.json")
MAX_TOOL_ITERS = 8
HISTORY_LIMIT = 120

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
    "\n(Reminder: emit TOOL: lines across iterations for multi-step tasks; never "
    "repeat identical calls. Available tools — one per line, nothing else:\n"
    "  TOOL:read_file({\"path\": ...})\n"
    "  TOOL:list_dir({\"path\": ...})\n"
    "  TOOL:write_file({\"path\": ..., \"content\": ...})\n"
    "  TOOL:edit_lines({\"path\": ..., \"start\": N, \"delete\": M, \"content\": ...})\n"
    "  TOOL:run_command({\"cmd\": ...})\n"
    "  TOOL:web_fetch({\"url\": ...})\n"
    "When all info gathered, write your final answer.)"
)

REFUSAL_RE = re.compile(
    r"can[’']?t (access|reach|read|open|use|run)|cannot (access|reach|read|open|use|run)|"
    r"unable to (view|access|read|open|see|use|run)|"
    r"don[’']t have (permission|access|authorization|ability)|"
    r"don[’']t have (any|the|a).*?(tool|tools)|"
    r"no (such )?(tool|tools)|no \w+ tool|"
    r"lack(s|ing)? (access|any|the|a)?.*?tool|not (have|possess).*?tool|at my disposal|"
    r"no (permission|authorization)|denied permission|access denied|"
    r"upload (them|it|files)|paste (the |its )?content|not your (machine|pc|computer)|"
    r"only (work|run|reach) (within|inside|in) (my|their|a) (sandbox|container)|"
    r"no (direct )?path (to|between)|there[’']s no (way|tunnel)|"
    r"i (don[’']t|cannot|can[’']t) have.*(access|visibility|permission)", re.IGNORECASE)

TOOL_NUDGE = (
    "(Immediate instruction from the operator: stop describing tool limitations. "
    "The TOOL: line from your instructions is executed locally on the user's real "
    "machine and it works. Reply with exactly one TOOL: line, e.g. "
    "TOOL:list_dir({\"path\": \".\"}), then I will send you the real output.)"
)

# Client-side variant (opencode path): references the SAME GLM-friendly names
# the curated contract teaches (run_command, list_dir, ...), so the nudge stays
# consistent with what GLM was told to emit. Used by _run_client_tools when
# GLM-5.2 refuses a tool request.
TOOL_NUDGE_CLIENT = (
    "(Immediate instruction from the operator: stop describing tool limitations. "
    "The TOOL: line from your instructions is executed locally on the user's real "
    "machine and it works. Reply with exactly one TOOL: line, e.g. "
    'TOOL:list_dir({"path": "."}) or TOOL:run_command({"cmd": "ls -la"}), then I will '
    "send you the real output.)"
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


_TOOL_NAME_RE = re.compile(r"TOOL:[ \t]*([A-Za-z_][A-Za-z0-9_./:-]*)")

_MD_FENCE_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\n(.*?)```", re.DOTALL)

_XML_CALL_RE = re.compile(
    r"<(?P<tag>tool_call|tool|function_call)\b(?P<head>[^>]*)>"
    r"(?P<inner>.*?)</(?P=tag)(?::[A-Za-z_][A-Za-z0-9_-]*)?>", re.DOTALL)

_XML_EMPTY_RE = re.compile(
    r"<(?P<tag>tool_call|tool|function_call)\b(?P<head>[^>]*?)/>", re.DOTALL)

_XML_NAME_RE = re.compile(
    r"\bname\s*=\s*(?:[\"'](?P<q1>[^\"']*)[\"']|(?P<q2>\S+))")


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


def _repair_multiline_json(raw):
    """Escape raw (unescaped) newlines inside double-quoted JSON strings so
    json.loads can parse multi-line / escaped string values intact."""
    out = []
    in_str = False
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            out.append(ch)
            out.append(raw[i + 1])
            i += 2
            continue
        if ch == '"':
            in_str = not in_str
            out.append(ch)
            i += 1
            continue
        if in_str and ch == "\n":
            out.append("\\n")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _load_args(raw):
    """Parse an args blob into a value (dict preferred), tolerating multi-line
    or escaped string values; return None if it cannot be loaded."""
    if not raw or not raw.strip():
        return None
    for source in (raw, _repair_multiline_json(raw)):
        try:
            return json.loads(source)
        except Exception:
            try:
                return ast.literal_eval(source)
            except Exception:
                continue
    return None


def _xml_args(inner, canonical):
    if not inner or not inner.strip():
        return {}
    args = _load_args(inner.strip())
    if args is None:
        args = _parse_bare_args(canonical, inner)
    elif not isinstance(args, dict):
        args = _map_positional(canonical, args)
    return args


_INTENT_CLUSTERS = {
    "exec": {"execute_command", "run_command", "bash", "sh", "shell", "exec",
             "run", "terminal", "cmd", "command"},
    "read": {"read_file", "view_file", "cat", "view", "read", "open"},
    "write": {"write_file", "write_to_file", "save", "create_file", "write"},
    "edit": {"edit_lines", "edit_file", "replace_lines", "str_replace", "edit"},
    "list": {"list_dir", "list_directory", "list_files", "ls", "dir", "glob", "tree", "scandir", "readdir", "walk"},
    "search": {"grep_search", "search_files", "grep", "find_files", "search"},
}

# When tools are validated against a CLIENT toolset (e.g. opencode's real 113
# tools), GLM-5.2 emits native <tool_call> with hallucinated / generic names.
# Map each intent to the client's actual tool so those calls fire.
INTENT_TO_CLIENT_TOOL = {
    "exec": "bash",
    "read": "read",
    "write": "write",
    "edit": "edit",
    "list": "glob",
    "search": "grep",
}


def _normalize(name):
    """Strip non-alphanumerics for fuzzy name matching."""
    return re.sub(r"[^A-Za-z0-9]", "", name).lower()


def _exec_like(known_names):
    """Return known names that look like an exec/command tool."""
    return [n for n in known_names
            if "command" in n.lower() or "cmd" in n.lower() or "exec" in n.lower()
            or "shell" in n.lower() or "bash" in n.lower()
            or n in _INTENT_CLUSTERS["exec"]]


def _intent_of(name):
    """Best-effort intent detection for a (possibly hallucinated) tool name."""
    for intent, members in _INTENT_CLUSTERS.items():
        if name in members:
            return intent
    n = name.lower()
    if not n:
        return None
    toks = {t for t in re.split(r"[_\-]+", n) if t}
    def has(*keys):
        return any(k in toks for k in keys)
    if has("list", "dir", "ls"):
        return "list"
    if has("read", "view", "cat", "open"):
        return "read"
    if has("write", "save", "create"):
        return "write"
    if has("edit", "replace"):
        return "edit"
    if has("search", "grep", "find"):
        return "search"
    if has("bash", "shell", "exec", "command", "run", "cmd"):
        return "exec"
    # Whole-name substring fallback — gated to keywords of length >= 4 so short
    # tokens (e.g. "cat") can't false-positive inside unrelated words
    # ("frobnicate"). Handles seamless names like "readfile".
    long_keys = {
        "list": ("list",), "read": ("read",), "write": ("write",),
        "edit": ("edit",), "search": ("search",), "grep": ("grep",),
        "find": ("find",), "bash": ("bash",), "shell": ("shell",),
        "exec": ("exec",), "command": ("command",),
    }
    for intent, keys in long_keys.items():
        if any(k in n for k in keys):
            return intent
    return None


def _desc_match(name, known_names, known_tool_dict):
    """Light fallback: match a guessed name to a client tool via name/desc
    token overlap. Returns a known name or None (prefer missing over misfire)."""
    if not known_tool_dict:
        return None
    n = _normalize(name)
    if len(n) < 4:
        return None
    tokens = {t for t in re.split(r"[_\-.]", n) if len(t) >= 3}
    best = None
    best_score = 0
    for kn in known_names:
        kd = _normalize(kn)
        if n in kd and len(n) >= 4:
            return kn
        if kd in n and len(kd) >= 4:
            return kn
        fn = known_tool_dict.get(kn) or {}
        desc = (fn.get("description") or "").lower()
        score = sum(1 for t in tokens if t in desc)
        if score > best_score and score >= 2:
            best_score = score
            best = kn
    return best


def _resolve_tool_name(name, known_names, known_tool_dict=None):
    """Return the exact known tool name to emit, or None if no match."""
    if known_names is None:
        return name
    if name in known_names:
        return name
    low = name.lower()
    for n in known_names:
        if n.lower() == low:
            return n
    norm = _normalize(name)
    if norm:
        for n in known_names:
            nn = _normalize(n)
            if nn and (norm in nn or nn in norm):
                return n
    cluster_hit = None
    for members in _INTENT_CLUSTERS.values():
        if name in members:
            cluster_hit = members
            break
    if cluster_hit is not None:
        for n in known_names:
            if n in cluster_hit:
                return n
        for members in _INTENT_CLUSTERS.values():
            if any(name == m or _normalize(m) == norm for m in members):
                for n in known_names:
                    if n in members:
                        return n
    # Client toolset: map GLM's hallucinated/generic name to the client's real
    # tool via intent, then a light description-based fallback.
    intent = _intent_of(name)
    if intent is not None:
        target = INTENT_TO_CLIENT_TOOL.get(intent)
        if target is not None and target in known_names:
            return target
    fallback = _desc_match(name, known_names, known_tool_dict)
    if fallback is not None:
        return fallback
    return None


_EXEC_ARG_KEYS = ("command", "cmd", "shell", "cli", "command_line")

_ARG_SYNONYMS = {
    "cmd": ("command", "cmd", "shell", "cli", "command_line", "script"),
    "shell": ("command", "cmd", "shell", "cli", "command_line", "script"),
    "cli": ("command", "cmd", "shell", "cli", "command_line", "script"),
    "command_line": ("command", "cmd", "shell", "cli", "command_line", "script"),
    "command": ("command", "cmd", "shell", "cli", "command_line", "script"),
    "path": ("path", "file", "filename", "filepath", "file_path", "directory",
             "dir", "folder"),
    "file": ("path", "file", "filename", "filepath", "file_path"),
    "filename": ("path", "file", "filename", "filepath", "file_path"),
    "filepath": ("path", "file", "filename", "filepath", "file_path"),
    "file_path": ("path", "file", "filename", "filepath", "file_path"),
    "directory": ("directory", "dir", "path", "folder"),
    "dir": ("directory", "dir", "path", "folder"),
    "content": ("content", "text", "body", "data"),
    "text": ("content", "text", "body", "data"),
    "body": ("content", "text", "body", "data"),
    "url": ("url", "link", "uri"),
    "start": ("start", "line"),
    "delete": ("delete", "count", "lines"),
}


def _remap_args(name, args, known_tool_dict):
    """Map common param synonyms to the tool's real parameter keys.
    known_tool_dict is the name->function-dict map (from _build_known)."""
    if known_tool_dict is None:
        return args
    if not isinstance(args, dict):
        return args
    fn = (known_tool_dict or {}).get(name) or {}
    params = {}
    props = (fn.get("parameters") or {}).get("properties") or {}
    params = {k.lower(): k for k in props}
    new = {}
    for key, val in args.items():
        k = str(key).lower()
        target = params.get(k)
        if target is None and k in _ARG_SYNONYMS:
            for cand in _ARG_SYNONYMS[k]:
                if cand in params:
                    target = params[cand]
                    break
        if target is None and k in _EXEC_ARG_KEYS:
            if "command" in params or "cmd" in params:
                target = params.get("command") or params.get("cmd")
        if target is None:
            target = key
        new[target] = val
    if name == "glob" and "pattern" not in new:
        new["pattern"] = "*"
    return new


def _xml_no_name_call(inner, known_names, known_tool_dict):
    """Handle a <tool_call> with no name= attribute. Returns a (name, args)
    tuple, or None if nothing resolvable (preserve 'prefer missing over misfire')."""
    inner = (inner or "").strip()
    if not inner:
        return None

    fence = re.match(r"```[ \t]*(?:bash|sh|shell)[ \t]*\n(.*?)```",
                     inner, re.DOTALL)
    if fence:
        cmd = fence.group(1).strip().strip("`").strip()
        if known_names is not None:
            exec_names = _exec_like(known_names)
            if not exec_names:
                exec_names = [n for n in known_names
                              if n in _INTENT_CLUSTERS["exec"]]
            name = exec_names[0] if exec_names else None
        else:
            name = "run_command"
        if name is None:
            return None
        name = _resolve_tool_name(name, known_names, known_tool_dict) or name
        args = {"cmd": cmd}
        return name, _remap_args(name, args, known_tool_dict)

    obj = _load_args(inner)
    if isinstance(obj, dict) and isinstance(obj.get("name"), str):
        name = obj.get("name")
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        resolved = _resolve_tool_name(name, known_names, known_tool_dict)
        if resolved is None and known_names is not None:
            return None
        resolved = resolved or name
        return resolved, _remap_args(resolved, args, known_tool_dict)

    m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_./:-]*)\s*\((.*)\)\s*$",
                 inner, re.DOTALL)
    if m:
        name, raw = m.group(1), m.group(2)
        resolved = _resolve_tool_name(name, known_names, known_tool_dict)
        if resolved is None and known_names is not None:
            return None
        resolved = resolved or name
        args = {}
        try:
            tree = ast.parse(raw)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if isinstance(kw.value, ast.Constant):
                            args[kw.arg] = kw.value.value
        except Exception:
            args = _parse_bare_args(resolved, raw)
        return resolved, _remap_args(resolved, args, known_tool_dict)

    return None


def _parse_md_calls(text, known_names=None, known_tool_dict=None):
    calls = []
    for m in _MD_FENCE_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        resolved = _resolve_tool_name(name, known_names, known_tool_dict)
        if resolved is None and known_names is not None:
            continue
        resolved = resolved or name
        calls.append((resolved, _remap_args(resolved, args, known_tool_dict),
                      m.group(0), m.start()))
    return calls


def _parse_xml_calls(text, known_names=None, known_tool_dict=None):
    calls = []
    for m in _XML_CALL_RE.finditer(text):
        head = m.group("head")
        inner = m.group("inner")
        nm = _XML_NAME_RE.search(head)
        if not nm:
            hit = _xml_no_name_call(inner, known_names, known_tool_dict)
            if hit is not None:
                name, args = hit
                calls.append((name, args, m.group(0), m.start()))
            continue
        name = nm.group("q1") or nm.group("q2")
        resolved = _resolve_tool_name(name, known_names, known_tool_dict)
        if resolved is None and known_names is not None:
            continue
        resolved = resolved or name
        args = _xml_args(inner, resolved)
        calls.append((resolved, _remap_args(resolved, args, known_tool_dict),
                      m.group(0), m.start()))
    for m in _XML_EMPTY_RE.finditer(text):
        head = m.group("head")
        nm = _XML_NAME_RE.search(head)
        if not nm:
            continue
        name = nm.group("q1") or nm.group("q2")
        resolved = _resolve_tool_name(name, known_names, known_tool_dict)
        if resolved is None and known_names is not None:
            continue
        resolved = resolved or name
        calls.append((resolved, {}, m.group(0), m.start()))
    return calls


def _build_known(known_tools):
    """Build (known_names, name->tool_dict) from an OpenAI-shape tool list.
    Returns (None, None) when known_tools is None (skips all remapping)."""
    if known_tools is None:
        return None, None
    names = set()
    by_name = {}
    for t in known_tools or []:
        fn = (t or {}).get("function") or {}
        n = fn.get("name")
        if isinstance(n, str) and n:
            names.add(n)
            by_name[n] = fn
    return names, by_name


def parse_tool_calls(text, known_tools=None):
    """Return [(name, args_dict, matched_text), ...] for ALL tool calls.
    Recognizes the TOOL: text convention plus markdown fenced JSON blocks
    and XML tool/function_call tags. Accepts bare names, bare JSON args, and
    positional or keyword args in parens. Balanced scans tolerate calls glued
    on one line and parens/braces inside string values.

    When known_tools is provided (OpenAI-shape list of tool dicts), every
    emitted name is resolved against the known set and args are remapped to the
    tool's real parameter keys. When known_tools is None, no remapping happens
    (identical to the historical behavior)."""
    known_names, known_tool_dict = _build_known(known_tools)
    calls = []
    i = 0
    while True:
        m = _TOOL_NAME_RE.search(text, i)
        if not m:
            break
        name = m.group(1)
        canonical = name if name in LOCAL_TOOL_NAMES else TOOL_SYNONYMS.get(name, name)
        resolved = _resolve_tool_name(canonical, known_names, known_tool_dict)
        if resolved is None and known_names is not None:
            i = m.end()
            continue
        resolved = resolved or canonical
        if known_names is None and resolved.lower() in ("lines", "calls", "tags"):
            i = m.end()
            continue
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
                args = _load_args(raw)
            if args is None:
                args = _parse_bare_args(resolved, raw)
            elif not isinstance(args, dict):
                args = _map_positional(resolved, args)
        else:
            eol = text.find("\n", k)
            if eol < 0:
                eol = len(text)
            next_tool = text.find("TOOL:", k)
            if next_tool >= 0 and next_tool < eol:
                eol = next_tool
            raw_str = text[k:eol]
            args = _parse_bare_args(resolved, raw_str)
            has_kv = bool(re.search(r'^[A-Za-z_][A-Za-z0-9_-]*\s*[:=]', raw_str.strip()))
            matched_end = eol if has_kv else j
        if resolved == "list_dir" and "path" not in args:
            args = dict(args)
            args["path"] = "."
        if name == "list_dir" and resolved == "glob":
            args = dict(args)
            args.pop("path", None)
            if "pattern" not in args:
                args["pattern"] = "*"
        args = _remap_args(resolved, args, known_tool_dict)
        calls.append((resolved, args, text[m.start():matched_end], m.start()))
        i = matched_end
    calls.extend(_parse_md_calls(text, known_names, known_tool_dict))
    calls.extend(_parse_xml_calls(text, known_names, known_tool_dict))
    calls.sort(key=lambda c: c[3])
    return [(n, a, t) for n, a, t, _ in calls]


def parse_tool_line(text, known_tools=None):
    """Return (name, args_dict, line) for the first TOOL: call, else None."""
    calls = parse_tool_calls(text, known_tools)
    return calls[0] if calls else None


def strip_tool_lines(text, known_tools=None):
    """Remove every TOOL: call (incl. several glued on one line)."""
    out = text
    for name, args, matched in parse_tool_calls(text, known_tools):
        out = out.replace(matched, "", 1)
    return out

