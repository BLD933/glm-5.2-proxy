import re
import ast
import json

_POSITIONAL_KEYS = {
    "read_file": ("path",),
    "list_dir": ("path",),
    "edit_lines": ("path", "content"),
    "run_command": ("cmd",),
    "write_file": ("path", "content"),
    "web_fetch": ("url",),
}

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

_TOOL_NAME_RE = re.compile(r"TOOL:[ \t]*([A-Za-z_][A-Za-z0-9_-]*)")


def _balanced_span(text, start, open_ch, close_ch):
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
    
    # 0. Invalid dict literal
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
            if not raw_str.strip():
                matched_end = j
            else:
                matched_end = eol
        
        if canonical == "list_dir" and "path" not in args:
            args = dict(args)
            args["path"] = "."
        
        calls.append((canonical, args, text[m.start():matched_end]))
        i = matched_end
    return calls


def strip_tool_lines(text):
    out = text
    for name, args, matched in parse_tool_calls(text):
        out = out.replace(matched, "", 1)
    return out


# Test suite
cases = [
    # Basic & YAML
    ('TOOL: read_file path: run.py', ('read_file', {'path': 'run.py'})),
    ('TOOL: read_file path: "path with spaces.txt"', ('read_file', {'path': 'path with spaces.txt'})),
    ('TOOL: list_dir path: /home/bld', ('list_dir', {'path': '/home/bld'})),
    ('TOOL: list_dir path: .', ('list_dir', {'path': '.'})),
    ('TOOL: run_command cmd: git status -s | head -n 5', ('run_command', {'cmd': 'git status -s | head -n 5'})),
    ('TOOL: run_command command: "pytest tests/"', ('run_command', {'cmd': 'pytest tests/'})),
    ('TOOL: web_fetch url: https://api.example.com/v1', ('web_fetch', {'url': 'https://api.example.com/v1'})),
    # Glued calls with key-value
    ('TOOL: list_dir path: . TOOL: read_file path: server.py', 
     [('list_dir', {'path': '.'}), ('read_file', {'path': 'server.py'})]),
    # Broken pseudo syntax
    ('TOOL: read_file({path="/tmp/x"})', ('read_file', {})),
    ('TOOL: edit_lines path: a.py start: 10 delete: 2 content: "x = 1"',
     ('edit_lines', {'path': 'a.py', 'start': 10, 'delete': 2, 'content': 'x = 1'})),
]

all_passed = True
for text, expected in cases:
    res = parse_tool_calls(text)
    if isinstance(expected, list):
        got = [(c[0], c[1]) for c in res]
    else:
        got = (res[0][0], res[0][1]) if res else None
    ok = got == expected
    if not ok:
        all_passed = False
    print(f"[{'PASS' if ok else 'FAIL'}] {text!r:70} -> {got!r}")

assert all_passed, "Some tests failed!"

# Test stripping
prose_sample = """Here is the plan:
TOOL: read_file path: run.py
And here is another check:
TOOL: list_dir path: glm_rev
Done."""

stripped = strip_tool_lines(prose_sample)
print("\n--- Stripped Prose Test ---")
print(repr(stripped))
assert "TOOL:" not in stripped
assert "path: run.py" not in stripped
assert "path: glm_rev" not in stripped
print("Prose stripping verified clean!")
