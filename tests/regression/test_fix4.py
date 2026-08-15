import sys

sys.path.insert(0, "/home/bld/glm-rev")

from glm_rev.config import parse_tool_calls, parse_tool_line, strip_tool_lines


def check(label, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"[{status}] {label}: {got!r}")
    return got == expected


results = []
ok = True

def case(label, text, expected_name, expected_args):
    global ok
    calls = parse_tool_calls(text)
    got = None
    if calls:
        got = (calls[0][0], calls[0][1])
    good = got == (expected_name, expected_args)
    if not good:
        ok = False
    results.append(check(f"{label:55} -> parse_tool_calls({text!r})", got,
                         (expected_name, expected_args)))


case("1 bare list_dir", "TOOL: list_dir", "list_dir", {"path": "."})
case("2 bare list_files (synonym)", "TOOL: list_files", "list_dir", {"path": "."})
case("3 positional paren", 'TOOL: list_dir("/tmp")', "list_dir", {"path": "/tmp"})
case("4 bare json brace", 'TOOL: list_dir {"path": "/var"}', "list_dir", {"path": "/var"})
case("5 positional run_command", 'TOOL: run_command("ls -la")', "run_command", {"cmd": "ls -la"})
case("6 bash synonym json", 'TOOL: bash {"cmd": "whoami"}', "run_command", {"cmd": "whoami"})
case("7 web_fetch keyword", 'TOOL: web_fetch({"url": "https://x"})', "web_fetch", {"url": "https://x"})
case("8 read_file positional", 'TOOL: read_file("/etc/hostname")', "read_file", {"path": "/etc/hostname"})
case("9 cat bare path", "TOOL: cat /etc/hostname", "read_file", {"path": "/etc/hostname"})
case("10 bare ls", "TOOL: ls", "list_dir", {"path": "."})

calls = parse_tool_calls("TOOL:list_dir TOOL:read_file(\"/x\")")
names_args = [(c[0], c[1]) for c in calls]
good = names_args == [("list_dir", {"path": "."}), ("read_file", {"path": "/x"})]
if not good:
    ok = False
results.append(check("11 glued calls", names_args,
                     [("list_dir", {"path": "."}), ("read_file", {"path": "/x"})]))

good = parse_tool_calls("no tools here") == []
if not good:
    ok = False
results.append(check("12 no tools", parse_tool_calls("no tools here"), []))

case("13 write_file paren in string", 'TOOL: write_file({"path": "/tmp/a", "content": "hi (x)"})',
     "write_file", {"path": "/tmp/a", "content": "hi (x)"})

calls = parse_tool_calls("TOOL: list_files please")
good = bool(calls) and calls[0][0] == "list_dir"
if not good:
    ok = False
results.append(check("14 bare w/ prose matches", calls and (calls[0][0], calls[0][1]),
                     ("list_dir", {"path": "please"})))

pl = parse_tool_line("TOOL: ls")
good = pl and pl[0] == "list_dir" and pl[1] == {"path": "."}
if not good:
    ok = False
results.append(check("15 parse_tool_line first call", pl, ("list_dir", {"path": "."}, "TOOL: ls")))

stripped = strip_tool_lines("TOOL: list_files please")
good = "TOOL" not in stripped and stripped == " please"
if not good:
    ok = False
results.append(check("16 strip_tool_lines leaves prose", repr(stripped), repr(" please")))

stripped2 = strip_tool_lines("TOOL:list_dir TOOL:read_file(\"/x\")")
good = "TOOL" not in stripped2
if not good:
    ok = False
print(f"[{'PASS' if good else 'FAIL'}] 17 strip glued calls: {stripped2!r} (no TOOL text)" )
results.append(good)

case("18 no-space bare", "TOOL:list_dir", "list_dir", {"path": "."})

calls = parse_tool_calls('TOOL: read_file({path="/tmp/x"})')
good = bool(calls) and calls[0][0] == "read_file" and calls[0][1] == {}
if not good:
    ok = False
results.append(check("19 invalid literal handled", calls and (calls[0][0], calls[0][1]),
                     ("read_file", {})))

print("\n" + ("ALL PASS" if ok else "SOME FAIL"))
sys.exit(0 if ok else 1)
