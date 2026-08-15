import os
import sys

sys.path.insert(0, os.getcwd())

from glm_rev.config import REFUSAL_RE

MATCH_CASES = [
    ("ASCII apostrophe", "I don't have permission to view the file contents"),
    ("U+2019 apostrophe", "I don\u2019t have permission to view the file contents"),
    ("ASCII can't access", "I can't access that"),
    ("U+2019 can't access", "I can\u2019t access that"),
    ("cannot read", "I cannot read that file"),
    ("can't open", "I can't open the file"),
    ("unable view", "Unable to view the file"),
    ("unable access", "unable to access the network"),
    ("access denied", "access denied"),
    ("don't have permission (you)", "You don't have permission"),
    ("don't have authorization", "I don't have authorization to do that"),
    ("no permission granted", "no permission granted"),
    ("there's no way", "there's no way to reach it"),
    ("can't reach", "I can't reach the server"),
    ("cannot access db", "I cannot access the database"),
]

NO_MATCH_CASES = [
    ("can access results", "You can access the results here"),
    ("please open file", "Please open the file and read it"),
    ("I can open it", "I can open it"),
    ("can be viewed by anyone", "the file can be viewed by anyone"),
    ("we have permission to proceed", "we have permission to proceed"),
    ("reading worked", "reading the file worked fine"),
]

failures = []

print("== MATCH assertions ==")
for name, text in MATCH_CASES:
    ok = bool(REFUSAL_RE.search(text))
    print(f"{'PASS' if ok else 'FAIL'}: {name!r}: {text!r}")
    if not ok:
        failures.append(f"MATCH failed: {name}: {text!r}")

print("== NO-MATCH assertions ==")
for name, text in NO_MATCH_CASES:
    ok = not bool(REFUSAL_RE.search(text))
    print(f"{'PASS' if ok else 'FAIL'}: {name!r}: {text!r}")
    if not ok:
        failures.append(f"NO-MATCH failed (false positive): {name}: {text!r}")

print("== RESULT ==")
if failures:
    print("FAIL: %d assertion(s) failed" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
else:
    print("PASS: all %d assertions passed" % (len(MATCH_CASES) + len(NO_MATCH_CASES)))
    sys.exit(0)