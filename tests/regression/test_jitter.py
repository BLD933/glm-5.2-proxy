import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from glm_rev.solver import COLLECT_JS, PATCH_JS, WARM_JS

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name}{' — ' + detail if detail else ''}")
    if not ok:
        failures.append(f"{name}: {detail}")


# 1. Zero-entropy yield removed.
check(
    "no i % 50 yield",
    "i % 50" not in COLLECT_JS,
)

# 2. Per-token jitter present: setTimeout(r, <num> + Math.random() * <num>).
jitter_re = re.compile(r"setTimeout\(r,\s*\d+\s*\+\s*Math\.random\(\)\s*\*\s*\d+")
check(
    "jitter expression present",
    bool(jitter_re.search(COLLECT_JS)),
    COLLECT_JS.strip(),
)

# 3. Rest of COLLECT_JS intact.
for marker in ("window.z_um.getToken", "typeof window.z_um", "return out"):
    check(f"COLLECT_JS contains {marker!r}", marker in COLLECT_JS)
check(
    "COLLECT_JS async-arrow signature (total)",
    re.search(r"async\s*\(\s*total\s*\)", COLLECT_JS) is not None,
)

# 4. JS syntax sanity check via node --check (skip if node absent).
node = subprocess.run(["command", "-v", "node"], capture_output=True, text=True).stdout.strip()
if node:
    body = "const __check = " + COLLECT_JS.strip() + ";"
    wrapped = body + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(wrapped)
        tmp = f.name
    try:
        rc = subprocess.run(["node", "--check", tmp], capture_output=True, text=True).returncode
        check("node --check syntax", rc == 0, "rc=%d" % rc)
    finally:
        os.unlink(tmp)
else:
    check("node --check syntax", True, "node not available — skipped")

# 5. PATCH_JS / WARM_JS unchanged markers.
check("PATCH_JS has fetch mock", "window.fetch" in PATCH_JS and "originalFetch" in PATCH_JS)
check("WARM_JS has initAliyunCaptcha", "initAliyunCaptcha" in WARM_JS)

print("== RESULT ==")
if failures:
    print("FAIL: %d assertion(s) failed" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
else:
    print("PASS: all jitter assertions passed")
    sys.exit(0)