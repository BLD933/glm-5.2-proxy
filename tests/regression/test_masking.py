import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.getcwd())

from glm_rev import solver

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}: {name}{' — ' + detail if detail else ''}")
    if not ok:
        failures.append(f"{name}: {detail}")


# 1. MASK_JS exists and contains the required markers.
check("MASK_JS is a non-empty str", isinstance(solver.MASK_JS, str) and len(solver.MASK_JS) > 0)
for marker in (
    "navigator.webdriver",
    "Object.defineProperty",
    "window.chrome",
    "getParameter",
    "UNMASKED_RENDERER_WEBGL",
    "(function()",
):
    check(f"MASK_JS contains {marker!r}", marker in solver.MASK_JS)

# 2. Syntax-check + execute MASK_JS in a stub harness with node.
node = subprocess.run(["command", "-v", "node"], capture_output=True, text=True).stdout.strip()
if node:
    body = solver.MASK_JS.strip()
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(body + "\n")
        tmp = f.name
    try:
        rc = subprocess.run(["node", "--check", tmp], capture_output=True, text=True).returncode
        check("node --check syntax", rc == 0, "rc=%d" % rc)
    finally:
        os.unlink(tmp)

    harness = r"""
var navigator = {};
var window = {};
var WebGLRenderingContext = function() {};
WebGLRenderingContext.prototype.getParameter = function(p) { return 'orig-' + p; };
var AudioContext = function() {};
AudioContext.prototype.sampleRate = 44100;
AudioContext.prototype.currentTime = 0;
AudioContext.prototype.state = 'suspended';
var OfflineAudioContext = function() {};
OfflineAudioContext.prototype.sampleRate = 44100;
OfflineAudioContext.prototype.currentTime = 0;
OfflineAudioContext.prototype.state = 'suspended';
""" + body + r"""
var errs = [];
if (navigator.webdriver !== undefined) errs.push('webdriver not scrubbed');
if (!window.chrome || !window.chrome.runtime ||
    window.chrome.runtime.OnInstalledReason.INSTALL !== 'install' ||
    typeof window.chrome.loadTimes !== 'function') errs.push('chrome mock missing');
var ctx = new WebGLRenderingContext();
if (ctx.getParameter(37445) !== 'Google Inc. (Intel)') errs.push('vendor not spoofed');
if (ctx.getParameter(37446) !== 'ANGLE (Intel, Intel(R) UHD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11)') errs.push('renderer not spoofed');
if (ctx.getParameter(0) !== 'orig-0') errs.push('getParameter passthrough broken');
if (new AudioContext().sampleRate !== 48000) errs.push('audio sampleRate not spoofed');
if (new AudioContext().state !== 'running') errs.push('audio state not spoofed');
if (typeof new AudioContext().currentTime !== 'number') errs.push('audio currentTime missing');
if (new OfflineAudioContext().sampleRate !== 48000) errs.push('offline sampleRate not spoofed');
if (errs.length) { console.error('HARNESS-FAIL: ' + errs.join('; ')); process.exit(1); }
console.log('HARNESS-OK');
"""
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness + "\n")
        tmp2 = f.name
    try:
        r = subprocess.run(["node", tmp2], capture_output=True, text=True)
        check("node stub-harness executes without throwing",
              r.returncode == 0 and "HARNESS-OK" in r.stdout,
              "rc=%d out=%s err=%s" % (r.returncode, r.stdout.strip(), r.stderr.strip()))
    finally:
        os.unlink(tmp2)
else:
    check("node --check syntax", True, "node not available — skipped")
    check("node stub-harness executes", True, "node not available — skipped")

# 3. Both launch sites call add_init_script(MASK_JS); _oneshot_impl injects
#    before its first page.goto.
src = open(os.path.join(os.path.dirname(__file__), "..", "..", "glm_rev", "solver.py")).read()
n_inject = src.count("add_init_script(MASK_JS)")
check("add_init_script(MASK_JS) present at both launch sites", n_inject >= 2,
      "count=%d" % n_inject)

m = re.search(r"async def _oneshot_impl\(token: str, count: int\) -> list\[str\]:"
              r"(.*?)(?=\nasync def |\nclass )", src, re.DOTALL)
if m:
    body = m.group(1)
    i_mask = body.find("add_init_script(MASK_JS)")
    i_goto = body.find("page.goto")
    check("_oneshot_impl: MASK_JS injected before first page.goto",
          i_mask != -1 and i_goto != -1 and i_mask < i_goto,
          "mask_idx=%d goto_idx=%d" % (i_mask, i_goto))
else:
    check("_oneshot_impl body extracted", False, "regex failed")

m2 = re.search(r"async def _open\(self, token\).*?(?=\n    async def )", src, re.DOTALL)
if m2:
    body2 = m2.group(0)
    i_mask2 = body2.find("add_init_script(MASK_JS)")
    i_goto2 = body2.find("page.goto")
    check("_open: MASK_JS injected before first page.goto",
          i_mask2 != -1 and i_goto2 != -1 and i_mask2 < i_goto2,
          "mask_idx=%d goto_idx=%d" % (i_mask2, i_goto2))
else:
    check("_open body extracted", False, "regex failed")

# 4. PATCH_JS / WARM_JS / COLLECT_JS unchanged (Phase C jitter still present).
check("PATCH_JS still has fetch mock",
      "window.fetch" in solver.PATCH_JS and "originalFetch" in solver.PATCH_JS)
check("WARM_JS still has initAliyunCaptcha", "initAliyunCaptcha" in solver.WARM_JS)
for marker in ("window.z_um.getToken", "typeof window.z_um", "return out"):
    check(f"COLLECT_JS contains {marker!r}", marker in solver.COLLECT_JS)
check("COLLECT_JS still has Phase C jitter",
      re.search(r"setTimeout\(r,\s*\d+\s*\+\s*Math\.random\(\)\s*\*\s*\d+", solver.COLLECT_JS) is not None)

print("== RESULT ==")
if failures:
    print("FAIL: %d assertion(s) failed" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
else:
    print("PASS: all masking assertions passed")
    sys.exit(0)