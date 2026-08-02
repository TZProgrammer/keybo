"""Shared preamble: pin threads, force MY worktree, assert D5. Import FIRST, before keybo."""
import os, sys, hashlib

MY_WT = "/local/home/zegertho/agent/workspaces/tournament/wt"
E2E = "/local/home/zegertho/keybo-e2e"
OUT_MODELS = "/local/home/zegertho/agent/workspaces/tournament/models"
ART = "/local/home/zegertho/agent/state/tournament/artifacts"
SHIPPED = "/local/home/zegertho/repos/keybo/data/models/k31"

# D5 — the venv's editable install points at the SHARED checkout, which is a LIVE SIBLING's
# working tree (pick2, on branch pick2-decision). A naive `import keybo` silently measures
# ANOTHER AGENT'S BRANCH. Verified firing 2026-08-02. This insert is load-bearing.
sys.path.insert(0, MY_WT + "/src")


def assert_d5():
    import keybo
    assert keybo.__file__.startswith(MY_WT + "/"), (
        f"D5 FAIL: keybo resolved to {keybo.__file__} -- NOT my worktree {MY_WT}. "
        "The shared checkout is a live sibling's tree; refusing to measure another branch."
    )
    return keybo.__file__


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# The 13-board field. Every string verified against a repo artifact or the registry (TOURNAMENT-1
# prereg section 1) -- NOT transcribed from the parent's brief.
BOARDS = {
    # --- TUNED / KEYBO (7) ---
    "arm-B":       "flmpg-yuo,sntdcireahkxbwv'.jzq",
    "BALL-1":      "flmpg-yuo,sntcdireahkxbwv'.jzq",
    "F(2.5)":      "flmpg-,uoysntdcireahkxbwv.'jzq",
    "F(2.0)":      "pyu.,gdfnlhieaocstrmkj'-qbwzvx",
    "candidate":   "pyu.,vdfnlhieaocstrmkj'-qgwbzx",
    "keybo-lsb":   "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    # --- COMMUNITY (5) --- registry strings (src/keybo/layouts.py)
    "colemak":     "qwfpgjluy;arstdhneiozxcvbkm,./",
    "colemak-dh":  "qwfpbjluy;arstgmneiozxcdvkh,./",   # variant B; variant A scored separately
    "graphite":    "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "semimak":     "flhvz'wuoysrntkcdeaixjbmqpg,.-",
    "dvorak":      "',.pyfgcrlaoeuidhtns;qjkxbmwvz",
    # --- BASELINE (1) ---
    "qwerty":      "qwertyuiopasdfghjkl;zxcvbnm,./",
}
# The registered colemak-dh ambiguity: two published strings in prior artifacts.
COLEMAK_DH_VARIANTS = {
    "colemak-dh/A(canon.json)":        "qwfpbjluy;arstgmneioxcdvzkh,./",
    "colemak-dh/B(charset_hamming)":   "qwfpbjluy;arstgmneiozxcdvkh,./",
}
