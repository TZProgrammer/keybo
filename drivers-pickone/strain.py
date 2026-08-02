"""Measured-strain + span arithmetic driver for pickone. Reads ONLY shipped keybo code."""
import os, sys, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "48"
WT = "/local/home/zegertho/agent/workspaces/pickone/wt"
sys.path.insert(0, WT + "/src")
import keybo
assert keybo.__file__.startswith(WT), f"WRONG KEYBO: {keybo.__file__}"

from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.scoring.utilization import DEFAULT_SLOWNESS, finger_name
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.data.corpus import production_corpus_dir

BOARDS = {
    "arm-B":       ("A", "flmpg-yuo,sntdcireahkxbwv'.jzq"),
    "F(2.5)":      ("A", "flmpg-,uoysntdcireahkxbwv.'jzq"),
    "BALL-1":      ("A", "flmpg-yuo,sntcdireahkxbwv'.jzq"),
    "F(2.0)":      ("B", "pyu.,gdfnlhieaocstrmkj'-qbwzvx"),
    "candidate":   ("B", "pyu.,vdfnlhieaocstrmkj'-qgwbzx"),
    "keybo-lsb":   ("C", _EXTRA_NAMED["keybo-lsb"]),
    "flagship-c3": ("C", _EXTRA_NAMED["flagship-c3"]),
    "archive-1846":("C", _EXTRA_NAMED["archive-1846"]),
    "graphite":    ("comm", NAMED_LAYOUTS["graphite"]),
    "semimak":     ("comm", NAMED_LAYOUTS["semimak"]),
    "colemak":     ("comm", NAMED_LAYOUTS["colemak"]),
    "colemak-dh":  ("comm", "qwfpbjluy;arstgmneiozxcdvkh,./"),
    "qwerty":      ("base", NAMED_LAYOUTS["qwerty"]),
}

out = {"provenance": {"keybo": keybo.__file__, "slowness_src": "keybo.scoring.utilization.DEFAULT_SLOWNESS",
                      "slowness": DEFAULT_SLOWNESS}}

# --- 1. layout-string validity: 30 chars, all distinct ---
val = {}
for n, (fam, s) in BOARDS.items():
    val[n] = {"len": len(s), "distinct": len(set(s)), "ok": len(s) == 30 and len(set(s)) == 30,
              "charset": "".join(sorted(s)), "family": fam, "string": s}
out["layout_validity"] = val
bad = [n for n, v in val.items() if not v["ok"]]
assert not bad, f"INVALID LAYOUT STRINGS: {bad}"

# --- 2. corpus: unigram keystroke share per finger (STRAIN primitive, not time) ---
cdir = production_corpus_dir(None)
bigrams = {}
with open(cdir / "bigrams.txt") as fh:
    for line in fh:
        p = line.split()
        if len(p) >= 2:
            bigrams[p[0] if len(p[0]) == 2 else p[-1]] = int(p[1]) if p[1].isdigit() else int(p[0])
# robust re-parse: detect column order
with open(cdir / "bigrams.txt") as fh:
    first = fh.readline().split()
out["bigram_file_first_line"] = first
bigrams = {}
with open(cdir / "bigrams.txt") as fh:
    for line in fh:
        p = line.rstrip("\n").split("\t") if "\t" in line else line.split()
        if len(p) < 2: continue
        a, b = p[0], p[1]
        ng, cnt = (a, b) if not a.isdigit() else (b, a)
        try: bigrams[ng] = int(cnt)
        except ValueError: pass
out["n_bigrams"] = len(bigrams)

# unigram counts from bigram table (first char), the corpus keystroke measure
uni = {}
for ng, c in bigrams.items():
    if len(ng) >= 1:
        uni[ng[0]] = uni.get(ng[0], 0) + c
out["n_unigram_chars"] = len(uni)

res = {}
for n, (fam, s) in BOARDS.items():
    lay = Layout(s, ROW_STAGGERED_30)
    per = {}
    tot = 0
    for ch, c in uni.items():
        # space is the THUMB (column 0) and is not an assignable slot: exclude it so the
        # shares below are per-LETTER-keystroke (the space-EXCLUDED kmstats convention).
        if ch == " " or not lay.has_key(ch): continue
        p = lay.pos(ch)
        hand = "L" if p[0] < 0 else "R"
        key = hand + {"pinky": "P", "ring": "R", "middle": "M", "index": "I"}[finger_name(p)]
        per[key] = per.get(key, 0) + c
        tot += c
    ks = {k: 100.0 * v / tot for k, v in per.items()}
    for k in ("LP", "LR", "LM", "LI", "RI", "RM", "RR", "RP"):
        ks.setdefault(k, 0.0)
    # measured-slowness-weighted strain index (pinky 1.43 / ring 1.21 / mid,idx 1.0)
    W = {"P": DEFAULT_SLOWNESS["pinky"], "R": DEFAULT_SLOWNESS["ring"],
         "M": DEFAULT_SLOWNESS["middle"], "I": DEFAULT_SLOWNESS["index"]}
    strain = sum(ks[k] * W[k[1]] for k in ks)
    res[n] = {"family": fam, "keystroke_share_pct": ks,
              "weak_left_pct": ks["LP"] + ks["LR"], "weak_right_pct": ks["RP"] + ks["RR"],
              "pinky_total_pct": ks["LP"] + ks["RP"],
              "pinky_asymmetry": abs(ks["LP"] - ks["RP"]),
              "strain_index_measured_slowness": strain,
              "coverage_of_unigram_mass_pct": 100.0 * tot / sum(uni.values())}
out["keystroke_strain"] = res
print(json.dumps(out, indent=1)[:400])
with open("/local/home/zegertho/agent/state/pickone/artifacts/strain_keystroke.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("\nWROTE strain_keystroke.json")
print("\n%-13s %3s %6s %6s %6s %6s %6s %6s %6s %6s | %6s %6s %6s %8s" % (
    "board","fam","LP","LR","LM","LI","RI","RM","RR","RP","Lweak","pinky","asym","STRAIN"))
for n, v in res.items():
    k = v["keystroke_share_pct"]
    print("%-13s %3s %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f | %6.2f %6.2f %6.2f %8.3f" % (
        n, v["family"], k["LP"],k["LR"],k["LM"],k["LI"],k["RI"],k["RM"],k["RR"],k["RP"],
        v["weak_left_pct"], v["pinky_total_pct"], v["pinky_asymmetry"],
        v["strain_index_measured_slowness"]))
