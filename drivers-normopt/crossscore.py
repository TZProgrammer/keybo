"""Score every produced layout under EVERY ruler + the field, and dump one JSON."""
import sys, json, glob, os
sys.path.insert(0, "/tmp/normopt/src")
import numpy as np
from keybo.scoring import model_norm as MN
from keybo.analysis import surfaces as S
from keybo.cli.analyze import _EXTRA_NAMED
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.data.corpus import load_frequencies, production_corpus_dir
from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30

ANCH = "/tmp/normopt/drivers-normgauge/anchors.json"
anchors = MN.Anchors.read(ANCH)
fits = MN.SurfaceFits()
SPEC_B = MN.BlendSpec(weights={"AALTO":0.5411,"COMMUNITY":0.3977,"POOL":0.0612}, rule="registered (c)")
SPEC_C = MN.BlendSpec(weights={"AALTO":0.5,"COMMUNITY":0.5}, rule="50/50 drop-pool")

# --- the ms/char ruler: the SAME scorer arm A optimized -------------------
model = XGBoostTypingModel.load("/tmp/normopt-scratch/models/bigram_reg31_seed0.json")
freqs = load_frequencies(str(production_corpus_dir(None) / "bigrams.txt"))
C30M = S.C30M
tbl = TableBigramScorer(model, freqs, target_wpm=90.0, chars=C30M)
TOTAL_CHARS = float(sum(freqs.values()))

def ms_total(lay):
    return float(tbl.fitness(Layout(lay, ROW_STAGGERED_30)))
def ms_per_char(lay):
    # ms/char on the campaign's convention: total predicted ms / total bigram mass
    return ms_total(lay) / TOTAL_CHARS

def rulers(lay):
    f = fits.fit_of(lay)
    n = anchors.normalize_many(f)
    return {
        "ms_total": ms_total(lay),
        "ms_per_char": ms_per_char(lay),
        "aalto_n": n["AALTO"], "comm_n": n["COMMUNITY"], "pool_n": n["POOL"],
        "blend_registered_c": SPEC_B.blend(n),
        "blend_5050": SPEC_C.blend(n),
    }

# --- load the 30 produced layouts ---------------------------------------
produced = {}
for arm in "ABC":
    for s in range(10):
        p = f"/tmp/normopt/runs/{arm}-s{s}.json"
        d = json.load(open(p))
        produced[f"{arm}-s{s}"] = {"arm": arm, "seed": s, "layout": d["layout"],
                                   "reported_fitness": d["fitness"], "src": p}

# --- the field ----------------------------------------------------------
field = dict(_EXTRA_NAMED)
for k in ("graphite","semimak"):
    field[k] = NAMED_LAYOUTS[k]
field["BALL-1"] = "flmpg-yuo,sntcdireahkxbwv'.jzq"     # PREREGISTRATIONS.md:9423
field["arm-B"]  = "flmpg-yuo,sntdcireahkxbwv'.jzq"     # blend-report gauge_table
field["arm-A"]  = "udy.,fgpmliheaocsntr-k'qjwzbvx"
# the normgauge campaign's own anchor/blend boards (blend-report.json)
field["ng:anchor-AALTO"]     = "lnfdg-,yehcrstmaoiupxzbwv.kq'j"
field["ng:anchor-COMMUNITY"] = "cstr,kdeaigflnmypo.uwzqbxvh-j'"
field["ng:anchor-POOL"]      = "cyea,krstpguoi-mlndfwj'.qhvxzb"
field["ng:registered-best"]  = "ufio,vdnrmyhea.ptsclkj'-qgbzxw"
field["ng:droppool-best"]    = "clndf,geihrmstp.aouywzxbvk-qj'"
field["ng:10M-AALTO-champ"]  = "lnfdg-,yehcrstmaoiupxqbwv.k'jz"

def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

out = {"produced": {}, "field": {}, "provenance": {
    "branch": "normopt-layouts", "base": "96e6138",
    "anchors": ANCH, "corpus": str(production_corpus_dir(None)),
    "model": "/tmp/normopt-scratch/models/bigram_reg31_seed0.json (gz-inflated k31 bigram_reg31_seed0)",
    "total_bigram_mass": TOTAL_CHARS,
    "resolution_floor_ms_per_char": 0.135,
}}
for k, v in produced.items():
    r = rulers(v["layout"]); r.update(v); out["produced"][k] = r
for k, v in field.items():
    if not S.is_c30m(v):
        out["field"][k] = {"layout": v, "skipped": "not C30M charset"}
        continue
    r = rulers(v); r["layout"] = v; out["field"][k] = r

json.dump(out, open("/tmp/normopt/runs/crossscore.json","w"), indent=1, sort_keys=True)
print("wrote crossscore.json")
print(f"total bigram mass = {TOTAL_CHARS:.6g}")
print(f"\nsanity: qwerty30m ms/char = {ms_per_char(C30M):.6f}")
for k in ("keybo-c30m","keybo-lsb","BALL-1","arm-B","graphite","semimak"):
    if k in out["field"] and "ms_per_char" in out["field"][k]:
        print(f"  {k:22s} {out['field'][k]['ms_per_char']:.6f}")
