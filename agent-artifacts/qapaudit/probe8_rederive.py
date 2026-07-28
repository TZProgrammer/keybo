"""FIND-pass probe 8: can any of the 6 registered certificate numbers be RE-DERIVED?

Each ledger cert is cert(F2, T2, fit_bi(best_perm)) for that round's F2/T2 (which depend
on the round's MODELS and CORPUS) and that round's champion layout. The champion layout
strings ARE in the ledger. The models are round-specific. This probe tests, for every
registered champion, what certificate the SHIPPED k31 models + blend-v1 corpus produce.

If a re-derived number lands on the registered one, that entry is reproducible.
If not, I must say which input is missing rather than assert a discrepancy (trap 20).
"""
import gzip, json, shutil, tempfile
import numpy as np
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.optimize.qap_bound import certificate, gilmore_lawler_bound

ROOT = "/tmp/qapaudit"; QW = NAMED_LAYOUTS["qwerty"]; geom = ROW_STAGGERED_30
def load_freq(p):
    o = {}
    for ln in open(p):
        q = ln.rstrip("\n").split("\t")
        if len(q) == 2: o[q[0]] = int(q[1])
    return o
def load_model(stem):
    d = tempfile.mkdtemp()
    for suf in (".json", ".meta.json"):
        with gzip.open(f"{ROOT}/data/models/k31/{stem}{suf}.gz", "rb") as fi, open(f"{d}/{stem}{suf}", "wb") as fo:
            shutil.copyfileobj(fi, fo)
    return XGBoostTypingModel.load(f"{d}/{stem}.json")

# The registered champions, verbatim from the pinned ledger (git show 106bfbc:PREREGISTRATIONS.md)
REG = [
  (287,  2.54, "bhaievlnsdpyo.utmrfcq;/,jgkwxz", "cond_rebuild / T3c tri-corrected"),
  (1195, 3.64, "ctsnhkuoepdwflr.iaygbjqmv,x/;z", "P8b w=0"),
  (1211, 4.38, "gaedinrtsw.oypumflcbq;jk,hxvz/", "P9 w=0 (F5M)"),
  (1884, 3.35, "cgldk.yuo,srthmpnieaxqwbvfj/;z", "P10 w=0"),
  (2423, 3.40, "uoy,.vldfgaeinprhtcs;/jkbmwxzq", "P11-ablation w=0"),
  (2463, 3.41, "cgldk.,yousrthmpnieaqxwbvfzj;/", "P11-FINAL w=0.5 (the pick)"),
]

for corpus_name, corpus_path in [("blend-v1", f"{ROOT}/data/corpus/blend-v1/bigrams.txt"),
                                 ("legacy-root", f"{ROOT}/data/corpus/bigrams.txt")]:
    bi = load_freq(corpus_path)
    bts = [TableBigramScorer(load_model(f"bigram_reg31_seed{s}"), bi, target_wpm=90.0, chars=QW)
           for s in (0, 1, 2)]
    T2 = np.mean([s._T for s in bts], axis=0); F2 = bts[0]._F
    lb = gilmore_lawler_bound(F2, T2)
    assert np.isfinite(lb) and lb > 0
    print(f"\n{'='*84}\nCORPUS {corpus_name}  |  shipped k31 bigram_reg31 seeds 0-2  |  GL lb {lb:.4f}")
    print(f"{'ledger':<8}{'reg%':>7}{'rederived%':>12}{'delta':>9}   champion")
    for ln, reg, lay, label in REG:
        try:
            perm = bts[0].permutation(Layout(lay, geom))
        except Exception as e:
            print(f"  :{ln:<6}{reg:>6.2f}   ERR {type(e).__name__}: {str(e)[:44]}")
            continue
        fit = float((F2 * T2[np.ix_(perm, perm)]).sum())
        assert np.isfinite(fit)
        c = certificate(F2, T2, fit)
        g = c["gap_pct"]
        assert np.isfinite(g)
        print(f"  :{ln:<6}{reg:>6.2f}{g:>12.4f}{g-reg:>+9.4f}   {label}")

# and: is the qwerty/colemak reference band stable across the two corpora?
print(f"\n{'='*84}\nSANITY: does the certified gap even depend on the CORPUS? (it must)")
for corpus_name, corpus_path in [("blend-v1", f"{ROOT}/data/corpus/blend-v1/bigrams.txt"),
                                 ("legacy-root", f"{ROOT}/data/corpus/bigrams.txt")]:
    bi = load_freq(corpus_path)
    bts = [TableBigramScorer(load_model(f"bigram_reg31_seed{s}"), bi, target_wpm=90.0, chars=QW) for s in (0,1,2)]
    T2 = np.mean([s._T for s in bts], axis=0); F2 = bts[0]._F
    lb = gilmore_lawler_bound(F2, T2)
    q = float((F2 * T2[np.ix_(bts[0].permutation(Layout(QW, geom)), bts[0].permutation(Layout(QW, geom)))]).sum())
    print(f"  {corpus_name:<12} lb {lb:>18.2f}  qwerty gap {(q-lb)/lb*100:.4f}%")
print("\nPROBE8-DONE")
