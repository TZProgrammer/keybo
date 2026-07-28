"""VERIFY PASS — hostile stranger re-reading my own FIND-pass findings.

Each block names what would REFUTE the finding, then tries to refute it.
"""
import gzip, json, shutil, tempfile
import numpy as np
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.models.xgboost_model import XGBoostTypingModel
from keybo.scoring.table_scorer import TableBigramScorer
from keybo.optimize.qap_bound import certificate, gilmore_lawler_bound

ROOT="/tmp/qapaudit"; QW=NAMED_LAYOUTS["qwerty"]; geom=ROW_STAGGERED_30; N=30
def load_freq(p):
    o={}
    for ln in open(p):
        q=ln.rstrip("\n").split("\t")
        if len(q)==2: o[q[0]]=int(q[1])
    return o
def load_model(stem):
    d=tempfile.mkdtemp()
    for suf in (".json",".meta.json"):
        with gzip.open(f"{ROOT}/data/models/k31/{stem}{suf}.gz","rb") as fi, open(f"{d}/{stem}{suf}","wb") as fo:
            shutil.copyfileobj(fi,fo)
    return XGBoostTypingModel.load(f"{d}/{stem}.json")
bi=load_freq(f"{ROOT}/data/corpus/blend-v1/bigrams.txt")
bts=[TableBigramScorer(load_model(f"bigram_reg31_seed{s}"),bi,target_wpm=90.0,chars=QW) for s in (0,1,2)]
T2=np.mean([s._T for s in bts],axis=0); F2=bts[0]._F
lb=gilmore_lawler_bound(F2,T2)
def fit_bi(p): return float((F2*T2[np.ix_(p,p)]).sum())
P6=json.load(open("/tmp/qapaudit/agent-artifacts/qapaudit/probe6.json"))
best_fit=P6["best_search_fit"]

print("="*86)
print("F3 RE-EXAMINATION — I claimed '>=2.34% is IRREDUCIBLE GL SLACK'. IS THAT DIRECTION RIGHT?")
print("="*86)
print("  Decomposition: found - lb = (found - OPT) + (OPT - lb),  with OPT <= found always.")
print("  My measurement: a deep search found F_best with gap (F_best-lb)/lb = 2.3410%.")
print("  Since OPT <= F_best  =>  (OPT-lb)/lb <= 2.3410%.")
print("  *** That makes 2.3410% an UPPER bound on the slack, NOT a lower bound. ***")
print("  => MY FIND-PASS WORDING WAS BACKWARDS. Corrected claim below.")
print()
print("  What IS airtight:")
print(f"    (a) The best layout a deep search on the CERTIFIED objective could find still")
print(f"        certifies at {(best_fit-lb)/lb*100:.4f}% — so the quoted gap does NOT approach 0")
print(f"        even for a layout the search cannot improve. The certificate's RESOLUTION")
print(f"        floor is ~{(best_fit-lb)/lb*100:.2f}%, whatever the split.")
print(f"    (b) 'within N% of optimal' is MATHEMATICALLY TRUE as written: OPT >= lb implies")
print(f"        (found-OPT)/OPT <= (found-lb)/lb. So each entry's claim is VALID, just LOOSE.")
print(f"    (c) The certificate therefore cannot distinguish 'near-optimal' from 'mediocre'")
print(f"        below its floor: qwerty {(fit_bi(bts[0].permutation(Layout(QW,geom)))-lb)/lb*100:.2f}%,"
      f" search-best {(best_fit-lb)/lb*100:.2f}%.")

print()
print("="*86)
print("F2 RE-EXAMINATION — is certificate()'s statement string really the ROOT CAUSE of the")
print("                    dropped qualifier at ledger :2423/:2463?")
print("="*86)
c=certificate(F2,T2,best_fit)
print(f"  statement: \"{c['statement']}\"")
print(f"  REFUTER 1: does the statement carry ANY scope qualifier? ->",
      any(k in c['statement'].lower() for k in ("bigram","component","quadratic","partial")))
print(f"  REFUTER 2: does the returned dict carry scope metadata a writer could copy? ->",
      [k for k in c if k not in ('lower_bound','found_fitness','gap_pct','statement')] or "NO extra keys")
print("  REFUTER 3: could the :2423 writer instead have copied a DRIVER log line that")
print("             DOES qualify? cond_rebuild.py:257 prints 'bigram-component certificate'.")
print("             -> but the P11-ablation driver is NOT in this repo, so I CANNOT verify")
print("                which string that entry was copied from. Causation is INFERRED (orange),")
print("                not verified. The IN-FILE defect (unqualified statement) is VERIFIED.")

print()
print("="*86)
print("F1 RE-EXAMINATION — does the component mismatch actually MOVE A CONCLUSION?")
print("="*86)
print("  REFUTER: if fit_bi and the searched objective are monotonically related, the")
print("           certificate on one transfers as an ordering statement to the other.")
print(f"  measured spearman(fit_bi, fit_tri_corrected) = {json.load(open('/tmp/qapaudit/agent-artifacts/qapaudit/probe3.json'))['spearman_bi_tri']:.4f}")
print("  -> HIGH but < 1. And an ORDERING correlation is NOT a bound transfer: a GL bound")
print("     on component A gives NO bound on A+B, because min(A+B) >= min(A) + min(B) and")
print("     min(B) is uncertified. Let me verify that inequality is the binding issue:")
# how much would we need to bound the trigram part to certify the combined objective?
P3=json.load(open("/tmp/qapaudit/agent-artifacts/qapaudit/probe3.json"))
q=P3["rows"]["qwerty"]
print(f"     qwerty: certified mass {q['bi']:.4g} ({q['bi_share_of_comb']:.2f}% of combined)")
print(f"             uncertified    {q['tri']:.4g} ({q['tri_share_of_comb']:.2f}%)")
print(f"     -> {q['tri_share_of_comb']:.1f}% of the searched objective's mass has NO bound.")
print("  VERDICT: the mismatch is REAL and the omitted term is the MAJORITY of the mass.")
print("           But it does NOT invalidate the entries that KEEP the qualifier — those")
print("           correctly scope the claim to the bigram component. It invalidates only")
print("           the UNQUALIFIED readings (:2423, :2463, and the module's own statement).")

print()
print("="*86)
print("F5 RE-EXAMINATION — are the t_in/f_in mutants reachable defects or contrived?")
print("="*86)
print("  REFUTER: if F and T were symmetric, row-vs-column is a no-op and the mutant is")
print("           unreachable in practice.")
print(f"  F2 symmetric? {np.allclose(F2,F2.T)}   T2 symmetric? {np.allclose(T2,T2.T)}")
print(f"  max|F2-F2.T| = {np.abs(F2-F2.T).max():.4g}   max|T2-T2.T| = {np.abs(T2-T2.T).max():.4g}")
print("  -> BOTH asymmetric on the real instance, so the transposition is live, not contrived.")
print("     The mutant class is reachable AND the shipped test does not catch it.")

print()
print("="*86)
print("SELF-KILL CHECK — did I run any control only AFTER using its result?")
print("="*86)
print("  probe2: indep_fitness positive-controlled vs shipped qap_fitness BEFORE the sweep. OK")
print("  probe5: bound_variant('shipped') controlled vs shipped bound BEFORE variants. OK")
print("  probe7: bound(mode='code') controlled BEFORE comparing to docstring version. OK")
print("  probe9: gl_cost controlled (assert) BEFORE the pinned-bound computation. OK")
print("  probe6: NO control — it uses only the shipped bound + a search. Its numbers")
print("          share a component with the target (the shipped bound IS the target).")
print("          -> F3's floor number is NOT independent of the code under test. Flagged.")
print("\nPROBE10-DONE")
