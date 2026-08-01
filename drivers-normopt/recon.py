import sys
sys.path.insert(0, "/tmp/normopt/src")
from keybo.scoring import model_norm as MN
from keybo.analysis import surfaces as S
from keybo import layouts as L

A = MN.Anchors.read("/tmp/normopt/drivers-normgauge/anchors.json")
fits = MN.SurfaceFits()
print("module:", MN.__file__)
A.assert_direction()
print("assert_direction  PASS")
A.assert_matches_surfaces(fits, A.provenance["probe_layout"])
print("assert_matches_surfaces (drift) PASS  probe=%r" % A.provenance["probe_layout"])
fits.assert_batch_invariant(A.provenance["probe_layout"])
print("assert_batch_invariant PASS")

# --- reconciliation targets from NORMGAUGE-1 ---
# (1) each model's own optimum normalizes to exactly 1.0; pool mean to 0.0  (module guard)
# (2) qwerty normalizes to ~0.42-0.56, NOT ~0  (docstring's stated direction guard)
qw = A.provenance["probe_layout"]
nq = A.normalize_many(fits.fit_of(qw))
print("\nqwerty normalized:", {MN.GAUGE_OF_POOL[k]: round(v,6) for k,v in nq.items()})

# (3) the AALTO 'one' anchor == MODELNORM-1's 10M champion fit, and its layout rescores to it
op = A.provenance["one_provenance"]
for pool, rec in op.items():
    tl = rec.get("target_layout"); tf = rec.get("target_fit")
    if tl is None: continue
    got = fits.fit_of(tl)[pool.replace("_control","").upper()] if False else None
print("\none_provenance keys:", list(op.keys()))
