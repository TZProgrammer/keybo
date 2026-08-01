import sys
sys.path.insert(0, "/tmp/normopt/src")
from keybo.scoring import model_norm as MN
fits = MN.SurfaceFits()
A = MN.Anchors.read("/tmp/normopt/drivers-normgauge/anchors.json")

print("=== RECON A: MODELNORM-1's 10M AALTO champion fit ===")
tgt = "lnfdg-,yehcrstmaoiupxqbwv.k'jz"
published = 223236317224.4177
got = fits.fit_of(tgt)["AALTO"]
print(f"  layout      {tgt!r}")
print(f"  published   {published!r}")
print(f"  reproduced  {got!r}")
print(f"  rel         {(got-published)/published:.3e}")
assert abs(got-published)/published < 1e-12, "MISMATCH"
print("  MATCH (to float rounding)")

print("\n=== RECON B: per-model 'one' anchors == fit of layout_of_record ===")
lor = A.provenance["one_provenance"]["layout_of_record"]
for pool, lay in lor.items():
    got = fits.fit_of(lay)[pool]
    want = float(A.one[pool])
    rel = (got-want)/want
    print(f"  {pool:10s} anchor {want!r}  fit-of-record {got!r}  rel {rel:+.3e}  norm={A.normalize(pool,got):.9f}")

print("\n=== RECON C: the zero anchor rebuilt from (n=100, seed=20260728) ===")
pool_layouts = MN.random_pool(A.provenance["zero_n"], A.provenance["zero_seed"])
import numpy as np
f = fits.fits_from_permutations(pool_layouts)
for n,p in enumerate(fits.pools):
    m = float(f[:,n].mean()); want=float(A.zero[p])
    print(f"  {p:10s} published zero {want!r}  rebuilt {m!r}  rel {(m-want)/want:+.3e}")
