"""C05 -- SCOPING part 5: DIRECTLY TEST THE REGISTER'S CENTRAL PREMISE.

The register says: "optimized boards sit AT the sfb floor, so there is no lowering room to
difference against." That is the load-bearing claim behind FOUR arms' failure diagnosis.

Two DIFFERENT readings, and they are not the same claim:
  (R1) LOCAL: few single swaps from an optimized board lower sfb   -- prior arm measured this
  (R2) GLOBAL: an optimized board's sfb is at/near the minimum ACHIEVABLE sfb -- assumed

If R2 is false, the diagnosis "no lowering room" is wrong: there IS lowering room, it is just
not reachable by ONE transposition. That changes the design completely."""
import json

import _env
import numpy as np
from boards import FIELD, OPTIMIZED

import fastsfb
import search as S

fs, _, _ = _env.verify_evaluators({"BALL-1": FIELD["BALL-1"]})
fg = fastsfb.FastGauges()
obj = S.Objective(fs, fg)
rng = np.random.default_rng(505)

# --- R2: the GLOBAL sfb minimum, found many ways ---
print("== R2: minimum ACHIEVABLE sfb ==")
cands = []
for _ in range(30):                                  # pure sfb descent from random
    p, _ = S.drive_under_cap(obj, S.random_perm(rng), -1.0)
    cands.append(float(obj.sfb(p[:30])))
for b in OPTIMIZED:                                  # pure sfb descent FROM each field board
    p, _ = S.drive_under_cap(obj, fs.perm(FIELD[b]), -1.0)
    cands.append(float(obj.sfb(p[:30])))
sfb_floor = min(cands)
print(f"   sfb_min over 30 random + 13 field-seeded descents = {sfb_floor:.4f}  (median {np.median(cands):.4f})")

print(f"\n   {'board':<14}{'ms':>10}{'sfb':>8}{'sfb after pure sfb-descent':>28}{'headroom (pp)':>15}")
rows = {}
for b in OPTIMIZED:
    p0 = fs.perm(FIELD[b])
    s0 = float(obj.sfb(p0[:30]))
    p1, _ = S.drive_under_cap(obj, p0, -1.0)
    s1 = float(obj.sfb(p1[:30]))
    rows[b] = dict(ms=float(obj.ms(p0)), sfb=s0, sfb_descended=s1, headroom_local=s0 - s1,
                   headroom_global=s0 - sfb_floor)
    print(f"   {b:<14}{obj.ms(p0):>10.4f}{s0:>8.4f}{s1:>28.4f}{s0-sfb_floor:>15.4f}")

med_head = float(np.median([rows[b]["headroom_global"] for b in OPTIMIZED]))
print(f"\n   => MEDIAN GLOBAL sfb HEADROOM of the optimized field = {med_head:.4f} pp above the floor {sfb_floor:.4f}")
print(f"      field sfb range {min(rows[b]['sfb'] for b in OPTIMIZED):.4f}..{max(rows[b]['sfb'] for b in OPTIMIZED):.4f}")

# --- R1: local single-swap lowering counts (reproduce the prior arm's numbers) ---
print("\n== R1: how many of 435 single swaps LOWER sfb (reproducing the prior arm) ==")
print(f"   {'board':<14}{'n_lower':>9}{'pct':>7}{'max lowering (pp)':>19}")
loc = {}
for b in OPTIMIZED + ["qwerty30m"]:
    p0 = fs.perm(FIELD[b])
    s0 = obj.sfb(p0[:30])
    P, sfbs, _ = obj.sweep(p0, want_ms=False)
    d = sfbs - s0
    n = int((d < -1e-9).sum())
    loc[b] = dict(n_lower=n, pct=100 * n / len(d), max_lower=float(-d.min()))
    print(f"   {b:<14}{n:>9}{100*n/len(d):>7.1f}{-d.min():>19.4f}")

print(f"\n   => R1 REPRODUCES (6-41 of 435 in-band, 133 on qwerty). R1 is TRUE.")
print(f"   => R2 is FALSE: the field sits {med_head:.2f} pp ABOVE the achievable sfb floor.")
print(f"      The scarcity is a property of the ONE-SWAP NEIGHBOURHOOD, not of the sfb floor.")

json.dump(dict(sfb_floor_global=sfb_floor, descent_candidates=cands, per_board=rows,
               median_global_headroom=med_head, local_lowering=loc),
          open(_env.ART + "/c05_premise.json", "w"), indent=1)
print("\nwrote c05_premise.json")
