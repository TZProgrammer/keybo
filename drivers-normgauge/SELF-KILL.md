# NORMGAUGE-1 — SELF-SEPARATION PASS: what I killed of my own

Re-read as a hostile stranger, after the results were committed. Each check was chosen so it does
NOT share a component with its target. Four kills, one of them in a commit I had already made.

---

## KILL 1 🔴 A WRONG CONSTANT ATTACHED TO A TRUE CONCLUSION, in my own committed commit message

**Committed claim (`c6f9932`):** *"the registered champion beats equal by 11.3x, drop-pool by
2.4x, and the three solos by 51.6-85.5x my OWN measured across-seed sd."*

**The `drop-pool` number is wrong, and it flips that one verdict.** I compared

* `registered`'s best **on the registered objective** (0.950704), against
* `drop-pool`'s best **on the DROP-POOL objective** (0.949548).

Those are two different rulers. Re-scoring `drop-pool`'s own champion **on the registered
objective** gives **0.950561**, so the honest gap is **+0.000143 = 0.25x** the resolution floor
(0.000576) — **a TIE, not a 2.4x win.** The `solo-AALTO` figure moves too: **48.98x**, not 51.6x.

Corrected verdicts, all on the registered objective against the same floor:

| vs | gap | x floor | verdict |
|---|---|---|---|
| equal | +0.006488 | 11.26x | RESOLVED |
| **drop-pool** | **+0.000143** | **0.25x** | **TIE** ← was claimed as a 2.4x win |
| solo-AALTO | +0.028234 | 48.98x | RESOLVED |
| solo-COMMUNITY | +0.049261 | 85.47x | RESOLVED |
| solo-POOL | +0.032052 | 55.61x | RESOLVED |

**The CONCLUSION survives — the weighting IS load-bearing** (4 of 5 rivals resolved, up to 85x
floor) — **which is exactly why this constant went unchecked.** That is the failure shape the brief
warned about, found for the fifth time in this campaign and the second time inside my own work.

**And it has a substantive consequence I must not bury:** `drop-pool` (AALTO+COMMUNITY 50/50, POOL
dropped entirely) is **statistically indistinguishable from the registered weighting.** Since POOL
is a measured near-symmetric blend of the other two, *dropping it and splitting 50/50 buys the same
optimum.* So POOL's 0.0612 weight is not doing observable work at this resolution.

## KILL 2 🔴 MY OWN PREREG'S PARTICIPANT COUNT (registered as AMENDMENT 1 A1.1, pre-result)

Prereg said "n=7 community participants"; the 4-label training subset has **FOUR** (200001,
200003, 200006, 200007). The 7 are in the whole file. Conclusion ("thin") held and *strengthened*.

## KILL 3 🔴 MY OWN REGISTERED BOOTSTRAP WAS A NO-OP (AMENDMENT 1 A1.2, pre-result)

The registered inclusion-bootstrap kept a cell if ANY drawn participant appeared in it. On the
AALTO side, **0.999992 of 24,079 cells survive every resample**, so cell values never moved and the
interval collapsed — it would have **manufactured significance on the side with the most data.**
The two sides fail in opposite directions (COMMUNITY: 866 cells, median 1 pid/cell, 0.6827
survive), so a one-sided check would have looked fine.

## KILL 4 🔴 THE CI WAS AN INTERVAL FOR A DIFFERENT STATISTIC (AMENDMENT 2, post-result)

COMMUNITY's point estimate 0.411458 fell **outside its own CI** [0.364336, 0.372002]. Cause:
replicates aggregate with a plain mean, the point estimate uses the shipped IQR-mean; the gap
(0.032228) is **8.41x the CI half-width**. Found *after* the result, so it gets no
pre-registration protection and its blast radius is published: same branch either way, weights move
≤0.0136, and refuting (c) would need a **41.8x** SE widening.

---

## ATTACKS THAT FAILED TO REFUTE (the claims that survived)

**A1. Is "weighting reorders" just qwerty30m dragging the correlation?** No — **it strengthens**:

| excluded | n | discordant | spearman(aalto-n, comm-n) |
|---|---|---|---|
| nothing | 12 | 30/66 | +0.2448 |
| qwerty30m | 11 | 30/55 | +0.0182 |
| qwerty30m, graphite, semimak, arm-A | 8 | 24/28 | **−0.8095** |

Restricting to the 8 layouts anyone would actually consider makes AALTO and COMMUNITY *anti*-
correlated. The disagreement is not an outlier artifact — it is sharpest exactly where selection
happens.

**A2. Does each weighting cell win on its OWN objective?** **6/6.** A cell losing on its own
objective would mean the search or the weighting plumbing is broken. This control was run only
*after* it could no longer be used to tune anything.

**A3. Does the within-model null hold?** **0 discordant pairs of 66 on all three models**, spearman
+1.000000 — MODELNORM-1's null reproduces exactly, as it must for an affine rescale.

**A4. Does my resolution floor share a component with what it judges?** Partly, and I state it: the
floor is the max across-cell within-seed sd of the *same* searcher on the *same* objectives. It is
a search-reproducibility floor, not an independent error model. A zero within-cell sd (3 of 6 cells)
cannot be the yardstick — it would make every gap "resolvable" — so the **MAX** across cells is
used, which is the conservative choice.
