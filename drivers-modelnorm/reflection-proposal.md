# MODELNORM-1 — reflection proposal (DRAFT for the parent to register; I do not edit PREREGISTRATIONS.md)

Two proposed registry entries and four proposed TOOLING-TRAPS additions. Every claim here is
🟢 VERIFIED on **blend-v1**, `.native` frame, 90 WPM baked, with the artifact named.

---

## Proposed registered finding 1 — MODELNORM-1

**A per-model 0-1 anchored normalization (0 = random-layout mean, 1 = per-model optimized
search) is STABLE at campaign budget and changes NO ranking; its product is an interpretable
weight, not a re-ordering.**

- 🟢 **Anchors stabilize completely.** Two independent seeds (20260728 / 20260901) at identical
  10M-unique budget returned the **identical champion layout for all three models**; seed gap
  **exactly 0.0 ms**, 40/40 islands within 0.10 %, champion last improved at epoch 4-12 of 55.
  Anchor-induced blend perturbation **0.000000** vs a **0.003284** decision margin.
  Artifact: `state/modelnorm/artifacts/anchors-evidence.json`.
- 🟢 **The n=100 "0" anchor is sufficient.** n=100 → n=1000 shifts it by **< 1 SE** on every
  model (max −0.979 SE = −1.70 % of span, AALTO); the candidate ranking is unchanged at n=100,
  1000 and 10 000. Artifact: `step1-zero-anchor.json`.
- 🟢 **Normalizing changes NO ranking**: 0 discordant pairs within each model (affine, positive
  scale — asserted), 0 vs the raw mean of the three surfaces, 0 vs raw mean saved %, **and 0 vs
  the prior ceiling-fraction anchoring**. It changes 2 pairs vs the scale-broken raw `min()`,
  with gaps 0.0122 / 0.0241 against a **0.231897** conservative normalized floor — neither
  clears it. Artifact: `rank-table.json`.
- 🟢 **The weight DOES become an interpretable preference.** AALTO normalized: **1.00000 at
  (1,0,0) → 0.93740 at (2,1,1) → 0.90286 at (1,1,1)**, monotone; solo champions are **24, 26, 24
  of 30** slots apart; the extreme cells span **4.31 ms/char** (254.0711 … 258.3823).
  Artifact: `sweep-report.json`.
- 🟢 **The blend champion loses to arm B.** `pctsk-reayfgdlm.niuobzvwxh,qj'`, 9,811,784 unique
  evals, **256.6268 ms/char vs 253.9006** (+2.7262); beats arm A (256.8466) and qwerty30m
  (264.1389). **No dominator** (best n_ge 5/10 blend, 7/10 solo); normalized floor **+0.902863**.
  Artifact: `judgement.json`.
- 🟢 **The correction is very unequal across models.** Search-anchoring beats ceiling-anchoring by
  **1.21 % (AALTO) / 14.74 % (COMMUNITY) / 15.19 % (POOL)** of span — AALTO is nearly saturated
  by existing layouts (arm B sits at 0.9879 of AALTO's own optimum).

**Relation to FLOOR-METHODOLOGY-1:** this confirms the *motivation* (raw aggregation over
per-surface quantities is scale-broken) while bounding the *payoff* — on this candidate set the
scale-break was not actually flipping any ordering, and the normalized and ceiling-fraction
anchorings agree on every rank.

## Proposed registered finding 2 — MODELNORM-1 addendum, two anchoring defects

- 🟢 **"qwerty30m ≈ 0" is FALSE and is an unsafe direction guard.** qwerty30m is at the
  **0.00-0.20 percentile** of a 1000-layout random pool (z = −2.5 … −3.1) and normalizes to
  **0.50-0.62**. A correctly-signed implementation FAILS the "qwerty ≈ 0" check, so using it
  invites reversing the sign — the exact inversion the campaign already shipped once
  (`oxey-style`). Use the random-pool anchor plus monotonicity instead.
- 🟢 **A random-layout "0" wastes ~90 % of the scale.** Excluding qwerty30m, the 7
  optimized/community layouts occupy only **0.1696 / 0.0895 / 0.0962** of the per-model range
  and **0.0934** of the blend range. A percentile or incumbent-based "0" would put the range
  where candidates actually live.
- 🟢 **Effective independent models = 1.1672 of 3** on a homogeneous n=10 000 random pool (PC1 =
  92.34 % of variance, Kaiser count 1; ρ(A,C)=0.8310, ρ(A,P)=0.8729, ρ(C,P)=0.9502) — so equal
  weights are NOT neutral. **But** the solo champions are 24-26/30 apart, so the wide-pool
  correlation does not describe the near-optimal band. Both numbers should be quoted together.
- 🟢 **FLAGSHIP-1's "seed = 78-83 % of SS" is an iWeb figure and does NOT transfer to blend-v1**,
  where the seed main effect is **0.74 % (raw ms) / 0.83 % (saved %)** on `COMMUNITY_BASE`, the
  only surviving per-seed family. Two orders of magnitude. Artifact: `seed-floor.json`.

---

## Proposed TOOLING-TRAPS additions

**T-A. A BLAS matmul's result depends on its operand SHAPE, so a batched objective must use a
constant tile shape or it is not a function of its input.**
A tiled `(B, 29791) @ (29791, 3)` evaluator returned results differing by **~1e-15 relative** for
the *same* layout depending on how many other layouts shared its batch (a partial final tile
dispatches a different kernel). That makes the objective batch-length dependent, so neither the
search nor its checkpoint-resume is reproducible. Fix: zero-pad the final partial tile so every
matmul is at exactly one shape; then assert **bit-exact** invariance across batch lengths. Also
record the tile size in the run's identity — the tile size *is* the shape, so it can never be
made bit-irrelevant, only frozen and named. (Same family as trap 36: assert the resume
reproduces on COUNTS **and VALUES**.)

**T-B. A resume is only "the same run" if every knob that shapes the SCHEDULE matches — and
`--epochs` is one of them.**
`per_epoch = budget * overshoot / (epochs * islands)`, so resuming with a different `--epochs`
rescales every remaining epoch's spend: the resumed run is a *different search* wearing the same
output filename, and it exits 0. Fix: stamp a `run_identity` over (objective, weights, budget,
islands, epochs, seed, polish_sweeps, ga_share, overshoot, corpus) into the checkpoint and refuse
a mismatched resume, naming the differing key. Discovered by *testing* resume rather than reading
the code — my first resume test compared a 2-epoch and a 4-epoch run and looked like a
non-reproducibility bug.

**T-C. `analyze`'s `--ref` row is a DIFFERENT CHARSET, so it breaks any invariance or aggregate
computed across "all rows".**
The default `--ref` is **classic** qwerty (`;./`), not `qwerty30m` (`'` and `-`). `sfr` counts
doubled *letters*, so a different **charset** legally changes it: 2.664409719629 (classic) vs
2.659577102696 (every C30M layout). Testing trap 23's invariance over the raw row set therefore
reports a **false non-invariance** with a spurious numpy std of 1.2e-3. Restrict every
cross-row statistic to the layouts you asked for. (Companion to trap 38's containment rule: the
extra row must be excluded from *statistics*, not merely tolerated in the *count*.)

**T-D. A paired floor computed on a `saved-vs-REFERENCE %` scale is DEGENERATE if the reference
is in the pool.**
`saved% = 100*(1 − fit/fit_ref)` per replicate makes the reference row identically `(0,0,0)`, so
`spread(X − ref) ≡ spread(X)` and the paired/unpaired ratio is forced to **exactly 1.0000** —
hiding the very cancellation the paired analysis exists to measure. Measured here: 1.0000 with
the reference in the pool, **0.5632** with it out. A ratio of exactly 1.0000 is the tell.
Exclude the reference row, and report the raw-unit floor alongside the normalized one since the
variance decomposition is **not** scale-invariant.

**T-E (small).** The 19-gauge frame is **not** all in `analyze`'s `gauges` block: that block
carries 15 (and lacks `bad-redir`, `onehand`), while `genkey`/`oxeylyzer1`/`oxeylyzer2`/`wfd` are
`community` scores. A win count printed as `n/14` while the frame is called "19-gauge" is a
denominator error of the same family as trap 23. Count them separately and state the denominator
on every line.
