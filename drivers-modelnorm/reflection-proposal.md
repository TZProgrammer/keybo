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

---

# ADDENDUM — reflection pass, 2026-07-28 (state-flush scope only; parent owns the KB pass)

Three items the parent asked me to flush because they are load-bearing and were not verifiable
from my callback. Plus loose learnings that did not fit a registry entry.

## ADD-1 (upgrade of proposed T-A) — BLAS shape-dispatch is a REPEATING class, and there is a THIRD instance

**This is the second independent instance in one day, in different code, by a different agent.**
`price_many` (`79cb175`, verified: *"`_design(...) @ coeffs` dispatches to a different BLAS kernel
by array shape, so the same level priced at n=1 and n>=2 differed in the last ULP"*, 9 of 14
curves) and my `fit_batch`. Neither of us was looking for it. **Both were found the same way: by
asserting BIT-EXACTNESS where the author expected the property to be trivially true, and having
the assertion fail.**

**Magnitude, measured over 400 batch lengths (not a single probe) — this is the number to quote:**

| quantity | value |
|---|---|
| max relative error | **1.5946e-15** (= **7.2 × float64 eps**) |
| mean relative error | 6.8675e-16 |
| median | 8.7658e-16 |
| **batch lengths affected** | **275 of 400 = 68.8 %** |
| worst absolute | 2.4414e-04 ms |
| ÷ tightest adjacent gap | 2.25e-09 → **cannot reorder** |

The prevalence figure is the one that matters and the one a single probe would have missed: this
is **not** a rare edge case.

**Fix classification (asked explicitly):** *stronger* than "pad to a constant tile" on the axis
that matters — every matmul is issued at exactly `(16, 29791) @ (29791, 3)` with the final tile
**zero-padded**, giving **bit-exactness across all 400 batch lengths** (vs 275 differing
unpadded), asserted plus a **mutation control** that fails if the unpadded path ever becomes
batch-invariant on another BLAS. *Weaker* on one axis and this must be stated: it does **not**
make the answer independent of `TILE` — the tile size *is* the operand shape, so that is
unachievable by padding. I froze `TILE = 16` and record it (with the numpy version) in
`identity()`, so every published number names its shape. `price_many`'s "one shape-invariant
implementation" is the strictly stronger fix; mine is "one **pinned** shape, declared in the
provenance" — sufficient for a lookup-table objective, insufficient for anything published as a
physical constant.

**🟢 THIRD INSTANCE, CITED (structural claim only — I did not re-measure it, out of scope):**
`state/keybo-optimization/artifacts/noanchor-1/drivers/fast_eval.py:283-291`
(`SixSurface.saved_batch`) is the *same construction* — per-row `np.bincount` then an **unpadded**
`W @ self.mean_flat.T`, `(B,29791) @ (29791,6)` — and its docstring asserts *"Verified identical
to the gather to <1e-11"*, i.e. it **consumes the result as if shape-invariant**.
`normfloor_batch` (L304-307) routes through it, so the **ceiling-fraction normalized floor — a
headline dominance axis — inherits the shape dependence.** Nothing there is wrong today (1e-11 is
~4 orders looser than the effect, so it neither catches nor is broken by it); the risk is a future
agent tightening that comparison, or diffing two artifacts produced at different batch sizes, and
reading reordering noise as a finding.

**Others I noticed (only what I saw — no hunt, per scope):** my own `search_modelnorm._neighbours`
scores a fixed 435-row block, so it is shape-stable **by luck** (always the same partial tile) —
stable in practice, fragile by construction, and a good illustration of how this class hides. More
generally the `bincount`-then-matmul idiom is the campaign's standard fast-evaluator pattern and
was **copied between arms** (mine is a rewrite of `fast_eval.py`'s), so the population at risk is
"every driver that batches a QAP objective". **The discoverable tell is a *tolerance-based*
equivalence assertion (`<1e-11`, `allclose`) standing in for a bit-exactness one.** A grep like
`allclose|<1e-1[0-9]` near `@ .*flat` would enumerate it; I did not run it.

**Proposed sharpening of the class name:** it is **not** "BLAS is nondeterministic". It is
**"a tolerance-based equivalence test cannot detect shape-dependence, and shape-dependence is
exactly what breaks checkpoint-resume and cross-artifact diffs."** That phrasing tells the next
agent what to *write* (a bit-exactness assertion over batch lengths), not just what to fear.

## ADD-2 — the standing rule for borrowing a resolution floor (requested wording)

*(Full paragraph is in `report.md` under "Proposed standing rule". The one-line form:)*

> **A resolution floor is a property of a (POOL × REPLICATE-STRUCTURE × SCALE × STATISTIC)
> quadruple, not of a metric or a corpus. It may be quoted for a second design only if all four
> match; if any differs it must be recomputed, and the quadruple must be printed beside every
> floor so a reader can check the match without re-deriving it.**

Per-clause, with what "match" means and the instance that broke it:
- **POOL** — same candidates *and same kind* (near-optimal vs random are different quantities;
  mixing them is a Simpson artifact, trap 26). Must **exclude the reference layout** of any ratio
  scale, or the floor goes degenerate (my paired/unpaired ratio was forced to **exactly 1.0000**
  with qwerty30m in, **0.5632** with it out — *a ratio of exactly 1.0000 is the tell*).
- **REPLICATE STRUCTURE** — *what counts as noise*: per-seed refits, per-model disagreement,
  bootstrap draws, cross-corpus draws are four different nuisances. Mine bounds **model
  disagreement** (0.2319 normalized); the seed floor bounds **fit noise** (0.3914 saved%).
  **Neither is a refinement of the other** and quoting one for the other is the core error.
- **SCALE** — raw / saved-vs-reference% / 0-1 anchored are related by transforms that are **not
  variance-preserving**, so a variance decomposition does not carry (seed share reads 0.74% on raw
  ms vs 0.83% on saved%, and a saved% scale removes part of the nuisance *by construction*).
- **STATISTIC** — max-pair-spread / median / SD / p95 are not interchangeable, and a p95 over few
  replicates is ~the maximum (trap 46), so the replicate **count** travels with the statistic.

**Operational half:** *absence of a match is not licence to use the nearest available number.*
Recomputing mine cost one driver and seconds of CPU, against a borrowed figure wrong by **two
orders of magnitude** (FLAGSHIP-1's iWeb "seed = 78-83% of SS" → **0.74%** on blend-v1; fails
clauses SCALE+POOL). All four non-transfers this session *looked* like metric-level constants
("the floor is ~0.2 ms/char") when every one was a quadruple-level measurement. **A floor quoted
without its four labels should be treated as un-sourced.**

## ADD-3 — a pre-registration must not weld a VERDICT to a BOUND, and must be checked against its own candidate list

From classifying my own 6 failures into (a) world-differed vs (b) badly-posed — a split I
recommend every arm adopt, because **only (a) is evidence about the object of study**:
- **(a) 3 failures → 2 distinct facts.** P6 (normalization re-orders nothing) and P1+P13 (AALTO is
  near-saturated: arm B at 0.9879 of its own optimum). P1 and P13 are **one fact counted twice** —
  a pre-registration should avoid two predictions that can only fail together, or the failure
  count over-weights one mechanism.
- **(b) 3 failures → 2 distinct mis-posings.** **P15 welded a verdict ("no dominator") to a bound
  ("n_ge ≤ 4")**: the verdict held, the bound failed, and the bound was never well-posed because I
  inherited its ceiling from arms whose `floor` axis is a *different quantity* than mine. **Rule:
  if you redefine an axis, you may pre-register the VERDICT but not a numeric bound calibrated on
  the old axis** — and report `n_ge` as non-comparable across such arms. **P17+P18** stated a
  threshold over "all 8 candidates" while qwerty30m *is* one of the 8 and is the sole outlier;
  mechanism confirmed, arithmetic inconsistent with its own list. **Rule: before committing a
  numeric threshold, evaluate it against the actual candidate list you will score.**

## ADD-4 — loose learnings (no registry entry proposed)

1. **`$?` read through a pipe is not the command's rc.** My first pytest invocation was
   `pytest ... | tail`, which printed `RC=0` while pytest had **failed to spawn** (`Failed to
   spawn: pytest`). This is trap 1's shape in a form the traps file does not name — the *pipe*,
   not the missing sentinel. *Fix pattern:* `{ cmd > log 2>&1; echo $? > rc.txt; }` — never read
   `$?` downstream of a pipe.
2. **Backticks in `git commit -m` are shell-evaluated and silently delete words.** `` `qwerty_row` ``
   vanished from a committed message, leaving only two innocuous-looking "command not found"
   lines. The commit had already succeeded. *Fix:* `git commit -F -` with a quoted heredoc for any
   message containing backticks or `$`. Caught only because I re-read the committed message —
   verify the artifact (the message), not the exit code.
3. **The Bash 10-min clamp is per-CALL, not per-job.** A loop of three ~3-min searches got killed
   mid-third. Per-epoch checkpointing made it a non-event (resume finished in 10 s and reproduced
   bit-exactly). *Fix:* detach anything running 3+ sub-jobs regardless of individual job size.
4. **`analyze`'s `--ref` row is a different CHARSET (classic qwerty `;./`), so it poisons any
   cross-row STATISTIC**, not just a count. It made `sfr` look non-invariant (2 values, numpy std
   1.2e-3) when within C30M it is one value, std exactly 0.0. Trap 38 covers excluding it from the
   *count*; this extends to excluding it from *statistics*.
5. **A tolerance chosen in absolute units is meaningless without the magnitude.** My first
   batch-vs-gather assertion used `< 1e-6` on sums of order 2.4e11 — below one ULP, so it could
   only ever fail. Relative tolerances, or eps-multiples, for any large-magnitude accumulator.
6. **`ticket --expect-callback` + a callback fired from the SAME subshell as the work (trap 50)
   worked flawlessly** on both detached batches — no separate watcher process existed to die, and
   both sentinels read 0. Worth keeping as the default recipe rather than a fallback.
