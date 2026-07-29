# normgauge — reflection proposal (loose reusable learnings)

Raw material for the parent's knowledge pass. **No shared-KB write performed** — that is the
parent's call. Each item is scoped to what I actually verified, with the evidence inline so the
parent can judge it without re-running anything.

Source: branch `normgauge` @ `477bd64`, base `dd04219` (+ local ledger commit `37b2dd5`).

---

## A. TOOLING HAZARDS — generalize beyond this repo

### A1 🟢 A size-preserving source mutation can make CPython execute the WRONG bytecode, silently

**Proposed home:** `.agent/knowledge/workflow/agent-tool-quirks.md`

CPython validates a `.pyc` on **`(source mtime truncated to the SECOND, source size)`**. A mutation
harness that (a) writes a mutant, (b) runs the suite, (c) restores the original — all within one
mtime second — with a **size-preserving** mutant leaves a cache CPython considers **valid** while it
holds the *other* version's bytecode.

**Verified directly, not inferred:** my mutant reorders `a - b` into `b - a` (same length). After
restore, the `.pyc` recorded `mtime 1785288965 / size 24429`, *exactly matching the restored source*,
and the subprocess kept running the mutant. It fails in **both** directions — a mutant can read as
"caught" when it never executed, or as "survived" when the original ran.

**Fix:** unlink `__pycache__/<module>.*.pyc` on every write. **What caught it:**
`testkit.assert_harness_detects_a_fatal_mutant`'s **restore-to-green** third check — the baseline and
mutated checks both passed. Without that third check this is invisible.

**Suggested addition to `keybo/testkit.py`** (repo-local, not KB): a shipped
`write_module_and_invalidate(path, text)`, since every mutation harness here needs it and the failure
mode is silent.

### A2 🟡 `argparse` cannot take a layout string that begins with `-` or `.`

A random C30M permutation can start with `-` or `.`, and argparse reads such a token as an option
(`error: unrecognized arguments: -.jwglicyepkhbzsmq…`). Round-tripping generated layouts through
`argv` breaks on ~7% of random permutations (2 of 30 leading chars). **Fix:** build the namespace
from defaults and assign the list directly (`args.layouts = distinct`). Bit me on a 300-layout pool
after a 12-layout smoke test passed — **the small smoke test could not surface it.**

### A3 🟡 `keybo analyze` refuses a duplicate layout under two names

`analyze.py:452` raises *"internal error: N layouts requested but M rows produced"* when two keys map
to the same board. In my arm that collision **was the positive control passing** (solo cells
rediscovered their own anchors), so the shipped guard flagged a success as an error. **Fix:** pass
each *distinct* board once and re-attach aliases yourself.

### A4 🟡 `keybo analyze` always emits its `--ref` row, and `qwerty`'s charset is NOT C30M

The default `--ref qwerty` uses `;` and `/` where C30M has `'` and `-`. **Measured:** all 12 C30M
field rows share `sfr = 2.6595771027` **exactly** while the ref row reads `2.6644097196`. Leaving the
ref row in made every pair read "15 of 15 contested" and would have made `sfr` look non-invariant —
**this is the documented trap-38 signature, reproduced from a new direction.** Drop any returned row
you did not request.

---

## B. STATISTICAL / METHOD HAZARDS — the highest-value items

### B1 🔴 An inclusion-only participant bootstrap is a NO-OP on cell-rich data, and it FAILS SILENTLY BY MANUFACTURING SIGNIFICANCE

**The generalizable rule:** a bootstrap that resamples *which units exist* rather than *the values
being correlated* propagates no uncertainty when the unit set is effectively invariant. **Measure the
survival fraction before trusting any interval it produces.**

**Both sides of my design, measured:**

| held-out side | cells | pids | median pids/cell | fraction of cells surviving a resample |
|---|---|---|---|---|
| COMMUNITY | 866 | 4 | 1.0 | **0.6827** (works) |
| AALTO | 24,079 | 55,404 | 139.0 | **0.999992** (no-op) |

The two sides fail in **opposite** directions, so **a one-sided check looks fine.** The no-op side is
the one with *more* data — i.e. it would have manufactured significance exactly where a reader would
least suspect it.

**Fix:** cluster-bootstrap that **re-aggregates each cell's value** from the drawn participants' own
samples. **And the residual caveat:** even after the fix the AALTO side's survival is 1.000, so its
interval width is a **lower bound on uncertainty**, not a precision estimate.

### B2 🔴 A POINT ESTIMATE OUTSIDE ITS OWN CI is a cheap, high-yield diagnostic — add it as a routine assertion

My COMMUNITY point estimate was `0.411458` against `CI95 [0.364336, 0.372002]`. **Cause:** the
replicates aggregated with a **plain mean** while the point estimate used the shipped **IQR-mean**, so
the interval was honest *for a different statistic than the one it was placed around*. Aggregation gap
`0.032228` = **8.41×** the CI half-width; the CI midpoint sat near the plain-mean estimate (gap 0.011)
not the IQR one (0.043).

**Proposed rule:** any bootstrap that reports `(point, lo, hi)` should assert `lo <= point <= hi` and
say so loudly when it fails. It is a two-line check that catches a whole class of
estimator/statistic mismatches, and nothing else I ran would have caught this.

### B3 🔴 A "resolution floor" needs the SAME RULER on both sides of the comparison, not just the same quadruple

The registered quadruple rule (*a floor is a property of a (pool × replicate-structure × scale ×
statistic) quadruple*) is **necessary but not sufficient**. I satisfied it and still got a verdict
wrong, because I compared **cell A's best on objective A** against **cell B's best on objective B**.
Two objectives = two rulers, even at one quadruple.

**Corrected form:** to compare champions from different objectives, **re-score every champion on ONE
chosen objective.** Doing so flipped my `registered vs drop-pool` verdict from "2.4× win" to
**"0.25× — a TIE"**, and that tie is what revealed POOL's weight does no observable work.

**Corollary:** a **zero** within-cell sd cannot be the yardstick (3 of my 6 cells had one) — it makes
every gap "resolvable". Use the **max** across cells.

### B4 🟠 A cross-source held-out validation may need NO refit — check for disjointness first

Worth surfacing because it converted an "out of scope, needs a training campaign" item into a
~25-minute measurement. If two sources' **training subsets are disjoint** (here: aalto pids <200000
vs community pids 200001–200007), then **each source's data is already out-of-sample for the other's
fitted surface** — no refit needed. Check `set(pids_A) & set(pids_B)` before concluding held-out
weighting is infeasible.

### B5 🟠 A prediction can be UNSCOREABLE by category error, and no gate catches it

My P10 predicted my search-noise sd would land "within 2× of 0.0492–0.0995". My sd is in
**normalized blend units**; those are **ms/char**. The ratio has no truth value — and I wrote it
*inside the section warning against borrowed rulers*. **Proposed practice:** the wrong-constant sweep
should extend to **predictions**, not just results. A prediction is a number nobody re-derives,
because it isn't a measurement.

### B6 🟠 SCOPE, not the count, must travel with a participant/fold number

Three mutually contradictory-looking numbers here are **all true at their own scope**:

* **7** = distinct participants in the whole community file (`{200001…200007}`)
* **4** = participants in the 4-label rowStagger **subset the COMMUNITY surface was fitted on**
* **9** = distinct strings from a naive `label.rsplit("#",1)[1]`, because `+pseudo`/`+rareboost` are
  **corpus tags on the same submitter**, not other people

Two independent agents got this wrong **in opposite directions on the same day** (my prereg said 7
where 4 was right; the parent's correction said 9 where 7 was right). **A bare participant count
should not be quoted without its scope attached.**

---

## C. DOMAIN FINDINGS (keybo-specific, for the ledger not the KB)

### C1 🟢 "Normalization reorders nothing" and "the weighting reorders nothing" are DIFFERENT CLAIMS

Only the first is true, and it is true necessarily (affine positive rescale): **0 discordant pairs of
66, spearman +1.000000 on all three models.** The second is **false**: solo-AALTO vs solo-COMMUNITY
differ on **30 of 66** (ρ +0.2448); registered vs solo-AALTO on 18/66; registered vs ms/char on
20/66. **Strengthens under attack** — restricted to the 8 realistic layouts, AALTO and COMMUNITY are
*anti*-correlated at **−0.8095**.

### C2 🟢 On the SHIPPED `.standardized` frame the three sources share AALTO's bigram tensor

`standardized − native` is **exactly independent of the third slot** (max variation over `c`: AALTO
0.0, COMMUNITY/POOL 1.14e-13) and identically 0 for AALTO. So the three differ **only** in their
conditional trigram increment and are **less independent than on `.native`**. Anyone shipping a
multi-source gauge on the resolver's frame inherits this and should print it (`frame_caveat()`).

### C3 🟢 POOL is a measured near-symmetric blend, and its weight does no observable work

Fit level over 400 random layouts: `POOL = 0.498757·AALTO + 0.508017·COMMUNITY`, **R² = 0.93881**.
Cell level agrees (0.454530 / 0.449591, R² 0.87400). **Consequence:** `drop-pool` 50/50 **ties** the
registered weighting (0.25× floor), so **AALTO+COMMUNITY 50/50 is the simpler equivalent.** Equal
weights are *not* neutral — effective loadings 0.4996 / 0.5027 / 0.0204, i.e. the correlated pair's
agreement counted ~1.5×.

### C4 🔴 The campaign's "643×" is scissor-neighbourhood + covered-pair-filtered, and that filter is ASYMMETRIC

AALTO's count is **identical** in the filtered and unfiltered artifacts while COMMUNITY loses
**92.1%** (151,365 → 11,930). Scopes: `642.9×` (filtered neighbourhood) / `50.7×` (unfiltered, same
groups) / `46.2×` (whole stroke table) / **`907.8×`** (my own scan on the 31³ surface-cell frame the
gauge uses: 26,368,247 vs 29,047). **The conclusion — AALTO is far better supported — is confirmed
and strengthened. The constant should not be re-quoted as a reliability ratio.**

### C5 🟡 The 2-opt polish, not SA cooling, does nearly all the work on the normalized blend

Measured through the shipped CLI: `--no-local-search` returns **0.523429** at `max_outer` **60 AND
300** (barely above qwerty's 0.522878); with the polish on, **0.941646**. Any test or sweep that
disables the polish on this objective will read as "the objective is not being optimized."

### C6 🟢 Free positive controls that cost nothing and should be reused

* AALTO's `.native` **==** `.standardized` **byte-identical** (max|d| = 0.0), so MODELNORM-1's
  10M-eval AALTO champion is a **calibrated yardstick on the standardized frame**. Mine reproduced to
  **rel −1.4e-16**, validating corpus + loader + fit arithmetic + charset **before** any result.
* **Unplanned:** my AALTO **zero** anchor `243118526775.9713` matched MODELNORM's
  `243118526775.97125` — the n=100 / seed-20260728 random pool reproduces across two **independent
  implementations** of both the pool constructor and the evaluator.
* A **conservative** anchor (slower of two seeds) makes a later better search score **>1.0**; mine hit
  `1.00027` and the excess equalled `(anchor − fit)/span` **exactly**. That is the one-sided-bound
  property showing through, **not** a bug — worth documenting so nobody "fixes" it.

### C7 🟡 The padded/tiled evaluator guard needs a live mutation control

Pinning the matmul tile makes fits bit-stable across batch lengths. But the **guard** is only
meaningful while the **unpadded** path is still batch-*dependent* — measured **5.6e-15** rel here. I
ship an assertion that the unpadded path *does* vary, so if a future BLAS makes it invariant the
guard fails loudly instead of silently testing nothing.

---

## D. PROCESS OBSERVATIONS

1. **Smoke the reporting path on a tiny pool first.** Three real bugs (A2/A3/A4) surfaced in seconds
   that way instead of after a multi-hour run. ⚠ But note A2 was *only* visible at 300 layouts, not
   12 — a small smoke catches shape bugs, not distribution bugs.
2. **Run the FULL suite, not just your new tests.** My 36 tests all passed while **two pre-existing
   tests went red** from `args.model_weight` on a hand-built `SimpleNamespace`. A new flag must not
   break a caller for merely existing → `getattr(args, "flag", None)`.
3. **Amend the pre-registration in its own commit, and label pre- vs post-result.** Amendment 1
   (pre-result) legitimately keeps prereg protection; Amendment 2 (post-result) does not, and saying
   so is what lets a reader discount it correctly.
4. **Publish the blast radius of a self-found defect, not just the fix.** "Same branch either way,
   weights move ≤0.0136, refuting would need a 41.8× SE widening" is what makes a post-hoc correction
   auditable rather than suspicious.
5. **A parent's correction is a claim.** The parent's own correction #2 carried a wrong constant
   (9 vs 7). Re-deriving a handed-down figure cost ~3 minutes and caught it.
