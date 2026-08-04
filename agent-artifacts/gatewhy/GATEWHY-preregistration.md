# GATEWHY-1 — preregistration

**Committed BEFORE any measurement number exists.** Question: WHY does every interpretability frame
fail the high-wpm non-regression gate structurally on 4/4 folds while the served frame passes?
Frame defect, gate artifact, or both.

Base: `hybridtri` @ `52f0e3f`. Worktree `/local/home/zegertho/repos/keybo-wt-gatewhy`, branch
`gatewhy`.

---

## §0 CORRECTIONS TO MY BRIEF, registered before measuring so they cannot be hindsight

**C0.1 — `state/gateaudit/report.md` audits a DIFFERENT GATE than the one refusing the frames, and my
brief conflates them.** My brief cites GATEAUDIT-1's "false-flag floor: a PERFECTLY calibrated
64-cell bucket lands outside a (0.90,1.10) band 49% of the time" as evidence about *this* gate. It is
not. GATEAUDIT-1 audits the **calibration-slope** gate (`calibration_report` / `require_calibration`
in `verdicts.py:363/422`), whose statistic is `slope(obs~pred)` against a band `(0.90,1.10)`. The
gate refusing interp.1 and hybrid-B is the **high-wpm non-regression** gate
(`bucket_regression_report` / `require_no_high_wpm_regression_in_report`, `verdicts.py:210/288`,
`validate.py:683`), whose statistic is a **per-bucket Spearman ρ DELTA vs an incumbent baseline**
against a tolerance of `HIGH_WPM_TOLERANCE = 0.005` ρ units. Different statistic, different
threshold, different failure geometry. The 49%-false-flag number therefore **does not transfer** and
I will not quote it as this gate's floor. **What transfers is only the qualitative point** (thin
buckets are noisy) and the *cell identity* (azerty b120 = 64 cells / 23 participants), which I
verified is the same cell in both instruments. **I must MEASURE this gate's own false-flag floor
(INVARIANT 8: measure every floor, borrow no constant).**

**C0.2 — the `support` the brief tells me to "just use" is ABSENT from the artifacts I must
diagnose.** `bucket_regression_report(..., support=...)` records support only when a caller passes
it. `validate.py:959` passes `_bucket_support(test_cells)` — but INTERPFRAME-1's and HYBRIDTRI-1's
gate verdicts were **not** computed through `validate()`'s own block: both drivers re-run
`bucket_regression_report` themselves passing `support=rec.get("bucket_support")`, and
**`"bucket_support"` is not a key `validate()` ever writes** (it writes `bucket_matrix`, which
carries `n` / `n_raw` / `n_participants` per bucket). VERIFIED: every `high_wpm_gate` block in
`agent-artifacts/interpframe/lolo.json` has `support: NONE`, and the drivers' re-computed blocks get
`support=None` too. ⇒ **the published refusals carry NO support at all**; I recover it from
`bucket_matrix` and say so. This is a *silent* `.get()`-returns-None bug of exactly the class my
brief warns about ("rc=0 with all-None output is a key-not-present bug").

**C0.3 — `hybrid-B` shares MORE with interp.1 than the monotone constraints, so H-MONO is not the
only shared-attribute candidate.** My brief says the constraints are "the one thing hybrid-B and
interp.1 SHARE that the served frame lacks". They also share: (i) **no `wpm` column**
(`BIGRAM_HYBRIDB_FEATURE_NAMES` inherits interp.1's drop — verified `to_ms` docstring
`base.py:95-100`), hence (ii) **a DIFFERENT pace path in `to_ms`** — `validate.py:632` computes
`needs_wpm = "wpm" not in feature_names` and passes an explicit per-cell `wpm` array for BOTH interp
frames, while the served frame recovers pace from its `X` column. So H-PACE applies to hybrid-B too
and is *not* excluded by the hybrid-B result. Registered because it means H-MONO and H-PACE are
**confounded across the two frames** and must be separated by construction, not by contrast.

---

## §1 HYPOTHESES, each with its refutation condition

Registered as decision rules with numeric bars where a number is possible.

**H-GATE (gate artifact, support-conditioned).** The 4/4 structural refusals concentrate in buckets
whose support is thin enough that this gate's *own* measured reseed/false-flag spread exceeds the
0.005 ρ tolerance by a wide margin, i.e. the refusals are not distinguishable from instability.
* SUPPORTED iff (a) the refused buckets' |Δρ| are mostly *within* the measured null spread of a
  same-frame reseed (the SEEDNOISE analogue) at their support, AND (b) the served frame also produces
  refusals at comparable rates once the *same* arithmetic asymmetry it enjoys is removed (see §2 GC2).
* REFUTED iff the refused buckets' Δρ are large multiples of the measured null spread at their
  support and the served control stays clean under a symmetric comparison.

**H-ASYM (NEW — my own, and my leading hypothesis; registered as primary).** The gate as *operated*
by INTERPFRAME-1/HYBRIDTRI-1 is **arithmetically incapable of refusing the incumbent**, because the
baseline is CUR's own per-fold **mean over its 3 seeds** while each candidate seed is compared
individually. A candidate seed must beat the *incumbent's mean*; the incumbent's own seeds are
compared to their *own mean*, so a bucket where CUR's seed-to-seed spread is s can produce at most a
~s/2-scale deviation for CUR but a full (level difference + s) deviation for a candidate. Under this
mechanism **any** frame with a slightly lower high-wpm ρ level fails structurally while CUR
mechanically cannot, regardless of frame quality.
* SUPPORTED iff a **symmetric leave-one-seed-out (LOSO) control** — comparing each CUR seed against
  the mean of CUR's *other* seeds, and each candidate seed against the same-cardinality CUR mean —
  makes CUR itself fail structurally on ≥1 fold, or shows the refused candidate Δρ to be of the same
  order as CUR's own LOSO deviations in those buckets.
* REFUTED iff CUR stays clean under the symmetric LOSO control AND the candidates still fail
  structurally on 4/4 (that would make the asymmetry immaterial).
* This is DISTINCT from H-GATE: H-GATE is about *support/noise magnitude*; H-ASYM is about a
  **baseline-construction asymmetry** that biases the comparison independent of support.

**H-EXTRAP.** Interp frames extrapolate worse specifically at high wpm. REFUTED (pre-emptively, and
I say so rather than testing it) if it cannot explain hybrid-B: hybrid-B cut the resolution floor 88%
and restored 573/765 rows, so a resolution-driven high-wpm story predicts hybrid-B improves; it did
not (Δwmae moved 0.03 ms, same 4/4). I therefore treat the *resolution* form of H-EXTRAP as already
dead per my brief, and test only the form NOT killed by that: whether the Δρ deficit **grows
monotonically with bucket wpm** (a gradient claim about ρ, not about the wmae floor).
* SUPPORTED iff mean Δρ vs CUR is monotone decreasing in bucket wpm with the high buckets
  significantly worse than the low ones, for BOTH interp frames.
* REFUTED iff the Δρ-vs-bucket profile is flat or non-monotone, or if the refused buckets are not
  the highest ones (e.g. b80/b100 refused while b120 passes).

**H-MONO.** The global monotone constraints make a genuinely non-monotone high-wpm regime
unrepresentable. Test: an UNCONSTRAINED variant of the SAME columns.
* SUPPORTED iff INTERP-NOMONO (same 10 columns, `monotone=False`) PASSES the gate where INTERP fails.
* REFUTED iff INTERP-NOMONO fails the same buckets on 4/4.
* Note `INTERP-NOMONO` **already exists** in `agent-artifacts/interpframe/lolo.json` (4 folds × 3
  seeds, matched) — its `bucket_rhos` are on disk, so this is scoreable without retraining. I
  register that I will score it as-is AND, if its verdict is decisive, verify the arm's identity
  (column count / monotone flag) rather than trusting the label.

**H-PACE.** A high-wpm-specific artifact of the LOGRAT→ms **conversion arithmetic** (not the wpm
column's information). The interp frames route through `to_ms(pred, X, wpm=<per-cell array>)`; the
served frame recovers pace from its own `X` column.
* ⚠ REGISTERED PREDICTION THAT WOULD KILL IT: the gate's statistic is a **within-bucket Spearman ρ**.
  `ms = exp(pred) * 12000/wpm` is a **strictly positive monotone rescaling per row**; within one wpm
  bucket the rows do NOT share one wpm (cells carry per-cell wpm), so the rescaling is NOT rank-
  preserving within a bucket and H-PACE is *not* excludable a priori. It IS excludable if the two
  paths are numerically identical.
* SUPPORTED iff feeding the served frame's pace through the *explicit-argument* path (or the interp
  frame's through a column) changes the refused-bucket set.
* REFUTED iff the two paths produce identical ms for the same model+cells (max |Δ| ≈ 0 at float
  tolerance), which would make the conversion arithmetic a non-difference.

**H-REAL (frame defect, the null I must be able to accept).** The interp frames genuinely regress
high-wpm ρ by more than the gate's measured floor, on well-supported buckets, under a symmetric
control.
* SUPPORTED iff refusals survive the symmetric LOSO control on buckets whose support is thick, with
  Δρ large vs the measured floor.
* REFUTED iff the refusals vanish or become noise-indistinguishable under the symmetric control.

**Not mutually exclusive.** I register in advance that "both" is a permitted verdict and state the
partition: how much of the 4/4 is (i) removed by fixing the baseline asymmetry, (ii) attributable to
thin support, (iii) a residual real ρ deficit.

---

## §2 MANDATORY CONTROLS — these run FIRST, before any frame-side story

**GC1 (as published):** re-derive the published gate verdicts for CUR / INTERP / HYBRIDB from the
on-disk `bucket_rhos`, reproducing HYBRIDTRI-1's exact structural sets. A reproduction failure means
my reader is wrong and everything after is void. **Must reproduce**: CUR PASS (noise-only azerty
[100,120], dvorak[100], qwertz[120]); HYBRIDB structural azerty[120] dvorak[80,100] qwerty[120]
qwertz[100]; INTERP structural azerty[120] dvorak[100,120] qwerty[120] qwertz[100].

**GC2 (the control the published runs did NOT do — SYMMETRIC LOSO):** for each fold and each seed
i of CUR, compare CUR seed i against the mean of CUR's OTHER two seeds; and compare each candidate
seed against a same-cardinality (2-seed) CUR mean, averaged over which seed is held out. This makes
the incumbent and the candidate face the SAME estimator. Report whether CUR then fails structurally.

**GC3 (SEEDNOISE analogue = this gate's OWN measured floor):** the null spread of the gate's
statistic under nothing-but-reseeding. Two routes, both registered:
* (a) **from existing data, no training**: CUR's per-(fold,bucket) seed-to-seed spread, and the
  distribution of `|seed_i − mean(other seeds)|` per bucket — the exact quantity GC2 thresholds.
* (b) **an additional CUR arm at NEW seeds [3,4,5]** if (a) is judged too thin (3 seeds). Register
  now: I will run (b) only if (a) leaves the verdict ambiguous, and I will say which.
* The floor must be reported **per bucket**, matched to that bucket's support (INVARIANT 8: a floor
  must match the comparison's DATA VOLUME).

**GC4 (support recovery, per C0.2):** recover `n_cells` / `n_participants` per (fold,bucket) from
`bucket_matrix` and state explicitly that the published gate blocks carried `support: None`.

**INVARIANT-3 compliance:** the gate's statistic is ρ(pred, obs) where `obs` is the **measured
duration** from `bistrokes31_v1.tsv` — NOT a served-frame-generated target. So the gate comparison is
free of the `_T2` tautology. I state this explicitly and will flag any quantity I compute that is
self-generated.

**INVARIANT-9 compliance:** I do not reuse `agent-artifacts/interpframe/metrics.py`. If I load any
sibling instrument I load it BY PATH and assert what it dispatched on.

---

## §3 DECISION RULE — how the one-sentence verdict is chosen

Evaluated in this order; the first that fires is the headline:

1. If **GC1 fails to reproduce** → report that and stop; no diagnosis is possible.
2. Else if **GC2 makes CUR fail structurally on ≥1 fold** → the published gate control is an
   **artifact of an asymmetric baseline**; verdict = GATE ARTIFACT (at least in part), and I then
   report the residual: do the candidates still fail under the symmetric control?
3. Else if candidates' refused-bucket |Δρ| ≤ the GC3 measured floor at matching support → GATE
   ARTIFACT (support/noise).
4. Else if refusals survive on buckets with thick support and Δρ ≫ floor → FRAME DEFECT (H-REAL),
   and I report which mechanism among H-EXTRAP / H-MONO / H-PACE survives.
5. "BOTH" is reported with the partition quantified, not as a hedge.

**Margin-vs-floor is reported BEFORE any p-value** (INVARIANT 8). Bootstrap stability of the verdict
(resampling which seeds fill the baseline) is reported beside the registered result, never instead.

---

## §4 WHAT I WILL NOT DO

* Not weaken, re-scope or re-threshold the gate in `src/`. If the finding is that a support floor or
  a symmetric baseline is needed, that is a RECOMMENDATION for the human (GATESUPPORT-1 left the
  threshold to them because it retroactively re-adjudicates past verdicts).
* Not adopt or promote any frame. `FEATURE_VERSION` untouched, `data/models/k31/` never written,
  `layouts.py` untouched.
* Not hunt for a variant that passes. The only new arms permitted are the ones registered above
  (a CUR reseed for the floor, if needed) and H-MONO's already-trained NOMONO scoring.
* Not read `state/gatefolds/*` until this file is committed.

## §5 MUTATION TESTING

Any new assertion I add gets mutation-tested with `pytest -B` and an explicit `__pycache__` purge
BEFORE and AFTER each mutation (FM4-1's false-survivor hazard). Every mutation asserted to have
changed the file; `rc` taken straight from pytest, never from a pipe tail. An assertion whose subject
cannot vary is vacuous (INVARIANT 5) — if a field is only ever checked at its default, I force the
other value somewhere.
