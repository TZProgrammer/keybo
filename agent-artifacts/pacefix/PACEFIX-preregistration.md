# PACEFIX-1 — preregistration

**Committed BEFORE any number of mine exists.** Branch `pacefix`, worktree
`/local/home/zegertho/repos/keybo-wt-pacefix`, base `gatefolds` @ `986f3a6`.

**THE QUESTION (parked with the human):** is a **pace-tracking interpretability frame ACHIEVABLE**,
and is the wpm / `CONSTFRAC` trade **STRUCTURAL** or **FIXABLE**? GATEFOLDS-1 established the causal
variable (the pace channel); DEAD-1 measured that simply restoring a `wpm` column leaves the
within-bucket ranking IDENTICAL 5/5 at ρ **exactly 1.000000**. My arm must EXPLAIN that 1.000000
before trying to break it, then intervene one variable at a time.

---

## 0. CORRECTIONS TO MY BRIEF — registered BEFORE measuring, from code reading alone

These are read-from-source (🟡 HIGH) facts that change what my arm can conclude. Registering them
now so the causal order (correction → measurement) is verifiable in git.

**🔴 C1 — MY BRIEF SAYS "at max_depth 3 with 11 columns, a wpm-×-geometry path is ALREADY REACHABLE
IN PRINCIPLE". THAT IS TRUE OF DEPTH BUT FALSE OF THE CONSTRAINT SET, AND THE CONSTRAINT IS THE
PART THAT MATTERS.** My brief's candidate (c) asks whether "a MONOTONE CONSTRAINT on some geometric
column, or on wpm itself" forbids the interaction — and offers it as one of four unranked guesses.
Read from source it is not a guess: `src/keybo/features/schema.py:365`

    BIGRAM_INTERP_WPM_MONOTONE = (*BIGRAM_INTERP_MONOTONE, -1)

so in the interp-wpm frame **EVERY ONE OF THE 11 COLUMNS IS MONOTONE-CONSTRAINED, `wpm` INCLUDED
(at −1)**. Verified on this tree: `replacement_frame("wpm")` returns
`mono=(1,1,1,1,1,1,1,1,1,-1,-1)` with `wpm` at index 10 → constraint −1. XGBoost's
`monotone_constraints` is enforced along EVERY root-to-leaf path, so a fully-constrained model is
constrained *jointly*, not per-column-in-isolation. **⇒ the interaction interp-wpm needed was not
merely un-encouraged, it was FORBIDDEN by the parameter the frame ships.** This makes candidate (c)
the leading structural hypothesis a priori, not the third of four.

**🔴 C2 — THE SERVED FRAME IS **NOT** MONOTONE-CONSTRAINED, so GATEFOLDS-1 §(a)/§(b)'s repeated
"the SERVED frame, … its OWN monotone constraints" (and my brief's inheritance of that phrase) is
WRONG.** `monotone_constraints` is set in EXACTLY ONE place — `training/train.py:436`, inside
`if interp:` (`grep -n monotone_constraints src/` = train.py:436 + the artifact record at :522).
There is no `BIGRAM_MONOTONE` tuple in `schema.py` at all. **⇒ the served frame trains
UNCONSTRAINED.** This matters directly: the served frame is the only pace-TRACKING frame, and it is
also the only frame with no constraints — so "constrained" and "cannot track pace" are **CONFOUNDED
across the served/interp comparison**, and C1's mechanism is exactly the confound. `CUR-INVARIANT`
does not break this confound (it pins wpm on an already-unconstrained frame).

**🔴 C3 — MY BRIEF'S "there are NO `interaction_constraints` anywhere to relax" IS CORRECT AND I
CONFIRM IT** (`grep -rn interaction_constraints src/ agent-artifacts/` = empty). Registering the
confirmation because my brief asked me to flag wrong premises, and this one is right.

**🟡 C4 — candidate (a) ("the trees never SPLIT on wpm") IS ALREADY NEAR-REFUTED BY DATA ON DISK.**
`agent-artifacts/gatefolds/invariance.json` records interp-wpm's raw LOGRAT spread across buckets as
**7.777e-02**, i.e. NONZERO — a model whose output moves with wpm must have split on wpm. So (a) is
unlikely to be the explanation, and (b) ("splits exist but act as a per-bucket SHIFT") is the
already-favoured reading. I will still measure gain attribution, because "nonzero spread" does not
by itself prove the split count is material, and because (a)-vs-(b) is cheap to separate.

---

## 1. HYPOTHESES — registered with the discriminator that decides each

**H-MONO-BLOCK (my PRIMARY, from C1).** The exact-1.000000 rank identity of interp-wpm is caused by
the **monotone constraint set**, not by depth and not by the absence of a wpm column. Mechanism: an
all-columns-constrained additive-in-sign model cannot express the sign pattern a pace-×-geometry
re-ordering requires, so `wpm` can only enter as a monotone per-bucket shift — and a monotone shift
is invisible to a within-bucket Spearman.
*Discriminator:* interp-wpm with `monotone=False` (ONE variable) must break rank identity
(ρ < 1.000000 in ≥1 bucket pair). **If it does NOT, H-MONO-BLOCK is REFUTED** and I will say so.

**H-DEPTH.** The identity is a depth/capacity limit; deeper trees create wpm-beneath-geometry paths.
*Discriminator:* interp-wpm at `max_depth=6` (ONE variable, constraints left ON) breaks rank
identity. **Registered prediction: it will NOT** (depth cannot buy a sign pattern the constraint
forbids). This is the arm my brief warned me not to assume, registered as a prediction I expect to
lose.

**H-TARGET (my brief's (d) — the STRUCTURAL answer).** The LOGRAT target `log(ms·wpm/12000)` has
already absorbed the pace structure, so there is genuinely little within-bucket pace-dependent
re-ordering left to learn, and NO intervention recovers it.
*Discriminator:* if BOTH interventions above leave ρ = 1.000000 on 5/5, H-TARGET is the surviving
explanation ⇒ the trade is **STRUCTURAL**. Conversely, if any intervention breaks rank identity,
H-TARGET is **REFUTED as a complete explanation**, and I must then report how much re-ordering was
recoverable (the amount is the answer to "structural or fixable", not the mere sign).

**H-SPLIT (my brief's (a)).** wpm gain ≈ 0; the column is present-but-unused.
*Discriminator:* booster split/gain attribution for `wpm` in the interp-wpm model. Near-refuted by
C4 already; measured for completeness.

**H-ROOT (my brief's (b)).** wpm splits exist but sit at the root / in their own subtree, giving a
per-bucket shift rather than a pair-dependent term.
*Discriminator:* per-tree depth-of-first-wpm-split distribution, and the count of trees where wpm
appears BELOW a geometric split on the same path (the literal definition of an interaction path).

---

## 2. THE MEASUREMENTS, in order, with bars fixed NOW

**M-A (diagnosis, model-free-ish, no LOLO).** Booster structure of interp-wpm at the production
recipe: total gain per feature, wpm's share, wpm split count, depth-of-first-wpm-split, and
**#trees where wpm co-occurs below a geometric column on one root-to-leaf path**. Bar: none — this
is descriptive, and it is the DIAGNOSIS my brief requires before intervening.

**M-B (rank identity, the same instrument for every arm).** Re-run GATEFOLDS-1's PREDICTION
invariance (path: `train_bigram_model` + `models/base.to_ms`, all 875 in-data position pairs × the 5
bucket midpoints): raw LOGRAT spread, within-bucket rank-identity count /5, and ρ(b40,b120).
Reference values I must reproduce for the UNCHANGED arms (a positive control on my own instrument):
served **0.793006** with 1/5 rank-identical; interp-wpm **1.000000** with 5/5, raw spread 7.777e-02.
**Bar for "rank identity BREAKS": ρ(b40,b120) < 1.000000 AND rank-identical buckets < 5/5.**

**M-C (accuracy, paired per-fold — MOR-FIX-1).** `validate()` LOLO, 4 folds × seeds [0,1,2],
`n_boot=10`, `ROW_STAGGERED_31`, against the SERVED baseline. Report ρ, ρ/ceiling, wmae, τ_heldout,
**paired per-fold Δ**. Bars registered now: an intervention is **ACCURACY-NEUTRAL** iff paired
per-fold Δwmae ≤ +1.0 ms vs SERVED (interp.1's cost was +5.77 ms = +58%; anything near that is NOT
neutral) **and** τ_heldout stays [1,1,1].

**M-D (interpretability trade).** MAXCORR + CONSTFRAC (and MEANCORR/MONOFRAC where available) on the
corpus-frequency-weighted serve grid. **`agent-artifacts/interpframe/metrics.py` DISPATCHES ON A
NAME SUBSTRING** (`"hand_conflict" in names` → INTERP grouping, else SERVED; line 55-64 verified on
this tree) so I MUST load it BY PATH and **ASSERT the grouping it returned is the one my frame
warrants**. Reference: served MAXCORR 0.9813 / CONSTFRAC 0.0579; interp.1 0.7037 / 0.0000;
interp-wpm CONSTFRAC 0.0010. Bar: a "fix" must keep MAXCORR ≤ 0.7850 (INTERPFRAME-1's registered
bar) — if MAXCORR returns toward 0.9813 it has re-imported the collinearity interp.1 existed to
remove and **is not a fix**.

**M-E (the gate, at BOTH thresholds, with a control that CAN fail).** Per-fold incumbent baseline
(the shipped construction), `bucket_regression_report`, reported at the shipped tolerance **0.005**
AND at the measured reseed floor **p95 = 0.0108** (`gatefolds/reseed.json`, which independently
reproduces gatewhy's 0.0117). **azerty b120 is treated as reseed-refusable** and excluded from every
count I rely on (SEEDNOISE/CUR-RESEED refuse it 3/3 on the served frame itself).
**I WILL NOT USE THE SHIPPED GATE CONTROL — it is a TAUTOLOGY** (the baseline is CUR's mean over the
same seeds being scored, so the deltas sum to ~0 and can never all be negative; measured
3.331e-16 / 0-of-20 by gatefolds, 200k adversarial trials by gatewhy). **My control is a SAME-FRAME
RESEED**: the already-measured `CUR-RESEED` (served, seeds [3,4,5]) — a control that DID fail, on
azerty b120. If an intervention's refusals are a subset of that control's, it has shown nothing.

---

## 3. INVARIANTS I BIND MYSELF TO

1. **No self-generated targets.** Every accuracy/rank number comes from `validate()`'s observed
   cell durations (real data), never from a `TableBigramScorer`/`_T2` surface built by the frame
   under test. Any spread I quote from a frame's own predictions is labelled a MODEL property, not a
   floor.
2. **One variable per arm**, named explicitly in the artifact, with the resolved param dict recorded
   (xgboost does NOT serialize `monotone_constraints`, so "present" must be recorded, and
   "effective" must be measured separately — present ≠ effective).
3. **Instrument dispatch asserted**, per M-D. A grouping-dependent LEVEL does not port across
   frames; only same-grouping DELTAS compare.
4. **`python -B -m pytest`** is the working form (`pytest -B` is invalid — `-B` is a python flag).
   Mutation-test any new assertion, purging `__pycache__` BEFORE and AFTER each mutation.
5. **No metric that is an algebraic function of its own outcome**, and no assertion whose subject
   cannot vary.
6. **Tree pinned:** `PYTHONPATH=<worktree>/src`, print `keybo.__file__` + branch in every driver,
   `require()` every symbol. 4 thread vars pinned before importing xgboost. Scratch in
   `/tmp/pacefix_wk` (a subdir, never bare `/tmp`). Long runs detached, polled via a SENTINEL FILE.
7. **Not doing:** no `FEATURE_VERSION` change in place, no new frame adopted or promoted, no gate
   weakened or re-thresholded, `data/models/k31/` and `layouts.py` untouched, no CODE pushed, no
   branch merged/deleted. If my finding implies the tolerance is wrong, that is a RECOMMENDATION.

## 4. WHAT WOULD MAKE ME REPORT "STRUCTURAL"

If `monotone=False` and `max_depth=6` BOTH leave ρ(b40,b120) = 1.000000 at 5/5 rank-identical
buckets, then no reachable intervention on this target recovers within-bucket pace re-ordering, and
I will report the trade as **STRUCTURAL**, naming H-TARGET as the surviving explanation. If an
intervention DOES break rank identity, the honest answer is two-sided and I must price it: the trade
is **FIXABLE only if** the break comes with M-C accuracy-neutrality AND M-D's MAXCORR bar intact.
**A break bought at interp.1's +58% wmae, or at MAXCORR back near 0.9813, is not a fix and I will
say so.**

---

## 5. POST-HOC ADDENDUM — a defect in MY OWN registered bar (added after measuring; labelled as such)

**C5.** §M-B registered: "rank identity BREAKS iff ρ(b40,b120) < 1.000000 AND rank-identical < 5/5".
`interp-wpm-depth6` measured **ρ = 0.9999998566886138** @ 1/5 — which satisfies that bar *literally*
while being a **1.4e-07** break, ~77,000× BELOW this gate's own measured p95 reseed floor (0.0108).
**The bar was stated at 6 decimals and therefore cannot distinguish a real re-ordering from float
noise.** The `n_rank_identical` count is likewise brittle: a 1.4e-07 perturbation flips it 5/5 → 1/5
while changing nothing material. Both readings are reported in the report; the substantive verdict
(depth does NOT restore pace tracking, exactly as registered under H-DEPTH) rests on ρ magnitude
measured against the floor. **Recommended pattern for future arms: report Δρ ÷ measured floor, never
the identity count alone.** Registered here rather than silently re-scored.
