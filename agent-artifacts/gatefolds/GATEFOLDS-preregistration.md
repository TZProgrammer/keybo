# GATEFOLDS-1 — preregistration

**Registered BEFORE any number of mine existed.** Base: `hybridtri` @ `52f0e3f`, branch
`gatefolds`, worktree `/local/home/zegertho/repos/keybo-wt-gatefolds`.

## §0 — The assigned prior, and what would refute it

**ASSIGNED PRIOR (mine to defend or break):** the high-wpm non-regression gate is RIGHT, and
interp.1 / hybrid-B really do regress high-wpm prediction. A sibling (`gatewhy`) is assigned the
opposite prior (gate artifact). **I have not read its report and will not before this file is
committed.**

**The prior is REFUTED (not merely unsupported) if either:**

* **R1 — the gate refuses the SERVED frame under conditions matched to the candidates.** If the
  incumbent fails the same gate whenever it is deprived of the same thing the candidates lack, the
  refusal is not evidence about interpretability.
* **R2 — the mechanism turns out to be a HARNESS/CONVERSION defect rather than a modelling loss**
  (the brief's own warning: the LOGRAT→ms conversion reads pace from the `wpm` column, and
  interp.1 has none).

I register **now** that I regard R1/R2 as live, because §1's H1 is a *structural* claim I verified
in code before writing this file, and it PREDICTS R1. So the honest registration is: **my assigned
prior is likely to lose, and the shape of the loss is pre-stated below** — that is a stronger
registration than pretending I had no read.

## §1 — Hypotheses

### H1 (PRIMARY, structural — MECHANISM) 🟢 already code-verified, stated here before it is *measured on the gate*
**Both failing frames are WPM-INVARIANT BY CONSTRUCTION: for a fixed position-pair, the feature
vector is bit-identical across every wpm bucket.** Verified on this tree
(`replacement_frame(True)` and `("hybridb")` produce `np.array_equal(v@50wpm, v@130wpm) == True`;
served and `interp="wpm"` produce `False`).

**Consequence, and it is the mechanism:** the model's raw LOGRAT output for a cell **cannot vary
with the bucket**. The only bucket-dependence left in the ms prediction is the *deterministic*
`to_ms` factor applied from `Cell.wpm`. Since the gate's per-bucket rho is a **within-bucket
Spearman**, a monotone deterministic rescale is rank-preserving *within* a bucket — so the
per-bucket rho of an invariant frame is the rho of ONE fixed ranking of position-pairs, replayed
against each bucket's own observed ordering. **A wpm-invariant frame therefore cannot track any
re-ordering of pairs across the wpm range, and the high-wpm buckets are exactly where the observed
ordering differs most from the pooled one.**

* **H1 predicts** the failure is *not* about interpretability, ordinals, or resolution — it is
  about **the absent `wpm` column**, i.e. the frame's inability to *re-rank* by pace.
* **H1 predicts R1 (my own prior's refutation):** the SERVED frame, made wpm-invariant, must fail
  the SAME way. This is the CONTROL in §2.
* **CONSISTENCY with the two dead hypotheses** (required by the brief):
  - Dead-1 ("it is the dropped `wpm` column", refuted because an 11-column interp-wpm variant made
    Δwmae **worse** and the gate refused the SAME buckets). **H1 is consistent and is NOT Dead-1.**
    Dead-1 is a claim about *wmae magnitude*, and about *adding a column back*. H1 is a claim about
    **rank re-ordering within high buckets**, and it makes a DIFFERENT, falsifiable prediction:
    interp-wpm should be **structurally different from interp.1 on the FAILING BUCKET SET even
    though it still fails** — because restoring the column removes the invariance but does not
    restore the served frame's *resolution* to express the re-ordering. §3 checks this by
    diffing the two frames' refused-bucket sets rather than their pass/fail bits. If interp-wpm's
    refused set is IDENTICAL to interp.1's *and* its per-bucket rho deltas are equal to within
    seed noise, **H1 is weakened** and I will say so.
  - Dead-2 ("feature-resolution floor / null-space collapse", refuted because hybrid-B cut the
    collapse floor 88% and the gate stayed structural 4/4). **H1 is consistent and EXPLAINS
    Dead-2's puzzle:** hybrid-B added 8 one-hots, all of which are wpm-invariant, so it bought
    *resolution* while buying **zero** pace re-ranking capacity. H1 predicts exactly what was
    observed: floor cut 88%, gate unmoved. **This is H1's strongest support and it is
    retrodictive, so it does not count as confirmation — §2/§3 are the prospective tests.**

### H2 (the brief's highest-value measurement — MONOTONE CONSTRAINTS)
**An UNCONSTRAINED variant of the SAME columns decides whether the monotone constraints cause the
refusal.** Passes gate ⇒ constraints are the cause. Fails 4/4 ⇒ constraints exonerated, cause is
the column BASIS (or, per H1, the invariance).

⚠ **REGISTERED BEFORE MEASURING, and it is a correction to my brief:** `INTERP-NOMONO` **already
exists** in `agent-artifacts/interpframe/lolo.json` (arm 3 of 4, `interp=True, monotone=False`).
So H2 is **already answered on disk** and I must not re-run it and present it as new. My job is to
(a) *report* it, (b) verify the arm is not confounded, and (c) note the direction. I register the
prediction: **H2 will EXONERATE the constraints**, and I predict the unconstrained arm is **no
better and plausibly WORSE**. (If it is worse, that is itself informative and consistent with H1:
removing constraints adds variance without adding pace capacity.)

### H3 (POPULATION — is high-wpm a different population?)
The high buckets may differ in participant mix / support. **Registered as DESCRIPTIVE, not
causal** — the brief already notes coarseness is not the story (Dead-2). I will report per-bucket
`n_cells` / `n_participants` / `n_raw` for every refused (fold, bucket) and the served control.
**H3 is NOT a mechanism claim** and I will not upgrade it to one.

### H4 (ORDINAL vs ONE-HOT basis) — registered and DEPRIORITIZED, with the reason
A basis claim distinct from resolution. **I register that H1 SUBSUMES the part of H4 that matters
and that hybrid-B already provides the discriminating evidence:** hybrid-B carries the served
one-hots BESIDE the ordinals (so the one-hot basis is present) and still fails 4/4. That is
evidence AGAINST "the ordinal spacing assumption is the cause". **I will report this rather than
run a new arm**, because a new arm cannot separate basis from invariance while both frames are
wpm-invariant. Naming it as not-run, with the reason, per invariant (g).

## §2 — THE DECIDING CONTROL (this is the experiment that adjudicates my prior)

**ARM `CUR-INVARIANT`: the SERVED 20-column frame, made WPM-INVARIANT, nothing else changed.**

Built by **pinning the `wpm` column to each cell's own bucket-independent constant** at BOTH train
and predict, so the frame retains all 20 served columns, all its resolution, all its one-hots, and
**no** monotone constraints — differing from the incumbent in EXACTLY the property H1 names.

⚠ **This is NOT the existing `CUR-NOWPM` arm and the difference is the whole point.**
`interpframe/lolo.json`'s `CUR-NOWPM` neutralized `wpm` to a **global constant** — which is the
correct ablation for "can a split use the column", and it **already fails the gate structurally on
4/4 folds** (azerty[120], dvorak[80,100], qwerty[100,120], qwertz[100]). I register that I
consider **`CUR-NOWPM` ALREADY SUFFICIENT to refute my assigned prior**, and §2's arm is a
*confirmatory* re-derivation on my own tree at my own hand.

**DECISION RULE (registered):**
* If `CUR-INVARIANT` **fails structurally on ≥3/4 folds** ⇒ **R1 fires, my ASSIGNED PRIOR IS
  REFUTED.** The gate is not detecting an interpretability cost; it is detecting **loss of pace
  adaptation**, which the served frame loses identically. I report that as the headline.
* If `CUR-INVARIANT` **PASSES** ⇒ my prior SURVIVES and H1 is wrong: something about the interp
  columns, not the invariance, is responsible. I then report H1 as refuted by my own control.
* Anything in between (1–2/4) ⇒ report as INDETERMINATE, no headline claim.

**GATE CONTROL (invariant 2), separately:** does the gate refuse the served/incumbent frame under
its own baseline? Registered prediction: **NO** (`highwpm.json` records
`gate_control_incumbent_passes: true`). I verify rather than borrow.

## §3 — Metrics, floors, and what I will NOT compute

* **NAME THE ROWS (invariant 1):** the actual refused (fold, bucket) list with per-bucket
  `n_cells` / `n_participants` / `n_raw` and per-bucket rho deltas, for **interp.1, hybrid-B, the
  SERVED control, `CUR-NOWPM`, `INTERP-NOMONO`, and my `CUR-INVARIANT`**.
* **FLOOR, measured not borrowed (invariant 7):** the comparison's own **seed-reseed floor** — the
  spread of per-bucket rho across the 3 training seeds *within* an arm, at the same data volume as
  the cross-arm delta. A cross-arm rho delta smaller than the within-arm seed spread is not
  resolvable. Reported as **margin-vs-floor BEFORE any p-value**.
* **BOOTSTRAP the VERDICT (invariant 7):** resample **seeds** (the unit the gate's
  structural/noise rule consumes) and report how often the structural verdict is reproduced per
  (arm, fold). A 3-seed structural call has a bounded stability and I will state it.
* **NO SELF-GENERATED TARGETS (invariant 3):** every rho here is against the **observed** held-out
  cell durations (`Cell.obs`, IQR-mean of real samples) built by `build_cells` from the stroke TSV.
  I compute **no** `TimeSurface`/`TableBigramScorer`-derived target and no frame-generated floor.
* **NO IDENTITY METRICS (invariant 6):** H1's test is `np.array_equal` on featurizer output vs the
  gate's independently-computed per-bucket rho. The invariance check cannot be an algebraic
  function of the rho outcome — they come from different code paths (`features/ngram.py` vs
  `training/validate.py`) and I state that explicitly.
* **NAME-SUBSTRING DISPATCH (invariant 8):** I do **not** reuse
  `agent-artifacts/interpframe/metrics.py`. If I load any sibling instrument it is BY PATH via
  `_boot.load_by_path` with an asserted frame. Levels are not ported across frames; only
  same-grouping deltas.
* **VACUITY (invariant 5):** any new assertion I add must be exercised at a NON-default value.
* **MUTATION TESTING (invariant 4):** new assertions get mutated with `__pycache__` purged BEFORE
  and AFTER each mutation, `pytest -B`, and the mutation asserted to have changed the file.

## §4 — What I will not do

* Not weaken/rethreshold the gate. A support floor is a RECOMMENDATION for the human
  (GATESUPPORT-1 precedent).
* Not adopt, promote, or land anything; `data/models/k31/` read-only; `layouts.py` untouched;
  `FEATURE_VERSION` not changed in place.
* Not push code. Ledger-only pushes.
* Not read `gatewhy`'s report before this file is committed.

## §5 — Outcome

*(filled in after measurement — nothing here yet by construction)*
