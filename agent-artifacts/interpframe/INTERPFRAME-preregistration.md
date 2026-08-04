# INTERPFRAME-1 — PREREGISTRATION

**Registered 2026-08-04, BEFORE any interpretability number exists.** The only numbers that
exist at registration time are the mandated NEGATIVE CONTROL (`negctl.json`), which
*reproduces shipped quantities* and creates no new claim, and the baseline-frame facts already
in the ledger (SHAPDIFF-1 / SHAPDIFF-TCOND).

---

## §0 — THE OBJECTIVE IS INVERTED, AND THAT IS REGISTERED

The user's words: *"we should craft a model which has BETTER FEATURES (NOT NECESSARILY which
will lead to a BETTER MODEL, but a model MORE SUITED TO BE INTERPRETED IN THIS LENS)."*

So the maximand is **INTERPRETABILITY UNDER THE SHAP-DIFF LENS**, not accuracy. A frame that
is slightly WORSE on held-out transfer but whose per-feature attributions are trustworthy and
mechanistic is the **WIN CONDITION**. I register that:

* I pre-register **NO accuracy bar**. A negative paired Δρ is an *acceptable and expected*
  outcome (CLOSING-2: nine feature-frame arms already returned NULL on accuracy — adding or
  changing columns does not help this model).
* I will NOT sell interpretability gains as accuracy gains.
* This is a **DESIGN + PROOF-OF-CONCEPT**. Nothing is adopted; production is not retrained;
  `FEATURE_VERSION` is not mutated; `data/models/k31/` is read-only. Adoption is a one-way-door
  architecture decision reserved for the human.

## §1 — WHAT IS ESTABLISHED FROM CODE BEFORE MEASURING (and one CORRECTION to the brief)

🟢 **CORRECTION 1 — `keybo shap-diff` is NOT on `main`.** `src/keybo/analysis/shap_diff.py`
exists only on the UNPUSHED branches `tcond` (14a614c — the newest, with `--channel` and the
BLOCK table) and `shapdiff` (aa47691 — T2 only). `git merge-base --is-ancestor tcond
origin/main` = **NO**; `origin/main` = 7759d2f carries the ledger entry but not the code. My
brief called the tool "productized"; it is written, tested (46 tests) and unmerged. This
worktree is therefore branched from **`tcond`**, not `main`.

🟢 **CORRECTION 2 — `bg1_*` and `bg2_*` are the SAME PREDICATE on DIFFERENT KEYS, not one
property split in two.** `_trigram_row_from_positions` builds `bg1_ = placement(a,b)` and
`bg2_ = placement(b,c)`, and the placement block describes the **second** key of its pair. So
`bg1_bottom` = "key **b** is bottom-row" and `bg2_bottom` = "key **c** is bottom-row". My brief
called their opposite signs (−0.2337 / +0.7382) "the same physical property split across two
blocks"; strictly they are one *predicate* applied to two *different keys*, so opposite signs
are not automatically an artifact. The **real** structural defect they expose is worse and is
what I register: in the trigram frame, key **a**'s absolute row/finger placement appears in NO
column at all (a → nothing, b → `bg1_*`, c → `bg2_*`), and in the bigram frame the FIRST key's
absolute placement is likewise invisible. The frame is asymmetric in the keys it can describe.

🟢 **VERIFIED FROM CODE (read, this tree):**
* `FEATURE_VERSION = "2026-07-05.3"` (`schema.py:29`); `FEATURE_VERSION_DIRECTION` and
  `FEATURE_VERSION_KITCHENSINK` are separate stamps derived from it; `models/base.py`
  hard-errors on a mismatch. A new frame gets a **NEW** constant. ✅ the mechanism the brief
  describes is real and is what makes this safe.
* `shap_diff.block_map()` (`shap_diff.py:160-176`) **REFUSES** an unregistered frame with a
  `ValueError` rather than bucketing the remainder — so registering a `_*_BLOCKS` dict IS the
  integration point, as the brief says. `_shap_tables()` additionally asserts the models'
  `feature_names` equal the SCHEMA list for their order (`shap_diff.py:757`), so BOTH must be
  taught.
* All six k31 artifacts are LOGRAT with `calibration: None`, so the LMDI identity transfers.
* `wpm` is passed as ONE scalar to the serve grid ⇒ it is a **constant column** at attribution
  time, and it still carries −0.0922 ms/char (**9.2%** of the T2 gap). Reproduced in `negctl`.

🟢 **PRIOR ART THAT CONSTRAINS ME — ADJ-2 PINKY-MONO already tried monotone constraints on this
repo and FAILED** (ledger ~L3275): *"monotone-constrained indicator columns learn ZERO
magnitude (served gap +0.0ms on all 8 probe pairs) … the within-layout collinearity with the
practice term still starves the columns of attributable variance"*, and LOLO failed too
(qwertz +1.57%, dvorak +1.64%). That arm constrained **binary indicator** columns for a
first-finger seam; mine constrains **graded geometry cost** columns, so it is a WARNING and not
a refutation — but the "flattens to zero" failure mode is a REGISTERED prior and §5(c) below
tests for it explicitly.

## §2 — THE FIVE FAILURE MODES, AS I WILL SCORE THEM

FM1 coupled-column credit splitting · FM2 one mechanism shattered across columns that then
fight · FM3 non-mechanistic features (wrong-signed physical story) · FM4 name collisions ·
FM5 aggregation that hides sign flips.

## §3 — THE INTERPRETABILITY METRICS AND THEIR BARS (registered BEFORE measuring)

All computed on the **corpus-frequency-weighted serve grid** — the population the attribution
is actually a weighted sum over. An unweighted 31×31 grid over-represents ~zero-mass cells.

| id | metric | direction | BAR |
|----|--------|-----------|-----|
| **M1** | `MAXCORR` — max off-diagonal \|Pearson r\| between feature columns (M1b = MEAN \|r\|) | lower | **M1_new ≤ M1_cur / 1.25** |
| **M2** | `CONSTFRAC` — share of total \|attribution\| on columns that are CONSTANT over the serve grid | lower | **M2_new == 0 exactly** |
| **M3** | `SPLITPAIRS` — # of same-property column pairs carrying OPPOSITE-signed attributions | lower | **strictly < M3_cur** |
| **M4** | `MONOFRAC` — share of \|attribution\| on columns that are monotone-constrained AND whose constraint is VERIFIED honored | higher | **M4_new ≥ 0.90** (M4_cur = 0) |
| **M5** | `SIGNSTAB` — share of columns whose attribution SIGN agrees across the two production corpora (default/C30M vs iweb); ALSO report column Spearman ρ | higher | **M5_new ≥ M5_cur AND ρ_new ≥ 0.8737** (SHAPDIFF-1's registered ρ) |
| **M6** | `SEEDSTAB` — share of columns with unanimous sign across 3 model seeds; mean pairwise ρ | higher | **M6_new ≥ M6_cur** |

**M1 rationale for the 1.25 factor:** these correlations are *structural* (deterministic
functions of geometry under a fixed corpus weight), so they carry no sampling noise and any
strict decrease is real. 1.25 demands a MATERIAL decrease rather than a hairline one.

**M3 same-property grouping, fixed now so it cannot be chosen after the fact.** Current bigram
frame: `{bottom,home,top}` (ROW one-hot), `{pinky,ring,middle,index,lateral}` (FINGER block),
`{dx,dy,distance}` (mutually functionally dependent travel), `{inwards,outwards}` (the
swap-invariant pair). Current trigram frame additionally: every `bg1_X`/`bg2_X` mirror pair.
New frame: any pair of columns derived from the same underlying per-key quantity.

### PRIMARY DECISION RULE
The frame is **MORE INTERPRETABLE** iff it wins **M1, M2, M3 and M4** *and does not lose* M5.
**PARTIAL** if it wins 3 of those 4 with M5 not lost. **NO** otherwise. M6 is reported either
way.

## §4 — THE PROPOSED FRAME `interp.1` (10 columns, bigram), EACH JUSTIFIED

Positions `a → b`; the thumb/space key (`hand(x)==0`) contributes **0** to every per-key term
(it is not a letter reach and is not pressed by a finger with a home column).

| # | name | definition | mono | fixes |
|---|------|------------|------|-------|
| 1 | `hand_conflict` | 0 = different hands, 1 = same hand different finger, 2 = same finger | +1 | FM1 (replaces the NESTED ladder `same_hand ⊃ adjacent ⊃ scissor` + `same_finger` with one ordinal); FM3 (`BigramClass`'s own documented speed ordering: ALTERNATE < SAME_HAND < SAME_FINGER) |
| 2 | `row_span` | `\|y_a − y_b\|` if same-hand two-finger else 0 (0/1/2) | +1 | FM1+FM3: the GRADED severity that subsumes `scissor` (dy==2 on adjacent fingers), `half_scissor` (dy==1) and `row_skip` (dy==2 any finger) — no threshold, so no layout-dependent blind spot |
| 3 | `lateral_span` | `keybo.features.classify.lateral_span(a,b)` — **the `lat-span` GAUGE's own per-cell quantity, unchanged** | +1 | **FM4 by construction** (the feature named `lateral_span` IS the gauge named `lat-span`); FM1+FM3 (replaces `lsb`, whose flagged share is layout-dependent: LSBWIDEN-1 measured a 2.20× coverage fold spread vs `lateral_span`'s 1.0000×) |
| 4 | `same_hand_travel` | `geometry.distance(a,b)` if same hand else 0 | +1 | **FM3, the headline mechanistic fix**: `distance` prices long travel CHEAPER because long travel proxies for CROSS-HAND (which is fast) — the confound that gives "distance explains X" the wrong sign. Conditioning on same-hand makes it monotone: within one hand, farther IS slower |
| 5 | `row_load` | `\|y_a−2\| + \|y_b−2\|` over letter keys (0..2) | +1 | FM1 (replaces the 3-way perfectly-collinear ROW one-hot with one ordinal); FM2 (ONE column carries "this stroke leaves the home row" instead of it being shattered) |
| 6 | `row_arrival` | `\|y_b−2\| − \|y_a−2\|` (−1..1) | +1 | FM1: the **orthogonal complement** of `row_load` — sum and difference of two equal-variance quantities are a 45° rotation, i.e. an ORTHOGONALIZED basis, which is exactly what INVARIANT 2 asks for. Also preserves the stroke-ORDER information a bare sum loses (`a→x` vs `x→a`) |
| 7 | `bottom_bias` | (#keys on the bottom row) − (#keys on the top row) (−2..2) | +1 | FM3: the up/down ASYMMETRY as its own signed mechanism. Not folded into `row_load`, because bottom is measured costlier than top (158.670 vs 137.0/140.2 ms, SHAPDIFF-TCOND) and a magnitude-only axis cannot say so |
| 8 | `finger_load` | `(3 − finger_kind(a)) + (3 − finger_kind(b))` over letter keys (0..6) | +1 | FM1: replaces the 5-way FINGER block, which is **not even a one-hot** — `lateral` co-fires with `index` (\|x\|=1) and `pinky` (\|x\|=6), so 5 columns encode 4 fingers plus an overlapping flag |
| 9 | `off_home_column` | # of keys sitting in a finger's off-home stretch column (\|x\| ∈ {1,6}) (0..2) | +1 | **FM4**: the column named `lateral` collided with the `lat-span` gauge while measuring something else entirely (an off-home COLUMN, not a stretch). Renamed to what it measures |
| 10 | `roll_inward` | +1 if the stroke travelled toward the index, −1 toward the pinky, 0 otherwise (`is_inwards_ordered`) | **−1** | **FM4 in its purest form**: `inwards`/`outwards` are SWAP-INVARIANT — 0 of 870 ordered pairs change under reversal — so the served names LIE about being directions of travel. This is the honest ordered predicate, already built and tested |

**DROPPED, and why:**
* **`wpm`** — CONSTANT at serve, yet carries −0.0922 ms/char (9.2% of the T2 gap) as pure
  interaction credit booked as a main effect. Removing it is the single largest artifact
  elimination available and is what makes **M2 == 0** attainable. ⚠ It does carry residual
  pace signal at TRAIN time (T-REL), so its removal has an accuracy cost that §5 measures as
  its **own arm** — I do not assume it is free.
* **`angle`** — a SIGNED rotation angle with no monotone mechanism (neither "more angle =
  slower" nor the reverse is defensible). Its information lives in `row_span` + `roll_inward`.
* **`adjacent`** — nested inside `same_hand`; once `row_span` and `finger_load` are present,
  adjacency has no standalone mechanism, only a correlation.
* **`dx`, `dy`, `distance`** — the mutually-dependent travel triple whose individual credits
  are not unique. Replaced by `lateral_span` (horizontal, coverage-invariant), `row_span`
  (vertical) and `same_hand_travel` (magnitude, same-hand-conditioned).
* **`inwards`, `outwards`, the row one-hot, the finger block, `scissor`, `lsb`, `same_finger`**
  — folded into 1–10 above.

**FEWER, CLEANER FEATURES:** 20 → 10 columns. Nothing is *added*; the frame is a re-expression
of the same geometry in a basis whose axes have names that mean one thing and signs a human can
defend.

**Stamp:** `FEATURE_VERSION_INTERP = f"{FEATURE_VERSION}+interp.1"`, a FOURTH population.
`FEATURE_VERSION` is not touched. `_BIGRAM_PLACEMENT_NAMES` / `_TRIGRAM_LEVEL_NAMES` (the
SHARED PREFIXES of the version-locked served lists) are not touched.

## §5 — MONOTONE CONSTRAINTS: THE VERIFICATION PROTOCOL (INVARIANT 4)

Present ≠ effective, and ADJ-2 PINKY-MONO's registered failure was *learns zero magnitude*.
Four checks, all reported:

* **(a) BOOSTER-LEVEL.** For each constrained column, sweep it alone across its observed range
  (all other columns held at their corpus-weighted median) and assert the prediction is
  non-decreasing (non-increasing for −1). This tests the booster, not the parameter dict.
* **(b) SHAP-LEVEL, HELD-OUT.** On held-out LOLO cells, Spearman(feature value, its own SHAP
  value) must be ≥ 0 for +1 columns and ≤ 0 for −1 columns.
* **(c) NON-DEGENERACY (the ADJ-2 trap).** Each constrained column must have mean \|SHAP\| > 0
  on the serve grid. A constrained column that learned nothing is **not** counted toward M4.
* **(d) THE CONSTRAINT'S OWN COST.** Train the SAME frame with constraints OFF and report the
  paired per-fold Δρ, isolating the constraint's price from the frame's.

## §6 — THE ACCURACY COST (reported honestly, no bar)

Matched-pair LOLO via the shipped `keybo.training.validate.validate()`: 4 folds × 3 seeds,
same seeds / same hyperparameters / same folds, the ONLY difference being the frame.
**PAIRED PER-FOLD deltas (MOR-FIX-1)** — a mean-of-ratios can reorder. Primary `rho`
(bucket-centered Spearman); secondary `rho_frac_ceiling`, `wmae`, `tau_heldout`.

**ARMS (registered now):**
1. `CUR` — the served 20-col frame (the incumbent baseline; the floor I MEASURE rather than
   borrowing SHAPDIFF-1's number, which came from the shipped models on a different data
   volume — five floor-confusions in this project, the newest being that a floor must match
   the comparison's DATA VOLUME).
2. `INTERP` — the 10-col frame, monotone constraints ON.
3. `INTERP-NOMONO` — the 10-col frame, constraints OFF (isolates §5(d)).
4. `CUR-NOWPM` — the served frame minus `wpm` (isolates the cost of the drop that makes M2==0).

**ONE registered refusal:** if the high-wpm gate raises **STRUCTURALLY** (every seed of a
fold), that is reported as a MATERIAL cost in the headline — SRROLL-1's precedent is that a
structural high-wpm regression is "worse than a plain null".

## §7 — CONTROLS

* **NC (already run, `negctl.json`)** — reproduce 11 shipped quantities before trusting any new
  number. **PASS**, \|diff\| ≤ 4.4e−5 on all 11.
* **NC2 — FRAME-SWAP SANITY.** Attribute the NEW model with the OLD frame's featurizer: must
  RAISE (the version guard / the `_shap_tables` name assert), not silently produce a table.
* **NC3 — SHUFFLE.** Run the shipped `--control shuffle` through the new frame: the INTERNAL
  bars must break while the external gauge tie survives, exactly as it does for the served
  frame. A control that RECONCILES means my identity is vacuous.
* **NC4 — MONOTONE PLACEBO.** Constrain a column in the direction OPPOSITE to its mechanism
  and confirm §5(a)/(b) DETECT it. A verification that cannot fail is not a verification.

## §8 — WHAT I AM NOT CLAIMING (INVARIANT 5, registered up front)

Some attribution non-uniqueness is **intrinsic to SHAP on any correlated frame**: no frame over
a 31-key board can make geometry columns exactly independent, because they are all deterministic
functions of two positions. Grouped / Owen-Shapley over blocks is a **TOOL** fix and interacts
with the frame fix. The §9 table separates frame-fixable from tool-fixable and is written
BEFORE the results.

## §9 — FRAME-FIXABLE vs TOOL-FIXABLE (registered prediction, to be scored)

| failure mode | my prediction | why |
|---|---|---|
| FM1 coupled columns | **PARTIALLY frame-fixable** | a smaller, rotated basis lowers but cannot zero the correlation; the residual needs grouped Shapley |
| FM2 shattered mechanism | **FRAME-fixable** | it is caused by having N columns for one mechanism; one column removes it |
| FM3 non-mechanistic / wrong sign | **FRAME-fixable** | conditioning (`same_hand_travel`) and constraints make the sign defensible by construction |
| FM4 name collisions | **FRAME-fixable, and free** | pure renaming + reusing the gauge's own predicate |
| FM5 aggregation hiding sign flips | **NOT fixable at bigram level** | the redirect sub-classes are TRIGRAM-level; a bigram POC cannot address it, and I will say so rather than claim it |

## §10 — DELIVERABLE

`/local/home/zegertho/agent/state/interpframe/report.md`, line 1 = YES/NO/PARTIAL with the
interpretability gain and the accuracy cost both as numbers. This is a **PROPOSAL**; the
ADOPT/DON'T decision is the human's.
