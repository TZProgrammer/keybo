# KITCHEN-SINK arm — pre-registration

Registered BEFORE any transfer score was computed. Branch `kitchen-sink`, built on `sfgated-eval`
(0da5c09) so it inherits the ordered-direction channel, the same-finger-gated redirect pair, AND the
high-wpm non-regression gate.

**The ask:** add every feature keycraft and the other external projects have that we lack, then
hyperparameter-tune, and measure whether the result is a BETTER MODEL. A clean NULL is an acceptable
answer; no positive is being hunted.

---

## 0. Corrections to the brief (verified first, per the standing rule)

| Brief claim | Verdict | Evidence |
|---|---|---|
| main @ e6a5b9e, 7 branch SHAs | 🟢 all correct | `git rev-parse` on each |
| `HIGH_WPM_FLOOR` / gate **NOT on main**, cherry-pick from `highwpm-gate` if absent | 🟢 correct AND **no cherry-pick needed** | 0 hits on `main`; present at `src/keybo/verdicts.py:183` on `sfgated-eval`, so it is inherited. The brief's own "verify" instruction is what caught this. |
| LOLO is 4 folds over azerty/dvorak/qwerty/qwertz | 🟢 correct | `cut -f1 bistrokes31_v1.tsv \| sort -u` = exactly those 4 |
| `FEATURE_VERSION = 2026-07-05.3`, models error on MISMATCH not on changed meaning | 🟢 correct | `features/schema.py:29`, `models/base.py:168-170` |
| `tune_lolo`'s tau gate near-useless at 4 layouts (7 values, 1/3 apart) | 🟢 correct | `tau_resolvable_step(4) = 4/(4·3) = 0.3333` |
| `GroupKFold(5)` raises on 4 groups; `grouped_cv` clamps | 🟢 correct | `tune.py:160-182`, clamp is `min(cv, n_groups)` |

**One number in the brief's supporting context is wrong:** the parent's memory quotes the training
tables as "2201 rows" / "16642 rows" in one place and "2202" / "16643" in another. `wc -l` gives
**2202** and **16643** and the files have **no header line**, so the higher pair is right — the
parent's own later correction, not the earlier figure. Not load-bearing for any verdict here.

**A gap the brief did not mention, found in the code:** `tune_lolo` has **no `direction`
parameter**. It calls `validate(...)` without one, so it can only ever tune the NARROW frame. Tuning
the widened frame requires threading the flag; that is part of this arm's work, not a pre-existing
capability.

---

## 1. Candidate audit — what is GENUINELY missing (run BEFORE implementing)

`agent-artifacts/kitchensink_audit.py`, over the FULL enumeration of `ROW_STAGGERED_30`
(870 ordered pairs / 24,360 ordered triples — never a corpus sample, which would confound
coverage with definition). `swap_asym` = pairs whose value changes under reversal; `R2` = OLS
recoverability from the existing frame.

| candidate | source | fires | swap_asym | R²(narrow) | R²(wide) | decision |
|---|---|---|---|---|---|---|
| `half_scissor` (HSB) | keycraft | 48 | 0 | 0.6857 | 0.6857 | **ADD** |
| `row_skip_anyfinger` | keycraft-ish | 100 | 0 | 0.5477 | 0.5480 | **ADD** |
| `pinky_off_home` (POH) | keycraft | 116 | 208 | 0.6924 | 0.6924 | **ADD** |
| `weak_finger_pair` | keycraft RED-WEAK@bigram | 60 | 0 | 0.3986 | 0.4487 | **ADD** |
| `finger_dist_ordered` | ours (graded IN/OUT) | 324 | 324 | 0.3560 | 0.8516 | **ADD** |
| `lsb_magnitude` (LSB-dist) | keycraft | 32 | 0 | 0.9768 | 0.9768 | **REJECT** — 97.7% recoverable |
| `onehand` (3RL) | keycraft | 756 | 0 | 0.5128 | 0.5338 | **ADD** |
| `onehand_in` (3RL-IN) | keycraft | 378 | 756 | 0.3200 | 0.4358 | **ADD** |
| `red_sfs` (RED-SFS) | keycraft | 972 | 0 | 0.5237 | 0.5819 | **ADD** |
| `alt_sfs` (ALT-SFS) | keycraft | 1440 | 0 | 0.6413 | 0.6731 | **ADD** |
| `sg_full_scissor` (FSS) | keycraft | 672 | 0 | 0.1836 | 0.1849 | **ADD** |
| `sg_half_scissor` (HSS) | keycraft | 1344 | 0 | 0.1897 | 0.1929 | **ADD** |
| `sg_lsb` (LSS) | keycraft | 896 | 0 | 0.1489 | 0.1554 | **ADD** |
| `red_weak` (RED-WEAK) | keycraft | 432 | 0 | 0.6853 | **1.0000** | **REJECT** — see below |

### Rejected, with proof

* **`red_weak` is already built.** R² = 1.0000 against the widened frame is not "nearly
  recoverable", it is an identity: over all 24,360 triples `red_weak` is **bit-identical** to
  `bad_redirect_sfgated` (both fire on exactly the same 432 triples, 0 differences). keycraft's
  RED-WEAK is REDIRGATE-1's same-finger-gated bad-redirect under another name — and `sfgated-eval`
  already measured that column NULL. Re-adding it would have re-measured a closed question.
  (It differs from the SERVED `bad_redirect`, which fires 648× — 216 more — which is why it looks
  novel at R²=0.685 against the narrow frame. The narrow comparison is the misleading one.)
* **`lsb_magnitude` is 97.7% recoverable** and fires on only 32 of 870 pairs. It is the graded form
  of a flag we already serve, on the smallest support of any candidate; KEYCRAFT-1 killed
  `2RL-IN+2RL-OUT` at R²=1.0000 by the same rule, and 0.977 with 32 firings is inside the same
  verdict.
* Also considered and NOT added: `sfb`/`sfs`/`alt`/`redir` (KEYCRAFT-1 measured our versions
  agreeing with keycraft to **0.0000%** relative — nothing to add); the `IN:OUT` ratio and `FLW`
  (both are linear combinations of columns in the table above, and keycraft itself weights FLW at
  +8.00 while zeroing every IN/OUT column — it computes direction then discards it);
  keycraft's `HLD`/`FLD`/`RLD`/`Hx`/`Fx`/`Rx` load-deviation family (whole-LAYOUT aggregates
  against user target loads, not per-n-gram features — they cannot enter a per-stroke model, and
  our `scoring/utilization.py` already covers that axis); cyanophage's `bigram_effort.json`
  ORDERED-pair table (CYANO-1's own method finding is that a single signed indicator captures only
  R²=0.0186 of it — copying the *table shape* means 870 free parameters on 2,202 training rows,
  which is a memorization frame, not a feature).

**12 new candidate definitions** survive: 5 bigram-level and 7 trigram-level (of which 3 are
skipgram-level). The COLUMN counts differ from the definition count because the bigram-level
definitions enter the trigram frame twice, once per constituent bigram:

* bigram frame: 22 (widened) **+ 5 = 27**
* trigram frame: 52 (widened) **+ 7 + 2×5 = 69**

The exact widths are asserted in code, not taken from this arithmetic (see §2's byte-identity test).

---

## 2. Implementation contract (version-locking)

Additive under a NEW stamp, following the `FEATURE_VERSION_DIRECTION` precedent exactly:

* `FEATURE_VERSION_KITCHENSINK = f"{FEATURE_VERSION}+kitchensink.1"` — never equal to
  `FEATURE_VERSION` or to `FEATURE_VERSION_DIRECTION`, so the three model populations can never be
  confused.
* Nothing is added to `_BIGRAM_PLACEMENT_NAMES` or `_TRIGRAM_LEVEL_NAMES` — those are the shared
  prefixes of the version-locked served lists, and appending there would silently widen the served
  frame for all six shipped k31 models.
* No existing column's meaning changes. `inwards`/`outwards` stay swap-invariant; `scissor` stays
  `dy == 2`; `bad_redirect` stays ungated.
* **Falsifiable byte-identity claim:** with the new flag OFF, the emitted feature matrix must be
  identical to the current `direction=` matrix at **max abs diff 0.000e+00**, for both the narrow
  and the widened frame, over the full 870-pair / 24,360-triple enumeration. If it is not
  0.000e+00, the implementation is wrong and the arm is void.

## 3. HPO grid

`tune_lolo` (transfer-scored), **not** `tune_hyperparameters` — the latter's CV is ungrouped and
optimistic by +0.0349, its winners have never shipped, and `shuffle=True` is 1.76× worse while
reporting the best CV number (KAGGLE-1 FINAL). `tune_lolo` needs a `direction`/frame parameter
threaded first (§0).

Grid (16 candidates, explicit — reproducible, not a random draw):

```
max_depth        ∈ {2, 3, 4, 5}      # 3 is the shipped value (goodhart-row-blindness)
n_estimators     ∈ {200, 400}
learning_rate    ∈ {0.05, 0.10}
min_child_weight ∈ {1, 3}
```
= 4×2×2×2 = 32 → subsampled to the 16 with `max_depth ∈ {2,3,4,5}` × `n_estimators ∈ {200,400}` ×
`learning_rate ∈ {0.05,0.10}`, `min_child_weight = 1` fixed (2 of the 5 knobs held to the shipped
value keeps the search inside what 4 folds can resolve).

Scored on seeds [0] during the search (cost), then the WINNER is re-run on seeds [0,1,2] for the
verdict. `min_margin = LOLO_MIN_MARGIN = 0.03` stays ON: a win inside the ceiling-reweighting
bound is a convention artifact. The tau gate is expected to be **saturated and to gate nothing** at
4 layouts (TAUGATE-1); that will be reported as a non-result, not as a passed check.

## 4. Decision rule (fixed before any score)

The candidate is a BETTER MODEL only if **all three** hold, on the trigram frame (the richer one)
with the bigram frame reported alongside:

1. **TRANSFER** — paired per-fold/seed deltas `rho_widened − rho_narrow` (MOR-FIX-1: paired
   deltas, never a mean of ratios). Requires mean paired delta **> 0** AND
   **≥ 3 of 4 folds sign-consistent across all 3 seeds**. 12 cells = 4 folds × 3 seeds.
2. **HIGH-WPM NON-REGRESSION (mandatory)** — `bucket_regression_report`, `HIGH_WPM_FLOOR = 80`,
   tolerance 0.005, baseline = the matched narrow cell's `bucket_rhos`. **Any** bucket ≥ 80
   regressing beyond tolerance on **all 3 seeds** = STRUCTURAL = FAIL regardless of mean rho.
   Inconsistent across seeds = NOISE, reported as such. `gated: False` = **UNSCOREABLE**, never
   "passed".
3. **NO RANKING COLLAPSE** — over the 15-name scoring catalog (the SEPARATE surface, not the 4 LOLO
   folds): Kendall tau and Spearman rho narrow-vs-widened, plus the argmin's stability and whether
   churn is confined to near-ties. NGRAM-FE improved fit and destroyed ranking (0.852 → 0.164);
   ARM-M did the reverse (+60.68% wmae). Both directions are checked.

## 5. Predicted outcome + falsifier

**Prediction: NULL** — specifically `{null}`, not `{fit-win/ranking-loss}`. **I agree with the
parent's prior**, and the audit sharpens the reason rather than merely restating it: the binding
constraint is 4 training layouts, and the 12 candidates that survived the audit are all
recoverable at R² between 0.15 and 0.85 from columns the model ALREADY has. A gradient-boosted tree
can synthesize an R²=0.69 column by spending depth; what it cannot synthesize is a distinction the
frame is blind to. Exactly one such distinction existed (stroke DIRECTION, swap_asym = 0 for every
served column), it has now been added and measured **NULL twice**, and every remaining candidate is
a reweighting of information already present. Two independent priors point the same way:
`red_weak` — the single candidate whose R² came back 1.0000 — turned out to be a column a previous
round already built and already measured null.

**Secondary prediction:** the high-wpm gate FAILS structurally on **dvorak b120** again, since the
last two rounds both did and the failure got *deeper* (1.4–5.3×) when columns were added rather
than shallower.

**Falsifier (what would make me wrong):** mean paired trigram delta > 0 with ≥3 of 4 folds
sign-consistent on all 3 seeds, AND zero structural high-wpm regressions, AND tau ≥ 0.85 on the
15-name catalog with a stable argmin. If that lands, "4 layouts is the binding constraint, not
feature count" is refuted and feature breadth does buy transfer.

**A NULL here is informative, not a failed experiment:** it would be the *fifth* independent
feature-addition round (NGRAM-FE, ARM-M, direction ×2, kitchen-sink) to land null on the same
4-layout frame, which is evidence about the FRAME, not about any one feature family.

## 6. Scope that travels with every n

* 4 LOLO folds = azerty/dvorak/qwerty/qwertz. **Not** the 13/15-layout scoring catalog.
* 12 cells per arm = 4 folds × 3 seeds.
* 870 ordered pairs / 24,360 ordered triples = full `ROW_STAGGERED_30` enumeration, not corpus-observed.
* Training rows: bigram 2,202 / trigram 16,643, 4 layouts, `data/models/k31` geometry.
* Ranking catalog: 15 named layouts (`NAMED_LAYOUTS` + `_EXTRA_NAMED`).
