# PREREG — Kaggle competition techniques audited against THIS problem

Registered **2026-07-29, BEFORE any arm was run.** Branch `kaggle-techniques`, worktree
`/tmp/kaggle`, base ledger commit `a6b3833`. Author: subagent `kaggle` of `keybo-optimization`.

The ask: *"audit Kaggle competition techniques in order to get the best model."* Deliverable is a
ranked, COSTED list of techniques that would help **this** problem plus a working demonstration of
the top one or two — not a listicle.

---

## 0. THE FRAME I MEASURED FIRST, because my brief's frame was wrong

My spawn brief supplied a data block as 🟢 VERIFIED. **Every figure in it is wrong.** I re-derived
each (the parent's own standing instruction: *"keep re-deriving every figure I hand you"*). Measured
with `load_strokes` under `assert_module_under('keybo','/tmp/kaggle')`:

| Quantity | Brief said | **Measured** | Instrument |
|---|---|---|---|
| layouts (aalto k31 bigram) | 11 | **4** — azerty, dvorak, qwerty, qwertz | `cut -f1 \| sort -u` + `load_strokes` |
| participants per layout | exactly 1, set `{1}` | **{64, 166, 485, 54690}** | distinct pids per layout |
| rows | 3,098 | **2,202** (`bistrokes31_v1.tsv`) | `wc -l` (no header; line 1 is data) |
| samples | 467,579 | **29,532,228** | sum of parsed sample tuples |
| trigram feats that are bigram-placement | 19 of 46 | **38 of 46** (19 `bg1_` + 19 `bg2_`) | `TRIGRAM_FEATURE_NAMES` |
| `EXPECTED_SIGN` sign priors | "registered per gauge" in this repo | **ABSENT from keybo entirely** — it lives in `evidence_scorer.py`, the *agent tooling* | `grep -rn` over `src/`, `tests/` |

**Direction of the error matters more than the error.** The brief's headline conclusion — *46 features
against very few independent folds, so volume-hungry Kaggle technique is mismatched* — is
**STRENGTHENED**: the real LOLO fold count is **4, not 11**. I am keeping that conclusion and
discarding its constants. This is the campaign's recurring
*wrong-constant-attached-to-a-true-conclusion* pattern; the conclusion's correctness is precisely
what protected the numbers from audit.

**Scope discipline (adopted rule: report what an n is an n OF).** Every n below is scoped:
- **4 folds** = *aalto layouts in the k31 training tables* (the frame these arms run on).
- **7 participants / 12 labels** = *the COMMUNITY dataset*, a different file. (I earlier reported "9"
  from a naive `label.split('#')[1]`; the `+pseudo`/`+rareboost` suffixes are corpus tags on one
  submitter. Registered `pids` map = 7, and `data/community/README.md` says "7 submitters".)
- **5 layouts** = *the served ranking frame* (`NAMED_LAYOUTS`: colemak, dvorak, graphite, qwerty, semimak).

### 0b. A brief claim this refutes outright (pending the running probe)
The brief states 🔴 *"`tune.py --objective lolo` CANNOT CURRENTLY SCORE AT ALL on this data"* because
the noise ceiling needs ≥2 participants per layout and "every layout has 1". On the **aalto k31
frame** every layout has **≥64** participants, so `split_half_ceiling` has participants to bisect.
That claim is true only of the **community** frame. Registered prediction, made before reading the
result: **≥3 of 4 aalto folds return a finite ceiling.** Probe `probe_ceiling.py` is running; its
output is registered as the adjudicator either way.

---

## 1. What is actually being optimized (the guard that kills most Kaggle technique here)

Fitness ranks layouts. The precedents in `PREREGISTRATIONS.md` that bind every arm below:

- **`NGRAM-FE`**: richer features gave the best full-model gain (**+0.0899**) while served geometry
  **collapsed 0.852 → 0.164**. Better fit, destroyed ranking.
- One arm won served-rho (+0.00879, CI above zero, survived BH-FDR) and **still failed** the ranking
  guard (margin-tau 0.822 < 0.905).

⇒ **A technique that improves fit and not ranking is a FAILURE and will be reported as one.**

---

## 2. ARMS — declared now, with the success criterion fixed in advance

Two arms only. This is deliberate: the 99-arm sweep already returned nothing under
multiplicity-aware inference, so adding arms costs multiplicity and buys little.

### ARM-G — `GroupKFold` on the `cv-mae` path (the leakage fix)
**Claim under test:** the legacy `cv-mae` objective splits UNGROUPED, so a layout's own rows appear
on both sides of a split, and the resulting "best params" are chosen partly on memorized
layout-specific timing.

Mechanism already verified statically (🟢): `check_cv(5, y, classifier=False)` → `KFold(shuffle=False)`,
and `is_classifier(XGBRegressor)` is `False` so there is no stratification either.
**⚠ The brief called this "random K-fold"; it is CONTIGUOUS-BLOCK, ungrouped `KFold`.** That
difference is load-bearing: with `shuffle=False` the leak is a function of ROW ORDER, and measured row
order is interleaved (`qwerty x506, qwertz x189, qwerty x3, qwertz x62, … azerty x187`), so blocks do
straddle layouts. **A naive `shuffle=True` "fix" would make leakage worse, not better.**

**What I will measure:** for each of {ungrouped `KFold` (status quo), `GroupKFold(groups=layout)`}:
the selected params, the CV score each *believes* it achieved, and — the honest number — the
**LOLO transfer** (`rho/ceiling`, plus layout tau) of a model trained with each selection.

**SUCCESS = the leaked path's optimistic CV score exceeds its true LOLO transfer while the grouped
path's does not** (i.e. leakage demonstrated as a *measured optimism gap*, not asserted from code
reading). Reporting a params *difference* alone is NOT success — params can differ by noise.

**FAILURE / kill condition, stated up front:** if grouped and ungrouped select params of
indistinguishable LOLO transfer, I report **"the defect is real but inert on this frame"** and the
recommendation drops to a hygiene fix, not a win.

**Blast radius (must be reported alongside, or the finding is inflated):** shipped default objective
is `lolo` (`cli/tune.py:27`); the only in-tree callers of `tune_hyperparameters` are `cli/tune.py:118`
and one test; `PREREGISTRATIONS.md:3548` already records the CV-MAE tuner as deprecated with winners
that **"never shipped"**. Pre-registered expectation: **blast radius on registered numbers ≈ ZERO**,
so this is a **latent-defect fix, not a retraction.** I will say so plainly even though it makes my
headline smaller.

**Second defect found while reading (registered so it is not passed off as a discovery later):**
`build_training_matrix` defaults `target_space="MS"`, but every shipped k31 model is `LOGRAT`
(verified in the meta sidecars). So the `cv-mae` path also tunes in the **wrong target space**.

### ARM-M — `monotone_constraints` from physical priors (small-n regularization)
**Claim under test:** with 4 folds and 20/46 features, the highest-value regularizer is a *prior*, not
a tuned penalty. XGBoost supports hard monotone constraints; a handful of typing-time relationships
have sign priors defensible from physics rather than from this dataset.

**Priors I will constrain — declared BEFORE running, and deliberately few** (bigram frame; `+1` =
predicted time non-decreasing):
`distance +1`, `dy +1`, `same_finger +1`, `scissor +1`, `lsb +1`, `wpm -1`.
Rationale, per feature: farther/more-vertical travel, same-finger repeats, scissors and lateral
stretches are all slower; higher session WPM means shorter keystroke times. **`EXPECTED_SIGN` does not
exist in this repo**, so these are MY priors, defended above — not "registered" ones, and I will not
claim otherwise.

**SUCCESS (all three, on the SERVED frame):**
1. served-frame ranking guard **does not degrade** — layout tau ≥ the unconstrained baseline's;
2. LOLO mean `rho/ceiling` within noise or better (not worse by > 0.005, the `tune_lolo` bar);
3. wmae not worse by > 0.91% (the ledger's incumbent-protection clause).

**FAILURE:** any served-frame ranking degradation ⇒ report as a FAILURE, however good the fit.
**Known risk, registered:** XGBoost docs state constraints under `hist`/`approx` "may produce
unnecessarily shallow trees" and *deliberately* remove structure. With `max_depth=3` in production,
shallow-tree damage is a live hazard, so a null/negative result here is genuinely likely.

---

## 3. Multiplicity discipline
Two arms, each with a pre-declared primary endpoint, so the 99-arm sweep's BH-FDR machinery is not
needed for arm *selection*. Where an arm reports across the 4 LOLO folds I report **per-fold numbers,
not a best-fold**, and any across-fold claim uses the mean with all 4 folds shown. **No best-of-N.**
If I end up testing more than these two arms I will apply **Bonferroni** over the added family and
say so.

## 4. Techniques I am declaring NOT-WORTH-TESTING here, with the reason (part of the deliverable)
Registered now so this cannot be mistaken for post-hoc rationalization of null results:
- **Hyperparameter search generally** — the 99-arm sweep found no arm beating the per-surface peak
  under BH-FDR *and* Bonferroni. Re-running it larger is not a technique, it is the same bet.
- **Stacking / blending / ensembling** — buys fit; fit is not the problem. `NGRAM-FE` is the
  precedent for "more capacity, worse ranking".
- **Pseudo-labelling / TTA** — cannot fabricate keystroke data.
- **Target encoding** — 98.7% of rows are one layout; any layout-keyed encoding is a memorization key
  (this is the OQ-1 frequency-feature finding in another costume).
- **Adversarial validation** — *worth naming, not running here*: with 4 train groups and a 5-layout
  served frame, the train-vs-served discriminator has ~9 units total. It cannot produce a
  discriminator AUC anyone should act on. **The NGRAM-FE collapse is the distribution shift it would
  be looking for, already measured directly and more cheaply.**
- **CatBoost ordered boosting** — the docs' small-data claim is real (`boosting_type=Ordered`
  "usually provides better quality on small datasets"; GPU default flips to Ordered at ≤50k objects)
  but it addresses small **n**, and we have 29.5M samples / 4 **groups**. Our scarcity is in groups,
  which ordered boosting does not address. The CatBoost *paper's abstract* makes **no** small-data
  claim at all. Also **not installed** (no lightgbm, no catboost), so it carries an install cost for a
  mechanism that does not match our scarcity axis.

## 5. Kill-my-own-result checks I commit to running
- `assert_module_under('keybo','/tmp/kaggle')` in **every** driver (the shared clone's `.venv` has an
  editable `.pth` into `repos/keybo/src`; without a `PYTHONPATH` prefix, probes silently test the
  WRONG TREE — I verified this failure mode is live in this environment).
- Report served-frame ranking for every arm, even when it kills the arm.
- No shipped artifact is retrained or overwritten; models are trained into `/tmp/kaggle-work/`.
- Re-derive any constant before it enters a conclusion; state the instrument next to each number.
