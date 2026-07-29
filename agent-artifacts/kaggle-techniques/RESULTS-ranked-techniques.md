# Kaggle-technique audit for keybo — RANKED, COSTED LIST + artifact index

Subagent `kaggle` of `keybo-optimization`. Worktree `/tmp/kaggle`, branch `kaggle-techniques`,
base `a6b3833`. Prereg `8168a82`, drivers `6b38467`. Nothing pushed, no CR, no shipped artifact
retrained. Dated 2026-07-29.

**Headline in one line:** the parent's suspected `cv-mae` leak is **REAL and now measured**
(ungrouped CV is optimistic by **+0.0349 MAE, 5/5 seeds**), the "obvious" `shuffle=True` fix is
**1.76× WORSE** than the status quo, and **monotone constraints — the technique with the best
prior — FAILED hard (+60.7% wmae).** Everything else in the Kaggle canon is mismatched to a
**4-group** problem, and I say so per technique rather than hedging.

---

## THE RANKED LIST

Ranking axis is *expected value to THIS problem*, = (evidence it helps) × (blast radius) ÷ cost.
"Served-frame guard" is mandatory per the NGRAM-FE precedent: a fit win that breaks layout
ranking is a FAILURE here.

### 🥇 1. Grouped cross-validation (`GroupKFold` by layout) — **ADOPT, but as HYGIENE, not a win**
| | |
|---|---|
| **Verdict** | 🟢 Defect real and measured. Fix is correct. **Blast radius on registered numbers ≈ ZERO.** |
| **Evidence** | Structural: shipped `KFold(shuffle=False)` puts **4/4 folds with ≥1 layout on both sides** of the split. Measured: ungrouped believed-CV is optimistic vs honest leave-one-layout-out MAE by **+0.0349 mean (range +0.0323…+0.0363, 5/5 seeds, always positive)**. |
| **Cost** | ~10 lines (`GroupKFold` + thread `groups=layouts` through `tune_hyperparameters`). Runtime unchanged. **Cheapest item on this list.** |
| **Served-frame guard** | Not required — it changes *how candidates are scored*, never what is served. It cannot be another NGRAM-FE by construction. |
| **What would refute it** | A demonstration that no one ever runs `--objective cv-mae`. Which is nearly true — see blast radius. |

**The honest boundary, stated because it shrinks my own headline:**
- The shipped default objective is **`lolo`** (`cli/tune.py:27`); `cv-mae` is opt-in and documented as deprecated. Only in-tree callers are `cli/tune.py:118` and one test. `PREREGISTRATIONS.md:3548` already records the CV-MAE tuner's winners as **"never shipped"**. So this is a **latent-defect fix, not a retraction of any published number.**
- Did the grouped split pick *better* params? Directionally yes — **3 wins, 2 ties, 0 losses** over 5 candidate-set seeds, mean **+0.44%** lower honest LOLO MAE — but the exact two-sided **sign test gives p = 0.25**, so ⚠ **the selection-quality improvement is NOT statistically established.** I am not claiming it. The *optimism* result (5/5, tight spread) is what carries.
- ⚠ **`GroupKFold`'s own optimism of exactly `0.000000` is partly TAUTOLOGICAL:** with 4 layouts and `n_splits=4`, `GroupKFold` **is** leave-one-layout-out, so "believed" and "honest" are the same estimator. Same reason its regret-vs-oracle is exactly 0 on 5/5 seeds. Do not quote either zero as an independent measurement.

🔴 **THE FINDING WITH THE HIGHEST SURPRISE VALUE — the naive fix is the worst option.**
`KFold(shuffle=True)` — what a competent engineer reaches for on seeing "ungrouped CV" — has
optimism **+0.0635 vs the status quo's +0.0361 (1.76×)** while reporting the **lowest** believed
MAE of all three (0.1967 vs 0.2241 vs 0.2581). It looks like the best option on the only number
a careless fixer would read. Mechanism: `shuffle=False` accidentally preserves some
same-layout contiguity (the TSV is layout-blocked in runs); shuffling destroys that and gives
every fold a near-iid mix of all 4 layouts. **Registered as a prediction in the prereg BEFORE
the run, and it held.**

### 🥈 2. Adversarial validation — **DO NOT RUN, but keep the concept; the answer is already known**
| | |
|---|---|
| **Verdict** | 🟡 Right instinct, un-runnable at this scale. |
| **Why** | It needs a train-vs-served discriminator. Train side = **4 layouts**; served side = **5 layouts** (`NAMED_LAYOUTS`). ~9 units total. No AUC from 9 points should move a decision. |
| **Better substitute (free)** | The NGRAM-FE collapse (served geometry **0.852 → 0.164**) already *is* the measured train/serve distribution shift, obtained directly and more cheaply than any discriminator. |
| **Cost if run anyway** | ~1h. Would produce an uninterpretable number. |

### 🥉 3. Monotone constraints from physical priors — **TESTED AND REJECTED** (see full result below)
Best a-priori match in the whole survey — XGBoost supports it natively, typing physics gives
real sign priors — and it **failed 2 of 3 pre-registered gates.** Ranked 3rd because the
*negative* result is valuable and cheap to reuse, not because it is recommended.

### 4. Nested CV — **CORRECT IN PRINCIPLE, NOT ACTIONABLE HERE**
🟡 sklearn calls grid-search-inside-CV "the best practice", and its mechanism (`best_score_` is
selection-optimistic) is exactly what ARM-G measured. But an outer loop over **4** layouts leaves
**3** for the inner loop. The estimator variance at that size exceeds the bias it removes.
**Use the existing LOLO harness as the outer loop instead — it already is one.** Cost to do
properly: high. Value: low.

### 5. Repeated CV / more seeds for variance reduction — **CHEAP, ALREADY PARTLY DONE**
🟢 The harness already runs multiple training seeds and the ledger requires ≥3. My ARM-M used 3
seeds and ARM-G3 used 5 candidate-set seeds; **the seed-to-seed spread was the difference
between "0.81% win" and "p=0.25, not significant"** — i.e. repeated sampling is what stopped me
publishing a one-draw artifact. Recommendation: keep doing it; it is the cheapest guard against
this ledger's most common failure. Cost: linear in seeds.

### 6. CatBoost / LightGBM (incl. ordered boosting) — **NOT WORTH IT, and the stated reason is wrong**
🟢 CatBoost docs *do* say `boosting_type=Ordered` "usually provides better quality on **small
datasets**", with the GPU default flipping to Ordered at **≤50k objects**. 🔴 **But that is small
*n*, and our n is 143,635 examples / 29.5M raw samples — we are nowhere near small-n.** Our
scarcity is **4 groups**, which ordered boosting does not address at all: it fixes *prediction
shift* from reusing the same rows for gradient estimation and leaf values, not group
generalization. Also 🟢 the **CatBoost paper's abstract makes no small-data claim whatsoever** —
that belief comes from the docs' parameter table, not the research. Neither library is
installed, so this carries an install + revalidation cost for a mechanism aimed at the wrong
axis. **Overturns the brief's "CatBoost's ordered boosting is designed for small data — worth
checking".**

### 7. Ensembling / stacking / blending — **CONFIRMED SCEPTICAL (parent's prior upheld)**
🟡 Buys fit. Fit is not the constraint; ranking is. NGRAM-FE is the precedent: **+0.0899 full-model
gain, served geometry 0.852 → 0.164.** Any capacity increase must clear the served-frame guard,
and the two things that *did* increase capacity here (NGRAM-FE features; my own monotone arm's
different bias) both failed. Not tested — declared in the prereg as not-worth-testing.

### 8. Hyperparameter search in general — **CLOSED BY PRIOR EVIDENCE, do not re-run**
🟢 The 99-arm sweep (85 two-knob stacks + 14 singles, 20 seeds) found **no arm beating the
per-surface peak under BOTH BH-FDR q=0.05 and Bonferroni**. Re-running it bigger is the same bet,
not a new technique. ⚠ **One correction:** the brief justified this partly with *"tune.py
--objective lolo CANNOT SCORE AT ALL"* — that is **false on this frame** (see refutations), so
"tuning is blocked on data" is not an available reason. The sweep's null stands on its own
evidence; the *stated reason* was wrong.

### 9. Target encoding — **ACTIVELY DANGEROUS HERE**
🟡 Any layout-keyed encoding is a memorization key when one layout dominates. This is the OQ-1
frequency-feature finding in another costume (frequency was measured to corrupt cross-layout
ranking and is deliberately excluded as a feature). Do not.

### 10. Pseudo-labelling / augmentation / TTA — **NOT AVAILABLE**
🟢 Cannot fabricate keystroke data. On the parent's specific suggestion of **hand mirroring**:
do not use it as an augmentation, and separately do not use it as a *test* — the parent's own
correction #2 records that a left-right mirror "maps the finger-index ordering onto itself and
cannot move a direction metric by construction", and that mirroring **does** move
`lsb`/`sfb-dist`/`sfs-dist` by up to 7.94pp, so the feature frame is **not** mirror-invariant.
A mirrored row is therefore a *differently-featured* row, not a free label.

---

## THE TWO DEMONSTRATIONS (hard requirement 2)

### ARM-G — grouped CV: **PASSES its pre-registered criterion**
`arm_g_results.json`, `arm_g3_results.json`. Real aalto k31 bigram frame, X = **143,635 × 20**,
LOGRAT target space.

| splitter | believed CV MAE | honest LOLO MAE | **optimism** | picked |
|---|---|---|---|---|
| `kfold_noshuffle` **(SHIPPED)** | 0.224079 | 0.260151 | **+0.036072** | #5 |
| `kfold_shuffle` (naive "fix") | 0.196670 | 0.260151 | **+0.063481** ⚠ worst | #5 |
| `groupkfold_layout` (FIX) | 0.258051 | 0.258051 | +0.000000 (tautological) | #2 |

Robustness over 5 candidate-set seeds: optimism **+0.0323…+0.0363, positive 5/5**; grouped
selection better on 3, tied on 2, **worse on 0**; sign test **p=0.25 (not significant)**.

### ARM-M — monotone constraints: **FAILED, reported as a failure**
`arm_m_results.json`. 3 seeds × 4 folds through the **shipped `validate()`**, so these are the
harness's own numbers. Constraints: `distance +1, dy +1, same_finger +1, scissor +1, lsb +1,
wpm −1` (verified mapped to the right columns by assertion).

| metric | BASELINE | MONOTONE | Δ | gate |
|---|---|---|---|---|
| `tau_heldout` mean/min | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0 | gate1 **PASS** (tie) |
| LOLO mean rho/ceiling | 0.9145 | 0.9005 | **−0.0139** | gate2 **FAIL** (bar 0.005) |
| **wmae** | 9.9382 | 15.9682 | **+60.68%** | gate3 **FAIL** (bar +0.91%) |
| umae | 16.2658 | 21.6237 | +32.9% | — |

**ADOPT = False.** Per-fold rho/ceiling: azerty .9916→.9786, dvorak .8853→.8578,
qwerty .7856→.7860, qwertz .9954→.9799.

**Interpretation (🟡):** the constraints preserve the *ordering* but wreck *magnitude
calibration* — and magnitude is load-bearing, because fitness is a weighted **sum**, so the
optimizer is invariant only to **affine** transforms, not monotone ones (`validate.py:150` says
exactly this). **This is the mirror image of the NGRAM-FE trap:** NGRAM-FE improved fit and
destroyed ranking; ARM-M held ranking and destroyed fit. XGBoost's own docs predicted the
mechanism — constraints under `hist` "may produce unnecessarily shallow trees" — and production
is already `max_depth=3`, so there was little structure to spare.

⚠ **Honest limit on gate1, which I must not let pass as a win:** the baseline `tau_heldout` was
**already saturated at 1.0**, so "ranking not degraded" could only tie or break. It is **not**
evidence the constraints help ranking. With 4 layouts, Kendall tau over 4 items is too coarse to
be the discriminating guard — the same defect `keybo.testkit.assert_discriminating` exists to
catch.

---

## REFUTATIONS OF MY OWN BRIEF (the parent invited this; all re-derived)

🔴 **1. Every figure in the brief's 🟢-tagged data block was wrong.**

| quantity | brief | **measured** |
|---|---|---|
| layouts | 11 | **4** (azerty, dvorak, qwerty, qwertz) |
| participants/layout | exactly 1, set `{1}` | **{64, 166, 485, 54690}** |
| rows | 3,098 | **2,202** |
| samples | 467,579 | **29,532,228** |
| bigram-placement trigram feats | 19 of 46 | **38 of 46** (19 `bg1_` + 19 `bg2_`) |
| `EXPECTED_SIGN` priors | "registered per gauge" here | **absent from keybo** — it is in the *agent tooling's* `evidence_scorer.py` |

The "11 labels / 1 participant each" frame is the **COMMUNITY** dataset (12 labels, **7**
participants), a different file. **Direction matters: real LOLO fold count is 4, not 11, so the
brief's CONCLUSION — volume-hungry Kaggle technique is mismatched — gets STRONGER while its
constants are discarded.** In-tree contradiction that was one grep away the whole time:
`train.py:63` names a **"64-participant layout"**, and dvorak has exactly 64.

🔴 **2. "`tune.py --objective lolo` CANNOT SCORE AT ALL on this data" is FALSE on the aalto
frame.** `split_half_ceiling` returns finite values on **4/4 folds**: azerty **0.8751**, dvorak
**0.7892**, qwerty **0.9906**, qwertz **0.9211**. True only of the *community* frame. So
"tuning is blocked on data, not effort" is not an available argument for the frame that matters,
and the `ObjectiveNotEvaluated` guard shipped today is a guard against the community frame, not
an epitaph for the objective. (Registered prediction before looking: "≥3 of 4 finite". Got 4/4.)

🟡 **3. "`RandomizedSearchCV(..., cv=5)` = random K-fold" is mechanically wrong.**
`check_cv(5, y, classifier=False)` → **`KFold(shuffle=False)`**, and `is_classifier(XGBRegressor)`
is False so there is no stratification either. It is **contiguous-block, ungrouped**. "Ungrouped
/ leaks" = TRUE; "random" = FALSE — and the difference is load-bearing, because it is exactly why
`shuffle=True` makes things **worse** (finding 🔴 above). A brief that said "random K-fold" would
have led straight to the harmful fix.

🟠 **4. A second defect the brief did not mention:** `build_training_matrix` defaults
`target_space="MS"`, but every shipped k31 model is **LOGRAT** (verified in the meta sidecars).
So the `cv-mae` path is doubly mismatched — wrong CV grouping **and** wrong target space. I
neutralized this in ARM-G by passing LOGRAT explicitly, so ARM-G tests the *split* alone.

---

## WHAT I KILLED OF MY OWN (self-separation, hard requirement)

1. 🔴 **A wrong constant in my own draft, caught pre-publication.** I wrote that the leaked path
   picks **depth 5** while the grouped path picks **depth 3**, and was about to bind it to the
   ledger's own "default depth-5 lost ~0.06 rho/ceiling to depth-3 while winning CV fit"
   (`tune.py:136`) as elegant confirmation. **Both picks are `max_depth=5`.** The real difference
   is **learning_rate 0.1337 → 0.0431** (and subsample 0.903 → 0.748). The story "grouped CV
   prefers less effective capacity" survives; the **knob was wrong**. This is the campaign's
   pattern exactly: the conclusion pointed the right way, so the number felt safe.
2. 🔴 **I fed the parent a wrong number.** I reported the community set as "**9** distinct
   participants" from a naive `label.split('#')[1]`; the `+pseudo`/`+rareboost` suffixes are
   corpus tags on one submitter. Truth is **7** (registered `pids` map; `README.md` line 40 says
   "7 submitters"). The parent propagated my 9 into a fleet-wide correction before it was caught.
3. ⚠ **I killed my own "0.81% better selection" headline.** One candidate draw showed grouped
   selection 0.81% better; five draws gave **p=0.25**. I demoted it from a result to a direction.
4. ⚠ **I refused to bank two tautological zeros.** `GroupKFold`'s optimism `0.000000` and its
   regret-vs-oracle `0.000000` are definitional at 4 groups, not measurements. Flagged in place.
5. ⚠ **My own instrument produced a FALSE DONE.** I armed a watcher on
   `pgrep -f "[a]rm_g_groupcv.py"`, then renamed the driver to `arm_g2.py` after killing v1 — so
   absence-of-process read as success and it fired "both arm drivers exited" while ARM-G2 was
   still running. Caught by checking `pgrep` before harvesting. Re-armed gated on the completion
   **marker** (`grep -q "^wrote …"`), with process-absence-without-marker mapped to an explicit
   FAILED. **Renaming a producer invalidates every watcher keyed to its old name.**

**Hostile re-read of my #1 recommendation:** is grouped CV premised on data volume we lack? No —
it is *more* important at 4 groups than at 400, because each group is a quarter of the data.
Would NGRAM-FE's guard have caught it? Not applicable: it changes scoring, not what is served.
Is it a *win*? **No — it is hygiene with ~zero blast radius on registered numbers**, and I
report it that way rather than inflating it.

---

## COST NOTE (all runtimes measured UNDER CONTENTION — not clean-box costs)
The host was at **load 414→745 on 192 cores** (~3× oversubscribed, sibling agents) throughout.
- TSV parse: **~236–290 s** (29.5M `ast.literal_eval`'d sample tuples; ~13,415 fields/row). Cached to `.npz` after the first pay.
- ARM-G v1 with `n_jobs=-1`: **killed at 27 min with 0 of 3 splitters done** — self-starving. Rerun with `OMP_NUM_THREADS=8`: **~180–290 s per splitter**. Pin threads on a shared box.
- ARM-M (2 arms × 12 fold-seed `validate()` runs + bootstrap CIs): **~35 min**.
- ARM-G3 (5 seeds × 8 candidates × 2 estimators): **~9 min**.

---

## ARTIFACT INDEX

| Artifact | What it is | Verdict/number |
|---|---|---|
| `arm_g_results.json` | ARM-G main run, 3 splitters | optimism +0.0361 / +0.0635 / 0.000 |
| `arm_g2.out`, `arm_g2.py` | its log + driver | leak: 4/4 folds straddle a layout |
| `arm_g3_results.json` | 5-seed robustness + sign test | 5/5 positive optimism; selection p=0.25 |
| `arm_g3.out`, `arm_g3_robust.py` | its log + driver | mean +0.44% (not significant) |
| `arm_m_results.json` | ARM-M monotone vs baseline | **ADOPT=False**, +60.68% wmae |
| `arm_m.out`, `arm_m_monotone.py` | its log + driver | gate2+gate3 FAIL |
| `ceiling_probe.json`, `.log`, `probe_ceiling.py` | split-half ceilings, all 4 folds | **4/4 finite** — refutes brief |
| `probe_bigram.py` | frame measurement | 4 layouts, pids {64,166,485,54690} |
| `cache_matrix.py` | `.npz` matrix cache | X=143,635×20 LOGRAT |

**In-worktree (committed, not pushed):** `8168a82` prereg, `6b38467` drivers, at
`/tmp/kaggle/agent-artifacts/kaggle-techniques/`. Every driver's first act is
`assert_module_under('keybo','/tmp/kaggle')`; the shared clone's `.venv` carries an editable
`.pth` into `repos/keybo/src`, so **without a `PYTHONPATH` prefix a probe silently tests the
wrong tree** — verified live in this environment.

## RECOMMENDED NEXT ACTION (one line, user-gated)
Apply `GroupKFold(groups=layout)` to `tune_hyperparameters` **and** fix its `target_space="MS"`
default — ~10 lines, no runtime cost, no served-frame risk — while labelling it a **latent-defect
fix with ~zero registered-number blast radius**, not a model improvement. Do **not** apply
`shuffle=True`. Do **not** adopt monotone constraints.
