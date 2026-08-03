# FREQCORRECT-1 — Artifacts Index

**Arm:** is the geometry/practice decomposition *CORRECT* (not merely ranking-invariant)?
**Verdict:** **CORRECT with ONE named defect.** The geometry/practice **attribution** is right —
`R_encode = 1.0614` CI95 [0.9761, 1.1656] (`b` encodes almost exactly the frequency dependence an
independent geometry-differenced measurement licenses) and `R²(b ~ served geometry) = −0.0151`
out-of-fold (`b` holds **no recoverable geometry**, surviving an equal-`n` falsifier). **The defect:**
the **"layout-independent"** label is overstated — `b` fitted on qwerty-only vs non-qwerty-only
disagrees beyond matched noise (`corr_true` **0.6682**, below the sample-matched floor p05 **0.9082**)
**at 1.249× matched-noise rms**.
**Biggest correction shipped:** the published `+9.906 d_wmae` for "restoring `b`" is an **arithmetic
double-count** (`_predict_cells` already adds `b`); the real comparison `g+b` vs `g` is **−16.1382,
12/12 cells, 4/4 folds** — `b` HELPS.
**My own registered predictions: 3 of 3 FAILED**, all reported as results (H-SATURATED refuted;
the level-share prediction refuted; the equal-`n` shrinkage and the artifact prediction both refuted).

Worktree: `/local/home/zegertho/repos/keybo-wt-freqcorrect` (branch `freqcorrect` off `origin/main`
583fc5f). Drivers: `drivers-freqcorrect/` — committed **before** each run (`/tmp` is tmpfs here).
Runner: `/home/zegertho/repos/keybo/.venv/bin/python` (the repo venv; a bare `python3` has no numpy).
Row cache: `/tmp/freqcorrect-drv/cache/rows_bi2_<size>_<mtime>.pkl` (409 MB, saves the 230 s TSV
parse; keyed on source size+mtime so a replaced TSV can never be served stale; regenerable —
deliberately NOT under `state/`).

**Causal order (verifiable in git):**
- PREREG `cdedc8f` @ **2026-08-03T16:54:40Z** — before any number of mine existed.
- ADDENDUM `31c9178` @ **2026-08-03T16:57:54Z** — the double-count, registered **before** measuring it.

## Runs

### Q01 — H-SATURATED, the negative control, and the double-count discriminator — 🟢 VERIFIED
- **`/local/home/zegertho/agent/state/freqcorrect/artifacts/q01_saturated.json`** (311.5 s)
- Driver: `drivers-freqcorrect/q01_saturated.py`; log `/tmp/freqcorrect-drv/q01.log`
- **N1 NEG CONTROL 🟢 PASS at |diff| = 0.000e+00 (EXACT)** — reproduced FREQGEO-1's
  `B(candidate) = −0.12673429794113286` (seed-mean `b` from the 3 SHIPPED `k31` metadata blobs,
  trigram first-transition marginal weighting); coverage mass 0.914210134 also matches.
- **P1 🟢 bijection HOLDS** — max **1** geometry per (layout, ngram). 724 ngrams; distinct-geometry
  hist [_,211,369,132,12] (211 = 29.1% at a single geometry across all layouts); layout-count hist
  [_,163,70,65,426].
- **P2 🔴 MY OWN H-SATURATED REFUTED** — R²(`b` ~ 19 served geometric cols), out-of-fold GroupKFold on
  ngram: **GBM −0.015122**, OLS +0.005593, **in-sample GBM 0.246063** (the gap is pure overfitting).
  Registered bar was ≥0.30 geometric / <0.10 refuted.
- **N3 🟢 placebo PASS** — `b` refit on shuffled ngram labels: R² **−0.091525** (instrument sound).
- **P3 🟢** geometric R² ≤ 0 at every `k ∈ {0, 10, 100, 1000, 10000}` (−0.0205 … −0.0989) ⇒ not a
  shrinkage artifact. sd(b) falls 0.3322 → 0.0781 across that sweep.
- **A1 🟢🟢 DOUBLE-COUNT CONFIRMED** — worst `|pred_path − g·exp(b)| = 1.1369e-13 ms` (bar 1e-6);
  **5854/5854 cells (100%) carry a `b`**; mean pred path 150.9645 ms vs `g`-alone 153.0975 ms.
- Shipped seed-mean `b`: n=724, mean **+0.037614**, sd 0.185496, range [−0.490475, +0.751974].
  ⚠ Note the freq-weighted `B = −0.1267` has the **opposite sign** to the unweighted mean — frequent
  bigrams get negative `b`. "The level" depends entirely on the weighting.

### Q02 — INVARIANT B: the matched-geometry contamination test — 🟢 VERIFIED (with one named caveat)
- **`/local/home/zegertho/agent/state/freqcorrect/artifacts/q02_matched.json`** (182.4 s)
- Driver: `drivers-freqcorrect/q02_matched.py`; log `/tmp/freqcorrect-drv/q02.log`
- Design: 5188 groups on **(layout, wpm-bucket, EXACT 19-col served geometry)**; **630 usable**
  (≥2 cells + frequency varies), 1296 cells, sizes 602×2 / 20×3 / 8×4. Median within-group log-freq
  spread **1.1624** of total range **11.0114** (**10.6%** — bounds the extrapolation).
- **PRIMARY 🟢 `β_freq` = −0.065084**, bootstrap **CI95 [−0.076199, −0.054053]** over GROUPS
  (10,000 draws, boot sd 0.005633) ⇒ **EXCLUDES 0**: at matched geometry, frequent bigrams are faster.
- **N4 / MEASURED FLOOR 🟢** — within-group frequency permutation, 2000 draws: null mean **−1.23e-06**
  (centred), sd 7.028e-03, **p95|β| = 1.385e-02**. **|β|/floor = 4.70×**, permutation p = 0 (<1/2000).
  *Floor measured for THIS design, never borrowed.*
- **`R_encode` 🟢🟢 = 1.0614, CI95 [0.9761, 1.1656]** (paired group bootstrap) — `β(b~logfreq)`
  = −0.069078 vs `β(obs~logfreq)` = −0.065084. Registered mis-attribution band ≥2 or ≤0.5 ⇒ **CLEAR**.
- **🟠 CAVEAT (largest in the arm):** within group `β(log n ~ log-freq) = 1.0051` — near-perfect
  collinearity; partialling out `log n` moves `β_freq` to **+0.024502** (sign flip). Frequency and
  per-cell sample count are **not separately identified** in this design. Participant Jaccard within
  group: median **0.127**, mean 0.191 (710 pairs) ⇒ participant mix is a live, uncontrolled confound.
- **ms/char conversion** (derived in `report.md`, not in the JSON): `exp(−0.065084)` = 0.9370×
  ⇒ **−6.30% per log-freq unit** ≈ **9.5 ms/keystroke** at the 150.96 ms mean — but **0.000 ms/char**
  against equal-coverage margins (`B_spread` = 0.0 exactly), and ~1.1 ms/char cross-coverage.

### Q03 — INVARIANT C: the 12-model LOLO and the corrected three-way comparison — 🟢 VERIFIED
- **`/local/home/zegertho/agent/state/freqcorrect/artifacts/q03_lolo.json`** (278.9 s)
- Driver: `drivers-freqcorrect/q03_lolo.py`; log `/tmp/freqcorrect-drv/q03.log`
- Config reproduces CALIB-1 `k03` exactly: 4 folds × 3 seeds, `ROW_STAGGERED_31`,
  `wpm_lo=40 hi=140 bucket=20 min_cell_samples=10`.
- **N2 NEG CONTROL 🟢 PASS at |diff| = 0.0000** — my `g+2b` vs `g+b` mean `d_wmae` = **+9.906534**
  vs published **+9.906533936853519**; azerty **12.565848804197481** and dvorak **7.604408050330945**
  exact to 15 digits (qwerty 6.216110 vs report's ~6.223, qwertz 13.239769 vs ~13.24).
- **🟢 THE INVERSION:** `b` **ONCE** (`g+b` vs `g`) = **−16.1382** (sd 2.3730), `d_umae` −12.3707,
  **12/12 cells better, 4/4 folds non-worse**. `b` **TWICE** (`g+2b` vs `g+b`) = **+9.9065**, 0/12.
  `g+2b` vs `g` = −6.2317, 12/12. ⇒ registered **A3 alternative HELD**: `b` helps at its fitted
  magnitude, overshoots when doubled.
- **C1 🔴 MY registered prediction FAILED** — level share **−0.129** (predicted ≥0.60): re-centering
  `b` makes the doubled penalty **worse** (+9.9065 → +11.1884) ⇒ CALIB-1's "`b` is a level shift" is
  **REFUTED**; the penalty is `b`'s *structure* double-applied.
- **C2 🟢** penalty is **100% on SEEN ngrams, exactly 0.0000 on unseen** (~67 cells/fold carrying
  **0.0001** of corpus mass) ⇒ the `b=0` default for unseen ngrams is a built-in placebo firing as
  designed; mis-transfer is **not** the mechanism.
- **🟢 `b` CALIBRATES the surface** — bucket-centered slope `g` → `g+b`: azerty 0.9963→1.0423,
  dvorak 0.4419→0.9248, qwerty 1.6207→**1.4067**, qwertz 1.1640→1.0217 (all four move toward 1.0).
- **S5 🟠 COINCIDENTAL-OR-UNRESOLVED (n=4, mechanism check NOT a test, flagged in the JSON itself)** —
  corr(slope_c, `b`-exposure) **−0.869** vs corr(slope_c, train/test ratio) **−0.892**: 0.023 apart.
  ⚠ qwerty has the **LOWEST** `b`-exposure (0.1459 vs 0.1839–0.2062) and the lowest train/test ratio
  (1.21 vs 3.16–6.33) ⇒ the exposure ordering points **against** "worst-calibrated on the practised
  layout".

### Q05 — the EQUAL-`n` FALSIFIER for my own INVARIANT B result — 🟢 VERIFIED, **§(b) SURVIVES**
- **`/local/home/zegertho/agent/state/freqcorrect/artifacts/q05_equaln.json`** (390.5 s)
- Driver: `drivers-freqcorrect/q05_equaln.py`; log `/tmp/freqcorrect-drv/q05.log`
- **Registered in `PREREGISTRATIONS.md` FREQCORRECT-1 ADDENDUM 2 (`b51e7e1` @ 19:37:26Z) BEFORE
  running**, with the decision rule written in the direction that costs me the result.
- Design: every cell in a matched-geometry group subsampled **without replacement** to the group's
  **minimum** raw-sample count; target = IQR-mean of the **drawn samples only**. Closes the
  noise-attenuation channel **by construction** rather than by regression adjustment. Common `n`:
  min 10, median 67.5, max 67029; **55.8% of samples discarded** (mean retained 0.442).
- 🟢 baseline reproduced in-process: `β_freq` = **−0.065084** (= q02's value).
- 🟢🟢 **EQUAL-`n` `β_freq` = −0.065400 ± 0.001189**, CI95 over 200 draws **[−0.067680, −0.063332]**;
  **shrinkage vs full-sample ×1.0049 — NO SHRINKAGE AT ALL.**
- 🟢 **re-MEASURED floor** (400 within-group permutations, NOT reused from N4): null mean −6.606e-04,
  sd 6.986e-03, **p95|β| = 1.396e-02** (vs q02's 1.385e-02 ⇒ the design's resolution did not degrade).
  **|β|/floor = 4.68×**, permutation p = 0. Placebo (equal-`n` targets + permuted freq) centred on 0.
- 🟢 **REGISTERED VERDICT: INVARIANT B SURVIVES the equal-`n` falsifier.** ⇒ the frequency effect at
  matched geometry is **not** a sample-count artifact; the q02 partialling sign-flip was collinearity
  absorbing the regressor.
- 🔴 **MY registered prediction FAILED (2nd of 3):** I predicted `β` would **shrink**; it came back
  **0.5% larger**. Registered in advance so an attenuated estimate could not be sold as confirmation.

### Q04 — INVARIANT D/E: cross-layout `b` transfer, first pass — ⚠️ **ITS FLOOR IS SUPERSEDED BY Q06**
- **`/local/home/zegertho/agent/state/freqcorrect/artifacts/q04_consequence.json`** (623.6 s)
- Driver: `drivers-freqcorrect/q04_consequence.py`; log `/tmp/freqcorrect-drv/q04.log`
- 🔴 **DO NOT QUOTE its `"DISAGREE BEYOND NOISE (contamination)"` verdict or its 7.01× rms ratio.**
  Its floor split-halves the QWERTY rows (both halves data-rich, ~49% of samples each) while the
  comparison pits **98.73% of samples against 1.27%** — not design-matched. Flaw registered in
  **ADDENDUM 3 (`5fc572a`)** before the correction was run; superseded by `q06`.
- 🟢 **SAFE parts:** the data-rich split-half reliability (12 splits, corr **0.991864** ± 0.001030,
  rms 0.020322) and the **INVARIANT D magnitudes**:
  `B(candidate)` full **−0.12673430** (`exp` 0.880968) vs geometric part **+0.01439795** (`exp` 1.014502);
  `B(qwerty)` full −0.13118422, geometric part +0.01381146; **cross-coverage residual full
  +0.00444992 log — reproduces FREQGEO-1's 0.004450 EXACTLY — of which only +0.00058649 (13.2%) is
  geometry-predictable ⇒ ≈0.15 ms/char of the ≈1.1.**
- 🟠 H1 sharp: `R²(Δb ~ Δgeometry)` oof **GBM 0.1294** / **OLS −0.0049**, placebo p95 −0.1309. Weak,
  nonlinear-only, and noise-dominated (Δb is dominated by `b_nonqwerty`'s noise). **Not a safe
  contamination claim.**
- ⚠️ **BUG WORTH KNOWING (cost one run):** the floor originally split the **ROW LIST**. A `StrokeRow`
  is unique per (layout, ngram) — **P1's bijection** — so the halves got **disjoint ngrams**,
  `set(b1) & set(b2)` was empty **by construction**, all 40 splits hit the `len(sh) < 20` guard, and
  `np.percentile` died on an empty list (the *"empty set intersection is the silent path into a nan
  cascade"* trap). Fix: split **samples within** each row; the driver now **asserts** the floor is
  non-empty rather than publishing a floorless comparison.

### Q06 — the CORRECTED cross-layout transfer test — 🟢 VERIFIED, **the arm's ONE defect**
- **`/local/home/zegertho/agent/state/freqcorrect/artifacts/q06_transfer.json`** (564.4 s)
- Driver: `drivers-freqcorrect/q06_transfer.py`; log `/tmp/freqcorrect-drv/q06.log`
- **Registered in ADDENDUM 3 (`5fc572a` @ 19:54:15Z) BEFORE running**, with my prediction written
  **against** my own draft finding.
- 🟢 **THE SAMPLE ASYMMETRY, measured: qwerty holds 98.73% of SAMPLES — 29,156,090 vs 376,138 (77:1).**
  This is why `q04`'s floor was invalid.
- **C-1** reliability of both sides (8 splits each): `rel_qwerty` **0.992421** ± 0.000666 ·
  `rel_nonqwerty` **0.950231** ± 0.003516.
- **C-2** disattenuation: `corr_true` = 0.6489 / √(0.9924·0.9502) = **0.6682** ⇒ registered rule ≤0.80
  ⇒ **GENUINE LAYOUT-SPECIFIC COMPONENT SURVIVES**. (Caveat in the JSON: half-size reliabilities
  undercorrect, so 0.6682 is a **lower bound** on the disattenuated agreement.)
- **C-3 SAMPLE-MATCHED floor** (qwerty subsampled to non-qwerty per-ngram counts, then split-halved;
  8 splits): corr **0.912131** ± 0.003153, **p05 0.908219**, rms 0.022458 — vs `q04`'s 0.991864.
  Cross-layout 0.6489 < p05 ⇒ **GENUINE NON-TRANSFER**.
- 🟢🟢 **C-4 PLACEBO — the most informative number in the arm, and it was only the placebo:** full
  `b_qwerty` vs a matched-**SIZE** *same-layout* subsample gives corr **0.762617**, rms **0.114017**.
  ⇒ **rms ratio cross-layout / matched-same-layout = 1.249×, NOT `q04`'s 7.01× (inflated ~5.6×).**
  Correlation drop 0.992 → 0.649 decomposes as **size 0.992 → 0.763**, **layout 0.763 → 0.649**.
- 🔴 **MY registered prediction FAILED (3rd of 3):** I predicted the non-transfer was artifactual; it
  is genuine. **But its magnitude is 1.25× matched noise, not 7×.**
- **READING (and the distinction matters):** a limit on the **"layout-independent"** label
  (`train.py:19-20`), **not** evidence geometry leaked into `b` — P2 refutes that out-of-fold.
  Practice that is itself partly layout-specific is a **different quantity** from geometry
  mis-attributed as practice.

## Reuse notes for the next agent
1. **`_predict_cells` ALREADY ADDS `b`** (`validate.py:574-577`). Any arm that "restores the practice
   term" by multiplying a `_predict_cells` output by `exp(b)` is applying `b` **twice**. This silently
   produced the published `+9.906`. Strip the practice block from a `deepcopy`'d metadata to get `g`.
2. **The served bigram frame is 20 columns and contains NO `log_freq`** (`features/schema.py`).
   FREQGEO-1's `f2`/`f3` SHAP numbers come from its own 21-column augmentation and are **not**
   properties of any shipped model.
3. **`candidate` is not in `keybo.layouts`** — `NAMED_LAYOUTS` doesn't carry it; reuse freqgeo's
   literal `"pyu.,vdfnlhieaocstrmkj'-qgwbzx"`.
4. **A bare `python3` lacks numpy** — use `/home/zegertho/repos/keybo/.venv/bin/python`.
5. **Frequency and per-cell sample count are collinear at slope ~1.0 within matched-geometry groups.**
   Any frequency-effect claim from this data must state whether it survives an equal-`n` design; the
   subsampling experiment is the standing top follow-up.
