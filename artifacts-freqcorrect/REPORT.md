# The **GEOMETRY/PRACTICE ATTRIBUTION is CORRECT**; the **"LAYOUT-INDEPENDENT" claim attached to it is WRONG BUT SMALL.** Net: **CORRECT with one named defect.**

**STRONGEST SINGLE PIECE OF EVIDENCE: `R_encode` = 1.0614, CI95 [0.9761, 1.1656].** The practice term
`b` encodes **almost exactly** the frequency dependence that an independent, geometry-differenced
measurement licenses — not 2× too much, not half. Combined with `R²(b ~ served geometry) = −0.0151`
out-of-fold (`b` holds **no recoverable geometry at all**, and that survives an equal-`n` falsifier),
**geometry has NOT been mis-attributed to practice.** That is the question the arm was asked.

⚠️ **THE ONE DEFECT I FOUND, and I am not letting the headline hide it:** `b` fitted on qwerty-only vs
non-qwerty-only data **disagrees beyond matched sampling noise** — disattenuated `corr_true` = **0.6682**
(registered rule: ≤0.80 ⇒ genuine), cross-layout corr 0.6489 below the **sample-matched** floor p05
**0.9082**. So `train.py:19-20`'s *"a layout-independent effect"* is **overstated**: the practice term
carries a **layout-specific component**. 🟢 **Its SIZE, from my own placebo, is 1.249× matched-noise
rms — not the 7.01× my first (mis-floored) run reported.** Size, not layout, drives most of the gap
(correlation drop 0.992 → 0.763 is size; only 0.763 → 0.649 is layout). **Ranking is untouched**
(cancellation follows from `b`'s *keying*, whatever `b` contains); **magnitude is touched only through
the already-bounded cross-coverage residual (≈0.15 ms/char of the ≈1.1).** Full treatment: **§(j)**.

⚠️ **THE PARENT'S READING OF MY OWN NUMBER IS WRONG, AND THIS IS THE MOST IMPORTANT CORRECTION IN THE
REPORT.** The parent read `β_freq = −0.0651` (CI excluding 0) as *"practice contamination MEASURED
DIRECTLY… the surface is NOT purely geometric."* **`β_freq` is not a contamination measurement. It is
a measurement of the SIGNAL `b` EXISTS TO CAPTURE.** It says frequency-dependent timing is real at
matched geometry — i.e. **the estimand is legitimate**, so modelling practice is *justified*.
Contamination would be `b` absorbing geometry (**tested: R² = −0.0151, refuted**) or `b` encoding the
wrong amount of the effect (**tested: `R_encode` = 1.06, refuted**). **A nonzero `β_freq` with
`R_encode ≈ 1` is the signature of a CORRECT decomposition, not a contaminated one.** Reported this
way in the ledger too, because the opposite reading inverts the arm's conclusion.

⚠️ **AND THE +9.906 d_wmae — the arm's loudest "correctness signal" — IS AN ARITHMETIC DOUBLE-COUNT.**
`_predict_cells` (`validate.py:574-577`) **already adds `b`**; its docstring says so (*"g(geometry,
wpm) + b(ngram) per cell — the model's full prediction"*). So CALIB-1's `base` was already `g+b` and
its `practice_b` was **`g+2b`**. I reproduced +9.906533936853519 to **|diff| = 0.0000** by deliberately
re-running the double-count, then ran the comparison never run: **`g+b` vs `g` = `d_wmae` −16.1382,
better on 12/12 fold×seed cells and 4/4 folds.** Adding the practice term **once cuts held-out
magnitude error ~64%**. It only hurts when applied twice.

---

## 0. CORRECTIONS TO MY BRIEF AND TO THE PARENT'S FRAMING, FIRST

| claim as given to me | truth | 🟢 |
|---|---|---|
| "restoring `b` WORSENS held-out error, mean d_wmae **+9.906**, 0 of 12 better" | that arm is `g+2b` vs `g+b`. The real question, **`g+b` vs `g`, gives −16.1382, 12/12 better, 4/4 folds**, `d_umae` −12.3707 | VERIFIED `q03` |
| CALIB-1: "`b` is a **level shift** that helps the contrast and hurts the level" | **REFUTED.** Re-centering `b` to zero mean makes the doubled penalty **WORSE** (+9.9065 → +11.1884), so it is `b`'s **structure** double-applied, not its level. (The freq-weighted `B = −0.127` and the unweighted `mean b = +0.0376` have **opposite signs**, so "the level" depends entirely on the weighting) | VERIFIED `q03` |
| FREQGEO-1: "R²(log-freq ~ geometry) = 0.0328/0.1139; `log_freq` ranks **3rd** by mean\|SHAP\|" | from a **21-column** matrix = `BIGRAM_FEATURE_NAMES + ["log_freq"]`, freqgeo's own augmentation. The **served frame has 20 columns and NO `log_freq`** (measured: `log_freq present = False`). Properties of an experimental model, **not of any shipped `k31` artifact** | VERIFIED `schema.py` + `q01` |
| PARENT (this message): `β_freq` ≠ 0 ⇒ "practice contamination measured directly; the surface is NOT purely geometric" | **`β_freq` measures the SIGNAL, not the defect.** See the banner above. Contamination is `R²(b~geom)` (−0.0151) and `R_encode` (1.06) — **both clear** | VERIFIED `q01`+`q02` |
| PARENT: "only 29.1% of ngrams have a single geometry" *(offered as weakening the bijection)* | those are **different facts**. The **bijection is per (layout, ngram) and HOLDS at max = 1**; the 29.1% is how many of 724 ngrams occupy one geometry *across all four layouts*. The 29.1% doesn't weaken the bijection — and the bijection is what makes `b` *able* to absorb geometry within a layout (which P2 then shows it does **not** do) | VERIFIED `q01` |
| my own **H-SATURATED** (my registered primary hypothesis) | **REFUTED BY ITS OWN THRESHOLD**, R² = **−0.0151** oof (bar ≥0.30 geometric / <0.10 refuted) | VERIFIED `q01` |

Causal order is in git: prereg `cdedc8f` @ **16:54:40Z** (before any number of mine existed);
double-count addendum `31c9178` @ **16:57:54Z** (before measuring its consequence).

---

## (a) INVARIANT A — THE THREE CLAIMS, SEPARATED. Every finding names its row.

| # | claim | verdict | which findings bear on it |
|---|---|---|---|
| **(i)** | **RANKING** — does `b` change which board wins? | **NO for equal coverage.** FREQGEO-1's `B_spread = 0.0`; I reproduced `B(candidate)` at **\|diff\| = 0.0 exactly** | N1 **only**. Nothing else here touches ranking. |
| **(ii)** | **CORRECTNESS** — is the split RIGHT? | **ATTRIBUTION CORRECT; the "layout-independent" LABEL is WRONG but small** | P2/P3/N3 (`b` is not geometric) · `R_encode` = 1.06 (`b` encodes the right amount) · INVARIANT B + `q05` (the estimand is real, survives equal-`n`) · A1/C1/C2 (the +9.906 is a double-count) · **`q06` (the one defect: `corr_true` 0.6682, at 1.249× matched noise) — §(j)** |
| **(iii)** | **MAGNITUDE** — are the absolute ms/char right? | **`b` IMPROVES them, and substantially** — held-out `wmae` 28.74 → **9.12 ms**; bucket-centered slope moves toward 1.0 on **all four folds** | the `g`/`g+b`/`g+2b` table; §(e) |

**What I did NOT show:** nothing re-opens row (i). `b` cancels within equal coverage whether or not it
is correctly attributed — independent facts. That conflation was the parent's original error; I have
not repeated it in the other direction.

---

## (b) THE MATCHED-GEOMETRY TEST (INVARIANT B) — interval, measured floor, and **in ms/char**

**Design** (registered before measuring): group cells by **(layout, wpm-bucket, EXACT 19-column served
geometry vector)**; regress the LOGRAT target on log-frequency **within** group; pool within-group
slopes. Fixed-effects — every between-group geometric difference is differenced out **by construction**.

| quantity | value |
|---|---|
| groups total / **usable** (≥2 cells, frequency varies) | 5188 / **630** (1296 cells; sizes 602×2, 20×3, 8×4) |
| median within-group log-freq spread | **1.1624** of a total range of **11.0114** (**10.6%** — a NARROW slice; see the extrapolation bound below) |
| **PRIMARY `β_freq`** | **−0.065084** log-units per log-freq unit |
| **bootstrap CI95 over GROUPS** (10,000 draws) | **[−0.076199, −0.054053]**, boot sd 0.005633 |
| **MEASURED FLOOR** — within-group frequency permutation, 2000 draws (N4) | null mean **−1.23e-06** (centred), sd 7.03e-03, **p95\|β\| = 1.385e-02** |
| **\|β\| / floor** | **4.70×** · permutation **p = 0** (<1/2000) — *floor beside p, per §9* |

🟢 **RESULT: CI95 excludes zero. At matched geometry, more frequent bigrams are typed FASTER.** Per
the registered rule this makes **the estimand `b` targets LEGITIMATE** — practice-dependent timing is
real and measurable. **It does NOT by itself say the surface is contaminated** (see the banner).

### ⚖️ (2) `β_freq` IN A UNIT THAT CAN BE JUDGED — the parent asked, and it is the number that decides whether this MATTERS

`β_freq` is a slope in LOGRAT space (log of a duration ratio), so it converts multiplicatively.

| conversion | arithmetic | value |
|---|---|---|
| effect over **one natural-log unit** of frequency | `exp(−0.065084)` | **0.9370×** ⇒ **−6.30%** per log-freq unit |
| effect over the **usable within-group span** (median 1.1624) | `exp(−0.065084 × 1.1624)` | **0.9271×** ⇒ **−7.29%** |
| effect over the **full observed range** (11.0114) | `exp(−0.065084 × 11.0114)` | **0.4886×** ⇒ **−51.1%** ⚠️ **EXTRAPOLATED ~9.5× beyond the identifying span — quote with that caveat or not at all** |
| at the surface's mean predicted cell time (**150.96 ms/keystroke**, measured `q01`) | 150.96 × 0.0630 | **≈ 9.5 ms per keystroke per log-freq unit** |
| **as ms/char at the corpus level** | this is the key subtlety ⇒ see below | **≈ 0 for the ranking-relevant part** |

🔴 **THE CRUCIAL POINT THE ms/char CONVERSION EXPOSES, and it is why "does it matter?" has a
different answer than "is it nonzero?":** a per-keystroke effect of ~9.5 ms/log-freq-unit is **large**
in absolute terms — but its corpus-weighted aggregate **is exactly the quantity FREQGEO-1 proved
cancels**. `b` is keyed on ngram identity, so its frequency-weighted total `B` is **identical to 8+
decimals for every board of equal coverage** (I reproduced `B(candidate) = −0.12673429794113286` at
|diff| = 0.0). So:

- **against the ~0.3 ms/char top-cluster margins: the effect contributes 0.000 ms/char** — `B_spread`
  is **0.0 EXACTLY** within the equal-coverage group all ten campaign boards share. Ratio: **0/0.3.**
- **against the 1.05 ms/char live-pair margin (`candidate` vs `flagship-c3`): also 0.000 ms/char** —
  both boards are in that same coverage group (covhash `3a8cfc66a3b1`). Ratio: **0/1.05.**
- **the only place it lands is CROSS-coverage magnitudes** (e.g. qwerty-vs-field), where FREQGEO-1
  measured the residual at **0.004450 log units (bigram) / 0.001802 (trigram) ≈ 1.1 ms/char** — which
  is **~3.7× a top-cluster margin** and therefore *not* negligible **for percentage claims**.

⇒ 🟢 **So the honest answer is: the practice effect is BIG per keystroke (−6.3%/log-unit, 4.70× its
floor), EXACTLY ZERO for every equal-coverage ranking, and ~1.1 ms/char for cross-coverage
magnitudes.** The magnitude question and the ranking question have genuinely different answers, and
neither may be quoted for the other.

### ⚠️ (4) CONFOUNDS — controlled, measured, and the ones I cannot clear

| confound | how handled | result |
|---|---|---|
| geometry | **grouping key** (exact 19 served columns) | differenced out by construction |
| layout | **grouping key** | differenced out |
| wpm / typist pace | **grouping key** (bucket) + LOGRAT pre-factors pace | differenced out |
| **per-ngram sample count (noise ⇒ attenuation)** | partialled out | 🟠 **`β(log n ~ log-freq) = 1.0051` within group — near-perfect collinearity.** Partialling flips `β_freq` to **+0.0245** (shift +0.0896) |
| **participant mix** | **measured, not controlled** | within-group participant **Jaccard median 0.127**, mean 0.191 (710 pairs) ⇒ groups share **few** participants: a **live** confound |
| bigram position-within-word | **NOT CONTROLLED** — the frame carries no word context | named as open |
| word-level context | **NOT CONTROLLED** | named as open |
| participant skill × frequency interaction | **NOT CONTROLLED** | named as open |

🟠 **The sample-count collinearity WAS the largest caveat in this report.** At a within-group slope of
~1.0, "controlling for `n`" is closer to *deleting the frequency variable* than to controlling a
confound — **frequency and sample count are not separately identified by the partialling
instrument**, so a noise-attenuation mechanism (thin rare cells → attenuated IQR-means → apparent
slope) could not be refuted that way. **So I ran the falsifier against myself. It cleared.**

### 🟢🟢 THE EQUAL-`n` FALSIFIER — registered before measuring, in the direction that costs me the result, and **INVARIANT B SURVIVES IT**

Registered in **ADDENDUM 2 (`b51e7e1` @ 19:37:26Z)**, run afterwards. **Design:** subsample every cell
in a matched-geometry group *without replacement* to that group's **minimum** raw-sample count, and
recompute each cell's IQR-mean target from the drawn samples only. Every cell in a group then carries
the **same** sample count, so the attenuation channel is closed **by construction** rather than by
regression adjustment.

| quantity | value |
|---|---|
| common `n` per group | min **10**, median **67.5**, max 67029 |
| **fraction of samples DISCARDED** | **55.8%** (mean retained 0.442) |
| baseline reproduced in-process | `β_freq` = **−0.065084** (= q02's published value) |
| **`β_freq` under EQUAL-`n`** (200 draws) | **−0.065400 ± 0.001189**, CI95 **[−0.067680, −0.063332]** |
| **shrinkage vs full-sample** | **×1.0049 — NO SHRINKAGE AT ALL** |
| **re-MEASURED floor** for this design (400 permutations) | null mean −6.606e-04, sd 6.986e-03, **p95\|β\| = 1.396e-02** (vs q02's 1.385e-02 — resolution did not degrade) |
| **\|β\| / its OWN floor** | **4.68×**, permutation **p = 0** |
| placebo (equal-`n` targets + permuted frequency) | 🟢 centred on zero |

🟢 **REGISTERED VERDICT: INVARIANT B SURVIVES.** Throwing away **55.8% of the samples** and equalizing
`n` across every cell in every group leaves `β_freq` **statistically identical** (−0.0654 vs −0.0651).
If the slope were driven by thin-rare-cell attenuation, equalizing `n` would have collapsed it into
the floor. **It did not move.** ⇒ **The noise-attenuation explanation is dead, and the partialling
sign-flip was collinearity absorbing the regressor — as I argued, now demonstrated rather than
argued.**

🔴 **MY OWN registered prediction FAILED again (2nd of 3):** I predicted `β` would **shrink** because
subsampling discards data and adds target noise. It came back **0.5% larger** (×1.0049). I registered
the shrinkage direction in advance precisely so an attenuated estimate could not be sold as clean
confirmation — instead there was nothing to attenuate, and I report the failed prediction as such.

⚠️ **Correction to a brief number:** the brief cites freqgeo's "574 of 718 groups, median spread 5.936
= 42.7% of range". My grouping is **stricter** (adds layout + wpm bucket) → **630 usable of 5188**,
median spread **1.1624 of 11.0114 = 10.6%**. Not a contradiction, a tighter design — but **42.7% must
not be quoted as this test's leverage**, and the 10.6% is what bounds the extrapolation above.

---

## (c) INVARIANT C — WHAT THE +9.906 ACTUALLY IS

12 fold×seed cells, paired per-fold deltas (MOR-FIX-1, never a mean of ratios):

| arm | `d_wmae` (sd) | `d_umae` | cells better | folds non-worse |
|---|---|---|---|---|
| **`b` ONCE — `g+b` vs `g`** ← *never asked before* | **−16.1382** (2.3730) | **−12.3707** | **12/12** | **4/4 w, 4/4 u** |
| **`b` TWICE — `g+2b` vs `g+b`** ← *CALIB-1's `practice_b`* | **+9.9065** (3.1849) | +7.3869 | 0/12 | 0/4 |
| `g+2b` vs `g` | −6.2317 | −4.9838 | 12/12 | 4/4 |
| C1: **centered** `b` once, vs `g` | −15.8641 | −12.2327 | 12/12 | 4/4 |
| C1: **centered** `b` twice, vs `g+b` | **+11.1884** | +8.1274 | 0/12 | 0/4 |

Per-fold `wmae` (seed 0), `g` → **`g+b`** → `g+2b`, with the bucket-centered slope:

| fold | `g` | **`g+b`** | `g+2b` | slope_c `g` → **`g+b`** |
|---|---|---|---|---|
| azerty | 28.74 | **9.12** | 21.72 | 0.996 → **1.042** |
| dvorak | 29.33 | **13.46** | 20.88 | 0.442 → **0.925** |
| **qwerty** | 22.37 | **9.01** | 15.06 | 1.621 → **1.407** |
| qwertz | 23.47 | **8.22** | 21.58 | 1.164 → **1.022** |

**Adjudicating the rivals I registered:**

- 🟢 **A1/A2 — DOUBLE-COUNT: CONFIRMED, and it is the answer.** Worst
  `|pred_path − g·exp(b)| = 1.137e-13 ms` (bar 1e-6); **5854/5854 cells (100%) carry a `b`**.
  **N2 reproduces the published mean at |diff| = 0.0000**, with azerty **12.565848804197481** and
  dvorak **7.604408050330945** exact to 15 digits. My registered **A3 alternative** — *"if `g+b`
  BEATS `g` while `g+2b` loses to `g+b`, then `b` helps at its fitted magnitude and merely overshoots
  when doubled — the opposite of the brief's premise"* — **is what happened.**
- 🔴 **C1 — LEVEL SHIFT: REFUTED, and MY OWN registered prediction FAILED.** I predicted ≥60% of the
  penalty would be level; measured level share **−0.129** (re-centering makes it *worse*).
- 🟢 **C2 — MIS-TRANSFER: REFUTED as the mechanism.** The penalty is **100% on SEEN ngrams, exactly
  0.0000 on unseen**. Unseen ≈67 cells/fold carrying **0.0001 of corpus mass** ⇒ the `b = 0.0` default
  is a **built-in placebo firing exactly as designed**. The brief's worry (question 4: "an unseen
  ngram gets no practice correction on exactly the most novel boards") is **arithmetically true but
  empirically ~empty at this coverage** — 0.01% of mass.
- 🟢 **C3 — GENUINE GEOMETRIC SIGNAL ABSORBED: REFUTED by P2** (§(e)).
- **C4 — SOMETHING ELSE:** not needed; A1 accounts for the sign and, via N2, the magnitude to 4 dp.

---

## (d) (5) IS THE QWERTY-WORST-FOLD LINK TO PRACTICE REAL? — **COINCIDENTAL-OR-UNRESOLVED**, and the exposure ordering points the WRONG WAY

| fold | slope_c (`g`) | slope_c (`g+b`) | train/test cell ratio | **`b`-exposure** (corpus-wt mean \|b\|) | frac seen |
|---|---|---|---|---|---|
| azerty | 0.9963 | 1.0423 | 4.85 | 0.2062 | 1.000 |
| dvorak | 0.4419 | 0.9248 | 6.33 | 0.1905 | 1.000 |
| **qwerty** | **1.6207** | **1.4067** | **1.21** | **0.1459** ← **LOWEST** | 0.975 |
| qwertz | 1.1640 | 1.0217 | 3.16 | 0.1839 | 1.000 |

corr(slope_c, `b`-exposure) = **−0.869** · corr(slope_c, train/test ratio) = **−0.892**.

🟠 **Indistinguishable at n = 4 (0.023 apart).** Prereg §5 registered in advance that this cannot be
resolved statistically and that I would report **COINCIDENTAL-OR-UNRESOLVED**. That is the verdict —
the parent hoped this was "the strongest corroboration available"; **it is not available at n=4.**

🔴 **And the direction is adverse to the practice reading:** qwerty has the **LOWEST** `b`-exposure of
the four (0.1459 vs 0.1839–0.2062). If `b` were absorbing qwerty-specific muscle memory, the qwerty
fold should be the **most** `b`-exposed, not the least. Meanwhile qwerty's train/test ratio is **1.21**
against 3.16–6.33 — it trains on the fewest cells and is tested on the most. **Support is the more
parsimonious reading; "worst calibrated on the practised layout" is NOT supported by the exposure
ordering**, though n=4 forbids calling it refuted. ⚠️ Note also `b` *improves* qwerty's slope
(1.621 → 1.407), which is the opposite of what a qwerty-contamination story predicts.

---

## (e) IS `g` A VALID BIOMECHANICAL ESTIMATE? (brief question 1) — evidence says **YES, defensibly**

Four independent measurements looked for qwerty muscle memory baked into `g`; none found it:

| test | registered threshold | measured | verdict |
|---|---|---|---|
| **P2** R²(`b` ~ served geometry), **out-of-fold**, grouped by ngram | ≥0.30 geometric / <0.10 refutes | **−0.0151** (GBM), +0.0056 (OLS); **in-sample 0.2461** | 🟢 **`b` is NOT geometric**; the in-sample/oof gap is pure overfitting |
| **N3** placebo `b` on **shuffled** ngram labels | must be ~0 or the instrument is broken | **−0.0915** | 🟢 instrument sound |
| **P3** does a geometric share hide behind the shrinkage? | — | R² ≤ 0 at **every** `k ∈ {0, 10, 100, 1000, 10000}` (−0.020 … −0.099) | 🟢 not a shrinkage artifact |
| **P1** is the estimand identifiable at all? | — | **bijection HOLDS** (max **1** geometry per (layout, ngram)); 724 ngrams; 211 (29.1%) at a single geometry across all layouts; 426 span all four | 🟠 the *worry* is well-founded; the *outcome* is not |

🔴 **This refutes MY OWN primary hypothesis and I report it as the result.** H-SATURATED said a
per-ngram intercept is an unconstrained function of geometry within a layout. **P1 confirms that
premise is structurally TRUE** — nothing in the functional form *prevents* `b` from absorbing
geometry. **P2 shows it doesn't happen:** a booster with 19 geometric columns cannot predict `b`
out-of-fold at all. **Structural possibility ≠ realised defect**, and I had banked the former as the
latter.

⇒ **On brief question 3** ("`b` is soaking up CLASS-STRUCTURED variation — a concrete correctness
defect"): the *observation* is real (sfbprice 49%, CALIB-1 57.8%, per-class +12% to +101%) but the
*interpretation* does not survive. `b` closing a class-level gap is what a **correctly specified**
additive term does when the class contrast is partly a practice contrast. It is **not** evidence `b`
holds geometry, because out-of-fold **`b` holds no recoverable geometry** (R² = −0.0151), and
`R_encode = 1.06` says it holds the **right amount** of the frequency effect.

⇒ **On brief question 4** (is the estimand — a per-ngram string lookup with frequency only in the
shrinkage denominator — right *in principle*?): 🟠 **In principle it is the weakest part of the
design, and in practice it is not biting.** A lookup cannot extrapolate to unseen ngrams (b = 0) and
cannot pool statistical strength across similar ngrams. But unseen mass is **0.0001** (C2), and a
functional form `b(log-freq)` would impose exactly the −0.065 slope the data already shows `b`
matching at `R_encode = 1.06` — so it would buy **shrinkage efficiency, not correctness**. **A
correct-in-principle form exists; it is not warranted by evidence here.**

---

## (f) WHAT A CORRECT DECOMPOSITION WOULD CHANGE (INVARIANT D) — answered for the outcome I got

**It is not wrong, so: it already does.** What my results change are **published readings**, not the model:

1. **Absolute ms/char — `b` IMPROVES it, substantially.** Held-out `wmae` **28.74 → 9.12 ms** (mean
   −16.14 over 12 cells). Any claim "the absolute magnitudes are suspect *because of* the practice
   term" is backwards.
2. **`b` is what CALIBRATES the surface**, on all four folds (dvorak 0.442 → 0.925; qwerty
   1.621 → **1.407**). 🟠 So CALIB-1's "the compression is one fold" is measured on the **`g+b`**
   surface and would be **far worse without `b`**.
3. **Hours-per-year and qwerty-vs-field %: UNCHANGED by this arm — and now BOUNDED.** They rest on
   cross-coverage `B` differences. 🟢 **I reproduced FREQGEO-1's cross-coverage residual exactly:
   `B(candidate) − B(qwerty)` = **+0.00444992 log** (freqgeo: 0.004450) ≈ 1.1 ms/char.** Decomposing
   it: only **+0.00058649 log (13.2%)** is geometry-predictable, i.e. **≈0.15 ms/char** even if one
   granted that the geometric share is mislabelled. And the geometric part of `B(candidate)` is
   **+0.0144** against `B` itself at **−0.1267** — **opposite sign, ~11% of the magnitude**
   (`exp`: 1.0145 vs 0.8810). ⇒ **the worst case for the sensitive quantity is ~1/7th of an already
   small effect.** CALIB-1's 3.68% → 5.60% propagation stands; **I found no reason to move it.**
4. **The sfb shadow price: UNCHANGED, explanation corrected.** `b` closing 49–58% is an
   **estimand-matching** effect, as sfbprice said — and `R_encode ≈ 1.06` now supplies the missing
   justification for *why* matching the estimand is right rather than merely convenient.
5. **Unequal-coverage rankings: NOT re-opened.** No fix was warranted, so per prereg §7 I did **not**
   retrain, re-optimize, or re-evaluate the field. `data/models/k31/` never written; `layouts.py`
   untouched; no layout adopted or promoted.

---

## (g) NEGATIVE CONTROLS — all four registered, all four ran

| control | bar | result |
|---|---|---|
| **N1** reproduce FREQGEO-1's `B(candidate) = −0.12673429794113286` | \|diff\| < 1e-9 | 🟢 **PASS at \|diff\| = 0.000e+00 (EXACT)**; coverage mass 0.914210134 also matches |
| **N2** reproduce CALIB-1's `practice_b` mean `d_wmae = +9.906533936853519` | within 0.10 | 🟢 **PASS at \|diff\| = 0.0000**; azerty **12.565848804197481** / dvorak **7.604408050330945** exact to 15 digits |
| **N3** placebo `b` on shuffled ngram labels shows ~no geometric R² | <0.10 | 🟢 **PASS, −0.0915** |
| **N4** matched-geometry design on permuted log-frequency | β = 0 within CI | 🟢 **PASS**, null mean **−1.23e-06**; **doubles as the measured floor** (p95\|β\| = 1.385e-02) |

Two reproduce *other agents'* published quantities from an independent code path, **both exactly**.
That is what makes the double-count finding safe: **I did not fail to reproduce +9.906 — I reproduced
it perfectly, then showed what it measures.**

---

## (h) WHAT WOULD FALSIFY THIS CONCLUSION (INVARIANT E)

1. 🟢 **~~TOP RISK — the frequency/sample-count collinearity~~ — RUN AND CLEARED.** I registered this
   falsifier against myself (`b51e7e1`) and ran it: equal-`n` `β_freq` = **−0.065400** (×1.0049 of the
   full-sample value) at **4.68× its own re-measured floor**, after discarding **55.8%** of samples.
   **§(b) survives.** This was the one result that could have overturned the arm; it did not.
2. **`b` being geometric in a richer basis.** Refuted at R² = −0.0151 on the **served** 19 columns
   (the frame the shipped model uses); a wider basis (kitchen-sink / direction frames, interactions)
   is a legitimate re-test.
3. **The +9.906 not being a double-count** — refuted at 1.137e-13 ms with 100% of cells carrying `b`.
   I regard this as **closed**: an arithmetic identity, not a statistical inference.
4. **A fix beating the incumbent** on held-out cross-layout transfer (paired per-fold, ≥3/4 folds,
   both `wmae` and `umae`, matched-complexity placebo + reseed arms). **`g+b` vs `g` at −16.14 with
   12/12 sets the bar** such a fix must clear.
5. 🟠 **The `q06` defect could be re-read as benign.** If practice attaches to motor sequences rather
   than letter pairs, a layout-specific residual is **expected**, and the right conclusion is
   "the label is wrong, the model is fine". **What would decide it:** more layouts (n=4 today), or a
   within-layout instrument that separates "same bigram, different geometry" *inside* one layout —
   which P1's bijection makes **impossible by construction**, so it genuinely requires new data.
   ⇒ **My "correct with one named defect" verdict would become plain "correct" if that reading holds,
   and would harden into a real modelling error if a 5th layout reproduced the non-transfer.**

---

## (i) WHAT REMAINS OPEN

1. 🟢 **~~The equal-`n` subsampling falsifier~~ — DONE, and §(b) survived it** (see §(b), `q05`). No
   longer open.
2. 🟠 **The qwerty-fold mechanism, UNRESOLVED at n=4** (support vs exposure differ by 0.023). Only
   more layouts settle it — which is CLOSING-2's standing conclusion: **layout diversity binds.**
3. 🟡 **Everything here is the BIGRAM surface.** `Tc` (trigram) carries its own practice term,
   unaudited. CALIB-1 flagged the same gap for calibration.
4. 🟡 **CALIB-1's four fix-route rows should be RELABELLED in light of A1:** its `practice_b` and
   `practice_b_then_affine_heldin` rows are **`g+2b`** arms. The two affine rows don't touch `b` and
   are unaffected; its *conclusions* (don't install a rescale; the defect is layout-specific) do not
   depend on the `b` rows, so I do **not** claim they fall.
5. 🟠 **Participant mix uncontrolled** (within-group Jaccard median 0.127). A participant-fixed-effect
   version of §(b) is the clean fix; not run.
6. 🟡 **H-SHRINK (γ/α) unmeasured** — still CALIB-1's standing top follow-up, untouched here.
7. 🟠 **DONE, and it is the arm's one defect: `b` carries a LAYOUT-SPECIFIC COMPONENT** (`corr_true`
   0.6682 ≤ 0.80; below the sample-matched floor p05 0.9082) **at 1.249× matched-noise rms** — see
   §(j). **Still open underneath it:** *why*. My leading (untested) reading is that practice attaches
   to motor sequences as much as to letter pairs, which would make the residual real and not a defect.
   Distinguishing that from a true modelling error needs more than 4 layouts.
8. 🟡 **Whether the "layout-independent" wording in `train.py:19-20` and `:28` should be amended.**
   My evidence says the claim is overstated; the fix is a docstring/scope correction, not a model
   change. **I did not edit it** (out of scope: no code changes were warranted).
9. 🟢 **Nothing retrained, re-optimized, or promoted.** No human typed on any board.

---

## (j) ⚠️ THE ONE TEST THAT CUT AGAINST "CORRECT" — AND THE FLAW I FOUND IN MY OWN FLOOR FOR IT

**The test (`q04`).** A genuinely **layout-independent** practice term must give the **same** `b` when
fitted on qwerty-only data and on non-qwerty-only data — they estimate the same quantity. Measured:

| quantity | value |
|---|---|
| `corr(b_qwerty, b_nonqwerty)`, 545 shared ngrams | **0.6489** |
| rms difference | **0.142384** |
| `sd(b_qwerty)` / `sd(b_nonqwerty)` | 0.181606 / **0.090864** (factor ~2) |
| `slope(b_nq ~ b_q)` | **0.3247** |
| `q04`'s split-half-of-qwerty floor (12 splits) | corr **0.991864** ± 0.001030, rms **0.020322** |
| ⇒ `q04`'s emitted verdict | *"DISAGREE BEYOND NOISE (contamination)"* at **7.01×** the rms floor |

🔴 **I am NOT publishing that verdict, because I found its floor to be invalid — and I registered the
flaw (`5fc572a`) before running the correction.** Three measured reasons:

1. 🟢 **QWERTY is 98.73% of the SAMPLES — 29,156,090 vs 376,138, a 77:1 ratio.** So `b_nonqwerty` is
   estimated from **1.3%** of the data, while *both* halves of `q04`'s floor hold ~49% each. **A floor
   built from two data-rich estimates cannot bound the agreement of one data-rich and one data-poor
   estimate.**
2. 🟢 **Differential shrinkage, visible in the numbers above.** The estimator is
   `b = Σ(c·r)/(Σc + 100)`, so a low-count population is shrunk toward 0 **harder** — which is exactly
   what `sd` 0.0909 vs 0.1816 and `slope` 0.3247 show. Much of the "disagreement" is the shrinkage
   denominator, not the layout. (Correlation is scale-invariant so shrinkage alone can't explain
   0.6489 — but **differential noise can**, and that is what the mis-matched floor fails to capture.)
3. 🟠 The sharp half — `R²(Δb ~ Δgeometry)` out-of-fold = **0.1294** GBM (placebo p95 −0.1309) but
   **−0.0049 OLS** — is weak, nonlinear-only, and **inherits the same noise problem** (`Δb` is
   dominated by `b_nonqwerty`'s noise). **Not a safe contamination claim either.**

### THE CORRECTED TEST (`q06`, registered in ADDENDUM 3 `5fc572a`) — **the component is REAL, and `q04`'s magnitude was inflated ~5.6×**

| instrument | result | registered verdict |
|---|---|---|
| **C-1** reliability of BOTH sides (8 splits each) | `rel_qwerty` = **0.992421** ± 0.000666 · `rel_nonqwerty` = **0.950231** ± 0.003516 | non-qwerty *is* less reliable — but only modestly |
| **C-2** disattenuation `corr_obs / √(rel_q·rel_nq)` | 0.6489 / 0.9711 = **0.6682** | ≤0.80 ⇒ **GENUINE LAYOUT-SPECIFIC COMPONENT SURVIVES** |
| **C-3** the **SAMPLE-MATCHED** floor (qwerty subsampled to non-qwerty's per-ngram counts, then split-halved) | corr **0.912131** ± 0.003153 (p05 **0.908219**), rms 0.022458 — vs `q04`'s mis-matched 0.991864 | 0.6489 < p05 ⇒ **GENUINE NON-TRANSFER** |
| **C-4** placebo: full `b_qwerty` vs a matched-**SIZE** qwerty subsample — **SAME layout, so truth = perfect agreement** | corr **0.762617**, rms **0.114017** | **the decisive number — see below** |

🔴 **MY REGISTERED PREDICTION FAILED — the third of my three to fail, and I report it as the result.**
I predicted the non-transfer would prove to be mostly artifact. **It is real by both registered
instruments.**

🟢 **BUT C-4 — which was only my placebo — is the most informative measurement in the arm, and it
rescales the finding by 5.6×.** Comparing `b_qwerty` against a **same-layout** subsample of matched
size gives corr **0.7626** / rms **0.1140**. Cross-layout gives corr 0.6489 / rms 0.1424. So:

- **rms ratio against the properly size-matched SAME-LAYOUT comparison = 0.142384 / 0.114017 =
  1.249×** — **not** the **7.01×** `q04` reported against its data-rich floor.
- decomposing the correlation drop **0.992 → 0.649**: **SIZE** accounts for 0.992 → **0.763**;
  **LAYOUT** accounts for only 0.763 → **0.649**.

⇒ 🟠 **NET VERDICT: a layout-specific component in `b` is REAL but SMALL (1.25× matched-noise rms).**
Both of my registered rules fire "genuine", so I do not get to dismiss it — and `q04`'s headline
magnitude was inflated ~5.6× by a floor I built wrong, so nobody gets to quote 7.01× either.

**WHAT IT MEANS, stated carefully because two very different readings are available:** this is a limit
on the **"layout-independent"** label (`train.py:19-20` calls the practice effect *"a layout-independent
effect"*), **not** a demonstration that geometry leaked into `b`. **Practice that is itself partly
layout-specific is a different quantity from geometry mis-attributed as practice** — and the second is
what "contamination" means and what P2 refutes out-of-fold (R² = −0.0151; `q04`'s
`R²(Δb ~ Δgeometry)` = 0.1294 GBM but **−0.0049 OLS**, weak and nonlinear-only, and noise-dominated).
🟠 A plausible substantive mechanism is that typists' practice attaches to **motor sequences** as much
as to letter pairs, so the same bigram at a different geometry is genuinely a partly different
practised act — which would make the residual **real, expected, and not a defect**. I did not test
that and am not claiming it.

🟢 **What this does NOT touch, in either direction:** P2/P3/N3 (`b` holds no recoverable geometry
out-of-fold), `R_encode = 1.0614`, the A1 double-count identity, and INVARIANT B's survival of the
equal-`n` falsifier are all **independent** of `q04`/`q06`'s floor. **The line-1 verdict rests on
those.** If `q06` finds a genuine layout-specific component, the correct amendment is *"`b` is
correctly attributed as practice but is partly layout-SPECIFIC practice"* — a limit on the
`layout-independent` label, **not** a finding that geometry leaked into `b`.

---

## PROVENANCE

**Causal order verifiable in git.** Branch `freqcorrect` @
`/local/home/zegertho/repos/keybo-wt-freqcorrect` (off `origin/main` 583fc5f):

| commit | UTC | what |
|---|---|---|
| `cdedc8f` | **2026-08-03T16:54:40Z** | **PREREG — before any number of mine existed**, incl. the FREQGEO SHAP correction (found by reading code) |
| `31c9178` | **2026-08-03T16:57:54Z** | **ADDENDUM — the double-count, registered BEFORE measuring it**, with its numeric discriminator and the reading under which it would be REFUTED |
| `3d96207`/`443f142`/`97350de`/`6115999` | — | drivers `q01`–`q04`, each committed **before** running (`/tmp` is tmpfs; a reboot already wiped a worktree here) |
| `8051c6f` | — | N1 fixed to freqgeo's actual recipe (shipped-metadata `b`, not a fresh fit) + row cache keyed on source size+mtime |
| `b51e7e1` | **19:37:26Z** | **ADDENDUM 2 — the equal-`n` falsifier for my OWN result, registered BEFORE running it**, with the rule written in the direction that costs me the finding |
| `5fc572a` | **19:54:15Z** | **ADDENDUM 3 — the flaw in my OWN `q04` floor, registered BEFORE the correction**, with my prediction written AGAINST my own draft finding |

**LEDGER PUSHED: `9d3ae8e` on `origin/main`** — prereg + 3 addenda + RESULT, **307 insertions,
`PREREGISTRATIONS.md` ONLY, 0 deletions.** Built by extracting the **added lines** of my four
ledger commits onto a **fresh detached worktree at `origin/main`** (which had moved twice — freqgeo,
mirror and losvar entries landed while I worked), never by `git cherry-pick` (which conflicts on this
append-only file, and `--theirs` deletes lines). **Verified before pushing** that `origin/main`'s
12797 lines are an exact **prefix** of the result, and **after pushing** with
`git merge-base --is-ancestor`. My 14 code/artifact commits stay **local** on branch `freqcorrect`.

**Artifacts** — `/local/home/zegertho/agent/state/freqcorrect/artifacts/`:
- `/local/home/zegertho/agent/state/freqcorrect/artifacts/q01_saturated.json` — N1, P1, P2, P3, N3, A1
- `/local/home/zegertho/agent/state/freqcorrect/artifacts/q02_matched.json` — INVARIANT B, N4/floor, `R_encode`, confounds
- `/local/home/zegertho/agent/state/freqcorrect/artifacts/q03_lolo.json` — the 12-model LOLO, N2, C1, C2, S5
- `/local/home/zegertho/agent/state/freqcorrect/artifacts/q04_consequence.json` — cross-layout transfer (first pass) + the INVARIANT D magnitudes. ⚠ **its floor is SUPERSEDED by `q06`**
- `/local/home/zegertho/agent/state/freqcorrect/artifacts/q05_equaln.json` — the equal-`n` falsifier INVARIANT B survived
- `/local/home/zegertho/agent/state/freqcorrect/artifacts/q06_transfer.json` — the corrected transfer test (C-1…C-4)

All six are **also committed into the repo** at `artifacts-freqcorrect/` on branch `freqcorrect`
(48K total), because `/tmp` is tmpfs and this fleet has had five tmux-server deaths.

**Discipline:** `data/models/k31/` **never written** (read via `gzip.open`); `src/keybo/layouts.py`
untouched; **no layout adopted or promoted**; **nothing pushed** except ledger lines; no branch merged
or deleted; shared checkout left on `main`. 🟢 **The D5 trap is real and my guard fired on it** — every
driver asserts `keybo.__file__` is inside my worktree and prints both checkouts' branches (the shared
one was on `main` throughout). All four thread vars pinned before importing xgboost; `require_finite`
on every aggregate; drivers committed; **report written incrementally** (this fleet has had five
tmux-server deaths, one of which hit me mid-arm — all four artifacts survived because each driver
flushed on completion).

**MY OWN REGISTERED PREDICTIONS: 3 of 3 FAILED.** (1) H-SATURATED said `b` would be materially
geometric — R² = −0.0151, refuted. (2) I predicted ≥60% of the +9.906 would be a level effect —
measured −0.129, refuted. (3) I predicted the equal-`n` subsample would shrink `β_freq` — it grew
0.5%; and I predicted the cross-layout non-transfer would prove artifactual — it is genuine. **Every
one is reported above as the result, not as a footnote.** What survived is what the *instruments* said,
not what I expected — which is the point of registering them first.

**Confidence:** 🟢 VERIFIED — N1/N2/N3/N4, the A1 double-count identity, the `g`/`g+b`/`g+2b` table,
P1's bijection, P2/P3's geometric-R² nulls, `β_freq` + CI + measured floor, the equal-`n` survival,
`R_encode`, C1's failure, C2's seen/unseen split, per-fold slopes, the served frame's 20 columns, the
ms/char conversions, the 98.73% sample asymmetry, `q06`'s C-1/C-2/C-3/C-4 numbers and the 1.249×
matched-noise ratio. 🟠 INFERRED — that the frequency effect is practice rather than some third
frequency-correlated cause (equal-`n` kills the noise route, not every route); that support rather than
practice explains the qwerty fold (n=4); that the `q06` layout-specific residual is motor-sequence
practice rather than a modelling error (**untested — see §(h) item 5**). 🔴 NOT MEASURED — a
participant-fixed-effect §(b); `Tc`'s decomposition; a wider feature basis for P2; H-SHRINK; any
5th-layout replication of `q06`.
