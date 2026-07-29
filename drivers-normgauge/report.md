# NORMGAUGE-1 — three normalized per-model gauges + one weighted objective

**Branch `normgauge` in `/tmp/normgauge`. Nothing pushed, no CR, `PREREGISTRATIONS.md` untouched,
no layout adopted or recommended.**

Everything below is **MODELLED**: shipped `.standardized` surfaces, geometry-only `g` frame (the
layout-independent `b(ngram)` term excluded), **BAKED 90 WPM** (not re-evaluable), corpus
`blend-v1`. **No claim about realized typing speed.**

---

## 1. THE ANSWER IN ONE PARAGRAPH

Three gauges `aalto-n` / `comm-n` / `pool-n` ship in `src/keybo/scoring/model_norm.py`, each 0 at a
100-random-layout pool's **mean** and 1 at that model's own searched optimum, plus a combined
objective wired into the shipped optimizer as
**`keybo optimize --model-weight aalto-n=… --model-anchors …`**. The pre-registered evidence rule
selected **held-out predictive skill**, giving **`aalto-n 0.5411 / comm-n 0.3977 / pool-n 0.0612`**.
The headline finding **separates two claims that were being conflated**: *within* one model,
normalization reorders **nothing** (0 discordant pairs of 66, spearman +1.000000 — MODELNORM-1's null
reproduces exactly, as it must for an affine rescale), but *across* models the **weighting reorders a
great deal** (solo-AALTO vs solo-COMMUNITY differ on **30 of 66** pairs, spearman **+0.2448**). So
"the scheme reorders nothing" is true only of the **scale**, not of the **weight** — and **my own
prediction P7, that the weight would not be load-bearing, is refuted.**

---

## 2. THE WEIGHTS AND THEIR JUSTIFICATION

### 2.1 What the evidence actually supports

| rule | identifiable? | outcome |
|---|---|---|
| (a) precision / sample size | yes | would give **0.9102 / 0.0286 / 0.0612** |
| (b) independence correction | yes, **measured** | sets POOL's weight at **0.0612** |
| (c) held-out predictive skill | **yes — and it fired** | **0.5411 / 0.3977 / 0.0612** ← shipped |
| (d) equal weights | reference only | **not neutral**, see §2.3 |

**(c) needed no refit, which is why it was affordable.** The two sources are **disjoint** (aalto pids
<200000; community pids 200001–200007), so each source's data is *already* out-of-sample for the
other's surface. Scored with the campaign's own cell machinery and its own bucket-centered Spearman
(bucket-centered because the wpm→duration axis is a model **input**), divided by the held-out
source's own split-half reliability ceiling:

| surface | held-out on | rho | ceiling | rho/ceiling | cells | pids |
|---|---|---|---|---|---|---|
| AALTO | COMMUNITY | +0.2786 | 0.5150 | **0.5410** | 866 | 4 |
| COMMUNITY | AALTO | +0.4115 | 0.9757 | **0.4217** | 23,714 | 55,404 |

gap 0.1402 > pooled SE 0.0822, neither CI crossing 0 → **(c) usable**. Two independent runs returned
bit-identical rho and ceiling values.

**(b) is measured, not asserted.** Over a 400-layout random pool at **fit** level (what the optimizer
sees): `POOL = 0.498757·AALTO + 0.508017·COMMUNITY + c`, **R² = 0.93881**, coefficient sum 1.00677.
Cell level agrees (0.454530 / 0.449591, R² 0.87400). POOL is a **near-exactly symmetric blend** of
the other two, so it is **not an independent third vote**; its weight is its measured unique variance
share, **1 − R² = 0.0612**.

### 2.2 🔴 A CONSTANT IN MY BRIEF THAT IS MIS-SCOPED — conclusion true, number wrong

The brief's *"AALTO 7,669,316 vs COMMUNITY 11,930 — 643x, the strongest single fact about relative
reliability"* is the **scissor-neighbourhood, covered-pair-filtered** ratio, and **that filter is
asymmetric**: AALTO's count is *identical* in the filtered and unfiltered artifacts while COMMUNITY
loses **92.1%** (151,365 → 11,930).

| scope | AALTO | COMMUNITY | ratio |
|---|---|---|---|
| `ss2d` scissor neighbourhood, covered-pair filtered | 7,669,316 | 11,930 | 642.9x ← the brief |
| `ss2` same groups, unfiltered | 7,669,316 | 151,365 | 50.7x |
| `ss2` whole stroke table | 18,535,823 | 401,543 | 46.2x |
| **my own scan, on the surface-cell frame the gauge uses** | **26,368,247** | **29,047** | **907.8x** |

**The conclusion is confirmed and strengthened** (AALTO is far better supported; COMMUNITY covers
1,044 of 29,791 cells at a median of 18 samples each, min 10). **643x is not used as a reliability
ratio anywhere in this arm.**

### 2.3 Equal weights are not neutral — with a number

Under `(1/3,1/3,1/3)`, POOL re-votes the AALTO+COMMUNITY consensus, so the effective source loadings
are **AALTO 0.4996 / COMMUNITY 0.5027 / unique-POOL 0.0204**: the correlated pair's agreement is
counted about **1.5x**, and POOL's own unique signal gets a third of an already small share.

### 2.4 ⚠ POOL's weight does no observable work

**`drop-pool` (AALTO+COMMUNITY 50/50, POOL removed) is statistically indistinguishable from the
registered weighting** — gap **+0.000143 = 0.25x** the resolution floor. Since POOL *is* a measured
near-symmetric blend of the other two, dropping it and splitting 50/50 buys the same optimum. This is
the honest simplification, and it only became visible after fixing a ruler error in my own work
(§5, Kill 1).

---

## 3. DOES THE COMBINED GAUGE FIND ANYTHING? — 18 cells, 6 weightings × 3 seeds

**Search-noise floor, measured HERE, not borrowed:** max across-cell within-seed sd **0.000576
normalized units**. Quadruple: *(pool = 6 weighting cells × 3 seeds on the blend objective over C30M
permutations, blend-v1, shipped `.standardized`) × (replicate structure = independent RNG seeds of
the same memetic-island searcher at an identical 5,000,000-unique-eval / 40-island budget) × (scale =
normalized blend units) × (statistic = across-seed sd of the best blend)*.
⚠ **NOT comparable to the campaign's 0.0492 / 0.0995 — those are ms/char-scale.**

| registered vs | gap | ×floor | verdict |
|---|---|---|---|
| equal | +0.006488 | 11.26x | RESOLVED |
| **drop-pool** | **+0.000143** | **0.25x** | **TIE** |
| solo-AALTO | +0.028234 | 48.98x | RESOLVED |
| solo-COMMUNITY | +0.049261 | 85.47x | RESOLVED |
| solo-POOL | +0.032052 | 55.61x | RESOLVED |

**6 distinct champions across 6 cells**, and **6/6 cells win on their own objective** (a positive
control run only after it could no longer tune anything).

**Against `ms/char`, both numbers reported because neither alone is honest:** over a **300-layout
random pool** the registered blend agrees with `ms/char` at **+0.9716** (a re-parameterization in the
bulk — P9 confirmed); over the **incumbent field** it disagrees on **20 of 66** pairs and picks a
different champion (`archive-1846` vs `arm-B`). **So the gauge earns its keep near the top — where
selection happens — and nowhere else. That is the answer to "what does it do that ms/char does
not".**

**NOT a dominator:** the registered champion dominates **0 of 12** field layouts on **contested**
axes (4–11 wins of 12–14 contested; `sfr` is a permutation invariant and `alt`/`imbalance` tie for
layouts sharing a hand partition, so those are excluded — never a bare n/15).

---

## 4. THE ANCHORS REPRODUCE

5,000,000 unique evals **requested**, **5,000,263–5,003,863 achieved** (achieved, not requested);
40 islands; 3 seeds; identical budget across models.

| model | zero (n=100, seed 20260728, mean) | one (conservative: slower of the seed bests) | span | seed spread |
|---|---|---|---|---|
| AALTO | 243118526775.9713 | 223241709941.1167 | 8.1758% | 0.0271% of span |
| COMMUNITY | 249483317974.6619 | 222447818165.8890 | 10.8366% | **exactly 0.0** |
| POOL | 247979864398.5926 | 227268377105.3342 | 8.3521% | **exactly 0.0** |

* **AALTO gate MET at +0.0024%** against MODELNORM-1's 10M-eval champion (bar was +0.05%); 2 of 3
  seeds hit its fit **223236317224.4177 exactly**. Valid as a control because AALTO's `.native` and
  `.standardized` arrays are **byte-identical** (max|d| = 0.0).
* **Unplanned free control:** my AALTO **zero** anchor `243118526775.9713` matches MODELNORM's
  `243118526775.97125` — the n=100 seed-20260728 pool reproduces across two independent
  implementations of both the pool constructor and the evaluator.
* **n=100 re-verified sufficient independently:** n=1000 moves the zero by −0.979 / −0.162 / −0.602
  SE. The user's n is not silently inflated.
* Every normalized score is an **UPPER bound** — an optimizer output bounds the true optimum from one
  side only. `solo-AALTO` correctly scores **1.00027** (it beat my deliberately conservative anchor,
  and the excess equals `(anchor − fit)/span` exactly).

**P3 confirmed, and the trap holds:** qwerty30m normalizes to **0.4621–0.5650**, *not* ≈0. The
direction guard is each model's own optimum → exactly 1.0, and a test pins that a **mid-range qwerty
is accepted**, so the sign-inverting "fix" cannot be reintroduced. **P4 confirmed:** the real
candidates occupy only 0.1696 / 0.0750 / 0.1215 of each range — a random-layout zero does waste most
of the scale.

---

## 5. WHAT I KILLED OF MY OWN (full detail: `drivers-normgauge/SELF-KILL.md`)

1. 🔴 **A wrong constant attached to a true conclusion, in my own committed result.** I claimed
   "registered beats drop-pool by 2.4x noise" — comparing registered's best *on the registered
   objective* against drop-pool's best *on the drop-pool objective*. **Two rulers.** Corrected:
   **0.25x — a TIE.** solo-AALTO moved 51.6x → 48.98x. The conclusion (weighting is load-bearing)
   survives, **which is exactly why it went unchecked**, and the correction produced the better
   recommendation in §2.4.
2. 🔴 **My prereg said "n=7 community participants"; the training subset has 4.** Conclusion held and
   strengthened. Registered pre-result (Amendment 1).
3. 🔴 **My registered bootstrap was a no-op on the AALTO side** — 0.999992 of 24,079 cells survived
   every resample, so it would have **manufactured significance on the side with the most data**. The
   two sides fail in *opposite* directions (COMMUNITY: median 1 pid/cell, 0.6827 survive), so a
   one-sided check would have looked fine. Registered pre-result; the fix widens intervals, making my
   own falsifier *easier* to fire.
4. 🔴 **The CI was an interval for a different statistic** — COMMUNITY's point estimate fell *outside
   its own CI*; the plain-mean/IQR-mean aggregation gap is **8.41x the half-width**. Found *after*
   the result, so published with its blast radius: same branch, weights move ≤0.0136, refuting (c)
   would need a **41.8x** SE widening (Amendment 2).
5. 🔴 **A test that would have "proved" the objective wasn't being optimized.** My first end-to-end
   test disabled the 2-opt polish — and on this objective the polish does nearly all the work
   (0.523429 with it off at max_outer 60 *and* 300, vs 0.941646 with it on).

**Attacks that failed to refute:** excluding qwerty30m leaves 30/55 discordant (+0.0182), and
restricting to the 8 realistic layouts gives **−0.8095** — AALTO and COMMUNITY are *anti*-correlated
exactly where selection happens, so the reordering finding **strengthens** under attack.

**Floor caveat, stated not hidden:** my resolution floor is the max across-cell within-seed sd of the
*same* searcher on the *same* objectives — a search-reproducibility floor, not an independent error
model. Three of six cells have a **zero** within-cell sd, which cannot be the yardstick (it would
make every gap "resolvable"), so the MAX is used — the conservative choice.

---

## 6. THE FRAME COST I AM NOT BURYING

`standardized − native` is **exactly independent of the third slot** (max variation over `c`: AALTO
0.0, COMMUNITY/POOL 1.14e-13) and identically 0 for AALTO. **The shipped standardization substitutes
AALTO's bigram tensor into all three sources**, so on the frame the gauge must use, the three differ
*only* in their conditional trigram increment and are **less independent than on `.native`**.
MODELNORM-1 chose `.native` for exactly this reason; I ship on `.standardized` because the shipped
resolver reads only that. `frame_caveat()` prints this on every report the gauge appears in.

---

## 7. DELIVERABLES

| # | deliverable | where |
|---|---|---|
| 1 | three normalized gauges + versioned anchors with provenance | `src/keybo/scoring/model_norm.py`, `drivers-normgauge/anchors.json` |
| 2 | combined objective in the **shipped** optimizer | `keybo optimize --model-weight GAUGE=W --model-anchors …` (`src/keybo/cli/optimize.py`) |
| 3 | tests, mutation-proven | `tests/scoring/test_model_norm.py` (27), `tests/cli/test_optimize_model_weight.py` (9) |
| 4 | the run + its analysis | `drivers-normgauge/blend-report.json`, `reorder_check.py`, `SELF-KILL.md` |
| — | pre-registration + 2 amendments | `drivers-normgauge/PREREGISTRATION.md` |

**Prediction scorecard:** P1 ✅ · P2 ✅ · P3 ✅ · P4 ✅ · P5 ✅ (0.0612 < 0.15) · **P6 ❌** ((c)
fired) · **P7 ❌** (the weight IS load-bearing) · P8 ✅ (no dominator) · P9 ✅ (+0.9716) ·
**P10 ⚠ untestable — my sd is in normalized units and the prior values are ms/char, so the
comparison the prediction asked for is a category error I registered without noticing.** That is a
sixth self-kill: I wrote a prediction that could not be scored, in the very section warning against
borrowed rulers.

**The mutation control found a real defect in itself,** which is the strongest argument for having
one: CPython validates a `.pyc` by *(source mtime truncated to the second, source size)*, and my
sign-inverting mutant is **size-preserving**, so mutate-then-restore inside one mtime second left a
cache CPython considered valid while it held the other version's bytecode (verified directly: pyc
mtime 1785288965 / size 24429 exactly matching the restored source). Caught only by `testkit`'s
restore-to-green check — without it the harness would have reported a caught mutant that never ran.

---

## 8. FINAL GATE STATUS

* **Full test suite: rc=0**, read from a sentinel (`/tmp/ngC.sentinel`), zero failures — including
  the two pre-existing tests my `--model-weight` flag had broken by reading `args.model_weight`
  directly on a hand-built `SimpleNamespace`. **My own 36 tests all passed while those two went
  red**, which is the argument for running the whole suite and not just the new file.
* **Harness positive-controlled before any pass was believed** (`assert_harness_detects_a_fatal_mutant`).
* **Evaluator bit-exact across batch lengths** with a live mutation control proving the unpadded
  path really is batch-dependent (5.6e-15), so the guard cannot silently retire.
* **`unique_evals` reported as ACHIEVED** (5,000,263–5,003,863), not requested. **rc from sentinels**,
  never from a callback: `drivers-normgauge/runs/.sentinel-*` for all 27 search cells.
* **`oxey-style` computed fresh** by today's code (post the nested-`bad_redirect` fix).
* Lint + format clean on every file touched.

## 9. ONE THING ONLY A HUMAN CAN DO

The branch `normgauge` is **committed and unpushed** (11 commits, `64c9ddf`…`12f4a45`). Landing it
touches a shipped CLI surface (`keybo optimize --model-weight`) and adds a new scoring module, so
**pushing it or raising a CR is the human gate** — my brief forbids both, and I have not done either.
