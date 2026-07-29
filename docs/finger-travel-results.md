# Results — `finger-travel` and `off-home` (the FT round, 2026-07-29)

Answers `docs/finger-travel-preregistration.md`. Every definition and all seven predictions were
committed **before** any layout was measured (`3df98cf`); this document scores them.

Corpus **blend-v1** (the CLI default), `trigrams.sha256=19806532ee35…`. Gauges and predicted
ms/char come from the shipped `keybo analyze --json` path; the two new metrics from
`keybo.analysis.finger_travel`. Field = the 15 registry layouts + the three campaign candidates
(`arm-B`, `BALL-1`, `armH-hdln`) grepped out of `PREREGISTRATIONS.md`, never retyped.

---

## 0. The headline, in four sentences

1. **`finger-travel` works as a descriptor and is an exact 100% partition, but its TOTAL is
   near-redundant with `sfb-dist` (|r| = 0.970) and should not become a 16th gauge.** Its value
   is the per-finger split.
2. **The user's off-home-pinky idea is the more INFORMATIVE of the two** — its closest single
   incumbent is only |r| = 0.605 — **but its relationship to predicted time runs the OPPOSITE way
   to the intuition behind it**: in this field, boards with *more* off-home pinky mass are
   *faster*.
3. **The user's cost claim ("pinky use is fine if it stays home") is UNSUPPORTED on this
   evidence.** Off-home adds +0.065 R² over a frequency control; pinky-*total* adds +0.069 — so
   off-home is not the term carrying the signal, and neither is large.
4. **Both metrics separate nine layout pairs the existing frame ties by construction**, which is
   the genuinely new thing here, and `finger-travel` additionally moves under corpus reversal
   where all 11 `kmstats` gauges are exactly blind — with a caveat that guts half the claim (§5).

---

## 1. Table: finger-travel shares (exact partition of 100%) + the LEVEL

Sorted by absolute total. **The `TOTAL` column is not optional**: normalizing destroys the level,
and two boards can share every percentage at very different totals (the `saved_vs_ref_pct`
artifact this ledger already registered).

| layout | L-P | L-R | L-M | L-I | R-I | R-M | R-R | R-P | **TOTAL** | max | pinky | L/R | gini |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| colemak | 0.74 | 3.34 | 12.66 | 29.38 | 31.79 | 12.32 | 9.12 | 0.65 | **2.208e8** | 31.79 | 1.39 | 0.86 | 0.490 |
| semimak | 5.82 | 11.12 | 11.42 | 11.09 | 21.61 | 11.41 | 21.29 | 6.25 | **2.426e8** | 21.61 | 12.07 | 0.65 | 0.234 |
| graphite | 3.31 | 11.05 | 14.89 | 18.56 | 17.85 | 20.82 | 11.69 | 1.83 | **2.440e8** | 20.82 | 5.15 | 0.92 | 0.291 |
| p13stab-win | 17.44 | 8.34 | 6.64 | 25.31 | 12.84 | 16.92 | 7.90 | 4.61 | **2.584e8** | 25.31 | 22.05 | 1.37 | 0.288 |
| flagship-c3 | 6.10 | 4.69 | 16.59 | 16.91 | 21.52 | 10.54 | 18.32 | 5.33 | **2.649e8** | 21.52 | 11.43 | 0.80 | 0.277 |
| keybo-lsb+lm | 6.01 | 4.41 | 8.78 | 24.94 | 22.20 | 10.38 | 18.04 | 5.25 | **2.690e8** | 24.94 | 11.26 | 0.79 | 0.333 |
| archive-1846 | 6.00 | 4.49 | 16.72 | 16.88 | 22.49 | 10.09 | 17.19 | 6.13 | **2.691e8** | 22.49 | 12.13 | 0.79 | 0.276 |
| arm-B | 6.08 | 10.87 | 7.10 | 21.85 | 26.79 | 7.82 | 16.33 | 3.16 | **2.691e8** | 26.79 | 9.24 | 0.85 | 0.344 |
| p16-balance | 6.54 | 17.13 | 15.16 | 21.18 | 17.13 | 4.61 | 16.11 | 2.15 | **2.696e8** | 21.18 | 8.69 | 1.50 | 0.286 |
| archive-1843 | 4.40 | 4.78 | 16.62 | 18.55 | 22.05 | 10.31 | 18.09 | 5.21 | **2.708e8** | 22.05 | 9.60 | 0.80 | 0.297 |
| dvorak | 1.10 | 3.10 | 3.56 | 28.96 | 26.01 | 8.73 | 18.19 | 10.35 | **2.729e8** | 28.96 | 11.45 | 0.58 | 0.444 |
| BALL-1 | 5.96 | 10.65 | 6.96 | 23.42 | 26.25 | 7.67 | 16.00 | 3.10 | **2.747e8** | 26.25 | 9.05 | 0.89 | 0.349 |
| armH-hdln | 5.92 | 10.59 | 7.17 | 22.68 | 25.89 | 7.62 | 15.91 | 4.22 | **2.763e8** | 25.89 | 10.14 | 0.86 | 0.331 |
| keybo-c30m | 5.65 | 4.23 | 7.85 | 24.53 | 18.36 | 12.41 | 17.27 | 9.70 | **2.809e8** | 24.53 | 15.35 | 0.73 | 0.296 |
| keybo-lsb | 5.73 | 4.21 | 8.38 | 23.80 | 21.18 | 9.91 | 17.22 | 9.57 | **2.818e8** | 23.80 | 15.30 | 0.73 | 0.302 |
| lsb-sib | 4.20 | 4.39 | 15.92 | 17.68 | 21.22 | 9.88 | 17.17 | 9.54 | **2.827e8** | 21.22 | 13.74 | 0.73 | 0.268 |
| qwerty | 0.32 | 1.44 | 22.83 | 26.03 | 27.16 | 9.63 | 10.34 | 2.25 | **5.114e8** | 27.16 | 2.57 | 1.03 | 0.467 |
| qwerty30m | 0.33 | 1.46 | 22.59 | 25.97 | 26.92 | 9.61 | 10.25 | 2.88 | **5.185e8** | 26.91 | 3.21 | 1.01 | 0.461 |

Reading notes:

* **The two qwerty boards travel ~1.9× as far as every optimized board** (5.1e8 vs 2.2–2.8e8).
  That is the one large, robust effect in the level, and it is the only travel-total comparison I
  would quote. Everything inside the optimized cluster spans just 2.21–2.83e8.
* **`colemak` is the extreme on dispersion, not on level**: gini 0.490, max share 31.79% on
  R-index, pinky share 1.39%. It concentrates travel on the strong fingers. `semimak` is the
  opposite (gini 0.234) — the most evenly-travelling board in the field.
* **`p13stab-win` puts 22.05% of all travel on the pinkies** (17.44% on L-pinky alone), ~16× what
  colemak asks of them. If a per-finger travel budget is ever used as a filter, this is the board
  it would reject.
* Observed (same-finger) travel is only **4.28%** of the total on graphite; the other 95.72% is
  the **modelled** from-home branch. Across the field the observed fraction runs **3.08%
  (semimak) to 11.66% (qwerty)**, median 4.52% — the optimized boards cluster near 4% and qwerty
  is an outlier because it has far more same-finger bigram mass. So the metric is **88–97%
  assumption** depending on the board. That ratio is published per row as
  `observed_fraction_pct`, and it is the single most important limitation of the metric (§5).

## 2. Table: off-home pinky usage — the user's second metric

`letter-freqs` convention (**the parent's**, reproduced to 0.0045 pp); the `restricted`
convention's off-home column is shown alongside because the choice moves numbers by up to ~0.9 pp.

| layout | pinky total | on-home | **OFF-home** | off % of own | (restricted OFF) | ms/char | off-home keys |
|---|---|---|---|---|---|---|---|
| colemak | 14.15 | 13.58 | **0.56** | 4.0 | 0.47 | 258.24 | q(r3) ;(r3) z(r1) /(r1) |
| qwerty | 9.72 | 7.14 | **2.58** | 26.6 | 2.73 | 263.71 | q(r3) p(r3) z(r1) /(r1) |
| graphite | 15.13 | 12.38 | **2.75** | 18.2 | 2.56 | 258.17 | b(r3) j(r3) x(r1) -(r1) |
| qwerty30m | 10.82 | 7.62 | **3.21** | 29.6 | 3.25 | 264.14 | q(r3) p(r3) z(r1) -(r1) |
| p16-balance | 13.65 | 9.62 | **4.03** | 29.5 | 4.09 | 254.75 | f(r3) k(r3) v(r1) z(r1) |
| BALL-1 | 13.99 | 9.62 | **4.37** | 31.3 | 3.59 | 253.97 | f(r3) ,(r3) k(r1) q(r1) |
| arm-B | 13.99 | 9.62 | **4.37** | 31.3 | 3.59 | 253.90 | f(r3) ,(r3) k(r1) q(r1) |
| armH-hdln | 14.26 | 9.62 | **4.64** | 32.6 | 4.30 | 254.04 | f(r3) y(r3) k(r1) q(r1) |
| archive-1843 | 14.15 | 9.39 | **4.75** | 33.6 | 5.09 | 254.84 | p(r3) m(r3) j(r1) q(r1) |
| dvorak | 18.16 | 13.21 | **4.96** | 27.3 | 5.45 | 255.02 | '(r3) l(r3) ;(r1) z(r1) |
| semimak | 17.57 | 12.50 | **5.06** | 28.8 | 4.55 | 257.39 | f(r3) y(r3) x(r1) -(r1) |
| flagship-c3 | 12.82 | 7.49 | **5.33** | 41.6 | 5.70 | 254.98 | p(r3) m(r3) k(r1) q(r1) |
| keybo-lsb+lm | 12.82 | 7.49 | **5.33** | 41.6 | 5.70 | 254.68 | p(r3) m(r3) k(r1) q(r1) |
| archive-1846 | 13.16 | 7.49 | **5.67** | 43.1 | 6.04 | 254.80 | p(r3) m(r3) k(r1) x(r1) |
| lsb-sib | 12.33 | 5.73 | **6.60** | 53.5 | 7.16 | 254.71 | f(r3) l(r3) z(r1) q(r1) |
| keybo-lsb | 12.82 | 5.73 | **7.09** | 55.3 | 8.00 | 254.63 | p(r3) l(r3) k(r1) q(r1) |
| keybo-c30m | 12.89 | 5.73 | **7.16** | 55.5 | 7.76 | 254.59 | f(r3) l(r3) k(r1) z(r1) |
| p13stab-win | 18.55 | 10.38 | **8.17** | 44.1 | 8.85 | 254.32 | r(r3) y(r3) x(r1) /(r1) |

**The parent's reversal claim is CONFIRMED as a fact and REFRAMED as an inference.** `keybo-lsb`
is indeed the worst of the parent's seven on off-home (7.09, 55.3% of its own pinky use) while
`analyze --attribution` shows it with light pinky *time*. But look down the `ms/char` column: the
ordering of this table is very nearly the *reverse* of the speed ordering. The three boards with
the least off-home pinky mass (colemak 0.56, qwerty 2.58, graphite 2.75) are the **three slowest
boards in the field** at 258.2 / 263.7 / 258.2 ms/char, and the two worst on off-home
(`keybo-c30m` 7.16, `p13stab-win` 8.17) are near the fastest. So this is not "the existing frame
missed a cost" — it is **"the optimizer deliberately spends pinky off-home mass, and the time
model rewards it."**

Also worth noting: **`semimak` is exactly the case the user's claim predicts should be fine** —
highest total pinky load in the field at 17.57% but a middling 28.8% off-home fraction — and it
is *slow* (257.39). The claim's own favourable case does not behave as the claim expects.

## 3. Redundancy — is either metric a restatement?

Registered bar: R² > 0.95 on the 15-gauge frame ⇒ a restatement, do not add it as a gauge. That
bar turns out to be **unusable as stated** and I am reporting it as such: with n = 18 *layouts in
this field* and k = 15 *gauges*
the frame fits *anything* (every candidate scores R² ≥ 0.96, `dof_warning = True`). This is the
ledger's own registered "~4–5 effective dof" problem. **The informative statistic is the closest
SINGLE gauge**, where a high value cannot be bought with degrees of freedom.

| metric | R² on frame | adj R² | **closest single gauge** | Spearman w/ ms/char |
|---|---|---|---|---|
| `travel_total` | 0.9994 | 0.995 | **`sfb-dist` 0.970** (comfort .968, sfb .963) | −0.086 |
| `travel_max_share` | 0.9952 | 0.960 | `alt` 0.616 | +0.042 |
| `travel_pinky_share` | 0.9745 | 0.783 | `lsb` 0.684 | −0.472 |
| `travel_gini` | 0.9945 | 0.953 | `sfs` 0.783 | +0.154 |
| **`pinky_off_home`** | 0.9667 | 0.717 | **`alt` 0.605** | −0.461 |
| `pinky_usage_total` | 0.9919 | 0.931 | `sfb` 0.561 | +0.164 |
| `pinky_off_fraction` | 0.9626 | 0.682 | `alt` 0.498 | −0.472 |

**Verdicts:**

* **`travel_total` is close to a restatement of `sfb-dist` (0.970).** Expected — the observed
  branch of travel *is* a same-finger distance sum. **Do not ship the total as a gauge.**
* **`pinky_off_home` is the most independent quantity in the round** (best single incumbent 0.605,
  and `alt` at that). It is a genuinely new axis. Cross-check: it is only r = +0.300 with
  pinky-*total*, so it is not "pinky usage in disguise", and r = −0.208 with `travel_total`, so
  the two new metrics are not each other.
* `travel_pinky_share` vs `pinky_off_home` is r = **+0.935** — those two ARE near-duplicates.
  Ship one. Since off-home is cheaper and more independent of the frame, ship off-home.

### 3b. ⚠ A correlation I killed of my own

My first pass reported **`travel_total` ~ ms/char at Pearson +0.82** and I nearly wrote "more
travel predicts slower typing," which is intuitive and would have passed review. It is an
**artifact**:

| subset | Pearson |
|---|---|
| all 18 | **+0.8181** |
| minus `qwerty` | +0.7002 |
| minus `qwerty30m` | +0.6797 |
| **minus both** | **−0.8694** |
| Spearman, all 18 | **−0.0857** |

The two qwerty boards sit at 5.1e8 against 2.2–2.8e8 for everything else; those two leverage
points alone create the positive sign. **Within the optimized field the relation is the opposite
sign, and the rank correlation over all 18 is essentially zero.** `travel_total` does not predict
predicted time. `pinky_usage_total` fails the same audit (+0.37 → −0.37). The five other
correlations are sign-stable. `leverage_audit()` now runs on every one of them.

## 4. Does it break ties the frame cannot see? — YES, and this is the best result

`alt` and `imbalance` depend only on the hand partition; `sfr` only on within-finger key identity.
So any two boards related by a **within-hand permutation** tie on all three *by construction*. I
verified that invariance myself rather than inheriting it: **200 random within-hand swaps × 3
gauges = 600 checks, zero movement.**

Nine such tied pairs exist in the field. Both new metrics separate **all nine**:

| tied pair | travel total gap | travel pinky-share gap | off-home gap | ms/char gap |
|---|---|---|---|---|
| keybo-lsb vs keybo-lsb+lm | −4.56% | −4.04 pp | −2.30 pp | +0.054 |
| flagship-c3 vs keybo-lsb | +6.39% | +3.87 pp | +2.30 pp | −0.345 |
| archive-1843 vs keybo-lsb | +4.08% | +5.70 pp | +2.91 pp | −0.213 |
| arm-B vs armH-hdln | +2.64% | +0.90 pp | +0.71 pp | +0.139 |
| BALL-1 vs arm-B | −2.01% | +0.19 pp | **0.00** | −0.066 |
| archive-1843 vs flagship-c3 | −2.17% | +1.83 pp | +0.61 pp | +0.133 |
| BALL-1 vs armH-hdln | +0.58% | +1.09 pp | +0.71 pp | +0.073 |
| archive-1843 vs keybo-lsb+lm | −0.66% | +1.65 pp | +0.61 pp | −0.159 |
| flagship-c3 vs keybo-lsb+lm | +1.54% | −0.17 pp | **0.00** | −0.291 |

Note the complementarity, which is the argument for keeping both: **`arm-B` vs `BALL-1` tie on
off-home at exactly 0.00** (they differ by a `cd`→`dc` transposition on non-pinky columns) **but
travel separates them by 2.01%**. Conversely `flagship-c3` vs `keybo-lsb+lm` tie on off-home and
barely move on pinky share. Neither metric dominates the other.

**This is the strongest case for adopting anything from this round**: a diagnostic that
discriminates where the frame is structurally silent is worth having even if it predicts nothing,
because it tells you *two boards the frame calls identical are not*.

## 5. Direction-sensitivity — a real finding with half of it retracted by me

All 11 `kmstats` gauges are **exactly** direction-blind: reverse every n-gram with the layout
fixed and every delta is `0.00e+00` (re-derived here, not taken on trust). Under the same
instrument `travel_total` moves **+2.91%**, a per-finger share moves **4.17 pp**, and the metric
**reorders 10 of 15 layouts** (delta range −1.62% … +7.68%, so not a constant offset).

**So travel carries a channel the incumbent frame provably cannot express. And here is the half I
killed:** that sensitivity is **100% in the MODELLED from-home branch.** The observed same-finger
branch moves by **exactly zero**, because `dist(k1,k2)` is symmetric per pair. Decomposed:
observed delta `+0`, modelled delta `+7.1e6` (+3.04%).

So the "new channel" is a property of **my return-model assumption** — which key is the *landing*
key — not an observed physical asymmetry. The first sentence of this section is true, publishable,
and misleading on its own; that is precisely the shape of the wrong-constant-behind-a-true-
conclusion failure this campaign has now hit seven times, so both halves are pinned by tests.
Instrument note: **corpus reversal is the correct test; a left-right mirror is not** (a mirror maps
the finger-index ordering onto itself and cannot move a direction metric by construction).

`off-home` is *exactly* direction-blind — it is a unigram metric and cannot see stroke order even
in principle. Stated as its honest limit.

## 6. The user's cost claim (C2) — UNSUPPORTED on this evidence

> "Pinky being used a lot is mostly fine, as long as it stays on the home row."

Split into a **measurement** half (the interesting quantity is off-home usage) and a **cost** half
(total does not hurt, off-home does). The measurement half is shipped. The cost half is testable
and I tested it, over 160 layouts, with the registered frequency control:

| model | R² on predicted ms/char |
|---|---|
| `sfb` alone (frequency control) | 0.5465 |
| pinky **total** alone | 0.0252 |
| pinky **off-home** alone | 0.0876 |
| `sfb` + pinky total | 0.6152 → **increment +0.0687** |
| `sfb` + pinky off-home | 0.6111 → **increment +0.0646** |

**C2 requires off-home to add materially more than total. It adds slightly LESS (+0.065 vs
+0.069).** The frequency control alone explains 8× more variance than either geometric term. This
is the same result `bad_scissor` produced: bigram frequency beats every geometric axis. **P5, the
null I registered expecting to fail to reject, is confirmed.**

⚠ **And the n is smaller than it looks — and the SCOPE of that n is half the fact.** Those 160
rows are 160 evaluations of one fitted surface whose generalization unit is **4 distinct LAYOUTS
in the Aalto/k31 tables the shipped time surface was fitted on** — verified on disk:
`bistrokes31_v1.tsv` (**2202** rows) and `tristrokes31_cond_v1.tsv` (**16643** rows) each contain
exactly `{azerty, dvorak, qwerty, qwertz}`; all six k31 sidecars are `target_space=LOGRAT`.

That scope qualifier is load-bearing, not pedantry: the Aalto side carries ~55k participant IDs,
and a bare "n = 55,000" or even a bare "n = 4" is unusable without saying *what it is an n of*.
For layout-level generalization the unit is **layouts**, and there are four. So every R² above is
against **model predictions, not measured time**, and `n=160` is a sampling density, not an
evidence count. Direction of that caveat, stated so it cannot be read the flattering way: it
**weakens any positive finding here and strengthens this negative one.**

(Row counts here are `awk END{NR}`, asserted not retyped. **The first line of each TSV is a data
row, not a header** — a `wc -l` minus one undercounts by exactly one, which is how 2202/16643 gets
mis-stated as 2201/16642. I propagated that very error into an earlier revision of this document
after my own terminal had printed the correct 2202: seeing a number and copying a supplied one are
different acts, and I did the second.)

## 7. Is the pinky special? — NO (P6 confirmed)

|r| of each finger's off-home mass with predicted ms/char, same 160-layout pool:

| finger | off-home ~ ms | usage ~ ms |
|---|---|---|
| **index** | **0.398** | 0.168 |
| ring | 0.319 | 0.110 |
| pinky | 0.296 | 0.159 |
| middle | 0.278 | 0.269 |

The pinky ranks **third**. If off-home mass matters at all it is not a pinky-specific effect, and
the index — the *strongest* finger — shows the largest association, which is the opposite of a
weak-finger story. **The user has found the pinky instance of a general "off-home use" axis, not a
pinky-specific cost.** For every finger the off-home association exceeds the total-usage one, so
the *row-restriction* idea is the durable part of the insight even though the pinky framing is not.

## 8. Is it optimizable? — YES, and it SHOULD NOT BE (P7 confirmed)

Greedy 1-swap descent on `travel_total` from `graphite` (12 accepted swaps, not converged):

* `travel_total` 2.440e8 → 2.192e8 — **−10.17%**
* predicted ms/char 258.17 → 260.85 — **+2.68 ms/char WORSE**
* result: `vlwdz'fx,gsrtnpyaoeihmbcqjku.-`

The metric is easily movable and its minimizer is **slower**. Optimizing it is optimizing the
ruler — exactly WSCISSOR-GEN-1's registered result for the scissor-severity axis. **Do not wire
either metric into the search objective.**

---

## 9. Scoring the pre-registered predictions

| # | prediction | outcome |
|---|---|---|
| P1 | travel highly correlated with `sfb-dist`, \|r\| > 0.8 | ✅ **CONFIRMED** — 0.970 |
| P2 | per-finger shares separate frame-tied pairs | ✅ **CONFIRMED** — 9/9 pairs |
| P3 | `off_home(pinky)` not ~1.0 with any gauge | ✅ **CONFIRMED** — best is `alt` 0.605 |
| P4 | keybo-lsb worst on off-home, not worst on ms/char | ⚠️ **PARTIAL** — worst of the parent's 7, but `p13stab-win` (8.17) and `keybo-c30m` (7.16) are worse in the full field. Directionally right, the superlative was wrong. Checked under **both** denominator conventions: both agree `p13stab-win` is worst and `keybo-lsb` is not, so the correction is convention-robust (only 2nd/3rd place swaps, which nothing here relies on). |
| P5 | C2 will NOT be cleanly supported | ✅ **CONFIRMED** |
| P6 | the pinky is not special | ✅ **CONFIRMED** — index > ring > pinky |
| P7 | travel movable but should not be optimized | ✅ **CONFIRMED** — −10.2% travel, +2.68 ms slower |

Six of seven confirmed, one partial. **P4's superlative failing is exactly why the field had to be
all 18 boards and not the parent's 7** — a claim of "worst" evaluated on a subset is a wrong
constant waiting to happen.

## 10. How to best utilize these — the recommendation

**Adopt `off-home` (all eight fingers) as a REPORTED DIAGNOSTIC. Do not adopt `travel_total` as a
gauge. Do not put either in the objective.**

1. **Ship `off-home` per finger in `analyze`, as a diagnostic column.** It is the most independent
   quantity measured in this round (best single incumbent 0.605), it separates all nine
   frame-tied pairs, it is cheap, and it is *auditable by eye* — `p(r3) l(r3) k(r1) q(r1)` is
   checkable in a way that `off_home = 7.09` is not. Report all eight fingers, not just the pinky,
   since the pinky is not special.
2. **Report it with the opposite sign convention to the intuition.** High off-home pinky mass in
   this field marks the *optimized, faster* boards. Presented naively as "lower is better" it
   would recommend colemak over keybo-lsb — a 3.6 ms/char regression. If it becomes a filter, it
   should be a **tripwire for the extremes** (`p13stab-win`'s 22% of all travel on the pinkies),
   not a term to minimize.
3. **Use `finger-travel` for its per-finger SPLIT and its dispersion statistics, never its
   total.** The total is `sfb-dist` restated. The split is not, and `gini`/`pinky_share` answer a
   question no shipped gauge does: *is this board's motion concentrated on the weak fingers?*
4. **Use both as TIE-BREAKERS in the specific place the frame is silent** — comparing boards
   related by a within-hand permutation, where `alt`/`imbalance`/`sfr` are constant by
   construction. That is a structural gap, not a coverage accident, so it will not close on its
   own.
5. **Do not present either as a comfort or speed claim.** Both are geometric descriptors. The one
   time-model test run here came out *against* the intuition, and the timing evidence behind it
   generalizes over only 4 layouts.
6. **If the campaign wants a travel metric in the objective, the honest blocker is data, not
   design**: **88.3–96.9% of the headline is the modelled from-home branch** (observed fraction
   runs 3.08% on semimak to 11.66% on qwerty, median 4.52% across the 18-layout field) because no
   raw text corpus ships. A metric that is ~90%+ assumption should not be optimized. Resolving
   that needs a corpus
   with sequence information, at which point definition (d) becomes measurable rather than
   modelled.

## 11. What I killed of my own

1. **`travel_total` ~ ms/char at +0.82** — an outlier artifact of two qwerty boards; the sign
   flips to −0.87 without them, Spearman is −0.09. Would have shipped as "more travel is slower."
2. **"Travel sees direction where the frame is blind"** — true, but 100% of it is the *modelled*
   branch; the observed branch moves by exactly zero. Retained with the retraction attached.
3. **P4's superlative** — keybo-lsb is not the field's worst on off-home; two boards are worse.
4. **Two of my own test expectations** — a trigram's leading character is a departure and is never
   charged; `w` is column −4 (**ring**, not middle). The implementation was right both times.
5. **`usage` "sums to 100"** under the `letter-freqs` convention — it sums to ~93.5%, because
   untypeable corpus mass stays in the denominator. Now published as `coverage_pct` rather than
   normalized away.
6. **The registered R² > 0.95 redundancy bar itself** — unusable at n = 18 *layouts* / k = 15
   *gauges*, where everything clears it. Replaced by the closest-single-gauge statistic and
   reported as a defect in my own pre-registration.
7. **My own row counts, 2201/16642** — I wrote the numbers I was handed while my own terminal had
   already printed 2202/16643. The first line of each TSV is data, not a header. Corrected in §6;
   the general form is the failure mode of this whole session: **seeing a number and copying a
   supplied one are different acts, and only the first is measurement.**

### The rule this round earned

**Report what an `n` is an `n` OF.** Three counts were in play today — 4 layouts (the fitted Aalto
subset), ~55k participant IDs (same side, different unit), 7 participants (a different file
entirely) — and each is correct *for its own scope* and wrong quoted bare. This is the same defect
as this ledger's unscoped `bad_scissor` share and its "GL certificate" cells. Every `n` in this
document now carries its unit inline, and `ft_analysis.json`'s `effective_generalization_unit`
block carries a `scope` field beside the integer. **A count without a scope is not a measurement,
it is a number.**

A second, harder rule from the same source: **the correction path is not privileged.** Of the
three corrections I received today, one contained its own wrong constant — and it survived for the
same reason the original did: it pointed the right way. So the discipline is not "trust the
correction over the brief", it is *re-derive both*. That is why every constant in this document is
either generated in-process or asserted against disk, and why §11 exists at all.
