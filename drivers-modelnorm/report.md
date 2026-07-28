# MODELNORM-1 — per-model 0-1 anchored normalization of the three fitted surfaces

**Every number below:** corpus **blend-v1** (`data/corpus/blend-v1/trigrams.txt`, md5
`c5066fa7bcc46dea1ecbc987fb465b4a`), the **`.native`** surface frame, family
`TRI_PS_FREQ_PRIOR`, geometry-only `g` frame, **BAKED at 90 WPM**.
🔴 **MODELLED ONLY** — tau saturated at 1.0, Phase-D cancelled. Nothing here is a claim about
realized typing speed. No layout is promoted or adopted.

Branch `modelnorm` in worktree `/tmp/modelnorm`, 7 commits on top of `main@dec1c3f`.
Not pushed, no CR. Shared clone left clean on `main`.

---

## The headline answers

1. **Do the anchors stabilize?** 🟢 **Yes, completely.** Two independent seeds at identical
   10M-unique-eval budget found the **identical champion layout for all three models** — the
   seed-to-seed gap is **exactly 0.0 ms** on every model. 40/40 islands landed within 0.10 %.
2. **Does normalizing change any ranking?** 🟢 **No — not one, anywhere it matters.** Not
   within a model (impossible by construction, asserted), not versus the raw mean of the three
   surfaces, and not even versus the *prior* ceiling-fraction anchoring. It changes exactly two
   pairs versus the scale-broken raw `min()`, and **neither clears the resolution floor**.
3. **What did the blend search produce?** 🟢 `pctsk-reayfgdlm.niuobzvwxh,qj'` at
   **256.6268 ms/char** versus arm B's **253.9006** — **slower by 2.7262**. It beats arm A
   (256.8466) and qwerty30m (264.1389); it loses to every campaign incumbent.

---

## A. The normalization, implemented and unit-tested

    norm_m(L) = (zero_m − fit_m(L)) / (zero_m − one_m)          1 = BEST (fastest)
    blend(L)  = Σ w_m · norm_m(L) / Σ w_m                        w = PREFERENCE

**Anchors of record** (predicted ms over blend-v1; `zero` = mean of n=100 random C30M
permutations, seed 20260728; `one` = the **slower** of two independent per-model searches — the
conservative choice, since an optimizer output is a *lower bound* on the true optimum):

| model | "1" layout | one | zero | span | span % of zero | zero SD | zero SE |
|---|---|---|---|---|---|---|---|
| AALTO | `lnfdg-,yehcrstmaoiupxqbwv.k'jz` | 2.232363e11 | 2.431185e11 | 1.988e10 | **8.178 %** | 3.42e9 | 3.42e8 |
| COMMUNITY | `mgndy-lea.tpscbkrouiwzxfqvh'j,` | 2.198280e11 | 2.549949e11 | 3.517e10 | **13.791 %** | 6.12e9 | 6.12e8 |
| POOL | `pctsm.reayfgdlk-niuobzvwx,hqj'` | 2.354386e11 | 2.590278e11 | 2.359e10 | **9.107 %** | 3.95e9 | 3.95e8 |

**Candidate table on the normalized scale** (1 = that model's own optimum):

| layout | AALTO | COMMUNITY | POOL | blend (1,1,1) | ms/char |
|---|---|---|---|---|---|
| flagship-c3 | 0.951240 | 0.852640 | 0.848070 | **0.883983** | 254.9761 |
| keybo-lsb | 0.962586 | 0.813997 | 0.831031 | 0.869205 | 254.6307 |
| keybo-lsb+lm | 0.960237 | 0.818018 | 0.819505 | 0.865920 | 254.6847 |
| arm-B | **0.987884** | 0.775900 | 0.818401 | 0.860728 | **253.9006** |
| graphite | 0.818285 | 0.838198 | 0.787631 | 0.814705 | 258.1696 |
| semimak | 0.839953 | 0.807498 | 0.760132 | 0.802528 | 257.3915 |
| arm-A | 0.856669 | 0.763106 | 0.751894 | 0.790556 | 256.8466 |
| qwerty30m | 0.564879 | 0.424298 | 0.523927 | 0.504368 | 264.1389 |

**Unit tests: 21, sentinel rc=0**, and the harness is mutation-controlled (a planted
`assert False` gives rc=1; restored gives rc=0). The guards that BITE:
- **native-frame guard** refuses standardized arrays (test builds a tree whose `.native` files
  *are* the standardized arrays and asserts construction raises);
- **direction guard** raises on an inverted sign;
- **`fit`** pinned against the shipped `surfaces.score_fit` (positive control on the arithmetic);
- **batch-length invariance** — see the load-bearing bug below.

### Anchor uncertainty

- **"1" (trap 1):** seed gap **exactly 0.0** on all three models; both seeds returned the same
  layout. Champion last improved at **epoch 4–12 of 55**, then 43–51 quiet epochs. Anchor-induced
  perturbation of the equal-weight blend: **0.000000** against a **0.003284** decision margin
  (the smallest adjacent gap among the 8 candidates) ⇒ **normalization is STABLE**.
- **"0" (trap 2):** the statistic is the **mean** — its SE falls as sd/√n, which makes the
  n=100-vs-n=1000 check a clean √n comparison; median reported alongside and agrees to well
  under one SE. **n=100 is sufficient:** moving to n=1000 shifts the anchor by **−0.979 SE
  (AALTO, = −1.70 % of span), +0.128 SE (COMMUNITY), −0.287 SE (POOL)**, and the candidate
  ranking is **unchanged at n=100, n=1000 and n=10 000** (flagship-c3's blend moves
  0.883983 → 0.883575 → 0.882677).

---

## B. Does normalizing change any ranking? **No.**

| comparison | discordant pairs | clears the floor? |
|---|---|---|
| within each model, raw ms vs normalized | **0** (impossible by construction — asserted in code) | n/a |
| blend (1,1,1) vs **raw mean of the 3 surfaces' ms** | **0** — rankings identical | n/a |
| blend (1,1,1) vs **raw mean saved-vs-qwerty %** | **0** — rankings identical | n/a |
| blend (1,1,1) vs **raw `min()`** (the scale-broken floor) | **2**: graphite>semimak, graphite>arm-A | **NO** — gaps 0.0122 / 0.0241 vs a **0.231897** conservative normalized floor |
| **search-anchored vs prior ceiling-fraction-anchored blend** | **0** — rankings identical | n/a |

That last row is the direct test of the design's own claim. Switching the "1" anchor from
best-of-a-fixed-set to a per-model optimized search shifts every candidate's blend by a nearly
**constant +0.058…+0.104** and reorders **nothing**. The improvement is real as *scale
provenance* (the anchor becomes a property of the model rather than of the sample) and
measurable in span (search "1" beats best-of-8 by **1.21 % / 14.74 % / 15.19 %** of span), but
it buys **zero ranking change** on this candidate set.

**The resolution floor, computed on MY pool** (a paired floor must name its pool):
- **Model-disagreement floor**, pool = these 8 candidates × 3 native surfaces: conservative
  (max pair spread) **0.231897** normalized, median 0.106982; only **15 of 28** pairs have a
  sign that agrees on all three models. Model main effect **14.10 %** of SS, interaction 5.79 %.
- **Seed-noise floor**, from `COMMUNITY_BASE` — the **only** per-seed family that survives
  (trap 14 confirmed: nothing survives for AALTO, POOL, or `TRI_PS_FREQ_PRIOR`): paired
  **0.3914 saved %** excluding the reference row. Positive control: the seed mean rebuilds the
  shipped native array to **exactly 0.0**.
- ⚠ **FLAGSHIP-1's "seed = 78–83 % of SS" does NOT transfer.** On blend-v1 the seed main effect
  is **0.74 % on raw ms / 0.83 % on saved %** — two orders of magnitude smaller. I computed it
  rather than reusing it, and on this corpus the paired-vs-unpaired distinction buys far less
  than it did on iWeb.

---

## C. The blend search at campaign budget

**Champion `pctsk-reayfgdlm.niuobzvwxh,qj'`** — blend **0.951258**, **9,811,784 unique evals**,
40 islands × 55 epochs, seed 20260728, per-epoch checkpointing, 40/40 islands within 0.01 %.

Independent ms/char via the shipped `keybo analyze --json` (a different quantity from the
search's own objective). **Set-containment of the requested layouts asserted, never a count** —
`analyze` adds a `--ref` row (`qwertyuiopasdfghjkl;zxcvbnm,./`), and the **frozen comparison set
reproduced to a worst absolute difference of exactly 0.0**:

| layout | ms/char | vs arm B |
|---|---|---|
| arm-B | 253.9006 | — |
| sweep (1,0,0) | 254.0711 | +0.1706 |
| keybo-lsb | 254.6307 | +0.7302 |
| flagship-c3 | 254.9761 | +1.0755 |
| sweep (2,1,1) | 255.7811 | +1.8806 |
| **blend (1,1,1)** | **256.6268** | **+2.7262** |
| arm-A | 256.8466 | +2.9460 |
| qwerty30m | 264.1389 | +10.2383 |

The champion **beats arm B on 2 of the 3 native surfaces** (COMMUNITY 221.05 G < 227.71 G,
POOL 235.78 G < 239.72 G) and loses on AALTO — which is exactly what a blend optimizing three
surfaces should do against a layout optimized for the served metric.

---

## D. The preference sweep — the weight behaves as a preference

All five cells at **identical budget, islands, epochs and seed**, so a difference between cells
is the **weight**, not the draw (asserted in the driver).

| cell | weights | champion | AALTO | COMM | POOL | ms/char |
|---|---|---|---|---|---|---|
| equal | 1,1,1 | `pctsk-reayfgdlm.niuobzvwxh,qj'` | 0.90286 | 0.96533 | 0.98558 | 256.6268 |
| aalto-only | 1,0,0 | `lnfdg-,yehcrstmaoiupxqbwv.k'jz` | **1.00000** | 0.75873 | 0.81118 | 254.0711 |
| community-only | 0,1,0 | `mgndy-lea.tpscbkrouiwzxfqvh'j,` | 0.82006 | **1.00000** | 0.97883 | 258.3823 |
| pool-only | 0,0,1 | `pctsm.reayfgdlk-niuobzvwx,hqj'` | 0.86220 | 0.96828 | **1.00000** | 257.6572 |
| aalto-pref | 2,1,1 | `csthg-reaypfdlm.nioubzvwkx,jq'` | 0.93740 | 0.93720 | 0.97262 | 255.7811 |

- 🟢 **End-to-end positive control.** A solo cell's blend maximum is 1.0 *by construction* at
  that model's "1" anchor. All three returned **own_blend = 1.000000000** AND **reproduced the
  anchor's own layout**. This is the check an anchor/objective mismatch could not survive, and
  no unit test can see it.
- 🟢 **The weight is a preference, monotonically.** AALTO's normalized score:
  **1.00000 at (1,0,0) → 0.93740 at (2,1,1) → 0.90286 at (1,1,1)**. Raising AALTO's weight
  moves the champion toward AALTO's optimum without reaching it.
- 🟢 **The three models disagree in the near-optimal band.** Solo champions are **24, 26 and 24
  of 30 slots apart** — despite a wide-random-pool participation ratio of only 1.17 of 3. High
  correlation on a wide pool does *not* imply agreement in the narrow band a search operates in.

---

## E. Admissibility — no dominator

**10-axis frame with the strict-win term required** (`n_ge == 10 AND n_strict ≥ 1`):

| champion | dominator? | best n_ge | best n_strict | normalized floor |
|---|---|---|---|---|
| blend (1,1,1) | **False** | 5/10 | 5 | **+0.902863** |
| sweep (1,0,0) | **False** | 7/10 | 7 | +0.758730 |
| sweep (0,1,0) | **False** | 6/10 | 6 | +0.820062 |
| sweep (0,0,1) | **False** | 6/10 | 6 | +0.862198 |
| sweep (2,1,1) | **False** | 6/10 | 6 | +0.937204 |

⚠ **The n_ge NUMBERS are not comparable to other arms', only the VERDICT is.** My `floor` axis
is *this arm's* min-over-three-models normalized score, not arm E's six-surface
ceiling-fraction floor — a layout optimized against it naturally scores well on it.

**19-gauge frame:** win counts **4–10 of 18 movable**. `analyze`'s `gauges` block carries only
**15** of the 19 (`bad-redir` and `onehand` are absent); `genkey`/`oxeylyzer1`/`oxeylyzer2`/`wfd`
are `community` scores, counted separately. **`sfr` is excluded** — verified a permutation
invariant over our 13 C30M layouts (one value, numpy std **exactly 0.0**), so counting it would
inflate every denominator by one.

---

## Defects found

### In the user's design (reported, and the user's version implemented anyway)

1. 🟢 **"qwerty30m must be ~0" is FALSE**, and is unusable as the direction guard the brief
   proposes. qwerty30m sits at the **0.00–0.20 percentile** of a 1000-layout random pool
   (z = −2.5 … −3.1), so it normalizes to **0.50–0.62** — mid-scale. Anyone using "qwerty ≈ 0"
   to check the sign would see it fail on a **correctly** signed implementation and might
   "fix" the sign, causing exactly the inversion trap 3 warns about. Pinned as a test. The
   direction guard I use instead is the random-pool anchor itself plus monotonicity.
2. 🟢 **The [0,1] scale spends most of its range where no candidate lives.** Excluding
   qwerty30m, the 7 optimized/community layouts occupy just **0.1696 / 0.0895 / 0.0962** of the
   per-model range and **0.0934** of the blend range. The "0" anchor being a *random-layout*
   mean means ~90 % of the scale is unused by real candidates — so normalized differences look
   numerically tiny, and that is a property of the anchoring, not of the layouts.
3. 🟢 **The correction the design buys is very unequal across models.** Search-anchoring beats
   ceiling-anchoring by 14.74 % / 15.19 % of span on COMMUNITY and POOL but only **1.21 %** on
   AALTO, because AALTO is already nearly saturated by existing layouts (arm B is at 0.9879 of
   AALTO's own optimum). A uniform-looking scheme delivers a very non-uniform correction.
4. 🟢 **Equal weights are not neutral.** Effective number of independent models on a homogeneous
   n=10 000 random pool: **participation ratio 1.1672 of 3** (PC1 = 92.34 % of variance, Kaiser
   count 1); ρ(A,C)=0.8310, ρ(A,P)=0.8729, ρ(C,P)=0.9502. The brief estimated ~2 independent
   votes; on the wide pool it is ~1.17. (But see D: they *do* disagree in the near-optimal band.)
5. 🟢 **Baked at 90 WPM.** 7 of 8 per-seed models are gone, so a 90–110 WPM objective **cannot**
   be honoured on these columns without a retrain.

### In my own work (found by testing, not by reading)

6. 🟢 **The zero-padding in `fit_batch` is load-bearing.** BLAS selects its kernel from the
   operand shape, so without a constant tile shape a layout's fit depended on **how many other
   layouts shared its batch** (~1e-15 relative). The objective would not have been a function
   of the layout, and neither the search nor its checkpoint-resume would be reproducible.
7. 🟢 **A resume with different `--epochs` is a different search.** `per_epoch` is derived from
   `--epochs`, so resuming with another value rescales every remaining epoch's spend under the
   same output filename. The checkpoint now stamps a `run_identity` over 10 knobs and refuses
   any mismatched resume, naming the differing key. Verified to bite.
8. 🟢 **`sfr` looked non-invariant** (2 values, numpy std 1.2e-3) because I tested across *all*
   `analyze` rows — including its `--ref` row, which is **classic** qwerty with `;./`. `sfr`
   counts doubled *letters*, so a different **charset** legally moves it; the invariance claim
   is over permutations of **one** charset.
9. 🟢 **A paired/unpaired ratio of exactly 1.0000** in the seed floor was a **degeneracy**, not
   a finding: `saved%` is computed per-seed against qwerty, so the reference row is (0,0,0) and
   `spread(X − qwerty) ≡ spread(X)`. Excluding it gives the readable 0.5632.
10. 🟢 **A real B023** in `seed_floor.floor_of` (closed over the loop's `qwerty_row` instead of
    taking it as a parameter) — latent today because it is called within the same iteration, a
    live bug the moment a second per-seed family survives.

**The lint cleanup touched every driver, so I re-ran the whole pipeline and compared artifacts
bit-for-bit rather than judging the diff cosmetic:** 6 of 7 **identical**; the 7th differs by
exactly the two provenance fields I added on purpose (`tile`, `numpy`), +34 bytes, every
measured number byte-identical (`gate-reverify.log`).

---

## Predictions: 11 held, 5 failed, 2 untestable

Full scoring in `artifacts/PREDICTION-SCORED.md`; pre-registered in commit `412e58f`, **before**
`runs/` existed. The failures, briefly:

- **P6 FAILED** and is the most informative: I predicted normalizing would re-order something
  versus a raw aggregate. It re-orders **nothing**. Normalization changes the *interpretation*
  of the weight (P14 held) while doing **no work** on the equal-weight ranking.
- **P1 and P13 FAILED** on AALTO for one shared reason (defect 3 above).
- **P15 FAILED as a bound** (n_ge 5/10 and 7/10 vs my ≤4) while **holding as a verdict** (no
  dominator) — I set the bound without accounting for my `floor` axis being a different quantity.
- **P17/P18 FAILED on the letter, held on the mechanism**: I forgot qwerty30m is one of the 8
  candidates. Excluding it, the window is *tighter* than I predicted — the defect is worse than
  I stated.
- **P3 UNTESTABLE** because all three models' seeds landed on the identical layout (0/0).

---

## What I would tell the user

The scheme is **correctly motivated and correctly implemented, and it does not change any
ranking on this candidate set.** Its real product is not a re-ordering — it is that the
**weight becomes interpretable**: the (1,0,0)/(0,1,0)/(0,0,1)/(2,1,1) sweep moves the champion
monotonically and by large amounts (24–26 of 30 slots between solo champions, 4.3 ms/char
between the extreme cells), which an unnormalized weighted sum could not have delivered
cleanly. If the goal was "make the weight mean preference", it worked. If the goal was "get a
faster layout than arm B", it did not: **256.6268 vs 253.9006**.

Two caveats to carry forward: the "0" anchor wastes ~90 % of the scale on random layouts (a
percentile or an incumbent-based "0" would spend the range where candidates actually live), and
with an effective model count of ~1.17 on the wide pool, "equal weights" is a choice that needs
defending rather than a neutral default.

---

## Artifacts

All under `/local/home/zegertho/agent/state/modelnorm/artifacts/`, and committed into
`drivers-modelnorm/` on branch `modelnorm`:

| file | what |
|---|---|
| `drivers/modelnorm_eval.py` | native-frame evaluator + the normalization |
| `drivers/test_modelnorm_eval.py` | 21 unit tests (sentinel `unit-rc.txt` = 0) |
| `drivers/search_modelnorm.py` | one engine for both anchor steps and the blend |
| `drivers/{step1_zero_anchor,build_anchors,rank_table,seed_floor,sweep_report,judge_modelnorm}.py` | the six analysis stages |
| `anchors.json` / `anchors-evidence.json` | anchors of record + stability/convergence evidence |
| `step1-zero-anchor.json` | the "0" anchor at n=100/1000/10000 × 4 seeds, correlations |
| `rank-table.json` | deliverables A and B |
| `seed-floor.json` | the seed-noise floor and the per-seed inventory |
| `sweep-report.json` | deliverable D |
| `judgement.json` | deliverables C and E |
| `runs/anchor-*.json` (6) · `runs/blend-*.json` (5) | every search, with per-epoch curves |
| `PREDICTION.md` / `PREDICTION-SCORED.md` | pre-registration and honest scoring |
| `gate-resume.log` / `gate-reverify.log` | the two gates |
