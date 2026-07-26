# Direction of travel: giving the gauge the channel it lacked, and measuring what changes

**Workspace:** `direction` (subagent of `keybo-optimization`) · **Date:** 2026-07-26
**Branch:** `direction-features` (local only — no push, no CR, no publish)
**Worktree:** `state/direction/wt-direction`, off `main` @ `9ce0563`
**Artifacts:** `state/direction/artifacts/` · **Drivers:** `state/direction/scratch/`

> **Frame, stated once and true of every number below.** Modelled only, on the **served
> frame** `g(geometry, wpm)`; the additive per-n-gram term `b(ngram)` is excluded throughout
> (layout-independent, ranking-irrelevant — `train.py`: "scoring deliberately ignores it").
> WPM 90, `ROW_STAGGERED_30` + space, 3 seeds, seed-mean tensors. Corpus-weighted quantities
> use `data/corpus/bigrams.txt` (iWeb, **single-source**, ledger `GAP-CORPUS-1`), md5
> `d115e052f215c77d34652a59e4c3901e`. Held-layout tau is saturated and Phase-D is cancelled,
> so **nothing here is a claim about realized human typing speed** and nothing is an adoption
> recommendation. **POOL is not independent** — it contains AALTO and COMMUNITY.

---

## The answer in one paragraph

I gave the bigram model a real direction-of-travel channel and refit. **The served surface did
not collapse** (optimizer-tensor Spearman 0.888–0.949 against v1; the NGRAM-FE disaster was
0.852 → 0.164), so this is **not a REJECT**. But **direction carries no cross-source signal**:
across 3 surfaces the incumbent board produced **10 pairwise flips and 0 that clear the ~1
ms/char resolution floor**, the held-out LOLO effect has **opposite sign by source** (AALTO
rho/ceiling −0.0134 and umae +0.314 ms *worse*; COMMUNITY rho +0.027 and umae −0.274 ms
better; POOL rho/ceiling −0.0199, the worst of the three), and on a **single common key set** only one of four roll classes even agrees in sign
— failing the magnitude bar by 20× (flat inrolls −10.42 / −0.52 / −1.60 ms, with COMMUNITY's
value inside its own seed spread). **So: direction is now expressible, and the honest answer
is that it changes essentially nothing about what we optimize for.** That retires the
community's inroll/outroll argument on a stronger footing than THEORY-1 could — not "the
gauge cannot see it" but "the gauge can now see it, and there is nothing there."

---

## 1. What I built, and the two things that nearly went wrong

### The v2 feature surface (commit `ab3ceee`, additive)

`FEATURE_VERSION` stays `2026-07-05.3` and the v1 column lists are untouched, so every
shipped artifact in `data/models/k31/` keeps loading and scoring bit-identically. The new
surface is opt-in via `direction=True` and stamps a **separate** `FEATURE_VERSION_DIRECTION =
"2026-07-26.1"`. That is the deliberate design: a real version bump that does **not**
invalidate the shipped models, because the two stamps coexist and `TypingModel.load` refuses a
mismatch in either direction.

Nine columns, each verified order-dependent **and** not already determined by v1:
`signed_dx`, `dir_dx_inward`, `dir_angle`, `dir_inwards`, `dir_outwards`, and the origin-finger
one-hot `o_pinky`/`o_ring`/`o_middle`/`o_index`.

### ⚠️ Finding: swap-dependence is necessary but NOT sufficient

The brief asked for the swap-difference count per candidate first. Every candidate passed that
test — and two of them are still **worthless**. Grouping all 870 ordered pairs by their exact
v1 vector and asking whether a candidate varies *within* a collision group shows:

| candidate | swap-differing pairs (of 870) | varies within v1 groups | verdict |
|---|---|---|---|
| `signed_dx` | 870 | 129 groups / 270 pairs | **new information** |
| `dir_dx_inward` | 360 | 6 / 24 | new information |
| `dir_angle` | 324 | 6 / 24 | new information |
| `dir_inwards` / `dir_outwards` | 324 each | 6 / 24 | new information |
| origin **finger** one-hot | 288–432 | 3 / 12 | new information |
| **`signed_dy`** | **600** | **0** | ⚠️ **already determined — rejected** |
| **origin ROW one-hot** | **600** | **0** | ⚠️ **already determined — rejected** |
| `o_lateral` | 288 | 0 | already determined — rejected |

**Mechanism 🟢:** `dx` is *stagger-adjusted*, and the per-row offsets differ
(`{bottom:+0.5, home:0.0, top:−0.25}`), so `dx` **leaks the origin row**. Concretely
`a=(−5,1)→b=(5,2)` gives `dx=9.50` while `a=(−5,3)→b=(5,2)` gives `dx=10.25` at identical
`dy=1` and `distance=10.0499`. Drop the stagger and origin-row ambiguity appears in 32 groups
/ 128 pairs. Had I trusted the swap test alone I would have shipped two null columns dressed
as a direction channel.

This also **sharpens THEORY-1**. The gauge's blindness is not "the origin key is invisible":
the origin row is recoverable, the origin key is ambiguous in only 159 of 699 distinct v1
vectors, and **only 30 of 870 ordered pairs (15 unordered, all cross-hand mirror pairs) have a
featurewise-identical reverse**. The missing quantity is specifically the **sign of travel**,
and it is a *small* channel — which bounds the achievable gain a priori. Under v2 the count of
featurewise-identical reverses goes 30 → **0**.

### ⚠️ The frame-width artifact, and why a placebo arm was mandatory

Going v1 → v2 changes **two** things: direction information is added *and* the frame grows by
9 columns, which by itself moves an XGBoost fit (colsample_bytree draws, split search,
effective regularization). Per `TOOLING-TRAPS #17` I built a **same-width placebo**: 9 columns
built only from v1-determined quantities (origin row, `signed_dy`, `o_lateral` + copies), so
it is exactly as wide and carries exactly zero new information. Every direction effect below
is reported as **placebo → v2**, never v1 → v2.

This was not bookkeeping. On the served tensor the **width artifact alone** moved COMMUNITY's
mean asymmetry by **+2.92 ms** while the attributable direction effect was **+2.13 ms** — the
artifact is *larger than the effect*. Reading v1 → v2 would have overstated direction by
roughly 2×. Its axis is deliberately nested in v2's information, which is the conservative
choice: it understates v2's marginal effect, so an "inert" verdict survives the bias.

### Two of my own errors, caught and fixed

1. **My v1-parity test was self-referential.** It compared v1 against v2 out of the *same*
   code path. A negative control (perturb `dy` by 0.001) left all 32 assertions **green**; the
   repo's frozen `golden_k30_features.npz` — an independent evidence path — caught it. The
   test now asserts against the golden matrix, and the same sabotage now fails it correctly.
2. **The placebo arm was silently building 20-column matrices** for 29-column models in my
   served driver. XGBoost's shape guard caught it (`expected: 29, got 20`); without that guard
   it would have been a wrong-arm comparison with plausible-looking output.

**Tests:** 86 feature tests pass; **full suite 788 passed, 3 skipped** (30 min). The one
warning is pre-existing on base `main` (verified in a throwaway worktree at `9ce0563`).

---

## 2. LOLO refit — per surface

Production `REG_LOLO` recipe, LOGRAT, 3 seeds, 4 folds. **`tau_min = 1.000` in every AALTO and
COMMUNITY arm** — the decisive gate holds there; direction neither helps nor breaks layout
ranking. POOL does not saturate tau (0.929 at v1, 8 layouts), and its v2 tau equals its v1 tau.

**AALTO** (2158 rows / 29.3M samples; ceilings well-defined: qwerty 0.982/54690 pids, qwertz
0.854/485, azerty 0.778/166, dvorak 0.652/64):

| arm | rho/ceiling | rho | umae | wmae | tau_min |
|---|---|---|---|---|---|
| v1 | **1.0245** | 0.8248 | 15.654 | 9.773 | 1.000 |
| placebo | 1.0170 | 0.8194 | 15.615 | 9.788 | 1.000 |
| v2 | **1.0036** | 0.8102 | 15.929 | 9.894 | 1.000 |
| **attributable** (placebo→v2) | **−0.0134** | −0.0092 | **+0.314** | +0.106 | — |
| (width) v1→placebo | −0.0075 | −0.0054 | −0.039 | +0.015 | — |

v1's `1.0245` **reproduces the registered REG-LOLO baseline of 1.0236** to 0.0009 — re-derived
independently rather than taken from the brief (`TOOLING-TRAPS #20`).

**COMMUNITY** (1775 rows / 51.8k samples):

| arm | rho/ceiling | rho | umae | wmae | tau_min |
|---|---|---|---|---|---|
| v1 | **n/a** | 0.6024 | 20.181 | 18.699 | 1.000 |
| placebo | n/a | 0.6041 | 20.036 | 18.704 | 1.000 |
| v2 | n/a | **0.6315** | **19.762** | **18.430** | 1.000 |
| **attributable** | n/a | **+0.0274** | **−0.274** | −0.274 | — |
| (width) | n/a | +0.0018 | −0.145 | +0.004 | — |

> ⚠️ **I cannot report rho/ceiling for COMMUNITY, and this is structural, not a gap in the
> run.** `split_half_ceiling` bisects *participants*, and every COMMUNITY layout has exactly
> **one** participant (colemak#alite 1, custom-aa426873#vg 1, custom-d42a1f92#ddn 1,
> mtgap-variant#richarddavison 1), so the ceiling is `nan` and the ratio is `None` for every
> fold. Raw centered rho, umae, wmae and tau are reported instead. Checked as keys, not prose
> (`TOOLING-TRAPS #19`).

**POOL** (3933 rows / 29.4M samples; **not independent** — contains both of the above):

| arm | rho/ceiling | rho | umae | wmae | tau_min | tau_mean |
|---|---|---|---|---|---|---|
| v1 | 1.0053 | 0.7260 | 17.838 | 14.432 | 0.929 | 0.929 |
| placebo | 1.0122 | 0.7322 | 17.708 | 14.190 | **0.857** | 0.905 |
| v2 | **0.9923** | 0.7346 | 17.789 | 14.203 | 0.929 | 0.929 |
| **attributable** | **−0.0199** | +0.0024 | +0.081 | +0.013 | — | — |
| (width) | +0.0069 | +0.0063 | −0.129 | −0.242 | — | — |

**⚠️ Cross-source split, and it is the load-bearing one.** The attributable direction effect on
held-out transfer has **opposite sign by source** on the magnitude metrics: AALTO degrades
(rho/ceiling −0.0134, umae **+0.314 ms worse**) while COMMUNITY improves (rho +0.027, umae
−0.274 ms better) and POOL is mixed (rho/ceiling **−0.0199**, the worst of the three, while raw
rho barely moves at +0.002). The two surfaces where rho/ceiling is computable **both degrade**
on it, by 1.8× and 2.9× their own width artifacts respectively — so the degradation is
attributable to the direction columns, not to frame growth. Per the campaign standard this is a
**per-dataset artifact, not a transfer win** 🟠.

**A side result worth recording:** POOL's placebo arm — which adds *nine columns of zero new
information* — moved `tau_min` from 0.929 to **0.857** on one seed, and moved wmae by −0.242
(larger than v2's own attributable +0.013). Frame width alone perturbs this pipeline's
layout-ranking gate. That is the empirical case for the placebo arm being mandatory, not a
formality 🟢.

---

## 3. The NGRAM-FE gate — served rho and the optimizer tensor

The trap the brief says must decide the verdict: *a model that fits better on the full frame
while the served geometry collapses is a **REJECT***. Measured on the 930 off-diagonal cells of
the 31×31 layout-neutral serve grid — the object the optimizer actually queries, not training
data (the corpus is ~98.7% qwerty; `OQ-1`: correlation is not price).

| surface | rho(v2,v1) | rho(placebo,v1) | **rho(v2,placebo)** | mean\|Δ\| v2−v1 | (width) | flips | **RESOLVED flips** | board rho |
|---|---|---|---|---|---|---|---|---|
| AALTO | 0.9495 | 0.9615 | 0.9281 | 3.42 ms | 2.41 ms | 2 | **0** | 0.9667 |
| COMMUNITY | 0.9344 | 0.9758 | 0.9245 | 8.36 ms | 4.70 ms | 3 | **0** | 0.9333 |
| POOL | 0.8880 | 0.9629 | 0.8725 | 6.80 ms | 4.42 ms | 5 | **0** | 0.8667 |

**This is NOT a served collapse, on any surface.** Per-seed rho is stable (AALTO
.948/.945/.943; COMMUNITY .932/.929/.932; POOL .876/.884/.893). The NGRAM-FE precedent went
0.852 → 0.164 with 0 % optimizer agreement; the worst number here is 0.872. So the change is
**not a REJECT** — and I say that in those words because the brief asked for the verdict in
them.

Note the width control: on every surface a large fraction of the raw v1→v2 tensor movement is
the placebo's, i.e. frame width, not direction.

### Does it reorder the incumbents?

**No.** Ten pairwise flips across the three surfaces, **none** of which clears the ~1 ms/char
resolution floor (per-seed layout spreads 0.70–0.99, ledger `E1`). The largest flip gap
anywhere is **0.2741 ms/char** against a seed spread of **0.2528** on that same pair.

AALTO board (ms/char, lower = faster, ±3-seed spread):

| layout | v1 | placebo | v2 | v2−v1 |
|---|---|---|---|---|
| keybo-lsb | 128.289±0.182 | 128.702±0.123 | **128.267±0.126** | −0.022 |
| keybo-lsb+lm | 128.329±0.183 | 128.717±0.124 | 128.294±0.134 | −0.035 |
| lsb-sib | 128.436±0.179 | 129.004±0.135 | 128.511±0.145 | +0.074 |
| archive-1846 | 128.437±0.184 | 128.968±0.128 | 128.494±0.150 | +0.057 |
| archive-1843 | 128.465±0.186 | 128.963±0.133 | 128.543±0.156 | +0.078 |
| flagship-c3 | 128.486±0.191 | 129.112±0.132 | 128.552±0.143 | +0.066 |
| graphite | 129.336±0.180 | 129.815±0.149 | 129.921±0.129 | +0.586 |
| semimak | 129.384±0.202 | 129.896±0.160 | 129.882±0.158 | +0.498 |
| qwerty | 133.179±0.170 | 133.792±0.081 | 133.153±0.108 | −0.026 |

The two AALTO flips are `lsb-sib ↔ archive-1846` (gaps 0.0004 → 0.0166, spread 0.184) and
`graphite ↔ semimak` (0.0484 → 0.0395, spread 0.202). The second is theory-1 `D5`'s
already-known-unresolved pair, so its flipping here is consistent with prior art rather than
new. **Leaders and qwerty-last are unchanged on every surface.**

---

## 4. What a direction-aware objective favours — the structural question

Estimator: **matched reverses** on the serve grid. For each unordered pair the tensor gives
both orderings, so the pair's direction effect is `T2[a,b] − T2[b,a]`; the attributable part is
that quantity in v2 minus the same in the placebo.

⚠️ **The first cut of this was confounded**, in exactly the way `TOOLING-TRAPS #16` warns: each
surface's "well-supported" pair set is a **different key set** (AALTO 100, COMMUNITY 61, POOL
112 of 162), so comparing their contrasts mixes the source with the composition. The numbers
below are on the **51 pairs supported on all three surfaces**.

### Inroll minus outroll (negative = inrolls faster), attributable, common 51-pair set

| class | n | AALTO | COMMUNITY | POOL | sign agrees? | all > own spread? |
|---|---|---|---|---|---|---|
| all spans | 51 | −4.187±0.140 | −0.263±0.770 | +0.081±0.226 | **no** | no |
| **flat (span 0)** | 19 | **−10.416±0.604** | **−0.515±0.871** | **−1.601±0.345** | **yes** | **no** |
| span 1 | 28 | −0.167±0.477 | −0.313±1.410 | +0.668±0.306 | no | no |
| span 2 | 4 | −2.740±0.465 | +1.287±0.678 | +3.957±2.096 | no | yes |

**Only one of four classes agrees in sign across sources, and it fails the magnitude bar.**
Flat rolls are where `dir_angle` genuinely adds separation (`rotation_angle` is non-zero on
216 of 870 pairs, `directed_angle` on 270; the extra 54 are exactly the flat rolls, because
outer→inner on a flat pair is always `atan2(0,+x)=0` and so collapses flat-inward with
flat-outward). Yet the three magnitudes span **20×**, and COMMUNITY's −0.515 sits **inside its
own 0.871 seed spread** — a null. **AALTO is the outlier, not the consensus.**

### Is the effect learned, or invented in a null space?

A direction column can only carry a *learned* effect where **both** orderings of a pair were
observed; elsewhere the model extrapolates into a null space — the
`goodhart-row-blindness` failure mode, where the optimizer queries off the training
distribution.

| surface | flat pairs both-supported | median min-n per flat pair | attributable, flat **supported** | attributable, flat **unobserved** |
|---|---|---|---|---|
| AALTO | 32/54 | 43 | −9.040±0.451 | −11.693±0.602 |
| COMMUNITY | 23/54 | **2** | −0.633±0.821 | **+62.661±6.766** |
| POOL | 37/54 | 163 | −1.750±0.603 | +18.749±4.149 |

On COMMUNITY and POOL the unobserved-pair effect is enormous and the supported-pair effect is
small or null — the model inventing structure where it has no data. On AALTO both are large.
**COMMUNITY's flat-pair support is desperately thin (median 2 samples), so its null may be a
power problem rather than a contradiction — that is a hypothesis, not a licence to prefer
AALTO.** POOL, which has by far the best support (median 163), lands at **−1.75 ms**, near
COMMUNITY and far from AALTO; but POOL is not independent, so it cannot break the tie by
itself.

### Hand symmetry — a falsification test the effect only half passes

A biomechanical inroll advantage is a property of the hand and should appear on both hands at
similar magnitude. Asymmetry points at the model fitting qwerty-specific placement instead.

| class | surface | left | right | L−R | same sign? |
|---|---|---|---|---|---|
| all spans | AALTO | −1.998±0.218 | −6.292±0.230 | +4.294 | yes |
| all spans | COMMUNITY | −0.656±0.963 | +0.115±0.653 | −0.771 | **no** |
| all spans | POOL | +0.705±0.181 | −0.519±0.370 | +1.224 | **no** |
| flat | AALTO | −9.034±0.875 | −11.952±0.306 | +2.918 | yes |
| flat | COMMUNITY | −0.276±0.679 | −0.781±1.115 | +0.505 | yes |
| flat | POOL | −1.664±0.705 | −1.531±0.126 | −0.133 | yes |

Flat rolls are sign-consistent across hands on all three surfaces — a genuine point in the
effect's favour, and I report it as such. But on all-spans, **two of three surfaces give the
two hands opposite signs**. The hand test therefore does not rescue the effect: it is
hand-consistent only in the single class whose magnitude is source-unstable by 20×.

### Which finger ends higher

Attributable (v2 − placebo): **AALTO −4.366 ms** vs **COMMUNITY +2.427 ms** — split. Note this
contrast is structurally confounded (reversing a two-row pair always swaps the landing key, as
theory-1's self-audit #3 found), which is precisely why only the placebo-subtracted delta is
quoted: the landing-row price is identical in both arms and cancels. Consistent with theory-1's
finding that `_PREFERRED_HEIGHT` does not replicate as a general rule.

---

## 5. So how does this change what we optimize for?

**It doesn't — and that is the result.**

* **Nothing collapsed.** Served tensor rho 0.888–0.949; no REJECT.
* **Nothing reordered.** 10 flips, 0 resolved above the ~1 ms/char floor, on 3 surfaces.
  Leaders (`keybo-lsb`, `keybo-lsb+lm` on AALTO) and qwerty-last are untouched.
* **No cross-source transfer gain.** The LOLO attributable effect is opposite-signed by source,
  and **both surfaces where rho/ceiling is computable degrade on it** (AALTO −0.0134, POOL
  −0.0199). `tau_min` is unchanged by direction on every surface.
* **No robust structural preference.** One of four roll classes agrees in sign, at magnitudes
  spanning 20×, with the best-supported surface near zero and the effect concentrated on
  unobserved pairs.

The community's inroll/outroll argument is now retired on a **stronger** footing than
THEORY-1's. THEORY-1 said the gauge *cannot represent* the question. This says: the gauge now
**can** represent it — 30 → 0 featurewise-identical reverses, nine order-dependent columns,
a full retrain on three surfaces — and where the data can constrain the answer, there is
**~0–1.75 ms/char** of it, below the instrument's own resolution and without cross-source
agreement. Upgrading theory-1 `C5` from "coin flip, and the wrong instrument" to
"**right instrument now, still nothing there**" 🟢.

**One boundary to keep.** This is about the **bigram** feature vector. The COMMUNITY *trigram*
inrolls/outrolls are genuinely directional (9720/9720 order-dependent) because the trigram
vector was already order-sensitive at the trigram level (`redirect` compares the two
constituent bigrams' directions of travel). Nothing here touches that.

---

## Recommendation

**Do not adopt v2 as the served surface.** It is not a REJECT — nothing collapsed — but it
buys no cross-source transfer gain, no resolved reordering, and no robust structural
preference, while widening the feature frame by 45 % and costing a retrain of every shipped
artifact. Keep the code as the **instrument** that settles the direction question, and keep
`FEATURE_VERSION` where it is.

Worth keeping regardless:

1. **The rejected-candidate finding is reusable.** "Swap-dependent" ≠ "new information", and
   the stagger-adjusted `dx` leaking the origin row is a concrete trap for the next agent who
   reaches for `signed_dy` or an origin-row one-hot.
2. **`is_inwards`/`is_outwards`/`rotation_angle` now carry docstrings** saying they are
   swap-invariant orientations, with pointers to the directed twins. That misnaming cost this
   campaign a whole arc.
3. **The same-width placebo pattern** should be standard for any future feature-frame change
   here: on this run the width artifact exceeded the effect being measured.

---

## Reproduction

Drivers in `state/direction/scratch/`: `swap_test.py`, `information_test.py`,
`mechanism_test.py` (the cheap decisive tests); `refit.py` (LOLO, 3 arms × 3 surfaces);
`served.py` (optimizer tensor + incumbent board); `structure.py` (matched reverses);
`support.py` (training-support split); `final_analysis.py` (common-set + hand symmetry).
Outputs and the 27 saved served tensors in `state/direction/artifacts/`. Logs in
`state/direction/logs/`. Run from the worktree with
`PYTHONPATH=$PWD/src /local/home/zegertho/repos/keybo/.venv/bin/python <driver>`.
