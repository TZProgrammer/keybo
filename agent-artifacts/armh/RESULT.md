# ARM H — RESULT

**Prereg `491138b` committed ALONE before any result existed. Drivers `a078611`. Judge
`2b90b47` committed WHILE PHASE 1 WAS EXECUTING** (138 worker processes live, 0 rc sentinels
written, 0 result JSONs on disk — verified at commit time). Every threshold below comes from
the prereg, which predates every run.

**MODELLED ONLY:** g-frame, baked 90 WPM, blend-v1, skipgrams `1-skip31`, and the **as-shipped
NESTED `bad_redirect` oxey convention** (a bad redirect charged +2.0 **and** +4.0 = +6.0) —
*the same convention* SPEEDTIE-1's 14.05× and every ARM G number were measured on, so these
numbers are comparable to those and **NOT** to a post-OXEYFIX board. Nothing here is a claim
about realized human typing speed. **No layout adopted or recommended. Nothing pushed, no CR,
`PREREGISTRATIONS.md` untouched.**

---

## THE HEADLINE, AND THE PART OF IT I KILLED MYSELF

🟢 **① COLLECTED — a 13-axis-feasible layout that is STRICTLY better than arm B on
`oxey-style`, at no cost on any of the other 13 axes. It exists, and there are THREE of them**
(a 676-layout archive sweep found 5 that are 13-axis-feasible, 4 FEASIBLE in-band, 3 COLLECTED):

```
                                        ms/char       Δms      oxey-style      Δoxey   axes   in-band under
arm B     flmpg-yuo,sntdcireahkxbwv'.jzq  253.900579   ——        8.611046      ——      13 ties  (reference)
BALL-1    flmpg-yuo,sntcdireahkxbwv'.jzq  253.966426  +0.065847  7.577429   -1.033616  13/13 ✓  ALL 3 rulers
MID       flmpg.yuo,sntcdireahkxbwv'-jzq  253.988534  +0.087955  7.769027   -0.842019  13/13 ✓  ALL 3 rulers
HEADLINE  flmpg-,uoysntcdireahkxvwb.'jzq  254.039627  +0.139048  4.446491   -4.164554  13/13 ✓  MINE ONLY
```

**MID** (from a warm run's top-50 archive, verified through the gate at all three bands) is
ruler-robust like BALL-1 but improves oxey slightly less. **The frontier is real and it is
ordered: the more oxey you collect, the more speed you pay** — 0.8420 at +0.0880, 1.0336 at
+0.0658, 4.1646 at +0.1390. (BALL-1 beats MID on both, so MID is not on the Pareto frontier of
the two; it is listed because it independently corroborates that the feasible set is not a
single point.)

🔴 **AND THE FIRST THING I KILLED IS MY OWN HEADLINE'S RULER-ROBUSTNESS.** "Speed-tied" is a
statement about a band, and the band is `2*sd_H`. My headline is in-band **only under my own
`sd_H`, which is the LARGEST of the three rulers ever measured on this objective:**

| ruler | sd | 2×sd | HEADLINE (+0.139048) | BALL-1 (+0.065847) |
|---|---|---|---|---|
| **`sd_H` (mine, PRIMARY)** | **0.09952542** | **0.19905085** | **IN-BAND** | **IN-BAND** |
| `sd_G` (ARM G's) | 0.049171 | 0.098342 | **OUT** | IN-BAND |
| borrowed (SPEEDTIE-1) | 0.0617 | 0.1234 | **OUT** | IN-BAND |

=> **The ruler-robust collected result is BALL-1, not the headline.** BALL-1 is in-band under
**all three** rulers; the headline under **one of three**. I verified this through the gate
directly: at `eps = 0.098342` the headline gates **rc=1 / INFEASIBLE on the speed leg with
`viol=0` axes**, while BALL-1 gates rc=0 / COLLECTED. So the honest two-line statement is:

> **the ruler-independent claim is BALL-1 (oxey −1.0336, 13/13 axes, in-band under every
> ruler); the headline is strictly stronger on the target (oxey −4.1646, 4.03× BALL-1's gain)
> but its speed-tie holds only against my own measured sd.**

⚠ **My prereg argued that any residual bias in my band would be TIGHTER (conservative, toward
EMPTY, against my own headline). It came out LOOSER — the same anti-conservative direction ARM
G was caught in.** What my structural fix *does* buy is that the search band and the verdict
band are **the same number by construction**, so a looser `sd_H` cannot open a gap between
them (ARM G's failure); it widens both equally. It cannot make the headline ruler-independent,
and I am not claiming it does.

---

## `sd_H` — MY OWN RULER, AND IT REFUTES MY OWN PREDICTION

**`sd_H = 0.09952542252893681 ms/char`** ⇒ `2*sd_H = 0.19905084505787363` ⇒ band edge
**254.09962994858392**.

**Quadruple, as the standing floor rule requires:** POOL = my 5 ARM-H-family baseline-control
champions (near-optimal, cold start, blend-v1) × REPLICATE-STRUCTURE = independent cold-start
search runs, one champion each × SCALE = raw ms/char × STATISTIC = sd, ddof=1, n=5.

| tag | seed | champion | ms/char | unique ACHIEVED | frac |
|---|---|---|---|---|---|
| baseline-r0 | 31337 | `flmpg-.uoysntdcireahxqbwvk,'zj` | 253.908205 | 1,041,020 | 104.1% |
| baseline-r1 | 136066 | **`flmpg-yuo,sntdcireahkxbwv'.jzq` = ARM B EXACTLY** | 253.900579 | 1,026,095 | 102.6% |
| baseline-r2 | 240795 | `wyou,kfmlrgeacidhtnsq'j.-pbvxz` | 254.141397 | 1,065,534 | 106.6% |
| baseline-r3 | 345524 | `flmpg-yo,usntdcireahkxwbv.'qzj` | 253.937032 | 1,053,550 | 105.4% |
| baseline-r4 | 450253 | `lmfbg.uoyprnstdieahcxzwkv-,'qj` | 253.997605 | 921,447 | 92.1% |

All 5 clear the pre-registered 80% floor, so **the floor excluded nothing and n=5 stands.**

🔴 **P5 FAILED, and the failure is the most transferable thing in this arm.** I predicted
`sd_H` within **1.5×** of the borrowed 0.0617; measured **1.613×** — outside. And note the
symmetry with ARM G: **it predicted ≥1.5× and got 1.255×; I predicted ≤1.5× and got 1.613×.**
Two arms, opposite predictions, both refuted. `sd_H` is **2.024× `sd_G`** on the *same engine,
same budget, same config, same statistic* — differing only in the **seed family**.

🟢 **AND I CAN ATTRIBUTE THAT 2.024× TO THE SEED FAMILY RATHER THAN TO MY ENGINE, because my
reproduction control is BIT-EXACT.** I ran one extra baseline at ARM G's own r=0 seed
(20260728), deliberately *outside* the `sd_H` pool so it could not contaminate my ruler:

```
ARM H repro-armg-r0 : pyu.,vdfnlhieaocstrmkj'-qgwbzx  253.9997247816586  1,044,667 unique
ARM G  baseline-r0  : pyu.,vdfnlhieaocstrmkj'-qgwbzx  253.9997247816586  1,044,667 unique
```

Identical layout, identical fitness **to all 16 digits**, identical unique-eval count. So the
`armh` arm and the repointed worktree literal changed **nothing** on the baseline path.

=> **REGISTERED: a search-noise sd is not transferable even when three of the four quadruple
legs match. The seed family alone moves it 2×.** That is a stronger statement of the standing
floor rule than either arm registered before, and it is measured, not argued.

---

## THE VERDICT, UNDER BOTH READINGS — AND F2 WAS MIS-SPECIFIED IN MY OWN PREREG

**Under the STRICT reading of my own F2, this arm reports ③ FAILURE.** F2 as I wrote it —
*"the champion gate rejects a returned champion"* — **FIRED**: all 5 `armh-cold` champions were
rejected as infeasible.

🔴 **But F2's registered WARRANT is FALSE BY MEASUREMENT, and the defect is mine.** F2's stated
justification was *"⇒ my hardness construction is broken, so nothing I report is
trustworthy."* Measured: **0 of 10 cross-path disagreements.** Every rejected champion was
*also* scored in the objective's infeasible branch (`fitness ≥ 1e6`), so `FastEval` and shipped
`analyze` **agree on every single champion's feasibility**.

**What I got wrong:** F2 conflated *"an infeasible layout was returned as a champion"* with
*"the construction is broken."* A run that finds nothing feasible **must still return
something**, and the engine's archive-best is *by construction* the least-infeasible layout —
so **F2 as written fires on the EXPECTED output of a correctly-working arm.** It should have
been scoped to a **cross-path disagreement** (objective says feasible, gate says not). I am
registering that as my defect rather than reading F2 the way that suits me.

**Resolution, and it does not depend on the choice:** gating the warm champions **alone**
returns **rc=0** (all FEASIBLE by both paths, verified). So *"a feasible, strictly-better
layout exists"* stands under **either** reading. The strict reading costs me the **label**, not
the result — **and I let the strict reading stand on the label.**

> **REPORTED VERDICT: ① COLLECTED (BALL-1 ruler-robustly; the headline under my own ruler
> only) + ③ FAILURE by the strict letter of my own F2, whose warrant I refuted.**

---

## PREDICTIONS: 5 HELD / 1 FAILED — and both self-adverse ones held

| # | Prediction | Outcome |
|---|---|---|
| **P1** | ≥3 of 5 warm seeds return a 13-axis-feasible champion | 🟢 **HELD 5/5** |
| **P2** | 🔴 *self-adverse:* cold returns **ZERO** feasible champions | 🟢 **HELD** — 0 of 5 |
| **P3** | the search beats BALL-1's oxey 7.577429 | 🟢 **HELD** — 4.446491, by **3.130938** |
| **P4** | 🔴 *self-adverse:* no champion is faster than arm B | 🟢 **HELD** — 0 of 10 |
| **P5** | `sd_H` within 1.5× of the borrowed 0.0617 | 🔴 **FAILED** — 1.613× |
| **P6** | `2*sd_H` admits BALL-1 (flip threshold 0.032924) | 🟢 **HELD** — 0.199051 ≥ 0.065847 |

The one that failed is the one about **my own ruler**, which is exactly the quantity this
campaign has been wrong about most often. Both deliberately self-adverse predictions held.

---

## THE LEVER IS REAL AND LARGE — AND *WHY* IT IS NOT FREE IS NOW MEASURED

This is the part that converts SPEEDTIE-1's "free headroom" into a mechanism, and it comes
from the arm that **failed** by its own prediction:

```
armh-cold-r2  fo,.yvgmrlhaeiucdtns'qjk-pwbxz   ms 254.204896 (+0.304)  oxey -5.273023   5 axes violated
armh-cold-r0  pyu.,mgfrlhieaodctnskj-q'bwzvx   ms 254.146098 (+0.246)  oxey  0.808179   5 axes violated
armh-warm-*   flmpg-,uoysntcdireahkxvwb.'jzq   ms 254.039627 (+0.139)  oxey  4.446491   0 axes violated
arm B         flmpg-yuo,sntdcireahkxbwv'.jzq   ms 253.900579  ——       oxey  8.611046   —— (reference)
```

🟢 **Unconstrained, `oxey-style` runs from 8.611 down to −5.273 — a 13.88-unit drop, 3.3× what
the constrained arm collects.** The cold arm went and got it. **It just could not keep the
other axes while doing so:** every cold champion violates **4–5** hard axes and sits
**+0.24 to +0.30 ms/char** out of band.

=> **The headroom SPEEDTIE-1 found is genuine and much larger than the collectable part. What
makes most of it uncollectable is not the speed band — it is the other 13 axes.** The
constrained arm collects **4.1646 of the 13.8841 available** = **30.0%**; the remaining 70% is
priced in axis violations.

**The binding constraint, per leg** (speed in ms/char, axes relative to arm B — reported side
by side, never compared on one scale, since the units differ):

- **speed leg:** relaxation needed = **0.0** — the speed band is **NOT** binding at `sd_H`.
- **axis legs with solo violators** (layouts in-band, improving oxey, violating *only* that
  axis): **`sfs`** and **`sr-roll`**.

⚠ **And the structural reason, registered in my prereg before I saw this:** `oxey-style` is
**R² = 0.9082** on its own six components (`sfb lsb scissor imbalance redir alt`) *in the
fastest 2% of my own 4000-layout random pool*. **Minimizing oxey while holding those six at arm
B's level is partly self-cancelling by construction** — the objective fights its own
constraints. That predicted a small feasible gain, and a small feasible gain is what happened.

---

## THE "12 BETTER / 0 WORSE" — CORRECT, AND SMALLER THAN IT SOUNDS

The headline vs arm B, per-pair **CONTESTED** counts (`sfr` **never** counted — permutation
invariant; `alt`/`imbalance` are **construction-ties** because the headline **shares arm B's
hand partition**, which I computed from live `Geometry`):

> **12 better / 0 worse of 12 CONTESTED, + 2 construction-ties, 0 genuine ties. Hamming 8/30.
> Cluster-corrected: 6 of 6 clusters better, 0 worse.** It **DOMINATES** arm B on the frame.

🔴 **BUT I KILLED THREE OF THE TWELVE.** A strict win above a 7.1e-15 numerical floor can still
be negligible. Scaled against **two yardsticks that share no pool** — the range over the six
frozen champions, and the range over arm B's 435-member 1-swap ball:

| axis | improvement | frac of six-range | frac of ball-range | |
|---|---|---|---|---|
| oxey-style | +4.164554 | 0.3166 | 0.0520 | substantive |
| lsb-dist | +0.715047 | 0.3150 | 0.0669 | substantive |
| roll | +0.464173 | 0.0639 | 0.0389 | substantive |
| sfb-dist | +0.416393 | 0.4629 | 0.0461 | substantive |
| lsb | +0.395975 | 0.4074 | 0.0810 | substantive |
| sfb | +0.287545 | 0.3774 | 0.0466 | substantive |
| scissor | +0.151505 | **0.8037** | 0.0905 | substantive |
| comfort | +0.089058 | 0.1406 | 0.0342 | substantive |
| redir | +0.061643 | 0.0598 | 0.0057 | substantive |
| **sr-roll** | **+0.015693** | **0.0021** | **0.0016** | 🔴 **MARGINAL** |
| **sfs-dist** | **+0.013159** | **0.0030** | **0.0022** | 🔴 **MARGINAL** |
| **sfs** | **+0.001047** | **0.0003** | **0.0003** | 🔴 **MARGINAL** |

=> **Honest restatement: 9 substantive improvements + 3 technically-strict-but-negligible
(< 0.3% of both yardsticks) + 2 construction-ties.** (The 1% cut is **my** judgement, made and
labelled as such; the raw fractions are published so a reader can set their own.)

⚠ **And the three marginal axes are exactly `sfs`, `sfs-dist`, `sr-roll` — three of the four
axes arm B is BEST-OF-SIX on.** That is not a coincidence: the constraint says *match arm B
where no sibling could*, so the optimizer's only feasible move there is to tie it to ~4
decimal places. **The dominance is real; three of its cells are the constraint binding, not a
win.** Further, `lsb`/`lsb-dist` are near-duplicates (spearman 1.0000, sibling-measured), so
the 12 wins span at most **11** independent axes, and 9 substantive ones at most **8**.

---

## WHAT ELSE I KILLED OF MY OWN

🔴 **The headline is WARM-ONLY, so it is not a cold-start discovery.** All 5 cold seeds returned
infeasible champions. `armh-warm` injects arm B into every island (fail-loud, asserted 20/20
with `V = 0`), so **it is a neighbourhood search around the incumbent.** Declared in prereg §3
and restated here rather than dropped. The honest description is **"a constrained local
improvement on arm B"**, not "a search found a better layout."

🔴 **An ordering hazard I created and disclosed:** the 1-swap-ball enumeration told me BALL-1
existed **before** I registered the verdict rules, so my prereg was written *knowing* a
feasible layout existed. I disclosed it in prereg §2 and registered **①b (enumeration-only)**
as a distinct, weaker outcome precisely so that knowledge could not be laundered into ①a. As it
turned out the search **did** beat the enumeration (P3 held by 3.130938), so ①a is earned — but
the disclosure stands on its own.

⚠ **Convergence, and it cuts against my own budget choice.** 5 of 5 warm runs hit the unique
target at epoch 9 of 12 and **4 of 5 returned the identical champion** — `n_distinct = 2` over
5 runs, mean Hamming over runs **2.4** vs over distinct champions **6.0** (both reported: either
alone supports the opposite reading). The warm arm is **converged and saturated**; a bigger
budget would very likely buy nothing, consistent with SPEEDTIE-BUDGET-1. But `n_distinct = 2`
also means **my warm arm's diversity is near zero**, so "the search found the best feasible
layout" is **not** supported — only "it repeatedly found *this* one."

⚠ **Three defects my own harness caught in itself, all pre-result:** (1) shipped `analyze`
**always injects a `qwerty` reference row** — my row-count assert fired on its first run
(fixed to assert the exact row-key *set*); (2) splitting `--layouts` on `,` shipped `.-` to the
CLI as a layout name — **a C30M layout contains `,` `.` `-`** (now whitespace + `len == 30`
assert); (3) a vestigial `from keybo.layouts import registry` (the symbol is `NAMED_LAYOUTS`),
caught by running rather than by reading.

---

## METHOD

- **16 of 16 runs rc=0, read from `.rc` SENTINEL files, never from a callback** (trap 50: a
  `while pgrep` watcher died three times in one session while the work completed fine).
  Phase 1 242.8 s, phase 2 366.7 s.
- **`unique_evals` reported ACHIEVED, never requested**, and **TRIPLY reconciled** — run JSON
  == ckpt `n_unique` == `keys.npy` length, **16/16 AGREE, 0 mismatches**. Range 921,447 –
  1,070,198 (92.1% – 107.0%). All 16 clear the 80% floor.
- **All 16 `.keys.npy` sidecars RETAINED** (129 MB) so `--resume` still works, unlike
  SPEEDTIE-BUDGET-1 which deleted 388 MB and permanently lost that ability.
- **Every constant GENERATED, never retyped**, and re-derived again at run start by
  `armh_assert_constants()`, which **refuses to run on drift**. Its run-start output on every
  ARM H run: `worst_ref 0.0`, `worst_ms 0.0`, **`V_armB` exactly `0.0`**,
  `fitness_armB == oxey(armB)`, BALL-1's two numbers to 1.4e-13 / 7.1e-15, and
  `fitness_qwerty = 1000071.68` (interval separation asserted numerically, not argued).
- **7 controls C1–C7 ALL ran BEFORE the prereg was written**, which itself predates every run:
  worktree isolation POSITIVE (`keybo.__file__`, `sys.prefix`, `FastEval.corpus_dir` all under
  `/tmp/armh`, trigrams md5 == trap-8 reference); arm B reproduces at **absdiff exactly 0.0**
  (FastEval) / 1.93e-12 (shipped, *identical* to ARM G's); **cross-path FastEval-vs-shipped
  worst rel 1.233e-14 with 11 of 15 bit-exact over 13 layouts**; **MUTATION-CONTROLLED** —
  planting `*1.000000001` on `oxey-style` ⇒ **rc=1**, removing ⇒ rc=0; directions **DERIVED two
  ways** (qwerty-is-worst 14/14; rank-correlation 13/14, sole miss `sfs` at rho −0.0157,
  *independently matching* ARME-1 and ARM G); numerical floor of my objective **7.105e-15**
  across {1, 2, 435, 20 000}-row batches with the same batch twice **exactly 0.0**.
- **The champion gate PROVEN TO BITE, four ways:** planted-infeasible (qwerty) ⇒ **rc=1**
  (13 axes violated); clean ⇒ rc=0; **tight band `eps=0.05` ⇒ rc=1 on the SPEED leg alone with
  `viol=0` axes** (so the two legs bite independently); and `eps=0.098342` ⇒ rc=1 on the
  headline. The first three ran **before phase 1 launched**.
- **Positive control on the gate itself, asserted rather than observed:** arm B must be
  feasible with **exactly 13 ties** or the gate raises.
- **EMPTY was tested against the whole archive, not just champions** (trap 4: an archive-only
  null is not a null): **676 distinct layouts** swept — every champion plus every run's top-50
  — of which 5 are 13-axis-feasible, 73 in-band, **4 FEASIBLE, 3 COLLECTED.**
- **Repointed hardcoded path declared (trap 35):** `search.py:427`'s `/tmp/armg/` literal was
  the **only** hit and is repointed to `/tmp/armh/`. Did **not** reuse `search_placebo.py`
  (`cwd=/tmp/optev`), `run_budget.py`/`analyze_budget.py` (`WORKTREE=/tmp/speedtie`).
  `evobj.py` and `evidence_scorer.RESTORED.py` are **byte-identical to ARM G's declared md5s**.
- **Seed family asserted DISJOINT** from ARM G's (`20260728 + 7919r`) and the placebo's
  (`900000 + 7919r`): `31337 + 104729r`.

---

## SCOPE AND WHAT I DO NOT CLAIM

- **No adoption, no recommendation to adopt.** Nothing pushed, no CR, `PREREGISTRATIONS.md`
  untouched, `oxey-partition-fix` **not** merged.
- **No claim that any gauge difference is PERCEPTIBLE.** SPEEDTIE-1's caveat binds: these
  differences are **free**, not necessarily felt. The +0.139 ms/char the headline pays is
  likewise modelled, not measured on a human.
- **No near-miss reported as a success**, and **no constraint relaxed mid-run.** The `sd_G` and
  borrowed-ruler gate runs are **sensitivity analyses** pre-committed in prereg §5 — they
  report the headline as **OUT of band** under those rulers rather than re-labelling it.
- **`oxey-style` is not independent evidence** alongside `sfb lsb scissor imbalance redir alt`
  (R² = 0.9082 in-band, my own measurement) — hence the cluster-corrected count, with oxey in
  its own cluster.
- **No multiplier for `scissor`'s mispricing** — direction only (SCISSORPRICE-1 unsettled),
  even though `scissor` is the headline's largest fractional gain (80.4% of the six-champion
  range).
- **`sd_G` = 0.049171 and the borrowed 0.0617 appear in NO primary verdict** — sensitivity only,
  and labelled as different quadruples throughout.
