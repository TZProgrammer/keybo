# ARM H — profiles and artifacts index

Workspace `armh`, branch `optimize-arm-h`, base `28942d7` (ledger HEAD at spawn).
**Nothing pushed. No CR. `PREREGISTRATIONS.md` untouched (verified zero diff on every commit).
No layout adopted or recommended. `oxey-partition-fix` NOT merged.**

**SCOPE ON EVERY NUMBER BELOW — MODELLED ONLY:** g-frame (geometry-only), **baked 90 WPM**
fitted timing surface, corpus **blend-v1** (`trigrams.txt` md5 `c5066fa7bcc46dea1ecbc987fb465b4a`),
skipgrams `1-skip31`, and the **as-shipped NESTED `bad_redirect` oxey convention** (a bad
redirect charged +2.0 AND +4.0 = +6.0). That is *the same convention* SPEEDTIE-1's 14.05× and
every ARM G number were measured on, so these numbers are comparable to those and **NOT** to a
post-OXEYFIX board. Nothing here is a claim about realized human typing speed.

## Commits, in causal order — each ALONE

| commit | contents | why the order matters |
|---|---|---|
| `491138b` | `PREREGISTRATION.md`, **1 file only** | committed **BEFORE any result existed**; every threshold, the flip number 0.032924, and both FAILURE/EMPTY definitions predate every run |
| `a078611` | drivers (17 files incl. generated constants + restored `evidence_scorer`) | the gate was proven to bite **before** phase 1 launched |
| `2b90b47` | `judge_armh.py`, `gate_armh.py` | committed **WHILE PHASE 1 WAS EXECUTING** — 138 workers live, 0 `.rc` sentinels, 0 result JSONs on disk (verified at commit time), so the judge's thresholds provably were not tuned |
| `c85623d` | `RESULT.md` + verification drivers | after |
| `835493e` | this index | after |
| `ba6819d` | arm G relay-response addendum (POST-HOC) | after |
| `<reflection>` | reflection pass: un-nested census, ratio audit, sd significance | after |

---

## RUN BLOCK — phase 1: baseline control (measures `sd_H`)

**Verdict: 6/6 rc=0 from `.rc` SENTINEL files (never a callback). 242.8 s wall.**
Engine config: `budget=1,000,000 unique · islands=20 · epochs=12 · overshoot=1.95 ·
ga-share=0.6 · polish-sweeps=40` — configuration-matched to the 1M placebo and ARM G.
Seeds `31337 + 104729*r`, r=0..4, **asserted disjoint** from ARM G's and the placebo's families.

| tag | seed | champion | ms/char | unique ACHIEVED | frac | keys.npy |
|---|---|---|---|---|---|---|
| baseline-r0 | 31337 | `flmpg-.uoysntdcireahxqbwvk,'zj` | 253.908205 | 1,041,020 | 104.1% | retained |
| baseline-r1 | 136066 | **`flmpg-yuo,sntdcireahkxbwv'.jzq` = ARM B EXACTLY** | 253.900579 | 1,026,095 | 102.6% | retained |
| baseline-r2 | 240795 | `wyou,kfmlrgeacidhtnsq'j.-pbvxz` | 254.141397 | 1,065,534 | 106.6% | retained |
| baseline-r3 | 345524 | `flmpg-yo,usntdcireahkxwbv.'qzj` | 253.937032 | 1,053,550 | 105.4% | retained |
| baseline-r4 | 450253 | `lmfbg.uoyprnstdieahcxzwkv-,'qj` | 253.997605 | 921,447 | 92.1% | retained |
| **repro-armg-r0** | 20260728 | `pyu.,vdfnlhieaocstrmkj'-qgwbzx` | **253.9997247816586** | **1,044,667** | 104.5% | retained |

**`sd_H = 0.09952542252893681`** (n=5, ddof=1 — the repro control is deliberately EXCLUDED from
the pool). Quadruple: POOL = my 5 ARM-H-family baseline champions (near-optimal, cold start,
blend-v1) × REPLICATE = independent cold-start runs, one champion each × SCALE = raw ms/char ×
STATISTIC = sd ddof=1 n=5. ⇒ `2*sd_H = 0.19905084505787363`, band edge **254.09962994858392**.

🟢 **REPRO CONTROL IS BIT-EXACT vs ARM G's `runs/baseline-r0.json`** — identical layout,
identical fitness **to all 16 digits**, identical 1,044,667 unique evals. So the `armh` arm and
the repointed literal changed nothing on the baseline path: this is a clean **ENGINE-IDENTITY**
control.
🔴 **CORRECTED IN THE REFLECTION PASS (see Q(d) below): I originally wrote that this makes the
2.024× `sd_H`-vs-`sd_G` gap "attributable to the SEED FAMILY, not to my engine". THAT
ATTRIBUTION IS WITHDRAWN.** Engine identity on ONE run does not license it, and the gap is not
distinguishable from n=5 sampling noise (F test p = 0.201; 95% CI on the sd ratio
[0.653, 6.273] contains 1.0). The control stands; the claim I hung on it does not.

## RUN BLOCK — phase 2: ARM H cold ×5 + warm ×5, at `EPS = 2*sd_H`

**Verdict: 10/10 rc=0 from SENTINELS. 366.7 s wall.** Same seeds, same budget as phase 1.

| tag | champion | search fitness | ms/char (shipped) | oxey (shipped) | axes viol | unique |
|---|---|---|---|---|---|---|
| armh-warm-r0 | `flmpg-,uoysntcdireahkxvwb.'jzq` | **4.446491** | 254.039627 | 4.446491 | **0** | 1,041,470 |
| armh-warm-r1 | `flmpg-,uoysntcdireahkxvwb.'jzq` | 4.446491 | 254.039627 | 4.446491 | **0** | 1,027,044 |
| armh-warm-r2 | `flmpg-yuo,sntcdireahkxbwv'.jzq` (= BALL-1) | 7.577429 | 253.966426 | 7.577429 | **0** | 1,054,992 |
| armh-warm-r3 | `flmpg-,uoysntcdireahkxvwb.'jzq` | 4.446491 | 254.039627 | 4.446491 | **0** | 1,070,198 |
| armh-warm-r4 | `flmpg-,uoysntcdireahkxvwb.'jzq` | 4.446491 | 254.039627 | 4.446491 | **0** | 1,064,679 |
| armh-cold-r0 | `pyu.,mgfrlhieaodctnskj-q'bwzvx` | 1000000.896250 | 254.146098 | 0.808179 | 5 | 1,029,972 |
| armh-cold-r1 | `pyu.,mgfrlhieaodctnskj-q'bwzvx` | 1000000.896250 | 254.146098 | 0.808179 | 5 | 968,732 |
| armh-cold-r2 | `fo,.yvgmrlhaeiucdtns'qjk-pwbxz` | 1000000.866735 | 254.204896 | **−5.273023** | 5 | 1,007,807 |
| armh-cold-r3 | `clfdk-.uyosrthgpnieaxqbmvw,z'j` | 1000001.135182 | 254.188919 | 2.536422 | 5 | 1,011,233 |
| armh-cold-r4 | `lcfkg.,uyprnstdoaeihvxwmb-'zjq` | 1000000.990146 | 254.142364 | 4.593809 | 4 | 1,050,439 |

**`unique_evals` reported ACHIEVED, never requested — and TRIPLY RECONCILED: run JSON == ckpt
`n_unique` == `keys.npy` length, 16/16 AGREE, 0 mismatches.** All 16 clear the pre-registered
80% floor, so the floor excluded nothing. **All 16 `.keys.npy` sidecars RETAINED (131.8 MB measured) so
`--resume` works** — unlike SPEEDTIE-BUDGET-1, which deleted 388 MB and lost that ability.
⚠ See the reflection pass's LOAD-BEARING vs BULK table: those sidecars are **97.5% of all
artifact bytes and bear ZERO published claims** (arm G's 99.8% pattern, reproduced on me).

**THE THREE COLLECTED LAYOUTS** (13-axis-feasible AND in-band AND oxey strictly better):

| layout | ms/char | Δms | oxey | Δoxey | in-band under |
|---|---|---|---|---|---|
| `flmpg-yuo,sntcdireahkxbwv'.jzq` (BALL-1) | 253.966426 | +0.065847 | 7.577429 | −1.033616 | **all 3 rulers** |
| `flmpg.yuo,sntcdireahkxbwv'-jzq` (MID) | 253.988534 | +0.087955 | 7.769027 | −0.842019 | **all 3 rulers** |
| `flmpg-,uoysntcdireahkxvwb.'jzq` (HEADLINE) | 254.039627 | +0.139048 | 4.446491 | −4.164554 | **mine only** |

---

## CONTROLS — every one ran BEFORE the result it bears on

| control | result | artifact | when |
|---|---|---|---|
| C1 worktree isolation, **POSITIVE** (trap 35) | `keybo.__file__`, `sys.prefix`, `FastEval.corpus_dir` all under `/tmp/armh`; trigrams md5 == trap-8 ref | `prereg-inputs.json` | before prereg |
| C2 arm B reproduces | FastEval absdiff **exactly 0.0**; shipped 1.93e-12 (*identical* to ARM G's) | `prereg-inputs.json` | before prereg |
| C3 cross-path FastEval vs shipped `analyze`, 13 layouts × 15 | worst rel **1.233e-14**, **11/15 bit-exact**, rc=0 | `pc_fasteval.json`, `pc-fasteval-CLEAN.log` | before prereg |
| C4 **MUTATION** on C3 | planted `*1.000000001` on oxey ⇒ **rc=1** | `pc_fasteval_MUTATED.json`, `pc-fasteval-MUTATED.log` | before prereg |
| C5 directions DERIVED 2 ways | qwerty-is-worst **14/14**; rank-corr **13/14** (sole miss `sfs` rho −0.0157, matches ARME-1 + ARM G) | `prereg-inputs.json` | before prereg |
| C6 numerical floor of the objective | oxey max dev across {1,2,435,20000}-batches **7.105e-15**; same batch twice **exactly 0.0** | `design-probe.json` | before prereg |
| C7 my harness bit on its own first run | `analyze` always injects a `qwerty` row ⇒ my row-count assert fired | — | before prereg |
| **G1 planted-infeasible gate** | qwerty as champion ⇒ **rc=1**, 13 axes violated | `gate-armh-PLANTED.json`, `gate-PLANTED-infeasible.log` | **before phase 1** |
| **G2 clean gate** | BALL-1 ⇒ rc=0, FEASIBLE, COLLECTED | `gate-warm-only.log` | before phase 1 |
| **G3 tight-band gate** | `eps=0.05` ⇒ **rc=1 on the SPEED leg alone**, `viol=0` axes (legs bite independently) | `gate-tight-band-speedleg.log` | before phase 1 |
| G4 gate positive control | arm B must be feasible with **exactly 13 ties**, else raise | `gate-armh.json` | with every gate run |
| G5 ruler sensitivity | headline ⇒ **rc=1 / OUT of band** at `eps=0.098342` | `gate-under-sdG-ruler.log` | after (a pre-committed sensitivity, prereg §5) |
| R1 run-start drift gate | `worst_ref 0.0`, `worst_ms 0.0`, **`V_armB` exactly 0.0**, `fitness_armB == oxey(armB)`, BALL-1 to 1.4e-13/7.1e-15, `fitness_qwerty=1000071.68` | every `runs/armh-*.log` | every ARM H run |
| R2 warm-start fail-loud (trap 10) | arm B injected **20/20 islands, V=0.0**, asserted | `runs/armh-warm-*.log` | every warm run |
| **1-swap ball enumeration** | all 435 transpositions of arm B: **exactly 1** is 13-axis-feasible; **0 are faster than arm B** | `ball-probe.json` | before prereg (quoted IN it) |

---

## ARTIFACT MANIFEST — all under `state/armh/artifacts/` (durable, verified present)

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | the prereg (copy of `491138b`'s file) |
| `RESULT.md` (in-worktree `agent-artifacts/armh/`) | the result document |
| `prereg-inputs.json` | every generated constant + C1/C2/C5 + the frozen feasibility table |
| `design-probe.json` | C6 numerical floor, throughput, the 200k random-pool structural probe |
| `ball-probe.json` | the exhaustive 435-member 1-swap ball, per-axis binding, hand partitions |
| `pc_fasteval.json` / `pc_fasteval_MUTATED.json` | C3 / C4 cross-path control, all 195 cells |
| `armh_constants.py` | the GENERATED constants (never hand-typed) |
| `runs/armh-summary.json` | all 16 runs, `sd_H`, its quadruple, triple reconciliation |
| `runs/*.json` (16) | per-run champion + top-50 archive |
| `runs/*.ckpt.json` (16) | per-epoch checkpoints (trap 7) |
| `runs/*.keys.npy` (16, 129 MB) | dedup sidecars — **RETAINED so `--resume` works** |
| `runs/*.log` (16) | full epoch traces incl. every run-start assert |
| `runs/*.rc` (16) | **the rc SENTINELS** — what rc=0 is read from, never a callback |
| `judgement.json` | the judge's output: verdict, 676-layout archive sweep, binding analysis, predictions |
| `verify-headline.json` | the F2 diagnostic, the per-axis table from shipped `analyze`, the ruler sensitivity |
| `self-separation.json` | the hostile-stranger pass: K1–K6, what I killed of my own |
| `gate-armh.json` | authoritative gate over all 10 ARM H champions (rc=1 under the strict F2 letter) |
| `gate-armh-PLANTED.json` | the planted-infeasible proof that the gate can fail |
| `gate-*.log` (5) | the gate's four bite-modes + the ruler sensitivity |
| `phase1-baseline.log`, `phase2-armh.log` | runner transcripts |
| `drivers/*.py` (11) | every driver, copied out of the worktree so they survive `--destroy` |

**Inputs, not just outputs (trap 14 — an index must cover a run's INPUTS):**
the six frozen champions were read from the **ORIGINAL** artifact
`state/keybo-optimization/artifacts/optevidence-1/search-noise-placebo.json` md5
`f5d78de67bf1c3c0f8e18a6b675942e0` (**not** ARM G's transcription of it); incumbent layout
*strings* from `state/armg/artifacts/D-prereg-input.json` with **every gauge value re-derived**;
engine `evobj.py` md5 `dc45ef503792576157a872a996d9e9d7` and `evidence_scorer.RESTORED.py` md5
`01f3a95ab7a0f53f8f9d5be057fc437e` — **byte-identical to ARM G's declared md5s**;
`search.py`'s single `/tmp/armg/` literal repointed to `/tmp/armh/` (trap 35).

---

## VERDICT

**① COLLECTED + ③ FAILURE by the strict letter of my own F2.**

- 🟢 **COLLECTED:** three 13-axis-feasible layouts strictly better than arm B on `oxey-style`.
  **BALL-1 (−1.0336) and MID (−0.8420) are in-band under all three rulers ever measured on this
  objective; the HEADLINE (−4.1646) only under my own `sd_H`.**
- 🔴 **F2 fired** (the gate rejected the 5 cold champions) so the strict reading is FAILURE —
  **but F2's registered warrant, "the construction is broken," is FALSE by measurement: 0 of 10
  cross-path disagreements.** F2 was mis-specified in my own prereg: it fires on the *expected*
  output of a correctly-working arm, because a run that finds nothing feasible must still
  return its least-infeasible archive entry. Registered as my defect; the strict reading stands
  on the label.
- 🟢 **Predictions 5 held / 1 failed. Both self-adverse ones held** (P2 cold finds nothing: 0/5;
  P4 nothing faster than arm B: 0/10). **P5 — about my own ruler — FAILED** at 1.613× vs a
  predicted ≤1.5×.
- 🟢 **The lever is real, large, and now MEASURED as not-free:** unconstrained, the cold arm
  drove `oxey-style` to **−5.273** (13.88 below arm B) while violating 4–5 hard axes. The
  constrained arm collects **4.1646 of 13.8841 = 30.0%**; the other 70% is priced in axis
  violations. **The speed band is NOT the binding constraint at `sd_H` (relaxation needed 0.0);
  the axis legs are** (`sfs` and `sr-roll` have solo violators).
- 🔴 **Self-killed:** 3 of the headline's 12 wins are < 0.3% of both independent yardsticks
  (`sfs`, `sfs-dist`, `sr-roll` — three of the four axes arm B is best-of-six on, i.e. the
  constraint binding rather than a win); the headline is **warm-only** (a neighbourhood search
  around the incumbent, not a cold-start discovery); the warm arm has `n_distinct = 2` over 5
  runs, so "the search found *the best* feasible layout" is **not** supported.

---

# REFLECTION PASS — LOAD-BEARING vs BULK, and four corrections to my own registered text

Added during the reflection pass, after result commit `c85623d`. Branch `optimize-arm-h`,
**actual tip read with `git rev-parse HEAD`: `ba6819d8dcd2927620328e28d6b8dbb1307c4f05`**
(pre-reflection; the reflection commit follows), `git status --short` **empty**.

## LOAD-BEARING vs BULK — arm G's 99.8% pattern reproduces on me at 97.5%

**`.keys.npy` is 131.8 MB of my 135.3 MB = 97.5% of all artifact bytes, and NO published claim
depends on it.** Everything load-bearing fits in **3.43 MB**.

| tier | files | bytes | what depends on it |
|---|---|---|---|
| **LOAD-BEARING — re-derives every verdict** | `PREREGISTRATION.md`, `RESULT.md`, this index, `runs/armh-summary.json`, `runs/*.rc` (16), `judgement.json`, `verify-headline.json`, `self-separation.json`, `armg-relay-response.json`, `reflect-unnested-oxey.json`, `reflect-ratio-and-sd.json`, `gate-armh*.json`, `pc_fasteval*.json`, `prereg-inputs.json`, `ball-probe.json`, `design-probe.json`, `armh_constants.py`, `drivers/*.py` (13) | **~1.5 MB** | every number, every control, every verdict |
| **SUPPORTING — audit trail, not needed to re-derive** | `runs/*.json` (16, champions + top-50 archives), `runs/*.log` (16, epoch traces + run-start asserts), `*.log` (9 gate/pc/phase transcripts) | ~1.9 MB | the 676-layout archive sweep re-runs from the 16 run JSONs; the logs prove the asserts fired |
| 🔴 **BULK — supports NO claim** | `runs/*.keys.npy` (16) | **131.8 MB (97.5%)** | **nothing.** They exist ONLY to make `--resume` possible on these exact runs. `unique_evals` is already TRIPLY recorded (run JSON == ckpt `n_unique` == keys length, 16/16), so deleting them costs the resume capability and **no claim**. |
| SUPPORTING | `runs/*.ckpt.json` (16) | ~2.7 MB | one leg of the triple reconciliation |

⚠ **The honest note, since I criticised SPEEDTIE-BUDGET-1 for deleting its sidecars:** retaining
them was the right call *for reproducibility*, but I should have said in the original index that
they are **97.5% of the bytes and bear zero claims** — the same thing arm G had to discover about
its own. Recorded here so a future reaper can drop them knowingly rather than guessing. **I have
not deleted them** (that is the parent's call, not mine).

## Q(b) 🟢 DOMINANCE SURVIVES THE UN-NESTED CONVENTION — AND THE GAP GETS LARGER

The question the parent most wanted answered. Probe `drivers/probe_unnested_oxey.py`, artifact
`reflect-unnested-oxey.json`. **`oxey-partition-fix` was NOT merged** — I read its
`_trigram_class` and reimplemented the *same delegation* into `community._v1_pattern` (present on
my base) in a scratch probe, which is the fix's own stated design point (trap 28: delegate,
don't re-derive).

**Four positive controls, all run BEFORE the census was read:**
- **PC1** my nested recomputation vs shipped `analyze`: worst absdiff **exactly 0.0** — my
  re-implementation of the scorer's arithmetic is faithful before I change one term.
- **PC2** reproduces OXEYFIX-1's published BEFORE values (qwerty30m 88.197171 → mine to 4.4e-7;
  graphite −7.148220 → 1.3e-8; arm B 8.611046 → 4.2e-7) **and** its registered consequence:
  every score drops **0.42–1.50** absolute. Mine: 0.432–1.429. ✓
- **PC3** over all 27,000 ordered slot triples: `onehand` **1080 → 756**, double-charged
  **540 → 0**, `bad_redirect` support **540 both ways** — OXEYFIX-1's exact registered census.
- **MUT** a planted +10% on the fixed-side `bad_redirect` moves the census ⇒ it can detect a
  change.

| layout | oxey NESTED | oxey FIXED | drop | gap vs arm B NESTED | gap FIXED |
|---|---|---|---|---|---|
| arm B | 8.611046 | 7.928391 | 0.682654 | — | — |
| BALL-1 | 7.577429 | 6.890308 | 0.687121 | −1.033616 | **−1.038083** |
| MID | 7.769027 | 7.120407 | 0.648620 | −0.842019 | **−0.807985** |
| HEADLINE | 4.446491 | 3.592720 | 0.853772 | −4.164554 | **−4.335672** |

> **Under the un-nested convention: HEADLINE 12 better / 0 worse of 12 CONTESTED (DOMINATES);
> BALL-1 6/6/0 (DOMINATES); MID 8/8/0 (DOMINATES). Identical to the nested census, and the
> HEADLINE's oxey gap GROWS from −4.164554 to −4.335672 (+4.1%).**

=> **The dominance claim is NOT convention-dependent.** No amendment needed on this axis, and the
scope caveat can be *strengthened*: the result holds under **both** conventions, with the fixed
convention slightly **more** favourable to the headline and to BALL-1 (BALL-1's gap also grows,
−1.033616 → −1.038083). MID's gap shrinks slightly (−0.842019 → −0.807985) but stays a strict win.

## Q(a) 🔴 THE 30.0% RATIO IS MIXED-FRAME — DO NOT REGISTER IT AS "30.0% OF THE LEVER"

Artifact `reflect-ratio-and-sd.json`. Footing audit of numerator vs denominator:

| leg | numerator (4.1646, `armh-warm-r0`) | denominator (13.8841, `armh-cold-r2`) | same? |
|---|---|---|---|
| oxey convention | nested | nested | ✓ |
| corpus / frame | blend-v1, g-frame | same | ✓ |
| seed family | 31337+104729r | same | ✓ |
| budget + 80% floor | 1M, cleared | same | ✓ |
| **search regime** | **WARM (arm B injected)** | **COLD** | 🔴 |
| **feasibility status** | **FEASIBLE, 0 caps violated** | **INFEASIBLE, 5 caps violated** | 🔴 |
| **band status** | **IN-BAND (+0.1390)** | **OUT OF BAND (+0.3043)** | 🔴 |

**"Collects 30.0% of the available headroom" reads as a fraction of a comparable total, and the
denominator is a quantity no feasible layout could ever attain.** This is the campaign's
"number wrong while the conclusion is right" class: the conclusion — *most of the oxey range sits
beyond the caps* — is correct and survives; the word **"of"** does not.

**Recommended canonical form (F3, two numbers, no ratio):**

> Under full non-inferiority the arm collected **4.1646** oxey-style units. **Abandoning the 13
> caps** (5 violated) **and the speed band** (+0.3043, out of band) reached **13.8841**. So the
> caps, not the speed tie, are what withhold the rest.

Secondary, if a ratio is wanted (F4) — **with its cross-frame label attached**:

> the constrained collect is **30.0% of what an unconstrained, INFEASIBLE, out-of-band cold
> champion reached** — a cross-frame comparison, not a fraction of available headroom.

Two decompositions I checked and could **not** estimate: the within-feasible-frame ratio is
**1.0 by construction** (the headline *is* the best feasible layout found, so it carries no
information about the lever), and the band-only relaxation is **empty in my archive** (0 layouts
hold all 13 caps while out of band) — registering the absence rather than substituting a
different denominator.

## Q(d) 🔴 "THE SEED FAMILY IS A FIFTH LEG" IS UNSUPPORTED — WITHDRAW IT

The parent was right to push. `sd_H` = 0.099525 (n=5) vs `sd_G` = 0.049171 (n=5, re-read from arm
G's own summary and cross-checked against its registered 0.049171 to < 5e-6), ratio **2.0241**:

- two-sided **F test on the variance ratio: p = 0.201**
- **95% CI on the sd ratio: [0.653, 6.273] — CONTAINS 1.0**
- at n=5 vs n=5 an sd ratio must exceed **3.099** to reach α=0.05
- 200,000-draw null simulation from **ONE** population: P(sd ratio ≥ 2.0) = **10.3%**
- 20,000-draw bootstrap on the observed data: median 1.98, 95% [0.84, 5.03]

=> **2.024× is NOT distinguishable from n=5 sampling noise. "The seed family is a fifth quadruple
leg" should be WITHDRAWN.** My bit-exact repro control proves **engine identity** on that one run
— which it does, cleanly — but engine identity plus a 2× sd difference does not license an
attribution to the seed family. **I over-read a control's scope: it was never powered to make the
claim I hung on it.**

**What survives, and it is weaker but more useful:** `sd_H` and `sd_G` are two estimates of the
same unknown quantity differing 2× — unremarkable at n=5, which means **neither is a usable ruler
on its own.** That argues the standing floor rule from the **precision** side rather than the
seed-family side: **a 2×-uncertain ruler cannot adjudicate a 1.4× band question**, which is
exactly what my headline turned on. That is the honest form, and it *reinforces* my own
self-demotion of the headline rather than rescuing it.

## Q(c) 🔴 THE MARGINALITY QUALIFIER WAS MISSING WHERE THE BARE COUNT IS ASSERTED

Swept my own artifacts. The qualifier is present in `self-separation.json` (K1) and RESULT.md's
K1 table — and **absent at every point where the count itself is asserted**:
`judgement.json.contested.armB` carries `n_better: 12` with construction-tie fields and **no
marginality field at all**; `verify-headline.json`'s `reportable_as` string stops at the
construction ties. **The named failure mode, in my own output.** Canonical form to use everywhere
(also in `state/armh/reflection-proposal.md`):

> **12 better / 0 worse of 12 CONTESTED**, +2 construction-ties (`alt`, `imbalance`), `sfr` never
> counted — **of which 9 are substantive and 3 (`sfs`, `sfs-dist`, `sr-roll`) are < 0.3% of both
> yardsticks and are the CONSTRAINT BINDING, not wins.** `lsb`/`lsb-dist` are near-duplicates, so
> the 12 span at most 11 independent axes and the 9 at most 8.

**BALL-1's census carries no such caveat** (6 better / 0 worse of 6 CONTESTED; its improvements
are substantive and it holds the rest at exact equality) — a further reason it is the cleaner
headline.

## Q(e) 🟢 YES — ARMH-1 SHOULD LEAD WITH BALL-1

Four independent lines now point the same way, three of them from the parent's own questions:

1. **Ruler-independent:** BALL-1 in-band under 3/3 rulers and **9/9** statistics; HEADLINE 1/3
   and 6/9, with all three excluding statistics being the **robust** ones.
2. **Census quality:** BALL-1 is 6/6/0 with **no** marginal wins; the HEADLINE's 12 include 3
   that are the constraint binding.
3. **Convention-robust:** both survive un-nesting, both gaps grow — so this does not separate
   them, but it removes the last reason to hedge on BALL-1.
4. **Provenance:** BALL-1 is reachable by **one transposition** from arm B (`d`↔`c`, both left
   index, row 2) and was found by **exhaustive enumeration** *and* independently returned by
   `armh-warm-r2` — the most reproducible of the three. The HEADLINE is warm-only, Hamming 8.

**Recommended ARMH-1 lead:** BALL-1 as the result (**oxey −1.0336 nested / −1.0381 un-nested at
+0.0658 ms/char, 13/13 caps, 6/6/0 contested, robust under every ruler and statistic tested**),
with the HEADLINE as *"a strictly larger collect (−4.1646) that is real but whose speed-tie holds
only against the loosest of three rulers and 6 of 9 statistics."* ⚠ And BALL-1's own caveat must
travel with it: **+0.0658 ms/char is not zero** — it is a *tie* only in the sense of being inside
a noise band, and per Q(d) that band is itself 2×-uncertain.
