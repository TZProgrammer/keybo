# ARM G — profiles and artifacts index

**Workspace** `armg` · **worktree** `/tmp/armg` · **branch** `optimize-arm-g` · **base** `66d0715`
**Nothing pushed. No CR. `PREREGISTRATIONS.md` untouched (verified: `git diff` clean on it across all commits). Shared clone `~/repos/keybo` never touched. No layout adopted or recommended.**

**MODELLED ONLY** — g-frame (geometry-only), **baked 90 WPM**, corpus **blend-v1**
(`trigrams.txt` md5 `c5066fa7bcc46dea1ecbc987fb465b4a`), skipgrams `1-skip31.txt`.
Nothing here is a claim about realized human typing speed.

**`oxey-style` convention:** AS-SHIPPED **nested `bad_redirect`** (my base predates OXEYFIX-1,
so a bad redirect is charged +6.0 not +4.0). This is *the same convention* SPEEDTIE-1's frozen
14.05× spread was measured on, so comparability with the target figure holds; comparability
with a post-OXEYFIX board does **not**, and no such comparison is made.

---

## Commits (in causal order — each precedes the evidence that could have biased it)

| SHA | What | When, relative to evidence |
|---|---|---|
| `0f606d1` | PREREGISTRATION.md, **alone** (1 file, 322 lines) | **Before any run existed** |
| `f58698e` | drivers + 9 positive controls + restored `evidence_scorer` | Before any run |
| `ceb85cd` | `judge_armg.py` — all thresholds as code | **While the 10 runs were still executing** |

---

## RUN — ARM G, n=5 seeds + n=5 baseline control (2026-07-28)

- **Verdict: FAILURE** by pre-registered condition **F1**.
- Engine `search.py` (inherited, `+armg` arm only), evaluator `evobj.py` (**byte-identical**
  to the speedtie copy, md5 `dc45ef503792576157a872a996d9e9d7`).
- Config: budget 1,000,000 unique/seed · islands 20 · epochs 12 · overshoot 1.95 ·
  ga-share 0.6 · polish-sweeps 40 — **configuration-matched to the 1M placebo** that
  produced the reference band (read from `search_placebo.py`, not a docstring).
- Seeds `20260728 + 7919*r`, r=0..4 — **deliberately disjoint** from the placebo family
  `900000 + 7919*r`.
- Wall clock **392.7 s** for all 10 runs (launched together; 192-core box, BLAS pinned to 1).
- **rc=0 verified from the SENTINEL** (`runs/RUN.rc`), not from the callback message
  (trap 50: callback-absence — or presence — is not a result). `n_ok=10/10`.

### `unique_evals` ACHIEVED (never requested) — and **triply recorded**

| run | ACHIEVED | % of 1M | run JSON / ckpt `n_unique` / `keys.npy` len |
|---|---|---|---|
| armg-r0 | 1,045,950 | 104.6% | all three agree |
| armg-r1 | 1,035,544 | 103.6% | all three agree |
| armg-r2 | 1,082,094 | 108.2% | all three agree |
| armg-r3 | 1,014,282 | 101.4% | all three agree |
| armg-r4 | 1,035,416 | 103.5% | all three agree |
| baseline-r0 | 1,044,667 | 104.5% | all three agree |
| baseline-r1 | 1,035,544 | 103.6% | all three agree |
| baseline-r2 | 1,086,807 | 108.7% | all three agree |
| baseline-r3 | 1,013,558 | 101.4% | all three agree |
| baseline-r4 | 1,033,434 | 103.3% | all three agree |

**Every seed EXCEEDED its target**, so the registered 80% exclusion floor removed **nothing**
and the primary n is the full **5 per arm**. (Contrast SPEEDTIE-BUDGET-1, where 1 of 6 fell to
77.9% and was excluded.)

⚠ **The `.keys.npy` dedup sidecars are RETAINED** (8.0–8.7 MB each, 82 MB total), so
`--resume` on these exact runs **is still possible** — deliberately unlike the prior arm that
deleted its sidecars and lost that capability.

### MY OWN ruler — measured, not borrowed

**`sd_G = 0.049171` ms/char** (2·sd = **0.098342**), from my own 5 baseline-control champions.
Quadruple printed as the standing rule requires:

| clause | value |
|---|---|
| **POOL** | my 5 ARM-G-family baseline-control champions (near-optimal, cold start, blend-v1) |
| **REPLICATE-STRUCTURE** | independent cold-start search runs, one champion each — *not* per-seed refits, *not* bootstrap draws |
| **SCALE** | raw ms/char, served K31 surface, baked 90 WPM |
| **STATISTIC** | sd, ddof=1, **n=5** |

`n_distinct = 5/5` · mean 254.0035 · min 253.9381 · max 254.0766 · range 0.138482.
The borrowed **0.0617** appears in exactly one place — setting the search band `EPS` before my
own seeds could exist — and in **no verdict**.

---

## Files

| Path | What | Durable? |
|---|---|---|
| `PREREGISTRATION.md` | the prereg, as committed at `0f606d1` | ✅ |
| `runs/armg-summary.json` | all 10 runs: layouts, fitness, ACHIEVED evals, rc, top50 | ✅ |
| `runs/{armg,baseline}-r{0..4}.json` | per-run champion + top50 archive + `armg_constants_check` | ✅ |
| `runs/{armg,baseline}-r{0..4}.ckpt.json` | per-epoch checkpoint (trap 7) | ✅ |
| `runs/{armg,baseline}-r{0..4}.keys.npy` | dedup sidecars — **retained**, so `--resume` works | ✅ |
| `runs/{armg,baseline}-r{0..4}.log` | full per-epoch trace | ✅ |
| `runs/RUN.log`, `runs/RUN.rc` | driver log + **rc sentinel** | ✅ |
| `armg-judgement.json` | the pre-committed judge's output: verdict, ruler, per-pair contested counts, placebo, Hamming both ways | ✅ |
| `armg-archive-analysis.json` | the 273-layout archive sweep that **refutes my own premise** | ✅ |
| `armg-self-separation.json` | hostile re-read A1–A6 | ✅ |
| `pc_fasteval.json` | cross-path control, 7 layouts × 15 axes | ✅ |
| `pc_armg_objective.json` | the 9 ARM G objective controls | ✅ |
| `shape-dependence.json` | BLAS batch-shape measurement (the refuted hypothesis) | ✅ |
| `gauge-directions.json` | directions DERIVED two ways | ✅ |
| `D-prereg-input.json` | the reference/scale constants + D of every existing layout | ✅ |
| `prereg-inputs.json`, `headroom-probe.json` | frozen-champion re-scores, headroom probe | ✅ |
| `drivers/*.py` | every driver, repointed to `/tmp/armg` | ✅ |

---

## Controls (all run BEFORE the result they bear on)

| Control | Result |
|---|---|
| worktree isolation | **POSITIVE**: `keybo.__file__` = `/tmp/armg/src/keybo/__init__.py`, `FastEval.corpus_dir` = `/tmp/armg/data/corpus/blend-v1` — not merely "no hardcodes found" (trap 35) |
| corpus identity | blend-v1 trigrams md5 == trap-8 reference |
| arm B reproduction (shipped CLI) | 253.90057910352797 vs frozen …604 → **1.93e-12** |
| six frozen 1M champions | worst **2.814e-12** — independently equals SPEEDTIE-BUDGET-1's reported worst |
| FastEval vs **shipped** `analyze` | worst rel **1.233e-14**; **10 of 15 gauges bit-exact** |
| ↳ **mutation control on that harness** | planted `*1.000000001` → **rc=1**; clean → **rc=0** |
| `ARMG_REF` / `ARMG_SCALE` / `ARMG_REF_MS` vs live | 5.3e-15 / **0.0** / **0.0** |
| `D(arm B)` and `F(arm B)` | both **exactly 0.0** (required by construction) |
| speed penalty dominates | qwerty `F` = 6.7e6 |
| deficit sign-flip mutation | bites: `D(graphite)` 3.2226 → 1.8402 |
| gauge directions | **13/14** by rank-correlation over 4000 random perms (sole miss `sfs`, rho −0.0157 ≈ 0 — matching ARME-1 independently); **14/14** by qwerty-is-worst |
| `unique_evals` triple-recording | run JSON == ckpt `n_unique` == `keys.npy` length, **10/10 runs** |
| shipped-`analyze` row-drop guard | fired correctly on a genuine duplicate (seed-900000 champion **is** arm B) — the trap-38 fix working as designed |
| ruff | my files add **zero** lint: `search.py` has 6 errors and the **unmodified inherited original has the same 6**; `run_armg.py` and `judge_armg.py` clean |

---

## Headline numbers

**arm B = 253.9005791035.** Verdict tie band (2·sd_G, measured) = **[253.8022, 253.9989]**.
Search band edge (EPS, from the borrowed sd) = **254.0240** — **looser than the verdict band
by 0.0251.**

| champion | ms/char | D | note |
|---|---|---|---|
| baseline-seed20276566 | **253.9381** | 1.2415 | fastest of all 10; **best in-band D — found by the CONTROL** |
| baseline-seed20284485 | 253.9976 | 3.1191 | |
| baseline-seed20260728 | 253.9997 | 3.1581 | |
| baseline-seed20292404 | 254.0056 | 2.8723 | **== frozen SPEEDTIE-1 s923757 exactly** |
| armg-seed20284485 | 254.0137 | 3.0476 | best armg by speed — still **outside** the verdict band |
| armg-seed20276566 | 254.0170 | **1.0594** | **selected**; global archive min D |
| armg-seed20260728 | 254.0188 | 3.0662 | |
| armg-seed20292404 | 254.0242 | 2.8186 | |
| armg-seed20268647 | 254.0766 | 2.4945 | **== frozen s907919**, and **== its own control champion** |
| baseline-seed20268647 | 254.0766 | 2.4945 | identical to the armg champion on this seed |

Existing layouts for reference: arm-B D=0 (by construction) · flagship-c3 1.4878 ·
keybo-lsb+lm 1.9092 · keybo-lsb 2.1317 · graphite 3.2226 · arm-A 0.4533 (but 2.95 ms slower).

**Selected champion** (registered rule applied for the record):
`flmpg.yo,usnctdireahvxwkb-'qjz` — 254.0170, D 1.0594, Hamming 16/30 from arm B.
vs arm B: **12 CONTESTED / 8 better / 4 worse / 2 tie-by-construction** (`alt`, `imbalance`);
**cluster-corrected 5 better / 3 worse of 9**. Worse on `sfs`, `sfs-dist`, `redir`,
`oxey-style`. **Not a dominator.**

---

## Full-gauge table — selected ARM G champion vs all six requested comparators

`flmpg.yo,usnctdireahvxwkb-'qjz` · ms 254.0170 · D 1.0594

| gauge | ARM G | arm-B | arm-A | keybo-lsb | keybo-lsb+lm | flagship-c3 | graphite |
|---|---|---|---|---|---|---|---|
| sfb | 2.3690 | 2.5391 | 1.4093 | 1.6231 | 1.6231 | 1.6538 | 1.5257 |
| sfs | 8.3891 | 6.7995 | 6.2724 | 7.6488 | 7.6488 | 6.7717 | 6.8778 |
| sfb-dist | 2.6640 | 3.0423 | 1.6216 | 1.9031 | 1.9029 | 1.9290 | 1.8548 |
| sfs-dist | 9.8128 | 8.0056 | 7.2442 | 8.9906 | 8.9870 | 8.1310 | 8.2252 |
| lsb | 0.8314 | 1.1411 | 1.0711 | 0.9219 | 0.9219 | 0.7967 | 0.5594 |
| lsb-dist | 1.8034 | 2.3227 | 2.2837 | 1.8960 | 1.8960 | 1.6852 | 1.2424 |
| alt | 37.1373 | 37.1373 | 43.7939 | 45.1561 | 45.1561 | 45.1561 | 44.0763 |
| roll | 45.6879 | 45.4421 | 42.1482 | 41.6249 | 41.6249 | 41.7608 | 42.4657 |
| sr-roll | 18.0079 | 17.8131 | 17.8343 | 12.6921 | 12.9856 | 12.9532 | 14.3417 |
| redir | 4.4298 | 4.4206 | 3.8952 | 3.3584 | 3.3584 | 2.4939 | 3.2131 |
| scissor | 0.1208 | 0.2567 | 0.1753 | 0.1429 | 0.1431 | 0.0889 | 0.5173 |
| imbalance | 4.8754 | 4.8754 | 0.6657 | 2.0779 | 2.0779 | 2.0779 | 2.4959 |
| oxey-style | 11.3958 | 8.6110 | −12.4932 | −3.2497 | −2.8585 | −7.8749 | −7.1482 |
| comfort | 3.3845 | 3.4140 | 2.9592 | 3.7109 | 3.5953 | 3.6056 | 3.9810 |
| *sfr (INVARIANT)* | *2.6596* | *2.6596* | *2.6596* | *2.6596* | *2.6596* | *2.6596* | *2.6596* |
| **ms/char** | **254.0170** | **253.9006** | **256.8466** | **254.6307** | **254.6847** | **254.9761** | **258.1696** |

`sfr` is identical across all seven — a permutation invariant, reported and never counted.

### Per-pair CONTESTED counts (never a bare n/15)

| vs | contested | ARM G better | worse | tie-by-construction | cluster-corrected b/w | ms delta | resolves on sd_G? |
|---|---|---|---|---|---|---|---|
| arm-B | 12 | 8 | 4 | 2 (`alt`, `imbalance`) | 5/3 | +0.1164 | yes |
| arm-A | 14 | 5 | 9 | 0 | 2/6 | −2.8296 | yes |
| keybo-lsb | 14 | 6 | 8 | 0 | 3/5 | −0.6137 | yes |
| keybo-lsb+lm | 14 | 6 | 8 | 0 | 3/5 | −0.6677 | yes |
| flagship-c3 | 14 | 3 | 11 | 0 | 1/7 | −0.9591 | yes |
| graphite | 14 | 4 | 10 | 0 | 2/6 | −4.1526 | yes |

**ARM G is not a dominator of anything**, and it is *dominated on the majority of contested
axes by every incumbent except arm B*. It is faster than all of them except arm B.

---

## 🔴 The sharpest self-kill: my objective was mis-designed for its own stated purpose

I built ARM G to **collect oxey-style headroom**. Its champion is **worse on oxey-style
(11.3958) than the layout it was collecting against (arm B, 8.6110)** — and every incumbent
is far better still (flagship-c3 −7.8749).

**Mechanism, verified by decomposing my own D:** `D` is an *unweighted sum of range-normalized*
excesses. `oxey-style` has the widest scale (s=13.15) so one normalized unit of it buys 13.15
raw points, while `redir` (s=1.03) buys 1.03. The optimizer therefore took cheap wins on eight
narrow axes and paid on the widest one. **`oxey-style` accounts for only 20.0% of my final
deficit** despite being the entire point of the experiment.

This is a **design defect in my objective, not a property of the headroom.** Range
normalization makes every axis equally *tradeable* — precisely wrong when you care about one
axis. It also compounds the trap-27 double-count I flagged in my own prereg §5 (oxey-style is
R²=0.9937 on {sfb,lsb,scissor,imbalance,redir,alt}), so the objective was partly fighting
itself: rewarding sfb/lsb/scissor gains while charging the composite that restates them.

**The premise is therefore NOT refuted by a correct instrument — it is untested by one.**
The next arm to register (NOT run here): minimize `oxey-style` **alone**, subject to **hard**
constraints `ms <= armB + 2*sd_measured` and `g <= g_armB` on the other 13 axes. Hard, not
summed — trap 51's lesson is that a maximizer does not read flags, and *a summed penalty is a
flag*.

---

## ⚠ WHICH FILES ARE LOAD-BEARING — read these, ignore the rest

The artifact set is **82 MB**, but **the ARM G verdict is re-derivable from 6 files totalling
~180 KB.** Everything else is bulk run output kept for reproducibility, not for reading.

### LOAD-BEARING (open these first — the verdict rests on them)

| File | ~size | Why it is load-bearing |
|---|---|---|
| `PREREGISTRATION.md` | 20 KB | the registered objective, ruler rule, 4 failure conditions, 6 predictions. **Nothing in the verdict is legitimate except by reference to this.** |
| `runs/armg-summary.json` | 54 KB | all 10 runs: champion layouts, ACHIEVED `unique_evals`, rc, and the **top-50 archives** that the 273-layout sweep is computed from. **This one file is sufficient input to re-derive every number below.** |
| `armg-judgement.json` | 39 KB | the pre-committed judge's output: verdict, measured `sd_G` + quadruple, per-pair CONTESTED counts, placebo, Hamming both ways |
| `armg-archive-analysis.json` | 5 KB | the 273-layout sweep — the evidence that the premise fails *independently* of the band defect |
| `armg-self-separation.json` | 4 KB | the hostile re-read, incl. the axis-win test (7.80 vs 7.80) that shows the null is not an artifact of judging D by D |
| `drivers/search.py` + `drivers/judge_armg.py` | 55 KB | the objective and the judge. `ARMG_REF`/`ARMG_SCALE`/`ARMG_DIR` live in `search.py` and the judge **imports them**, so the two cannot diverge. |

**Re-derivation recipe (verified 2026-07-28, 0 mismatches on 24 numbers):**
`drivers/audit_reproduce.py` recomputes `sd_G`, the band edges and gap, min/mean D per arm,
the 273 count, the 7-in-band count, the axis-win means, and the headline self-kill figures
**from `runs/*-r?.json` alone** — not from any summary or judgement file. Run it to confirm the
verdict before trusting any prose.

### SUPPORTING (open only if auditing a specific control)

`pc_fasteval.json` (cross-path control) · `pc_armg_objective.json` (9 objective controls) ·
`shape-dependence.json` (the refuted BLAS hypothesis) · `gauge-directions.json` (directions
derived two ways) · `D-prereg-input.json` (the reference/scale constants) ·
`prereg-inputs.json`, `headroom-probe.json` · `armh-feasibility-warning.json` (the arm H
warning) · `armg-ruler-robustness.json` (the ddof/statistic sensitivity)

### BULK ARCHIVE — 82 MB, ~99.8% of the bytes, needed by NO claim

| File | Size | Purpose |
|---|---|---|
| `runs/*.keys.npy` (10 files) | **~82 MB total** | dedup sidecars. Their **only** use is enabling `--resume` on these exact runs. `unique_evals` is *triply* recorded (run JSON == ckpt `n_unique` == sidecar length, verified 10/10), so **no reported number depends on them.** Safe to delete if space is needed — that only forfeits `--resume`. |
| `runs/*.ckpt.json` (10 files) | ~1.7 MB | per-epoch island state (trap 7 insurance). Redundant with the run JSONs for every published figure. |
| `runs/*.log` (10 files) | small | per-epoch traces — useful for convergence questions, no claim depends on them |

**One-line summary for a future reader:** to check ARM G, read `PREREGISTRATION.md`, then run
`drivers/audit_reproduce.py` against `runs/armg-summary.json`. Do not open the 82 MB.
