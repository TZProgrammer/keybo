# MODELNORM-1 — profiles and artifacts index

All runs are LOCAL CPU (no pod, no hardware allocation — this arm needs none). Every path below
is durable: it lives under `state/modelnorm/artifacts/` and is additionally committed into
`drivers-modelnorm/` on branch `modelnorm` in worktree `/tmp/modelnorm` (**8 commits** on
`main@dec1c3f`, HEAD `2ec398a`). **Nothing here is pod-local.**

Common to every run: corpus **blend-v1**, `.native` frame, `TRI_PS_FREQ_PRIOR`, **BAKED 90 WPM**,
numpy 2.5.0, evaluator `TILE=16`. 🔴 MODELLED ONLY.

---

## Sweep block — step 2: the per-model "1" anchors (6 runs)

| run | objective | seed | unique evals | champion | fitness (ms) | artifact |
|---|---|---|---|---|---|---|
| anchor-AALTO-s1 | solo:AALTO | 20260728 | 8,021,356 | `lnfdg-,yehcrstmaoiupxqbwv.k'jz` | 223236317224.4177 | `runs/anchor-AALTO-s1.json` |
| anchor-AALTO-s2 | solo:AALTO | 20260901 | 9,902,351 | `lnfdg-,yehcrstmaoiupxqbwv.k'jz` | 223236317224.4177 | `runs/anchor-AALTO-s2.json` |
| anchor-COMMUNITY-s1 | solo:COMMUNITY | 20260728 | 9,229,103 | `mgndy-lea.tpscbkrouiwzxfqvh'j,` | 219828038256.7231 | `runs/anchor-COMMUNITY-s1.json` |
| anchor-COMMUNITY-s2 | solo:COMMUNITY | 20260901 | 9,992,986 | `mgndy-lea.tpscbkrouiwzxfqvh'j,` | 219828038256.7231 | `runs/anchor-COMMUNITY-s2.json` |
| anchor-POOL-s1 | solo:POOL | 20260728 | 9,847,686 | `pctsm.reayfgdlk-niuobzvwx,hqj'` | 235438602522.1889 | `runs/anchor-POOL-s1.json` |
| anchor-POOL-s2 | solo:POOL | 20260901 | 9,415,303 | `pctsm.reayfgdlk-niuobzvwx,hqj'` | 235438602522.1889 | `runs/anchor-POOL-s2.json` |

Budget per run, identical across all six (required — trap 1): 10M unique target, 40 islands,
55 epochs, overshoot 1.95, ga-share 0.6, polish-sweeps 40.
**VERDICT: both seeds returned the IDENTICAL layout for every model — seed gap exactly 0.0 ms.**
Runner `drivers/run_anchors.sh`; sentinel `anchors-rc.txt` = **0**.

## Sweep block — step 4 + deliverable D: the blend and the preference sweep (5 runs)

| run | weights | unique evals | champion | blend | ms/char | artifact |
|---|---|---|---|---|---|---|
| blend-equal | 1,1,1 | 9,811,784 | `pctsk-reayfgdlm.niuobzvwxh,qj'` | 0.951258 | **256.6268** | `runs/blend-equal.json` |
| blend-aalto-only | 1,0,0 | 8,021,356 | `lnfdg-,yehcrstmaoiupxqbwv.k'jz` | 1.000000 | 254.0711 | `runs/blend-aalto-only.json` |
| blend-community-only | 0,1,0 | 9,229,103 | `mgndy-lea.tpscbkrouiwzxfqvh'j,` | 1.000000 | 258.3823 | `runs/blend-community-only.json` |
| blend-pool-only | 0,0,1 | 9,847,686 | `pctsm.reayfgdlk-niuobzvwx,hqj'` | 1.000000 | 257.6572 | `runs/blend-pool-only.json` |
| blend-aalto-pref | 2,1,1 | ~9.8M | `csthg-reaypfdlm.nioubzvwkx,jq'` | — | 255.7811 | `runs/blend-aalto-pref.json` |

Same budget/islands/epochs/seed as the anchor runs. Runner `drivers/run_sweep.sh`; sentinel
`sweep-rc.txt` = **0**.
**The three solo cells are an end-to-end POSITIVE CONTROL: each hit blend = 1.000000000 and
reproduced its own anchor's layout.**

---

## Analysis artifacts (verdicts, not numbers alone)

| artifact | verdict / content | sentinel |
|---|---|---|
| `step1-zero-anchor.json` | the "0" anchor at n=100/1000/10000, 4 seeds; **n=100 sufficient (<1 SE)**; **effective independent models 1.1672 of 3** | `step1-rc.txt`=0 |
| `anchors.json` + `anchors-evidence.json` | anchors of record; **STABLE** (perturbation 0.000000 vs margin 0.003284); convergence curves | `anchors-build-rc.txt`=0 |
| `rank-table.json` | deliverables A+B: **normalizing changes NO ranking** vs raw mean; 2 changes vs raw min, neither clears the 0.231897 floor | `rank-rc.txt`=0 |
| `seed-floor.json` | seed floor from `COMMUNITY_BASE`, the ONLY surviving per-seed family; **seed = 0.74 % of SS (raw ms)**, so FLAGSHIP-1's iWeb 78-83 % does NOT transfer | `seedfloor-rc.txt`=0 |
| `sweep-report.json` | deliverable D: **the weight acts as a preference** (1.00000 → 0.93740 → 0.90286); solo champions 24-26/30 apart | `sweep-report-rc.txt`=0 |
| `judgement.json` | deliverables C+E: **256.6268 ms/char**, frozen set reproduced to **exactly 0.0**; **no dominator**, best n_ge 5/10 (blend) and 7/10 (solo) | `judge-rc.txt`=0 |
| `unit.log` | **21 unit tests pass**; harness mutation-controlled | `unit-rc.txt`=0 |
| `gate-resume.log` | **resume reproduces the uninterrupted run on COUNTS and VALUES** (614,709 unique, identical champion + fitness); the run-identity guard bites on `--epochs` and `--objective` mismatches | — |
| `gate-reverify.log` | after the lint cleanup touched every driver, the full pipeline was re-run: **6 of 7 artifacts BIT-IDENTICAL**, the 7th differs only by two deliberately added provenance fields | `reverify-rc.txt`=0 |
| `PREDICTION.md` / `PREDICTION-SCORED.md` | 18 pre-registered (commit `412e58f`, before `runs/` existed); **11 held, 6 FAILED, 1 untestable** (corrected at reflection from my callback's "5 failed / 2 untestable" — arithmetic slip, no verdict changed), each failure classified **(a) world-differed** vs **(b) badly-posed** | — |

## Inputs (an index must cover a run's INPUTS, not just its outputs — trap 14)

| input | location | note |
|---|---|---|
| the 3 native surfaces | `state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces/*_TRI_PS_FREQ_PRIOR.native.npy` | ⚠ OUTSIDE this workspace, in a **destroyed** workspace's state dir. sha256 of each is recorded in every artifact's `identity.surface_sha256`. The shipped `keybo.analysis.surfaces` **cannot reach these** — it resolves `.standardized` only. |
| per-seed material | same dir, `COMMUNITY_BASE.{bigram,conditional}.seed{0,1,2}.npy` | the ONLY surviving per-seed family (trap 14) |
| corpus | `/tmp/modelnorm/data/corpus/blend-v1/trigrams.txt` | md5 `c5066fa7bcc46dea1ecbc987fb465b4a`; sha256 in every artifact's `identity` |

---

## Reflection-pass addendum (2026-07-28) — provenance closeout

**No hardware runs, no pods, ever.** This arm is local CPU only, so rule 7c's pod/profile/NTFF
columns are structurally N/A. Every artifact listed above is durable in two independent places:
`state/modelnorm/artifacts/` **and** committed into `drivers-modelnorm/` on branch `modelnorm`
(HEAD **`2ec398a`**, worktree `/tmp/modelnorm`, `git status --short` verified **empty**).

**Harvest status (parent-confirmed):** branch harvested, bundle verified at HEAD `2ec398a`; the
work is registered to the ledger as `### MODELNORM-1`, pushed as `181f324` (now `origin/main`).
⚠ The parent reported "9 patches"; I count **8** commits on both `dec1c3f..HEAD` and
`origin/main..HEAD`. Most likely a bundle boundary-ref off-by-one — **the authoritative count is
8**, and if the harvest genuinely holds 9 objects, one of them is not mine.

**Parent's independent reproduction through the shipped CLI (all matched my numbers):**
blend champion **256.63**, COMMUNITY-solo **258.38**, POOL-solo **257.66**, and the
qwerty30m-normalizes-to **[0.5649, 0.4243, 0.5239]** defect.

### Late-added measurement, logged here because it is a reusable verdict

| probe | what it establishes | where |
|---|---|---|
| BLAS shape-dispatch, 400 batch lengths × 3 columns | **max rel 1.5946e-15 (7.2 × eps), mean 6.8675e-16, median 8.7658e-16; 275 of 400 lengths (68.8 %) affected**; worst abs 2.4414e-04 ms vs tightest gap 1.0854e+05 ms → ratio 2.25e-09, **cannot reorder**. Padded fix **bit-exact: 0 of 400 differ.** | `report.md` § "The BLAS shape-dispatch defect, in full"; re-derivable from `drivers/modelnorm_eval.py` (`fit_batch`, `TILE=16`) |
| third-instance citation | `noanchor-1/drivers/fast_eval.py:283-291` (`SixSurface.saved_batch`) has the same unpadded `(B,29791)@(29791,6)` and asserts *"identical to the gather to <1e-11"*; `normfloor_batch` (L304-307) routes through it, so the **ceiling-fraction normalized floor inherits the shape dependence**. Structural claim only — **not re-measured** (out of scope). | same section |

**Inputs reminder (trap 14 — an index must cover INPUTS):** the three `.native` surfaces live
*outside* this workspace, in the **destroyed** `keybo-selmethod` state dir
(`state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces/`).
Their sha256s are embedded in every artifact's `identity.surface_sha256`, which is the only thing
that will let a future agent prove it loaded the same arrays if that directory is ever pruned.
The shipped `keybo.analysis.surfaces` **cannot reach them** — it resolves `.standardized` only.
