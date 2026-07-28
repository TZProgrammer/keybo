# Memory — modelnorm

## Current State
- **Status:** done
- **Task:** MODELNORM-1 (the per-model 0-1 anchored normalization + blend search) is COMPLETE and
  reported. The current pass was the **reflection state-flush only** (agent-reflection-step steps
  0 and 1); the knowledge pass (steps 2-6) is explicitly the parent's job and was NOT done.
- **Next action:** none — flush complete, callback sent. **If anything resumes here: the parent
  must RE-HARVEST at `a3c96de`; its verified bundle stops at `2ec398a`** (see below).
- **Blocked on:** nothing

## Setup (🟢 VERIFIED)
- Worktree `/tmp/modelnorm`, branch `modelnorm`, from `main@dec1c3f`. Shared clone left clean on main.
- venv synced in worktree (`uv sync`); numpy 2.x, python 3.13.
- Corpus: **blend-v1** (production default). md5 `c5066fa7bcc46dea1ecbc987fb465b4a` for `data/corpus/blend-v1/trigrams.txt`; iWeb `50cab38b6859b6e6520ba5d6ec6553b1`. Re-derived, matches traps file.

## Re-derived facts
- 🟢 **TRAP 5 CONFIRMED, independently.** `std − nat` is EXACTLY c-independent (max variation over c = 1.14e-13) and **EXACTLY 0.0 for AALTO** on all three families (BASE / FREQ_PRIOR / TRI_PS_FREQ_PRIOR). COMMUNITY max|std−nat| = 1.2155e+02, POOL = 5.0743e+01. Decomposition verified: `native = T2_own[:,:,None] + conditional` (residual EXACTLY 0.0), `standardized = T2_aalto[:,:,None] + conditional` (residual 1.14e-13). So **.standardized substitutes AALTO's bigram tensor into every source** → must use `.native`.
- 🟢 The shipped `data/surfaces/*.standardized.npy.gz` are **bit-identical** (0.0) to the archived `.standardized.npy`. And `surfaces.py::_resolve` (L92-98) ONLY looks for `.standardized.npy{,.gz}` — there is no `.native` code path in the shipped resolver. So the served gauge is standardized-only.
- 🟢 Native surfaces survive at `/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces/` (all 3 `*_TRI_PS_FREQ_PRIOR.native.npy`).
- 🟢 **Frozen ms/char re-derived via shipped `keybo analyze --json` on blend-v1** (all 5 match the brief EXACTLY): armB 253.90057910352797, keybo-lsb 254.6307495925403, flagship-c3 254.9761188060974, armA 256.846570694692, qwerty30m 264.13891657883323. Also graphite 258.1695631301549, semimak 257.3915378008198, keybo-lsb+lm 254.68467763206198. `analyze` returned exactly 8 rows for 8 requested (with `--ref qwerty30m` = one of the 8, so no extra row here) — still assert SET-CONTAINMENT.
- 🟢 Host: 192 cores, 369G RAM. Arm E did 10.0M unique evals in 544 s with 40 islands (evidence objective).

## Engine template
Arm E's `search_arme.py` (island memetic, mixed operators, 2-opt polish, multi-start restart stream, per-epoch ckpt, blake2b-8 unique keys). Reuse its structure; swap the objective.

## Run configuration (frozen; identical across ALL 11 searches — required by trap 1)
10M unique-eval target, 40 islands, 55 epochs, overshoot 1.95, ga-share 0.6, polish-sweeps 40,
~200 s/run on this 192-core host. Anchor seeds s1=20260728, s2=20260901; every blend cell at
s1. Runners `drivers/run_anchors.sh` and `drivers/run_sweep.sh`, both detached with a callback
fired from the same subshell as the work (trap 50) and gated on an rc sentinel.

## Commits on branch `modelnorm` (worktree /tmp/modelnorm) — 9, HEAD `a3c96de`
- 762ac06 evaluator + normalization + 18 unit tests
- d50a0a7 step 1 zero anchor + 4x evaluator speedup (padding load-bearing)
- 49942c1 search engine + verified resume + resume-identity guard
- 412e58f PREDICTION.md (18 predictions, pre-registered BEFORE any search)
- 61bb8eb step 2-4: anchors of record (STABLE), deliverables A/B, blend champion
- 548e7ad deliverables D and E: preference sweep works, no dominator
- 3ca74d8 score the predictions + lint clean (real B023 fixed)
- 2ec398a report, artifacts index, reflection proposal  ← **the parent's verified bundle stops here**
- a3c96de reflection flush: corrected tally, (a)/(b) classification, 400-length BLAS
  quantification, third-instance citation, floor-quadruple rule

## RESULTS (all 🟢 VERIFIED, corpus blend-v1, .native frame, 90 WPM BAKED)

### Step 2 anchors — EXCEPTIONALLY STABLE (trap 1 satisfied)
Two independent seeds (20260728 / 20260901) at IDENTICAL 10M-unique budget found the
**IDENTICAL champion for all three models** — seed gap EXACTLY 0.0 ms (0.0000% of span),
40/40 islands within 0.10%, champion last improved epoch 4-12 of 55.
Perturbation 0.000000 vs decision margin 0.003284 => STABLE.
| model | "1" layout | one (ms) | zero (ms) | span | span % of zero |
|---|---|---|---|---|---|
| AALTO | `lnfdg-,yehcrstmaoiupxqbwv.k'jz` | 2.232363e11 | 2.431185e11 | 1.988e10 | 8.178% |
| COMMUNITY | `mgndy-lea.tpscbkrouiwzxfqvh'j,` | 2.198280e11 | 2.549949e11 | 3.517e10 | 13.791% |
| POOL | `pctsm.reayfgdlk-niuobzvwx,hqj'` | 2.354386e11 | 2.590278e11 | 2.359e10 | 9.107% |

vs prior ceiling-fraction anchoring: search "1" beats best-of-8-candidates by
**1.212% (AALTO) / 14.736% (COMMUNITY) / 15.193% (POOL)** of span.

### Deliverable A — normalized table (1 = per-model optimum, 0 = random-pool mean)
flagship-c3 0.883983 > keybo-lsb 0.869205 > keybo-lsb+lm 0.865920 > arm-B 0.860728 >
graphite 0.814705 > semimak 0.802528 > arm-A 0.790556 > qwerty30m 0.504368  (equal-weight blend)
Per-model: arm-B is 0.9879 on AALTO (near-optimal) but only 0.7759 on COMMUNITY.

### Deliverable B — P6 FAILED
- per model: NO change (affine positive scale, asserted in code)
- vs raw MEAN of 3 surfaces: **0 discordant pairs** — ranking IDENTICAL
- vs raw MIN (scale-broken floor): 2 discordant pairs (graphite>semimak, graphite>arm-A)
- NEITHER clears the floor: gaps 0.0122 / 0.0241 vs conservative normalized floor **0.231897**

### Seed floor (COMMUNITY_BASE only — trap 14)
seed main effect **0.74% of SS raw ms / 0.83% saved%** — FLAGSHIP-1's iWeb 78-83% does NOT
transfer. Paired floor 0.3914 saved% excluding ref (ratio 0.5632). Including qwerty30m forces
ratio EXACTLY 1.0000 = DEGENERACY (ref row is (0,0,0), spread(X-qwerty)==spread(X)).

### Step 4 — equal-weight blend champion
`pctsk-reayfgdlm.niuobzvwxh,qj'` blend 0.951258, 9,811,784 unique evals, 40/40 islands within 0.01%.
**ms/char = 256.6268** (shipped `keybo analyze`) vs arm B 253.9006 = **+2.7262**.
P8 HELD (loses to arm B). P9 HELD (in [254.0,257.5]; beats arm A 256.8466, qwerty30m 264.1389).

## Live resources — NONE
All detached work finished before the reflection pass. Both batch runners exited rc=0
(`artifacts/anchors-rc.txt` = 0, `artifacts/sweep-rc.txt` = 0); no pod was ever allocated (this
arm is local CPU only); no watcher subshell or deadman is still armed. Nothing to reap.

## FINAL (2026-07-28) — all deliverables complete
- **A** normalization + 21 unit tests (rc=0, harness mutation-controlled). Anchors STABLE.
- **B** normalizing changes NO ranking: 0 discordant within-model, 0 vs raw mean ms, 0 vs raw
  mean saved%, **0 vs the prior ceiling-fraction anchoring**. 2 changes vs raw min(), gaps
  0.0122/0.0241 vs the 0.231897 floor → neither clears it.
- **C** blend champion `pctsk-reayfgdlm.niuobzvwxh,qj'` = **256.6268 ms/char** vs armB 253.9006
  (+2.7262). 9,811,784 unique evals. Frozen set reproduced to worst abs diff EXACTLY 0.0.
- **D** sweep: 5 cells, all 3 solo cells hit blend=1.000000000 at their own anchor layout
  (end-to-end positive control). Weight IS a preference: 1.00000→0.93740→0.90286 monotone.
  Solo champions 24/26/24 of 30 apart. ms/char span 254.0711 … 258.3823.
- **E** no dominator (best n_ge 5/10 blend, 7/10 solo, strict-win term required). Normalized
  floor +0.902863. 19-gauge: 4-10 of 18 movable (sfr excluded, verified invariant std=0.0).
- **Predictions 11 held / 6 failed / 1 untestable** — `artifacts/PREDICTION-SCORED.md`.
  ⚠ **CORRECTED at reflection.** My DONE callback and report v1 said "5 failed / 2 untestable":
  an arithmetic slip in my own tally (I double-counted P15's "FAILED (half)" label as an
  untestable), **no verdict changed**. Six ❌: P1, P6, P13, P15, P17, P18. One ⚠: P3.
  Now classified per the parent's request — **(a) the world differed: P6, P1, P13**, which reduce
  to **2 distinct facts** (P1/P13 are both the AALTO-near-saturation fact); **(b) badly posed:
  P15, P17, P18**, which reduce to **2** (P17/P18 are the same mis-posing). Only (a) is evidence
  about keyboards. P3's untestable is closer to (a): it went 0/0 because the world was *better*
  behaved than any branch I wrote.
- Report `report.md`; index `artifacts/profiles-and-artifacts-index.md`; reflection draft
  `reflection-proposal.md` (parent registers; I did NOT touch PREREGISTRATIONS.md).
- **Scope respected:** 8 local commits on branch `modelnorm` in `/tmp/modelnorm`. NO push, NO
  merge, NO CR, no corpus/default change, no layout promoted. Shared clone left clean on `main`.

## Parent-side confirmations (received 2026-07-28, reflection prompt)
- 🟢 Work **registered to the ledger as `### MODELNORM-1`, pushed `181f324`** (that is now
  `origin/main`; my 8 commits sit on top of it, unpushed, by design).
- 🟢 Branch **harvested — bundle verified, HEAD `2ec398a`**. ⚠ The parent said "9 patches"; at
  that point I counted **8** commits on `dec1c3f..HEAD` *and* on `origin/main..HEAD`. Likely a
  bundle-header off-by-one (a bundle records a boundary ref) rather than a missing commit — but
  **the authoritative count at `2ec398a` was 8**, and if the harvest really held 9 objects one
  was not mine.
- ⚠⚠ **RE-HARVEST NEEDED: HEAD is now `a3c96de`, NOT `2ec398a`.** The reflection flush added one
  commit (`a3c96de`, "reflection flush: correct my own prediction tally, quantify the BLAS defect
  properly, and cite a THIRD instance"), so the branch is now **9 commits** on `origin/main..HEAD`
  and the parent's verified bundle at `2ec398a` **does not contain it**. That commit carries the
  corrected prediction tally, the (a)/(b) failure classification, the 400-batch-length BLAS
  measurement, the third-instance citation, and the floor-quadruple rule — i.e. exactly the three
  things the parent asked to be flushed. **Re-harvest at `a3c96de` or the flush is not captured.**
  (The state files under `state/modelnorm/` are the other durable copy and are already current, so
  nothing is lost either way — but the branch copy is the one the parent verified.)
- 🟢 Parent independently reproduced through the shipped CLI: blend champion **256.63**,
  COMMUNITY-solo **258.38**, POOL-solo **257.66**, and the qwerty30m-normalizes-to
  **[0.5649, 0.4243, 0.5239]** defect. All matched my numbers.

## Reflection pass (2026-07-28) — state-flush only, per explicit instruction
- Step 0 (child cascade): **verified no-op.** `ticket modelnorm --roster` → "No children found
  (no roster, no state/modelnorm/children)". I spawned zero subagents, so there is nothing to
  cascade to and no subtree proposals to merge.
- Step 1 (state flush): done — memory.md, events.log, report.md, summary.md, artifacts index,
  reflection-proposal.md all refreshed; `git status --short` **verified empty** at `2ec398a`.
- Steps 2-6 (knowledge pass): **NOT done, by instruction** — the parent owns it. No shared-KB
  write, no PREREGISTRATIONS.md edit, no critic subagents.

## Improvement Proposals
1. **`ticket --expect-callback` + a same-subshell push callback worked perfectly** (trap 50):
   both detached batches (`run_anchors.sh`, `run_sweep.sh`) fired their callback and both
   sentinels read 0. No watcher subshell died because there was no separate watcher.
2. **The Bash 10-min clamp (trap 22) bit once** — my first anchor loop ran 3 searches
   foreground and POOL-s1 was killed at epoch 53/55. Per-epoch checkpointing (trap 7) made it a
   non-event: the resume finished in 10s. *Proposal:* the traps file should say "detach anything
   that runs 3+ sub-jobs, even if each sub-job is only ~3 min" — the clamp is on the WHOLE Bash
   call, and per-job sizing is the wrong unit.
3. **Backticks in a `git commit -m` message got shell-evaluated** and silently deleted a word
   from a committed message (`` `qwerty_row` `` → nothing, plus two "command not found" lines
   that looked like harmless noise). *Proposal:* add to the traps file — always use
   `git commit -F -` with a quoted heredoc for any message containing backticks or `$`. Caught
   it only because I re-read the committed message; the commit had already succeeded.
4. **A `[WATCHDOG]`-style "task tools haven't been used" reminder fired repeatedly** during long
   analysis/compute stretches. Correctly ignorable per AGENTS.md, but it fired ~12 times.
5. **What I would do differently:** my very first pytest invocation piped through `tail`, so
   `$?` read 0 while pytest had actually failed to spawn (trap 1's exact shape, in a form the
   traps file does not name — the *pipe*, not the sentinel). I caught it because the rc looked
   implausible. *Proposal:* add "never read `$?` through a pipe; write the sentinel inside the
   same `{ ...; echo $? > f; }` group" to trap 1.
