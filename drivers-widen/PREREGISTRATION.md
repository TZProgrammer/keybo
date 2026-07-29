# WIDEN-AALTO — PRE-REGISTRATION (locked before any accuracy-gate output)

Subagent `widen-aalto`, branch `widen-aalto`, base `5e95981` (main HEAD at spawn).
Committed BEFORE running the accuracy gate. Direction predictions (W3) are locked here and
must not be edited after seeing any gate result. The descriptive funnel (counts only) may be
run before this commit to inform the *widening design*; it cannot reveal an accuracy direction.

## What I am testing (context, verified by reading the ledger tail 2026-07-29)
A blended objective (aalto-n / comm-n, e.g. drop-pool 50/50) is HELD, not landed. GATE-3 +
AMENDMENT (`normgauge` @ af8cf07) resolved "is the blend better?" as **SOURCE-RELATIVE**:
`ms/char` wins AALTO-held-out (4/4 folds, and 4/4 on the *unshared* native frame too);
the blend wins COMMUNITY-held-out. Each surface wins its own source. The reason it stays a
HOLD is a **POWER ASYMMETRY**: AALTO ≈ 55,404 participants vs **2 leakage-clean** community
participants (7 community pids total, 4 spent on the COMMUNITY fit, 1 yields no cells).

## W1 — What result would MOVE the hold? Do I agree with the parent?
Parent's position: only evidence that the blend beats `ms/char` on a frame that is BOTH
leakage-free AND has ≥2 independent participants. More AALTO participants CANNOT supply that —
it makes the better-powered side better powered, so it ARGUES FOR `ms/char`.

**I AGREE, with one registered nuance.** The reasoning:
- On an AALTO-source frame, `ms/char` (the AALTO surface) wins essentially by construction —
  GATE-3-AMENDMENT confirmed this on the *native* (unshared-tensor) frame, so it is not a
  shared-tensor artifact. More AALTO data makes that verdict MORE reliable, not less.
- The one thing that would move the hold — a leakage-free, ≥2-independent-participant,
  NON-AALTO frame on which the blend wins — is exactly what more AALTO data cannot buy.
- **Registered nuance (the one way I can imagine it mattering, and why it still doesn't):**
  dvorak is the only non-QWERTY-*family* layout, so a widened dvorak fold is the closest
  thing AALTO has to an "independent geometry". IF the blend flipped to beat `ms/char` on
  widened-dvorak, that would be the first AALTO-source signal for the blend. But (i) GATE-1
  already had `ms/char` winning dvorak by its LARGEST margin (+0.8066 vs +0.5509), so a flip
  is unlikely, and (ii) even a dvorak flip is one fold of the AALTO source, not the
  independent-frame evidence W1 requires — so it would be *interesting* but would NOT formally
  move the hold. I register it as the arm-1b falsifier below.

## W2 — What would NOT move it (accepted from parent)
- A larger effect on the same 2 clean community participants.
- Any win on folds inside COMMUNITY's fit set (the 4 `@rowStagger` labels). Arm 1 does not
  touch the community frame, so this is not at risk here; recorded for completeness.

## W3 — Predicted direction + falsifier, per arm (LOCKED)

### ARM 1a — headroom (descriptive, re-derived from SESSION wpm not AVG_WPM_15)
The shipped k31 table was built by `k31_extract.py` calling
`load_participant_metadata(min_wpm=40)`, i.e. it ALREADY includes every participant that
passes the process-time filter (FINGERS=="9-10", AVG_WPM_15≥40 floor [no upper bound],
KEYBOARD_TYPE∈{full,laptop}, LAYOUT∈4). So there is NO separate "we used only 21.9%" gate at
participant selection; the shipped table's non-qwerty pids ≈ all filter-passing non-qwerty
pids. The only pids "lost" after the participant filter are those dropped by
`load_strokes(min_samples=10)` (per-row) and `build_cells(min_cell_samples=10)` (per-cell).
- **Prediction:** the true recoverable headroom is FAR smaller than the withdrawn 2.27x when
  measured as (filter-passing pids) vs (pids surviving min_samples/min_cell_samples). The
  large gap the parent saw was an artifact of comparing against an AVG_WPM_15 window that the
  pipeline never applies at participant level. The dominant non-qwerty loss is at the
  PARTICIPANT filter (I expect FINGERS≠"9-10" to be the single biggest cut), and recovering
  those requires REPROCESSING with a relaxed filter (a population change), not merely lowering
  a load-time floor.
- **Falsifier:** distinct non-qwerty pids in the shipped table are ≥2× fewer than
  filter-passing non-qwerty pids — i.e. a large pool is lost purely at load_strokes/build_cells
  floors and is recoverable without touching the participant filter or min_cell_samples.

### ARM 1b — accuracy gate over AALTO layouts, widened frame (PAIRED PER-FOLD DELTAS)
Copy `drivers-normgauge/gate_accuracy.py` machinery: leave-one-layout-out over the 4 AALTO
layouts, fixed surfaces, bucket-centered Spearman rho, split-half ceiling. **Decision on
PAIRED PER-FOLD DELTAS** (ceiling cancellation is exact per fold, NOT across folds with
different ceilings — GATE-3-AMENDMENT qualification). Clears the gate iff the blend beats
`ms/char` by more than `verdicts.reweighting_margin_bound` on EVERY fold (or on the aggregate
paired-delta test, reported both ways). Widest DEFENSIBLE filter relaxation, justified per
relaxation; min_cell_samples NOT lowered below the registered 10 without reporting BOTH.
- **Prediction:** `ms/char` STILL wins the AALTO-held-out gate after widening, on paired
  per-fold deltas, because it remains an AALTO-source frame (confirmed source-relative on the
  native frame). Widening raises power → strengthens, not weakens, `ms/char`'s win.
- **Falsifier:** the blend (drop-pool 50/50 or registered-c) beats `ms/char` by more than the
  margin bound on ≥1 AALTO fold after widening — most notably dvorak.

### ARM 1b' — dvorak specifically (the whole off-qwerty question, on 64 pids today)
- **Prediction:** `ms/char` wins dvorak-held-out even after widening dvorak (GATE-1 had it
  winning dvorak by the largest margin of all 4 folds).
- **Falsifier:** the blend beats `ms/char` on the widened-dvorak fold by more than the margin
  bound.

### ARM 2 — within-AALTO geometry contrast on KEYBOARD_TYPE / FINGERS (never attempted)
The shipped table records only (wpm, duration, pid, hold) per sample — no KEYBOARD_TYPE /
FINGERS column — so a contrast requires mapping pid → covariate via the metadata and
partitioning the table's pids, then running the same 3-objective gate with the covariate level
as the held-out unit.
- **FINGERS prediction:** UNINFORMATIVE by construction. The process-time filter fixes
  `FINGERS=="9-10"` (keystrokes.py:315), so the fit set has ZERO FINGERS variance — there is
  no ≥2-level partition to hold out. Report UNINFORMATIVE as an ANSWER (W4), do not retry.
  **Falsifier:** the fit set contains ≥2 FINGERS levels each with ≥2 pids (would mean the
  filter does not actually fix it — I will verify against the table's pids).
- **KEYBOARD_TYPE prediction:** runnable as a binary full-vs-laptop contrast IF both levels
  have ≥2 pids and enough cells — BUT it CANNOT substitute for the community frame's 2 clean
  participants. Reason: full and laptop are both AALTO-source and both mapped to the SAME
  ROW_STAGGERED geometry by the pipeline (KEYBOARD_TYPE is a covariate on identical geometry,
  not a distinct geometry). So a KEYBOARD_TYPE-held-out fold behaves like the AALTO frame:
  `ms/char` wins or ties, and the arm KILLS the hope that a within-Aalto covariate supplies the
  missing independent-frame evidence.
  **Falsifier:** the blend beats `ms/char` on a KEYBOARD_TYPE-held-out fold by more than the
  margin bound (would suggest KEYBOARD_TYPE captures a real geometry-relevant axis after all).

## W4 — UNINFORMATIVE is an ANSWER
If a frame cannot supply the statistic (all-nan ceilings, no ≥2-pid partition, degenerate
cells), report UNINFORMATIVE and STOP — do not retry elsewhere and do not silently drop a fold.
**Guard against the GATE-2 shape:** the parent's `gate2_accuracy.py:138` did
`rho/ceiling if ceiling>0 else nan`, and `nan>0` is False, silently overwriting 36 already-
computed rhos → a false "no ordering". I will (a) NOT use that idiom — I gate on
`np.isfinite(ceiling) and abs(ceiling)>0.05` like the shipped `validate.py:719`, and (b) report
raw rho AND rho/ceiling side by side, and count finite rhos explicitly, so a silent-overwrite
cannot masquerade as "no result".

## Rules I am operating under
Work on branch `widen-aalto` in worktree `/tmp/widenaalto`; commit locally only. DO NOT push,
merge, land, adopt a layout, or touch `cb907aa`. Do not edit PREREGISTRATIONS.md (parent
registers). Full test suite must pass if I change `src/`. Every code claim cites file:line.
Findings tagged VERIFIED / HIGH / INFERRED / UNCERTAIN. Negative and UNINFORMATIVE results are
findings, reported plainly — no hunting for a positive.
