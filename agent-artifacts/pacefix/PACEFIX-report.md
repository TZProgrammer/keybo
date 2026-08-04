# PACEFIX-1 — report (HANDOVER: diagnosis COMPLETE and MEASURED; the accuracy/gate half is UNFINISHED)

**LINE 1 — A PACE-TRACKING INTERPRETABILITY FRAME IS ACHIEVABLE ON THE RANK AXIS, AND THE ρ = 1.000000
RANK IDENTITY IS A *FIXABLE DEFECT*, NOT A STRUCTURAL PROPERTY OF THE LOGRAT TARGET. The cause is the
frame's own monotone constraint tuple, which constrains `wpm` at −1 and thereby FORBIDS the
wpm-×-geometry sign change a within-bucket re-ordering requires — at ANY depth. Measured, on the SAME
11 columns, changing ONE variable (`monotone=False`): wpm's gain share rises 0.000635 → 0.812641
(1280×), wpm splits 4 → 441, interaction leaves 8/1020 → 625/1668, and the within-bucket rank identity
BREAKS — ρ(b40,b120) 1.0000000 → 0.9210093. ⇒ real within-bucket pace re-ordering IS learnable on this
target, so my brief's candidate (d) — "LOGRAT has already absorbed the pace structure" — is REFUTED as
a complete explanation. ⚠ BUT "achievable" is proven only on the RANK axis: the wpm/CONSTFRAC/accuracy
PRICE of this fix is NOT MEASURED (see §c), and INTERPFRAME-1 + GATEFOLDS-1 both measured
constraint-removal making other things WORSE — so this is NOT yet a recommendation to adopt.** 🟢

**HANDOVER STATE (parent directed a stop for the laptop migration):** the DIAGNOSIS half (§a, §b, §e)
is complete and measured. The PRICE half is not: `lolo.py` is **written, committed, and UNRUN** (M-C
accuracy paired per-fold + M-E gate at both thresholds); the M-D interpretability driver
(MAXCORR/CONSTFRAC) is **unwritten**. **§H is the migration checklist — read it first on the laptop.**

---

## 🔴 CORRECTIONS TO MY BRIEF AND TO GATEFOLDS-1 — first, as instructed. All registered in the prereg BEFORE measuring; parent has independently verified C1/C2

**C1 — THE INTERACTION WAS NOT "UN-ENCOURAGED", IT WAS FORBIDDEN BY A SHIPPED PARAMETER.** My brief's
sharpest point — *"at max_depth 3 with 11 columns, a wpm-×-geometry path is ALREADY REACHABLE IN
PRINCIPLE — yet interp-wpm still produced rho exactly 1.000000. So 'just add depth' may NOT be the
fix"* — is right about depth and **wrong about reachability**. `src/keybo/features/schema.py:365`:

    BIGRAM_INTERP_WPM_MONOTONE = (*BIGRAM_INTERP_MONOTONE, -1)

Verified through the shipped registry: `replacement_frame("wpm")` → `mono = (1,1,1,1,1,1,1,1,1,-1,-1)`,
`wpm` at index 10 → **−1**. **All 11 columns are monotone-constrained, `wpm` included.** XGBoost
enforces `monotone_constraints` along **every root-to-leaf path**, so the model is constrained
*jointly*. A within-bucket re-ordering needs wpm's effect to change SIGN with geometry; the constraint
forbids exactly that. ⇒ the path DEAD-1 needed was **structurally unavailable at any depth**. So the
right diagnosis was never "not enough capacity" but "the capacity was licensed away". 🟢 **CONFIRMED by
measurement below.**

**C2 — THE SERVED FRAME IS *NOT* MONOTONE-CONSTRAINED; GATEFOLDS-1's PUBLISHED TEXT IS WRONG AND MY
BRIEF INHERITED IT.** GATEFOLDS-1 §3/§(a)/§(c) describe CUR-INVARIANT as *"the SERVED frame, all its
resolution, all its one-hots, **its OWN monotone constraints**, ZERO interp columns"*; my brief repeats
the phrase. **Measured:** `monotone_constraints` is set in **exactly one place** in the tree —
`src/keybo/training/train.py:436`, **inside the `if interp:` branch** — and **no `BIGRAM_MONOTONE`
tuple exists in `schema.py`** (asserted in-driver: `served_has_monotone_tuple_in_schema = False`).
Confirmed empirically: my served arm trained with `constraints = NONE`. **⇒ the served frame trains
UNCONSTRAINED.**

**Why C2 is load-bearing, not bookkeeping.** The served frame is simultaneously (i) the only
pace-TRACKING frame and (ii) the only UNCONSTRAINED one ⇒ "constrained" and "cannot track pace" were
**CONFOUNDED** across every served-vs-interp comparison on this line. **`CUR-INVARIANT` does not break
that confound** — it pins `wpm` on a frame that was already unconstrained, so it varies the pace
channel while holding constraints at *none*. My `interp-wpm-nomono` arm is the arm that DOES break it,
and it shows the constraint was doing the work. GATEFOLDS-1's headline (pace is causal for the
*refusal*) survives; what does **not** survive is the implicit inference that the interp frames'
wpm-invariance is intrinsic to the *basis* rather than imposed by their *constraint tuple*.

**C3 — my brief's "there are NO `interaction_constraints` anywhere to relax" is CORRECT.** Confirmed
(`grep` empty; `_DEFAULT_PARAMS` has `max_depth: 3`; only `monotone_constraints` is ever passed).

**C4 — candidate (a) was already near-refuted by data on disk before I trained anything.**
`gatefolds/invariance.json` records interp-wpm's raw LOGRAT spread as **7.777e-02** — nonzero, so it
DID split on wpm. Now measured directly: **4 splits, gain share 0.000635.**

**C5 — ⚠ MY OWN REGISTERED BAR WAS TOO LOOSE, AND I REPORT IT AGAINST MYSELF (see §b).** I registered
"rank identity BREAKS iff ρ(b40,b120) < 1.000000 AND rank-identical < 5/5". `interp-wpm-depth6` returns
**ρ = 0.9999998566886138** — which *literally* satisfies that bar, yet the break is **1.4e-07**, five
orders of magnitude below the gate's own measured 0.0108 reseed floor. A bar stated at 6 decimals
cannot distinguish a real re-ordering from float noise. **Reported both readings rather than claiming
the letter of my own bar.**

---

## (a) WHY ρ WAS EXACTLY 1.000000 — THE DIAGNOSIS, MEASURED 🟢

Two code paths, so neither number is an algebraic function of the other: **M-A** reads the booster's
own JSON dump (tree structure); **M-B** is the prediction path (`train_bigram_model` + `to_ms`), 875
in-data position pairs × 5 bucket midpoints.

**POSITIVE CONTROL — my instrument reproduces GATEFOLDS-1 on both unchanged arms:** served
ρ(b40,b120) = **0.7930056160581591** vs published 0.793006 (|diff| **3.84e-07**), rank-identical
**1/5**; interp-wpm **1.0** vs 1.000000 (|diff| **0.00e+00**), rank-identical **5/5**. So the
instrument is sound and DEAD-1's exact 1.000000 is confirmed independently.

| arm | ONE variable | wpm gain share | wpm splits | wpm split depths | interaction leaves | ρ(b40,b120) | rank-ident |
|---|---|--:|--:|---|---|--:|--:|
| served (20c) | reference, **unconstrained** | **0.751573** | 405 | {0:112, 1:142, 2:151} | **850/1841** (46.2%) | 0.7930056 | 1/5 |
| **interp-wpm (11c)** | **baseline, all 11 constrained** | **0.000635** | **4** | **{2:4}** — all at max depth | **8/1020 (0.78%)** | **1.0000000** | **5/5** |
| interp-wpm-nomono | `monotone=False` | **0.812641** | **441** | {0:112, 1:147, 2:182} | **625/1668** (37.5%) | **0.9210093** | 1/5 |
| interp-wpm-depth6 | `max_depth` 3→6 | 0.001066 | 7 | {2:4, 3:3} | 14/1217 (1.15%) | 0.9999999 | 1/5 |

**THE MECHANISM, STATED EXACTLY.** A within-bucket Spearman is invariant to any positive monotone
transform, so two things are structurally invisible to it: `to_ms`'s per-bucket `×12000/wpm`, and any
`wpm` **main effect** (it shifts every pair in a bucket by the same monotone amount). Only a term where
wpm's effect **depends on geometry** can move the ranking. Under the shipped constraint tuple that term
is all but absent — and the tree structure shows *how* it is suppressed, not merely *that* it is:

- **wpm carries 0.0635% of total gain across 4 splits in 4 of 300 trees.** It is present-but-inert.
- **Every one of those 4 splits sits at depth 2, the maximum**, and **`geom_BELOW_wpm = 0`** — wpm
  **never** sits above a geometric split. So wpm can only nudge individual leaves; it can never
  *reorganize* a subtree, which is what a re-ordering requires.
- Remove the constraint and the **same 11 columns** put wpm at **81.3% of gain / 441 splits**, with
  splits at depths 0/1/2 (**112 at the root**) — and the rank identity breaks.

⇒ **the constraint, not the basis and not the target, is what produced the EXACT 1.000000.** The
exactness is explained: a suppressed-to-inert interaction plus two provably invisible channels leaves
*nothing* that can reorder, which yields exactly 1.000000 rather than approximately.

## (b) THE INTERVENTIONS — ONE VARIABLE EACH, AND WHAT EACH DID 🟢

**1. `monotone=False` (H-MONO-BLOCK, my registered primary) — SUPPORTED. The rank identity BREAKS
materially:** ρ(b40,b120) **1.0000000 → 0.9210093**, rank-identical **5/5 → 1/5**, raw LOGRAT spread
7.777e-02 → **5.381e-01**. The break (**0.0790**) is **7.3× the gate's measured p95 reseed floor
(0.0108)**, so it is a real effect, not noise. **This is the arm that answers the deliverable question
on the rank axis.**

**2. `max_depth` 3→6 (H-DEPTH — I registered that this would LOSE, and it did, in substance) —
REFUTED as a fix.** ρ = **0.9999998566886138**: a break of **1.4e-07**, i.e. **~77,000× SMALLER than
the reseed floor** and immaterial by any standard. wpm gain share moves only 0.000635 → 0.001066 and
`geom_BELOW_wpm` stays **0** — extra depth adds 3 more max-depth wpm splits and still never lets wpm sit
above geometry. **⇒ depth was never the binding constraint, exactly as C1 predicted from source.**
⚠ Under the *letter* of my registered bar this arm "breaks" (ρ < 1.000000 and 1/5 rank-identical) —
that is a defect in my bar (C5), and the honest reading is that depth does **not** restore pace
tracking. Note the `n_rank_identical` metric is itself brittle here: a 1.4e-07 perturbation flips it
from 5/5 to 1/5 while changing nothing that matters, which is why I report ρ magnitude against the
measured floor rather than the identity count alone.

**3. NOT RUN: `interaction_constraints`.** Deliberately not a first move — constraining GROUPS also
FORBIDS cross-group interactions, so used naively it makes this *worse*.

## (c) ACCURACY vs SERVED + MAXCORR/CONSTFRAC — ⚠ NOT MEASURED. THIS IS THE UNFINISHED HALF

`lolo.py` is written, committed and **unrun**; the M-D driver is **unwritten**. So **the price of the
§b fix is unknown**, and the two-sided criterion (INVARIANT 3) is therefore **not satisfied**: breaking
the rank identity is necessary but NOT sufficient. **Registered bars, fixed before any number:**
accuracy-neutral iff mean paired per-fold Δwmae vs SERVED ≤ **+1.0 ms** AND τ_heldout = [1,1,1];
MAXCORR must stay ≤ **0.7850**.

**Why I explicitly do NOT recommend adopting the nomono fix on this evidence — two prior measurements
point the other way:** INTERPFRAME-1 measured the constraints *helping* (Δρ **+0.01265**, W/L 12/12),
and both GATEFOLDS-1 and GATEWHY-1 measured `INTERP-NOMONO` failing the high-wpm gate **worse** than
the constrained frame (**8 refused buckets vs 5**; dvorak b120 −0.0289 → **−0.1041**). Those are the
10-column frame, not this 11-column one, so they are **suggestive, not decisive** — but they make an
"unconstrain everything" fix the *least* likely version to survive pricing. 🟡

**Consequently the intervention I would run next is NOT the one I proved works.** A **partial tuple** —
keep all 10 geometric constraints, set `wpm` alone to **0** (unconstrained) — is the minimal change that
licenses the interaction while retaining the constraint benefits and the sign-defensibility that
`MONOFRAC` rests on. hybrid-B already precedents a partial tuple (`BIGRAM_HYBRIDB_MONOTONE` carries 0s),
so this needs no new machinery. **This is one variable, ~8 min, and it is the highest-value untested arm
on this line.** 🟠

## (d) THE GATE AT BOTH THRESHOLDS — ⚠ NOT RE-RUN (`lolo.py` unrun)

The instrument is wired and committed: per-fold incumbent baseline, reported at the shipped **0.005**
AND the **measured reseed floor p95 = 0.010760** (read from `reseed.json`, never hard-coded; it
independently reproduces gatewhy's 0.0117); `azerty b120` excluded as **reseed-refusable** (64c/23p,
refused 3/3 on the served frame merely reseeded); the thick cell of record is **qwerty b120**
(477c/10,811p). **My control is the same-frame reseed (`CUR-RESEED`, served @ seeds [3,4,5]) — a control
that DID fail — never the shipped control, which is a tautology** (deltas sum to ~0: 3.331e-16 over
20 cells, 0/20 all-negative; 200k adversarial trials never produced a self-refusal). `lolo.py` flags any
refusal set that is a **subset** of that control's as having shown nothing. **No gate verdict of mine
exists; do not infer one.**

## (e) WHICH OF THE FOUR EXPLANATIONS SURVIVED 🟢

| # | explanation | verdict | deciding evidence |
|---|---|---|---|
| (a) | trees never SPLIT on wpm | 🔴 **REFUTED** | 4 splits, gain share 0.000635 — present but inert (predicted by C4) |
| (b) | wpm splits act as a per-bucket SHIFT, not a pair-dependent term | 🟢 **SUPPORTED — and mechanized** | 8/1020 interaction leaves (0.78%); all 4 splits at max depth; **`geom_BELOW_wpm` = 0** ⇒ wpm can never reorganize a subtree |
| (c) | a MONOTONE CONSTRAINT forbids the required sign pattern | 🟢 **CONFIRMED — the CAUSE** | same 11 columns unconstrained: gain 1280×, ρ → 0.9210093, break 7.3× the floor |
| (d) | LOGRAT already absorbed the pace structure ⇒ **STRUCTURAL** | 🔴 **REFUTED as a complete explanation** | re-ordering IS learnable on this exact target once the constraint is lifted |

**(b) and (c) are the same finding at two altitudes:** (c) is the *cause*, (b) is the *mechanism by
which it bites*. **(d) — the answer that would have made this a structural trade — does not survive**,
and that is the substantive result: the ρ = 1.000000 was an artifact of a licensing choice, not a
property of the pace-normalized target.

## (f) WHAT I DID NOT DO, AND WHY
- **Did not measure the PRICE (§c/§d).** Parent directed a stop for the laptop migration; a run
  finishing on this host is wasted. `lolo.py` committed unrun; M-D unwritten. **Stated as the
  UNFINISHED half rather than papered over — the deliverable is two-sided and I have one side.**
- **Did not run an `interaction_constraints` arm** — it can make things worse (§b3).
- **Did not run the partial-tuple arm I now recommend** (§c) — it is a new arm and the stop came first.
- **Did not adopt, promote, or re-threshold anything.** `FEATURE_VERSION` untouched; `data/models/k31/`
  and `layouts.py` untouched; the gate NOT weakened; **`src/` is byte-identical to base** (my whole
  change is `agent-artifacts/pacefix/`); **nothing pushed**; no branch merged or deleted; no CR.
- **No ledger entry.** PREREGISTRATIONS.md writes are pre-approved, but the arm is half-measured and a
  ledger entry that implied a priced fix would be wrong. **The C2 correction to GATEFOLDS-1's published
  text is the one thing here that deserves a ledger line, and the parent is carrying it.**
- **No tests / mutation battery** — I added no `src/` behaviour; the drivers are measurement-only. Had
  I shipped an assertion, INVARIANT 6 would have required mutation-testing it.
- **Did not run the trigram channel** — all refusals here are bigram; TRIGRAM-CALIB-1 measured the
  trigram channel *inverting* the bigram calibration pattern, so transfer is not assumed.

## (g) OPEN ITEMS
1. **Price the fix (§c/§d) — the unfinished half.** `lolo.py` runs as-is.
2. **Run the PARTIAL-TUPLE arm (`wpm` at 0, geometry constrained) — the highest-value untested arm.**
   §c argues it dominates the nomono fix I proved works.
3. **The C2 confound deserves its own arm in the opposite direction:** a **served frame WITH constraint
   signs applied**. Combined with `interp-wpm-nomono`, that would fully de-confound
   constraints-vs-basis-vs-pace.
4. **My registered bar was too loose (C5)** — a rank-identity bar must be stated against the measured
   reseed floor, not at 6 decimals. Recommend the pattern for future arms: report ρ magnitude ÷ floor,
   never the identity count alone (it flips 5/5 → 1/5 on a 1.4e-07 perturbation).
5. **GATEFOLDS-1's "its OWN monotone constraints" (C2) should be corrected in the ledger** — parent has it.
6. **A recommendation, NOT a change:** the gate's 0.005 tolerance sits below its own 0.0108 reseed
   floor. Re-thresholding retroactively re-adjudicates past verdicts, so it stays the human's call.

---

## §H — HANDOVER / MIGRATION CHECKLIST (read this FIRST on the laptop)

**Branch `pacefix`** (LOCAL, **unpushed**), worktree `/local/home/zegertho/repos/keybo-wt-pacefix`, base
`gatefolds` **`986f3a6`**. **`src/` is byte-identical to base — no product code changed.**

| commit | what |
|---|---|
| **`c6d7841`** | **prereg — committed BEFORE any number existed** (2026-08-04T14:13:49-04:00); causal order verifiable in git timestamps |
| `3aae5e7` | drivers + vendored inputs (migration fix) |
| `5dc5dcb` | driver fix: the `interp_frame` metadata key |
| **`d97de23`** | **`diagnose.json` + `diagnose.log` — the measured diagnosis** |

### ⚠ BLOCKER 1 — THE STROKE DATA IS NOT IN GIT AND IS 582 MB (I cannot fix this)
`/local/home/zegertho/keybo-e2e/bistrokes31_v1.tsv` — **609,486,399 bytes**, **NOT tracked**
(`git ls-files | grep bistrokes31` → 0). **Every arm on this whole line needs it.** Transfer separately
or regenerate. Deliberately NOT staged into `state/` (far over the 100 MB / 1 GB bright line). *This
blocks pacefix and every sibling that retrains.*

### ✅ BLOCKER 2 — SIBLING ARTIFACTS: FIXED BY VENDORING
`lolo.py` used to read `…/keybo-wt-gatefolds/agent-artifacts/gatefolds/{reseed,rows}.json`, but
**`gatefolds` is LOCAL and UNPUSHED**, so that path is machine-local. All three JSONs are now
**committed on this branch** at
`/local/home/zegertho/repos/keybo-wt-pacefix/agent-artifacts/pacefix/vendored-gatefolds/`,
md5-verified byte-identical (~104 KB total):

| file | md5 |
|---|---|
| `reseed.json` | `363ef0df0a8d04f93b049813c9db444e` |
| `rows.json` | `a86ebf225515db7a1cbcb30a7d10f663` |
| `invariance.json` | `4deb48475009285e684009ffe1c906c1` |

`lolo.py` resolves **vendored-first**, falls back to the sibling path, and **PRINTS which copy it
used**. *Worth checking whether other siblings carry the same unpushed-sibling-path dependency.*

### To resume on the laptop
```bash
cd <worktree>          # branch pacefix
export PYTHONPATH=<worktree>/src
<venv>/bin/python -B agent-artifacts/pacefix/lolo.py       # M-C accuracy + M-E gate — THE UNFINISHED HALF
<venv>/bin/python -B agent-artifacts/pacefix/diagnose.py   # only to re-verify; result already committed
```

### Environment facts, MEASURED (not assumed) — each cost me a real failure
- **`pandas` is NOT installed** in `/home/zegertho/repos/keybo/.venv` ⇒ `trees_to_dataframe()` raises.
  `booster_structure()` uses `booster.get_dump(dump_format="json", with_stats=True)` and **sets
  `booster.feature_names` FIRST** — an unnamed dump labels splits `f10`, would find **zero** `wpm`
  splits, and would have manufactured a false "(a) SUPPORTED". xgboost **3.3.0**.
- **The frame record is `metadata.extra["training"]["interp_frame"]`, NOT `["frame"]`**
  (`train.py:535`/`:558`). My `.get("frame")` returned `{}` — i.e. "no constraints" for a **fully
  constrained** model, on the exact claim C1 rests on. Caught **only** because my arm-identity
  assertion demands constraints be PRESENT for a mono arm; it now raises with the real key list. *This
  is the "rc=0 with all-None output is a key-not-present bug" hazard, realized.*
- **`load_strokes` costs ~210 s.** `_boot.load_rows_cached()` memoizes to
  `/tmp/pacefix_wk/rows-<hash>.pkl`, keyed on `(path, mtime_ns, size)` + load params, so a changed
  input can never be served stale. Cache hit took the 4-arm rerun from 242 s to **69.5 s**. **`/tmp` is
  tmpfs and is wiped** — expect one 210 s repopulate per boot.
- `_boot.assert_tree()` hard-fails unless `keybo.__file__` resolves inside THIS worktree **and** the
  branch is `pacefix` (the venv otherwise resolves `keybo` to the shared checkout silently).
- 4 thread vars pinned before importing xgboost; long runs detached; completion signalled by a
  **sentinel file**, never `wait $PID`.
- **Instrument hazard to carry forward:** `agent-artifacts/interpframe/metrics.py:55-64`
  `same_property_groups()` **dispatches on a NAME SUBSTRING** (`"hand_conflict" in names`) and will
  silently apply the wrong frame's grouping. Load it **by path** and **assert the grouping matches your
  frame**. A grouping-dependent LEVEL does not port across frames; only same-grouping DELTAS compare.

### Reusable numbers already on disk — do NOT re-measure
served ρ(b40,b120) **0.7930056** @ 1/5 · interp.1 **1.000000** @ 5/5 (raw spread 0.000e+00) ·
interp-wpm **1.000000** @ 5/5 (raw spread 7.777e-02) · hybrid-B 1.000000 @ 5/5 · reseed floor p50
0.00163 / **p95 0.010760** / max 0.01574, 27.8% of high cells exceed 0.005 · interp.1 costs: wmae
9.9382 → 15.7036 (+58%), MAXCORR 0.9813 → 0.7037, CONSTFRAC 0.0579 → 0.0000 · interp-wpm CONSTFRAC 0.0010.

**Artifacts index:** `/local/home/zegertho/agent/state/pacefix/artifacts/profiles-and-artifacts-index.md`

**CONFIDENCE.** 🟢 VERIFIED: C1–C4; §a; §b arms 1–2; §e. Rests on two independent code paths (booster
dump vs prediction path) with a positive control reproducing published numbers to |diff| ≤ 3.84e-07.
🟡 HIGH: §c's claim that constraint-removal is the *wrong* fix (borrowed from INTERPFRAME-1/GATEFOLDS-1
measurements on the 10-column frame). 🟠 INFERRED: the partial-tuple recommendation — **untested**.
**UNMEASURED: all of §c and §d.** A green diagnosis is not a priced fix; do not quote §a/§b as
license to adopt.
