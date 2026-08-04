# Handover — keybo, session ending 2026-08-04

Written for a fresh agent (or you) on another machine. Read §1 before cloning anything: **the most
important facts here are about what is *not* in this repository.**

Ledger tip when written: `origin/main` = `c1c0bc8` (plus this document). `PREREGISTRATIONS.md` = 13,730 lines / 231
registered entries. Mainline suite: **1281 passed / 3 skipped / 0 failed**.

---

## 1. ⚠ THREE THINGS A FRESH CLONE WILL NOT GIVE YOU

### 1.1 BLOCKER — the stroke data is 28 GB and is not in git

| path | size | tracked? |
|---|--:|---|
| `~/keybo-e2e/bistrokes31_v1.tsv` | **609,486,399 B (582 MB)** | **NO** (`git ls-files \| grep bistrokes31` → 0) |
| `~/keybo-e2e/` (whole dir, incl. tristrokes) | **28 GB** | NO |

**Every arm that trains or validates needs `bistrokes31_v1.tsv`.** Without it, `load_strokes` raises
`FileNotFoundError` and *all* LOLO/gate/accuracy work is impossible. The trigram arms additionally need
`~/keybo-e2e/tristrokes31_cond_v1.tsv` (the **conditioned K31** set — note the name; a sibling wasted a
run pointing at `tristrokes31_v1.tsv`, which does not exist).

**Options:** transfer `keybo-e2e/` out of band (582 MB gets you the bigram line; 28 GB gets everything),
or regenerate via the `fetch-data`/`process-data` pipeline. **What still works without it:** the shipped
`data/models/k31/` surfaces (1.6 MB, *in* the repo), so `keybo analyze`, `keybo compare`, `keybo
frame-collapse` and every layout comparison run fine. Only *retraining* is blocked.

### 1.2 BLOCKER — every branch with this session's code is LOCAL-ONLY

`origin` has **only `main`** plus ledger-staging branches. Measured: **90 of 91 local branches have no remote
counterpart**, and **268 commits are unreachable from any remote ref**. All 14 branches in §3 exist on **one
machine**, and a fresh clone loses them.

**A verified bundle now exists** (this was the migration's real blocker, so it is done, not left as a step):

```
/local/home/zegertho/agent/state/keybo-optimization/artifacts/keybo-all-branches.bundle
99,976,986 bytes   md5 8e096f94d25ec82c7da9d496a90b97c9   91 branches
```

It was validated by **actually cloning from it**, not by trusting `git bundle verify`: all **91/91 branch
tips byte-identical**, file content md5-matched on 7 probe paths across 6 branches (`main:PREREGISTRATIONS.md`
13,730L, `tcond:shap_diff.py` 1,390L, `framediag:frame_collapse.py` 634L, `pacefix:schema.py` 455L,
`los:los.py` 301L, `productize:compare.py` 154L, `main:docs/HANDOVER.md`), **`git fsck` clean** (0
non-dangling complaints), and a **negative control** confirming an absent path still fails — so the passes
are real and not a vacuously-succeeding probe.

Restore on the laptop with either:
```bash
git clone keybo-all-branches.bundle keybo                                   # fresh, all 91 branches
git fetch ../keybo-all-branches.bundle 'refs/heads/*:refs/heads/*'          # into an existing clone
```

⚠ **Copy that file off this host before it is decommissioned.** It is the only copy of ~60 branches. It is
*not* in git (100 MB), so it travels the same way as the data in §1.1.

### 1.3 Rescued artifact not in git

`state/keybo-optimization/artifacts/seed-tables/tables_seed{3..24}.npz` (22 files, 4.8 MB) —
layout-independent per-seed (T2, Tc) tables. An n=25 board comparison takes **4 seconds** with these
instead of a retrain. Worth copying.

---

## 2. WHAT SHIPPED (on `origin/main`, nothing to do)

| landed | what |
|---|---|
| `keybo.analysis.los` | the LOS confidence instrument, 4 estimands, all 4 null bars pass |
| `validate()` geometry fix | it accepted `geometry` and never forwarded it to training (3rd "accepted but not honoured" bug) |
| high-WPM verdicts carry `support` | per-bucket cells + participants recorded (not thresholded) |
| `train.py` docstring | "layout-independent effect" corrected to what is measured |

## 3. WHAT IS BUILT BUT UNLANDED (all local-only, in dependency order)

**Land in this order — `fm4` and later cannot be applied before `productize`.**

| # | branch | contains | suite |
|--:|---|---|--:|
| 1 | `tcond` `14a614c` | `shap_diff` + `--channel t2\|tcond\|both` | 1327/3 |
| 2 | `productize` `738f6a7` | **`keybo compare`** + `docs/compare.md`; the two-column `mean_a`/`mean_b` view; block-first default; `[COUPLED]`/`[NO-DIFF]` flags; refuses to print on a failed external tie | 1356/3 |
| 3 | `fm4` `196c6dc` | display-name layer: `lsb→lsb_dx1p5`, `redirect→redirect_ungated`, `bad_redirect→bad_redirect_ungated`, `lateral→landing_off_home`; `scissor` deliberately unchanged | 1375/3 |
| — | `framediag` `7b5362c` | **`keybo frame-collapse`** — model-free frame diagnostic (independent of 1–3) | 1434/3 |
| — | `goodhart` `bd94e7e` | exploitability harness **+ a real fix: `--polish-incumbent` was silently inert under `--gauge-objective`** | — |
| — | `interpframe` `b973f39` | `interp.1`, the 10-column interpretability frame (**attribution only — see §5**) | 1393/3 |
| — | `hybridtri` `52f0e3f` | hybrid-B + **two real shipped-path bug fixes** (see §6) | 1455/3 |
| — | `gatewhy` `e56b12f` / `gatefolds` `986f3a6` | the gate diagnosis; `src/` byte-identical to base | 1467/3, 1480/3 |
| — | `pacefix` `d97de23` (moving — prereg `c6d7841`) | **IN FLIGHT at handover** (§7) | — |
| — | `calib` `c28b37e` | the calibration gate — **needs an end-to-end `validate()` run before landing** | 1276/3 |
| — | `gateaudit-proposal` `f098ca8` | the audited calibration-gate variant (land *this*, not raw `calib`) | 1295/3 |
| — | `domain-hard` `79cb175` | `valid_domain` — **78 files**; the only "LAND" from the branch audit | 1257/3 |

## 4. THE HEADLINE RESULTS

**The layout question.** `candidate` = `pyu.,vdfnlhieaocstrmkj'-qgwbzx` is the proposal, but the top ~6
boards are one equivalence class: all pairwise gaps sit **below the measured resolution floor**. Leaving
QWERTY is worth **3.68%** (~85× the intra-cluster margin); choosing *within* the cluster is not
evidence-backed. `flagship-c3` is a legitimate alternative — it wins 8/11 gauges (though `lsb`/`lsb-dist`
are the same gauge counted twice, and only `lat-span` correlates with speed). **`candidate` vs
`flagship-c3` is DECIDED-on-model (LOS 1.000) and UNDECIDED-for-a-human (LOS_valid ≈ 0.70).**

**Why `candidate` beats `graphite`, fully decomposed (this is what `keybo compare` is for).** Gap
+3.1934 ms/char = **T2 bigram +0.9981 (31.3%) + Tcond trigram +2.1953 (68.7%)**. In both channels the
dominant feature is a **bottom-row landing key** (`bottom` 23.3% of the full gap; `bg2_bottom` 23.1%) —
the same physical property in two channels, ~46% together, **not additive**. Mechanism, measured without
SHAP: graphite puts **1.57× more corpus mass on a bottom-row third key**, priced 158.7 ms vs 137.0/140.2.

**Calibration.** Compression is **UNIFORM**, not differential, and is **essentially one fold** (qwerty
1.407; azerty 1.042 / dvorak 0.925 / qwertz 1.022). Not fixable by rescaling — all four routes *and the
oracle* fail held-out, because the correction is a property of the **layout**, not the surface. Orderings
are affine-invariant, so **no adoption verdict moves**; only ms/percent magnitudes re-price (and the
qwerty-vs-field percent moves *favourably*, so published figures are lower bounds).

**Frequency / practice.** The practice term `b` is **provably irrelevant to layout choice**: `B_spread =
0.0` exactly within every equal-coverage group, and the optimizer's 2-opt/3-opt moves are
charset-preserving, so `b` is a constant over the entire reachable search space. Frequency *is* nearly
orthogonal to geometry (R² 0.033/0.114) so the decomposition is identified — it is just worth ~nothing
here. A SHAP-based frequency-removal arm **lost to its own shuffled-frequency placebo** (+0.0860 vs
+0.1085): the gain was capacity, not practice information.

## 5. `interp.1`: ATTRIBUTION ONLY — NEVER A SCORING FRAME

It wins all seven interpretability bars at zero rank cost. **But it is EXPLOITABLE by the optimizer, and
structurally so:** a control with **zero model error** (every cell = its interp-class mean of the truth)
is *still* exploited at 2.6× its floor. No amount of training fixes it. Mechanism is **within-group
adverse selection** — the search picks a below-average member of an indistinguishable class.
Attribution fidelity: 100% sign agreement but a **[0.578, 1.452]** magnitude range, worst on
searched boards. **Read it as "which mechanism, which way", never "worth N ms/char".** hybrid-B is worse
still: **not adoptable for anything** (SPLITPAIRS 24 vs the incumbent's 7).

## 6. THE DEFECT CLASSES THIS SESSION FOUND (the most reusable content here)

**Seven instances of verification that cannot fail.** Each looked green and proved nothing:
1. A SHAP control defined `base := p − Σshap` then asserted `base + Σshap == p` — printed exactly `0.000e+00`, structurally incapable of failing.
2. A shift-share read **−118.4%** where 100% is required; small residuals missed it, the **known endpoint** caught it.
3. Three tests green while not testing their own names — one "magnitude floor" test used a positive value so the *sign* rule short-circuited first; **deleting the floor left it green**.
4. A mechanism metric that was an **algebraic identity** of the outcome it explained (1.4e-14) — guaranteed positive whenever the effect existed.
5. Two assertions whose **subject could not vary** (a field only ever checked at its default).
6. `metrics.py:60` dispatches on a **name substring**, silently applying the wrong frame's grouping (a metric would read 4 and PASS instead of 24 and FAIL).
7. **The worst: the mandatory gate control was a TAUTOLOGY.** Its baseline is the incumbent's mean over the *same seeds being scored*, so deltas sum to zero and can never all be negative — verified at 0 self-refusals in 200,000 trials. **"Gate control passed" licensed nothing in two published arms.** The first six invalidated a *measurement*; this invalidated a *licence to read measurements*.

**Rules earned, in priority order:**
- **Mutation-test new assertions, and purge `__pycache__` before and after each mutation.** A `.bak` restored in the same second at the same byte size satisfies CPython's `(mtime, size)` `.pyc` check, so **mutated bytecode runs against restored source** — one arm got 3 *false* survivors. `python -B -m pytest` is the working form; **`pytest -B` is invalid** (`-B` is a python flag).
- **A self-consistency identity validates arithmetic, never the CHOICE of quantity.** A wrong corpus weighting passes "Σcontributions == prediction" at 2.1e-16 while decomposing the wrong thing. Always tie to an independently shipped number.
- **No self-generated targets.** `TimeSurface` builds `_T2` via the *served* featurizer, so identical served rows necessarily get identical targets — a "floor" computed that way is a tautology.
- **Build decompositions with a checkable endpoint.**
- **Measure every floor; borrow none.** Five floor-confusions occurred. A floor must match the comparison's **data volume**, not just its design; a `wmae` floor is minimized by the weighted **median**, not the mean.
- **`rc=0` + all-`None` output is a key-not-present bug, not a measurement.**
- **The venv resolves `keybo` to the shared checkout, silently.** Set `PYTHONPATH=<worktree>/src` and print `keybo.__file__` **and its branch** in every driver.
- **An exhaustive-equality result names its operands** — check they are the two objects your question is about. (A "predicate-equal" claim turned out to be *gauge-vs-gauge*, not frame-vs-gauge; I relayed it and it would have blocked a correct fix.)

## 7. OPEN ITEMS

### 6.8 A registered bar too loose to separate signal from float noise (self-caught)
`PACEFIX-1` registered "rank identity BREAKS iff rho(b40,b120) < 1.000000 AND rank-identical < 5/5". Its
`depth6` arm returned rho = 0.9999998566886138 — which satisfies that bar **literally** while being a
**1.4e-07** break, ~77,000× below the same gate's measured 0.0108 reseed floor. `n_rank_identical` is equally
brittle: that same 1.4e-07 flips it 5/5 → 1/5 while changing nothing material. The agent reported **both**
readings rather than banking the letter of its own bar, and filed a prereg addendum.

**Rule earned: state a break as Δ divided by the measured floor, never as a fixed-decimal threshold or an
identity count.** A 6-decimal bar cannot distinguish a real re-ordering from float noise. This is the eighth
instance in this section and the only one an agent caught **against itself** before it reached a verdict.

### 7.1 SOLVED at handover — why every interpretability frame failed the gate
**Cause (code + measured, `pacefix`):** `schema.py:365` sets
`BIGRAM_INTERP_WPM_MONOTONE = (*BIGRAM_INTERP_MONOTONE, -1)` — **`wpm` itself is monotone-constrained at
−1.** XGBoost enforces monotone constraints along every root-to-leaf path, so a within-bucket re-ordering
(which needs wpm's effect to change *sign* with geometry) is **forbidden at any depth**. Hence
`interp-wpm`'s rho of exactly 1.000000.

**Confirmed by measurement, not just code reading:** the constraint **suppresses wpm gain 1280×** —
gain-share 0.000635 with 4 splits and 8-of-1020 interaction leaves, versus **the same 11 columns
unconstrained** at 0.812641 / 441 splits / 625-of-1668. Under the constraint wpm never sits *above*
geometry (`geom_BELOW_wpm = 0`, all 4 splits at max depth), so it can only nudge leaves. Positive control
reproduced the published served numbers exactly (rho 0.7930056, |diff| 3.8e-07).

**The design question in §7.2 is therefore ANSWERED: the trade is FIXABLE, not structural.** Unconstrained
rho(b40,b120) falls to **0.9210093**, i.e. real within-bucket pace re-ordering *is* learnable on this
target — so the LOGRAT pre-factoring has **not** already absorbed the pace structure (candidate (d),
which I had flagged as the likely structural answer, is **refuted**).

⚠ **And the agent caught its own bar being too loose, which matters for anyone re-running this:** the
`depth6` arm gives rho = 0.9999998566886138, which *literally* satisfies the registered "rho < 1.000000
AND < 5/5" — but it is a **1.4e-07 break, five orders of magnitude below the gate's own 0.0108 reseed
floor, i.e. immaterial.** Depth is not the fix; removing the constraint on `wpm` is. Both readings are on
the record.

**What remains unmeasured on this line:** the accuracy side (`lolo.py` is written and committed but was
not run — paired per-fold Δwmae, and the gate at both 0.005 and the measured p95 0.010760 with a
same-frame-reseed control) and the MAXCORR/CONSTFRAC trade. **So "fixable" is established for the
*mechanism*, not yet for the *cost*:** dropping the `wpm` constraint necessarily moves CONSTFRAC off
0.0000, and nobody has measured what it does to wmae or to the interpretability bars. That is the first
thing to run on the laptop, and it needs `bistrokes31_v1.tsv` (§1.1).
**Do NOT read this as "ship unconstrained."** `INTERPFRAME-1` measured the constraints *helping*
(drho +0.01265, W/L 12/12), and both `GATEFOLDS-1` and `GATEWHY-1` measured `INTERP-NOMONO` failing the gate
*worse* (8 refused buckets vs 5; dvorak b120 −0.0289 → −0.1041). Those are the 10-column frame, so they are
suggestive rather than decisive — but they make "unconstrain everything" the **least** likely version to
survive pricing.

**HIGHEST-VALUE NEXT ARM (~8 min, one variable, not run): the PARTIAL tuple — keep all 10 geometric
constraints, set `wpm` ALONE to 0.** `hybrid-B` already precedents a partial tuple, so this needs no new
machinery. Breaking the rank identity is **necessary, not sufficient**; the criterion is two-sided and only
one side is measured.

Resume from **§H of `state/pacefix/report.md`**. **It vendored three JSONs (md5s in its report) because
its drivers read from the *unpushed* `gatefolds` worktree — check other branches for the same
unpushed-sibling-path dependency.**

### 7.2 (answered — see 7.1) The pace-free-frame question
Was: "is an accuracy-neutral pace-free interpretability frame achievable, or is dropping `wpm` a
structural trade?" **Mechanism answer: fixable — the block is a shipped *parameter*, not the basis and not
the target.** Cost answer: **still open**, per the last paragraph of §7.1.

### 7.3 Corrections to published ledger text (not yet applied)
- **GATEFOLDS-1 says the served frame has "its own monotone constraints". It does not.** `monotone_constraints` is set in exactly one place — `train.py:436`, inside `if interp:` — and no `BIGRAM_MONOTONE` tuple exists. **The served frame trains unconstrained**, so across every served-vs-interp comparison "constrained" and "cannot track pace" are **confounded**. GATEFOLDS-1's headline (pace is causal) is untouched; the inference that interp wpm-invariance is intrinsic to the *basis* is not.
- INTERPFRAME-1's within-group floor is a valid **upper** bound, not the greatest lower bound (a `wmae` floor takes the weighted **median**: 1.9964, not 2.2399 → share 34.63%, not 38.9%).
- `ADJ-2`'s "monotone constraints learn zero magnitude" is scoped to **binary indicators**; on graded geometry constraints bind and *help*.

### 7.4 Also open
- **The gate's tolerance (0.005) is below its own reseed noise** (p95 0.0108–0.0117; 27.8% of same-frame reseed pairs exceed it, measured independently twice). Replacing the tautological control with a same-frame reseed (~11 min/arm) is the fix; **changing the tolerance retroactively re-adjudicates past verdicts, so it is a human decision.**
- **`GATESUPPORT-1`'s support plumbing is inert**: all 36 published gate blocks carry `support: None` because the drivers read `bucket_support`, a key `validate()` never writes.
- **Why does *any* retrain move high-WPM rho 0.02–0.03 on thick cells while low buckets improve?** Unexplained.
- The trigram channel's split-pair conflicts are **more expensive, not more frequent** (per-opportunity only 1.25×; conflict mass 6.55×, exceeding the channel's own net gap). The indicated fix is a **tool** change (grouped/Owen-Shapley over blocks), not a frame — and **no `bg0_` block exists**, so a block table cannot be made symmetric in the three keys by regrouping alone.
- **`valid_domain` is absent from mainline** (`domain-hard`, unlanded) — so the search currently runs with an **unbounded objective**.
- **39 branches** were assessed DISCARD by a full audit; deletion is a human act. Two commits (`ce1acc0`, `78cca40`) were pushed to `main` **accidentally** — content is sound and tested, but you never reviewed them first.
- **The binding constraint on everything is layout diversity: 4 folds.** Near-clone `sigma_diff` is *flat* in training-layout count, so more seeds cannot help. What is needed is **observed strokes on two boards from within the tuned cluster**.

## 8. FIRST 15 MINUTES ON THE LAPTOP

```bash
git clone <repo> && cd keybo
uv sync                      # or the project's documented setup
python -m pytest -q          # expect 1281 passed / 3 skipped
python -m keybo.cli analyze qwerty colemak graphite semimak   # works with the vendored k31 models
```
Then: (a) confirm the 14 local-only branches arrived (`git branch -a`) — **if not, they are lost**;
(b) decide on `keybo-e2e/` (582 MB minimum) if you intend to retrain; (c) read
`PREREGISTRATIONS.md` from the bottom up — the last ~40 entries are this session.

**Standing conventions:** the ledger is the law and is append-only (**never** resolve a ledger conflict
with `--theirs`; extract added lines onto a fresh base). `data/models/k31/` is read-only. Adopting a
layout, publishing, and pushing non-ledger code are human decisions.
