# PREREGISTRATION — normopt: what does the normgauge objective PRODUCE?

**Registered 2026-08-01, BEFORE running any cross-arm comparison.** At registration time I have
run exactly two single-seed timing probes (arm A seed 0, arm B seed 0) to size the compute budget,
and I have read their two `--out` JSONs. I have NOT compared them, NOT scored either under the
other's objective, NOT run `analyze` on either, and NOT run arm C at all. Everything below is
fixed in advance of the comparison this document registers.

Branch: `normopt-layouts` (worktree `/tmp/normopt`), off `main` @ `96e6138`.

---

## 0. Two corrections to my brief, established before registering

- 🟢 **VERIFIED — `normgauge` is ALREADY MERGED into `main`.** `git merge-base --is-ancestor
  c9e1337 main` returns 0; `src/keybo/scoring/model_norm.py` and `drivers-normgauge/` are both
  tracked in `main`. TASK 1's "rebase or cherry-pick into YOUR branch; if it does not compose
  with main, report what conflicts and STOP" is moot: there is nothing to compose. The brief's
  `9290e9d` is also 3 commits stale (branch tip `c9e1337`).
- 🟢 **VERIFIED — the objective is already wired into the SHIPPED `keybo optimize`**, via
  `--model-weight GAUGE=W` (repeatable) + `--model-anchors`. NORMGAUGE-1's own closing line says
  so (`aba7c69`, "deliverable 2, not a driver path"). No driver script is needed for TASK 2; I run
  the shipped CLI.

## 1. TASK 1 gate — already PASSED at registration time

Recorded here because the gate is a precondition, not a result of the comparison.

| check | outcome |
|---|---|
| corpus `trigrams.txt` sha256 vs anchors provenance | 🟢 match (`19806532…`) |
| 3 surface sha256 vs anchors provenance | 🟢 all match (`5353b783…`, `aa09df63…`, `50e2f655…`) |
| `Anchors.assert_direction()` | 🟢 PASS |
| `Anchors.assert_matches_surfaces()` (anchor-drift refusal) | 🟢 PASS |
| `SurfaceFits.assert_batch_invariant()` | 🟢 PASS |
| RECON A: MODELNORM-1's 10M AALTO champion fit | published `223236317224.4177` vs reproduced `223236317224.41766`, rel **−1.37e−16** 🟢 |
| RECON B: each `one` anchor == fit of its `layout_of_record` | rel **exactly 0.0** ×3; each normalizes to `1.000000000` 🟢 |
| RECON C: `zero` anchors rebuilt from (n=100, seed=20260728) | rel **exactly 0.0** ×3 🟢 |
| qwerty normalizes to ~0.42–0.56, NOT ~0 (docstring's direction guard) | aalto-n 0.565032 / comm-n 0.462077 / pool-n 0.545283 🟢 |

→ I am running what I think I am running.

## 2. The three arms

Shipped `keybo optimize`, **search hyperparameters at DEFAULTS** (`--alpha 0.999`, no
`--max-outer`, 2-opt polish ON, `--attempts 1`). I vary **only `--seed`** and the objective flags.
Sibling `searchparams` owns restart/annealing tuning and `multiwpm` owns the wpm-range objective;
I touch neither.

| arm | objective | flags |
|---|---|---|
| **A** (control) | `ms/char` — the incumbent | *(no `--model-weight`)* |
| **B** | normgauge `registered (c)` | `--model-weight aalto-n=0.5411 --model-weight comm-n=0.3977 --model-weight pool-n=0.0612` |
| **C** | normgauge 50/50 (user-APPROVED, drop-pool) | `--model-weight aalto-n=0.5 --model-weight comm-n=0.5` |

Common to all three: `--start` = C30M qwerty (`qwertyuiopasdfghjkl'zxcvbnm,.-`), `--ngram bigram`,
`--corpus blend-v1` (default), `--target-wpm 90` (default), model
`data/models/k31/bigram_reg31_seed0` (gz-inflated to `/tmp/normopt-scratch/models/`, since
`XGBoostTypingModel.load` resolves the sidecar with `.with_suffix('.meta.json')` and so cannot
read a `.gz` path directly — arm A needs the model, arms B/C never load it).

**Seeds: 0–9, i.e. n=10 per arm, 30 runs total.** ≥5 was required; measured cost is ~2.5 s/run
(A) and ~18 s/run (B/C), so 10 is affordable and buys a real within-arm spread estimate. Seeds
are the SAME 0–9 across arms, so the arms are paired.

## 3. The measurements (all pre-specified)

For every one of the 30 layouts:
1. **Both objectives, both directions.** `ms/char` (arm-A ruler) and the normgauge blend under
   BOTH the registered-(c) and 50/50 weightings, plus all three raw gauges `aalto-n` / `comm-n` /
   `pool-n`. Every layout gets every ruler — no layout is scored only on the ruler that flatters it.
2. **The 15-gauge + sg_dist frame** from `keybo analyze` (sfr, sfb, sfs, sfb-dist, sfs-dist, lsb,
   lsb-dist, alt, roll, sr-roll, redir, scissor, imbalance, oxey-style, comfort, + sg_dist).
3. **Key-position Hamming distance** between arms' winners (and within-arm, across seeds), as a
   count over the 30 positions.
4. **Systematic character:** hand balance, per-row share, per-finger load and per-finger time
   share, home-row share.
5. **TASK 4 reproduction:** exact-string and Hamming-distance match of every produced layout
   against the shipped field: `keybo-c30m`, `keybo-lsb`, `keybo-lsb+lm`, `flagship-c3`,
   `archive-1843`, `archive-1846`, `lsb-sib`, `p16-balance`, `graphite`, `semimak`, `arm-A`,
   `arm-B`, `qwerty30m`, and the 5 normgauge anchor/blend boards from `blend-report.json`.
   ⚠ `BALL-1` and `p16-balance` are named in my brief but I have not yet located their layout
   strings; if I cannot, I will name them as missing rather than silently drop them.

**Winner definition, fixed now:** each arm's winner is its best-of-10 **on its own objective**
(arm A: lowest ms/char; arms B/C: highest their own blend). This is the honest reading of "what
the objective produces" — and it is deliberately NOT "best on ms/char", which would rig the
cross-arm ms comparison in arm A's favour by construction.

## 4. What would count as WHICH answer — the decision rule, registered

**The resolution floor is `median 0.135 ms/char` over 91 board pairs** (PREREGISTRATIONS.md
:10405; mean 0.136, p90 0.243). A gap under the floor is **the same layout for practical
purposes** and I will say so.

### On the ms/char ruler (the primary axis, because it is the shipped one and has a floor)
- **MATERIALLY DIFFERENT** — arm B or C's winner differs from arm A's winner by **> 0.135 ms/char**
  AND that gap **exceeds the within-arm across-seed spread** (I will report |between| / sd(within);
  I require **|between| > 1× the pooled within-arm sd** as well as > floor).
- **NO MATERIAL DIFFERENCE** — the winner gap is **< 0.135 ms/char**. Then the objective choice
  produces the same board at this budget, whatever the layout strings look like.
- **UNRESOLVED** — gap > 0.135 but **< the within-arm spread**. Then the search noise swamps the
  objective choice and I will say the objective does not matter *at this search budget*, which is
  itself the answer (my brief requires this call explicitly).

### On layout identity (the secondary axis, no floor — reported, not adjudicating)
Hamming distance between arm winners. I state now that I expect Hamming to be LARGE even in the
NO-MATERIAL-DIFFERENCE case, because a 30-key permutation has many near-isometric relabelings
(swapping two rare keys costs ~nothing). **So a large Hamming distance is NOT evidence of a
material difference** and I will not read it as one. Hamming is descriptive of *identity*; ms/char
vs the floor is what adjudicates *materiality*.

### Predictions I am registering so they can be refuted
- **P1** — Arm B and arm C winners will be within the floor **of each other** (registered-(c) vs
  50/50), because NORMGAUGE-1 measured POOL's 0.0612 to do no observable work and the two
  weightings to be statistically indistinguishable. If B and C differ by > 0.135, P1 is refuted
  and POOL's weight DOES do work in the search (a new finding, opposite to the ledger).
- **P2** — Arm A's winner will be **better on ms/char** than B's and C's, and the normgauge
  winners **better on the blend** than A's: i.e. each objective wins on its own ruler. If any arm
  loses on its own ruler, the search failed and I must say so rather than interpret it.
- **P3** — The cross-objective cost will be **asymmetric in magnitude**: because
  `spearman(ms/char, solo-AALTO) = +1.0000` (NORMGAUGE-1) and the blend is 0.5411 AALTO, I expect
  A's penalty on the blend to be SMALL and B/C's penalty on ms/char to be LARGER. 🟠 INFERRED.
- **P4** — At least one normgauge board will show a **higher home-row share** than arm A's. The
  two timing probes already hint this (A 28.4%, B 37.5%), so **P4 is weak — it is nearly
  post-hoc** and I mark it as such rather than claiming a clean prediction.
- **P5** — **No** produced layout will exactly reproduce a shipped field board (a 30! space makes
  exact re-derivation essentially impossible); the interesting quantity is the *minimum Hamming
  distance* to the field, and whether normgauge lands nearer the field than arm A does.

## 5. What this output is NOT

Descriptive only. I am not adjudicating the deadlocked accuracy question, and **no sentence of my
report will recommend adopting a layout or landing an objective** — both are user-gated. If the
evidence happens to look favourable to normgauge, that is still not a recommendation.

Everything here is a property of the **FITTED MODEL** (surfaces fitted on 4 training layouts,
baked at 90 WPM), never a claim about realized human typing. The `.standardized` frame caveat
applies throughout: the three surfaces share AALTO's bigram tensor and are LESS independent than
the `.native` arrays.

## 6. Known limits, stated up front

- **n=10 seeds is a spread estimate, not a distribution.** I report min/median/max and sd; I do
  not run significance tests on 10 stochastic search outputs.
- **The `one` anchors are searched optima, so every normalized value is an UPPER bound** on the
  true normalized score (the module says so). A blend of 0.946 is not "94.6% of optimal".
- **One model seed (`bigram_reg31_seed0`) for arm A.** Model-seed variance is not in scope here;
  the campaign's resolution floor was measured across seeds and I use it as given.
- The normgauge objective has **no QAP table fast path**, so arms B/C get ~7× fewer evaluations
  per wall-clock second than arm A. Since I hold *iterations* (not wall-clock) at defaults, the
  arms are matched on search effort, not on time.
