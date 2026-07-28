# SPEEDTIE-BUDGET — PRE-REGISTRATION

**Committed BEFORE any 10M result exists.** Nothing below may be edited after the first 10M
champion is read. Any post-hoc addition goes in a clearly-labelled ADDENDUM section with its
own commit, and the verdict stated here is the one that binds.

Author: subagent `speedtie` of `keybo-optimization`. Branch `speedtie-budget`, worktree `/tmp/speedtie`.

---

## 1. THE QUESTION

SPEEDTIE-1 registered that six cold-start searches of arm B's own served objective (baseline =
minimize predicted ms/char on the served K31 surface at 90 WPM, blend-v1) landed within
**0.1760 ms/char** of each other (= 2.85x that objective's own 6-seed sd of **0.0617**) while
their 15-gauge profiles spanned **17.70x** on `imbalance`, **14.05x** on `oxey-style` and
**3.76x** on `scissor`, with no sibling dominating any other.

That was measured at a **1,000,000-unique-eval** budget. The campaign's real arms ran ~10M
(arm B actually achieved 9,252,349). The open question:

> **Does the free gauge headroom SURVIVE at the full budget, or is it an artifact of
> under-convergence?**

- **H-REAL** — the objective is genuinely near-indifferent over a broad set of layouts. At 10M
  the speed spread stays ~flat while the gauge spread stays large. The headroom is real.
- **H-UNDER** — at 1M the runs had simply not converged, so the spread is search slack. At 10M
  the champions converge toward each other: speed spread shrinks AND gauge spread shrinks AND
  Hamming distances fall.

---

## 2. DESIGN, AND THE ONE FACTOR IT VARIES

**Paired, seed-matched, one-factor.** Same engine, same objective (`--arm baseline`), same
corpus (blend-v1), same 90 WPM surface, same seed formula the placebo used:

    seed(r) = 900000 + 7919 * r,   r = 0..5

so run *r* at 10M is the **same draw** as run *r* at 1M and the comparison is run-for-run
paired. An unpaired comparison of two different seed sets would confound seed with budget.

**Engine config — chosen so `init` is BIT-IDENTICAL to the 1M run.** `search.py:318-323`
builds the initial population as `islands x 64` uniformly random C30M permutations from
`np.random.default_rng(args.seed)` — I read this myself and confirm there is **no incumbent, no
warm start, no injected layout**. `init` is therefore a function of `(seed, islands)` ONLY. So I
hold `islands = 20` — the placebo's value — and the 10M run starts from **the identical
population** the 1M run started from. Only the number of epochs changes:

| | islands | epochs | overshoot | calls/island/epoch | budget |
|---|---|---|---|---|---|
| placebo (1M, frozen) | 20 | 12 | 1.95 | 8,125 | 1,000,000 |
| **this run (10M)** | **20** | **120** | **1.95** | **8,125** | **10,000,000** |
| (campaign's real arm, for reference) | 40 | 55 | 1.95 | 8,863 | 10,000,000 |

`per_epoch = int(budget * overshoot) // (epochs * islands)`, so 19.5e6/(120*20) = **8,125** —
**exactly** the placebo's per-epoch spend. This run is literally *"the same search, continued
10x longer"*: identical initial population, identical per-epoch spend, identical operators,
more epochs. That is the cleanest possible under-convergence test, and it means a difference
CANNOT be attributed to a changed engine.

**n.** n = 6 seeds if affordable (matching the placebo exactly, giving 6 paired runs). Minimum
n = 4 per the brief. Decision thresholds below are stated so they read on either n; where n
matters I say so.

**Scoring.** Every champion is scored with the **shipped CLI** (`uv run --no-sync keybo analyze
--json`), never a hand-rolled scorer (trap 28 / trap 13: a hand-rolled reimplementation loses
the validated constructor's permutation guard). The driver's own `_ms_per_char` is used only as
the search's internal objective; every number I report comes from the CLI.

**Frame.** 15 gauges are computed; **`sfr` is EXCLUDED from every spread, ratio, dominance and
win-count** because it counts doubled letters and is a permutation invariant (trap 23:
numpy reports its std as ~1.9e-14, not 0, so a `std>0` filter would keep it and
rank-correlate pure noise). **All dominance/spread arithmetic is on the 14 LIVE gauges.**
I state this again in the writeup.

---

## 3. THE STATISTICS, DEFINED BEFORE THEY ARE COMPUTED

Let the 10M champions be `C10 = {c_1..c_n}` and the seed-matched 1M champions `C1`.

1. **Speed spread.** `range = max - min` and `sd = stdev(ddof=1)` of CLI ms/char.
   `R_speed = range_10M / range_1M`, `R_sd = sd_10M / sd_1M`.
2. **Gauge spread.** Per live gauge *g*: `range_g = max - min` and `ratio_g = max/min`.
   Headline aggregate: **`M_gauge` = median over the 14 live gauges of
   `range_g(10M) / range_g(1M)`.** A median (not a mean) so one gauge cannot carry the verdict.
   I also report the count of gauges whose `range_g` shrank by >2x.
3. **Hamming.** All `C(n,2)` pairwise distances between 10M champions (chars differing by
   position, 0..30). `mean_H_10M` vs `mean_H_1M`. Also report `min_H`.
4. **Dominance.** On the 14 live gauges, direction taken from the campaign's registered
   `EXPECTED_SIGN`-corroborated convention. A champion *a* dominates *b* iff *a* is at-least-as-good
   on all 14 AND strictly better on >=1 (**trap 33: a strict-win term is mandatory, or ties count
   as wins**). Report `n_ge` and `n_strict` separately, and the full n x n matrix.
5. **Convergence diagnostic (per ARME-1's registered lesson: diagnose convergence by whether
   best-fitness has STOPPED IMPROVING, not by budget fraction).** From each run's epoch trace I
   report the epoch at which the champion last improved, and the improvement over the final
   50% of epochs. This is reported for every run whatever the verdict.

---

## 4. DECISION RULE — NUMBERS, FIXED NOW

Read in this order. The first rule that fires decides.

**H-UNDER** is called iff **ALL THREE** hold:
  - `R_speed <= 0.50` (the speed range at least halves), **AND**
  - `M_gauge <= 0.50` (the median per-gauge range at least halves), **AND**
  - `mean_H_10M <= 0.75 * mean_H_1M` (mean Hamming falls by >=25%, i.e. <= 20.05 of 30 given
    the 1M mean of 26.7333).

  Rationale: H-UNDER's whole content is *convergence toward each other*, which must show on
  all three axes at once. Any one of them alone is a weaker claim.

**H-REAL** is called iff **BOTH** hold:
  - `M_gauge >= 0.80` (the median per-gauge range is essentially preserved — retains >=80%), **AND**
  - the 10M gauge spread is still large in absolute terms: **>= 2 of the 14 live gauges have
    `ratio_g >= 5.0`** at 10M (the 1M run has 2 such gauges: imbalance 17.70x, oxey-style 14.05x),
    **AND** no champion dominates any other on the 14 live gauges (`n_strict`-correct predicate),
    i.e. the objective still cannot break the tie.

  Note H-REAL deliberately does **not** require the speed spread to stay flat — see the
  asymmetric case below, which is the point the brief singled out.

**INDETERMINATE** otherwise, including specifically:
  - `M_gauge` in `(0.50, 0.80)` with no other rule firing;
  - any case where the dominance and spread criteria disagree;
  - n < 4 completed runs;
  - any run whose achieved `unique_evals` is below **8,000,000** (80% of the requested 10M) —
    such a run is a DIFFERENT experiment and is reported as one, not pooled.

### 4a. THE ASYMMETRIC CASE — speed spread shrinks but gauge spread does NOT

This is a real possible outcome and the brief is right that it favours H-REAL *more strongly*
than the clean case. Registered handling, decided now:

> If `R_speed <= 0.50` **but** `M_gauge >= 0.80`, I call **H-REAL (STRONG FORM)**.

Reasoning, registered in advance so it cannot be reverse-engineered from the data: a shrinking
speed spread with a preserved gauge spread means the extra 9M evaluations bought **agreement on
the objective** while buying **no agreement on the gauges**. The runs did converge — they
converged *in objective value* — and the gauge disagreement survived that convergence. That is
a direct measurement of the objective's indifference: at the same ms/char, to tighter tolerance
than before, the layouts remain 14x apart on gauges. The 1M result could be explained by "these
runs are all still wandering"; this pattern cannot, because the wandering has demonstrably
stopped on the axis being optimized. **The headroom is then not slack — it is a level set.**

Conversely, if `R_speed >= 1.50` while `M_gauge >= 0.80`, I will report H-REAL as **NOT**
established and flag the run as suspect (a *growing* speed spread at 10x budget indicates a
scheduling or convergence pathology in my own harness, not a property of the objective), and
I will say so rather than bank the gauge half.

### 4b. Guards on my own reasoning

- **A gauge ratio is unstable when its min approaches 0** (`imbalance` min at 1M is 0.2755).
  So I report BOTH `range_g` (additive, stable) and `ratio_g`, and **every threshold above that
  uses a spread is on `range_g`**, never on `ratio_g`. `ratio_g` is used only in the
  "still large in absolute terms" leg, where its instability cannot manufacture a shrink.
- **A shrinking spread of a MAXIMUM is expected from more search even under H-REAL** for the
  *objective* (more search = better minima = tighter top end). That is exactly why the
  asymmetric case is registered as pro-H-REAL rather than as evidence for H-UNDER.
- I will not upgrade any finding to VERIFIED on the strength of a single seed, and I will not
  substitute a seed if one fails (brief's hard constraint): n is what completed.

---

## 5. POSITIVE CONTROLS ALREADY PASSED (before the run)

Recorded here because a control run after seeing the result is not a control.

1. **Cold start read, not assumed.** `search.py:318-323` — `seed_rng.permutation(30)`,
   `islands x 64`, no warm start. Verified by reading the source in my own worktree.
2. **Worktree isolation is POSITIVE, not merely the absence of a hardcode.**
   `grep -rn "repos/keybo" drivers/*.py` -> no hits, all `sys.path.append` (trap 35), AND
   `FastEval.corpus_dir` resolves to `/tmp/speedtie/data/corpus/blend-v1` — my own worktree.
3. **Frozen-number reproduction, all six 1M champions, in MY worktree:** worst
   `|got - frozen|` = **2.814e-12**, and the arm-B champion reproduces at **exactly 0.0**
   (253.90057910352604). Float summation order only; nothing substantive differs.
4. **Corpus md5 re-derived, not cited:** blend-v1 `trigrams.txt` =
   `c5066fa7bcc46dea1ecbc987fb465b4a`, matching TOOLING-TRAPS' reference.
5. **The ms/char code path is bit-identical to the commit the 1M run used.**
   `git diff 1b4a4d8..HEAD` over `timecard.py, kmstats.py, corpus.py, geometry.py,
   classify.py, comfort.py, oxey.py` and all of `data/` = **zero lines**. (My HEAD is NOT a
   descendant of that commit; the only differing files are the evidence-scorer ones, which the
   `baseline` arm never calls — `_objective` returns `g["_ms_per_char"]` directly for
   `arm == "baseline"`, before `evidence_score` is ever reached.)
6. **Trap 38 fix present at my HEAD:** `analyze.py:452` asserts `len(rows) == len(specs)`, so a
   truncation collision cannot silently drop a champion's row.

## 6. KNOWN DEFECTS IN THE INHERITED DRIVER THAT I DO NOT INHERIT

`search_placebo.py` is the wrong runner for this job and I do not use it as-is:
- `cwd="/tmp/optev"` — hardcoded to **another agent's worktree at a different commit** (trap 35
  wearing a subprocess's clothes; the copied file's own `sys.path` hygiene does not save you if
  the subprocess is launched in someone else's tree).
- `timeout=3600` — would kill a long run and, per trap 22/trap 1, a killed run looks exactly
  like a missing sentinel.
- writes to `/local/home/zegertho/agent/state/optevidence/artifacts` — another workspace.
- hardcodes `islands=20, epochs=12`, so it cannot express this experiment at all.

My runner sets `cwd` to **my** worktree, has **no** subprocess timeout, writes only under
`/tmp/speedtie` + my own state dir, and takes islands/epochs as arguments. `search.py` and
`evobj.py` are used **unmodified** (md5-verified against the originals).

## 7. SCOPE — WHAT THIS CANNOT SAY

Everything is **MODELLED**: g-frame, baked 90 WPM, blend-v1, `skipgrams=1-skip31`. Nothing here
is a claim about realized human typing speed. **No layout is adopted or recommended for
adoption.** n<=6 seeds, ONE objective (baseline served), ONE corpus. A verdict here is about
*this objective at these two budgets*, and does not transfer to the evidence arms, whose own
1M search-noise sd is 5.6x larger (0.3440 vs 0.0617).
