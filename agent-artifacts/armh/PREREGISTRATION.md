# ARM H — PRE-REGISTRATION

**Written and committed BEFORE any ARM H search result exists.** No search of this arm's
objective has been launched at commit time, and **no baseline-control run has been launched
either** — so `sd_H`, the ruler every verdict below is judged against, does not yet exist as a
number. Everything numeric in this file is one of:

- **(a)** a frozen figure from a prior arm's artifact that **I re-derived myself in this
  worktree** (never transcribed — §7 lists every one with its diff), or
- **(b)** a design constant I am choosing now, or
- **(c)** a **frozen-geometry enumeration** I performed before writing this file: the
  exhaustive 435-member 1-swap ball around arm B (§2). That is a property of the *board*, not
  a search result, of the same kind as ARM G's D-of-existing-layouts table.

**MODELLED ONLY.** Every number is on the g-frame (geometry-only), a **baked 90 WPM** fitted
timing surface, corpus **blend-v1**, skipgrams `1-skip31`, and the **as-shipped NESTED
`bad_redirect` oxey convention** (§0). Nothing ARM H produces is a claim about realized human
typing speed. **No layout will be adopted or recommended.**

---

## 0. Pre-flight and the convention declaration

`grep -c _v1_pattern src/keybo/scoring/oxey.py` = **0**; `oxey.py` last touched by `c37a080`.
So my `oxey-style` is the **as-shipped nested convention** — a bad redirect is charged +2.0
**and** +4.0 = **+6.0**, per the PENALTYAUDIT-1 CORRECTION. This is *the same convention*
SPEEDTIE-1's 14.05× spread and every ARM G number were measured on, so comparability with the
figure I am targeting is preserved. Comparability with a post-OXEYFIX board is **not**, and I
will not quote one. **I will not merge `oxey-partition-fix`** (OXEYFIX-1 is a user gate).

**Controls C1–C7, all run BEFORE this file was written.** (§7 gives sources.)

| # | Control | Result | Verdict |
|---|---|---|---|
| C1 | worktree isolation, **POSITIVE** (trap 35, not "no hardcodes found") | `keybo.__file__` = `/tmp/armh/src/keybo/__init__.py` · `sys.prefix` = `/tmp/armh/.venv` · `FastEval.corpus_dir` = `/tmp/armh/data/corpus/blend-v1` · `trigrams.txt` md5 `c5066fa7bcc46dea1ecbc987fb465b4a` == trap-8 reference | 🟢 |
| C2 | arm B reproduces in my tree | FastEval **253.90057910352604**, absdiff from frozen **exactly 0.0**; shipped `analyze` 253.90057910352797 (diff 1.93e-12, *identical* to ARM G's) | 🟢 |
| C3 | **cross-path** FastEval (search) vs shipped `analyze` (judge), 13 layouts × 15 quantities | worst relative **1.233e-14**; **11 of 15 bit-exact**; rc=0 | 🟢 |
| C4 | **MUTATION control on C3** — a control that cannot fail tests nothing | planting `*1.000000001` on `oxey-style` (my objective) ⇒ **rc=1**; removing it ⇒ rc=0 | 🟢 bites |
| C5 | gauge directions **DERIVED two ways**, never assumed (trap 5) | qwerty-is-worst **14/14**; rank-correlation with predicted ms over 4000 random perms **13/14**, sole miss `sfs` at rho **−0.0157** ≈ 0 — *matching* ARME-1 and ARM G independently | 🟢 |
| C6 | numerical floor of my own objective (batch-shape + repeat) | `oxey-style` max deviation across {1, 2, 435, 20 000}-row batches = **7.105e-15**; `_ms_per_char` 1.023e-12; **same batch twice = exactly 0.0** | 🟢 |
| C7 | **my harness bit on its own first run** | my row-count assert fired: shipped `analyze` **always injects a `qwerty` reference row** (2 requested ⇒ 3 rows). Rewritten to assert the exact row-key **set** | 🟢 |

**Engine — REUSED, not rewritten.** ARM G's engine is validated (14 controls) and the brief
requires reusing it. Copied verbatim into `agent-artifacts/armh/`:
`evobj.py` md5 **dc45ef503792576157a872a996d9e9d7** and
`evidence_scorer.RESTORED.py` md5 **01f3a95ab7a0f53f8f9d5be057fc437e** — both **byte-identical
to the md5s ARM G declared**; `search.py` md5 `9245ef074101c72cf23e23faeb06ebe5`.

**Repointed hardcoded path, declared (trap 35).** `grep -rn "repos/keybo|/tmp/armg|/tmp/speedtie|/tmp/optev"`
over all three files returns **exactly one hit**: `search.py:427`'s
`startswith("/tmp/armg/")` worktree assert. I repoint it to `/tmp/armh/`. I do **not** reuse
`search_placebo.py` (hardcodes `cwd="/tmp/optev"`), `run_budget.py` or `analyze_budget.py`
(hardcode `WORKTREE=/tmp/speedtie`). My runner hardcodes `/tmp/armh` and writes only into my
own worktree and my own state dir.

**Driver dependency, declared.** `evobj.py:42` imports `keybo.analysis.evidence_scorer`, which
is **absent at my base commit `28942d7`**. Restored byte-identically (md5 verified after copy).
`grep -rn evidence_scorer src/ tests/ keybo-e2e/` = **0 hits**, so restoring it changes no
shipped scoring path. ARM H passes `weights_json=None` and uses **no fitted curve**: it
optimizes shipped gauges and the served surface only.

---

## 1. The objective — minimize `oxey-style` ALONE, hardness by LEXICOGRAPHIC construction

ARM G's registered diagnosis is my premise: its `D` was an **unweighted sum of
range-normalized excesses**, so the widest axis was the cheapest to trade away, and
`oxey-style` is **48.5% of the whole board's gauge range** (5.3× the next widest). Its
prescription, which is my design: *minimize `oxey-style` ALONE subject to HARD constraints —
hard, not summed, because a maximizer does not read flags and A SUMMED PENALTY IS A FLAG
(trap 51).*

**Frame (pre-declared).** `LIVE_GAUGES` is read from live code, not typed: the shipped 15-gauge
frame minus `sfr`, which counts doubled letters and is a **permutation invariant** (trap 23) —
a tie by construction that cannot be earned, and **never counted**. That leaves **14 live
axes**: `sfb sfs sfb-dist sfs-dist lsb lsb-dist alt roll sr-roll redir scissor imbalance
oxey-style comfort`. ARM H **minimizes `oxey-style`** and **hard-constrains the other 13**.

**The feasible region, as an equation.** With `dir_g` = +1 (lower better) for `sfb sfs sfb-dist
sfs-dist lsb lsb-dist redir scissor imbalance oxey-style comfort` and −1 (higher better) for
`alt roll sr-roll` (C5-derived), a layout `L` is **FEASIBLE** iff

```
(A13)  dir_g * ( g(L) - g(armB) )  <=  TOL          for all 13 constrained axes
(Spd)  ms(L)                       <=  253.90057910352604 + 2*sd_H
```

- `TOL = 1e-9`. **Derived, not chosen:** it is ~1.4e5× my measured `oxey-style` numerical
  floor of 7.105e-15 (C6) and ~1e-9 of the smallest real axis difference in §2, so it can
  neither admit float noise as a violation nor mask a real one. The same `TOL` is the
  **minimum resolvable margin** for a *strict* win (MARGIN-GATE-1: a champion chosen inside
  the resolvable margin is indistinguishable from a real one in the output).
- `sd_H` = **my own measured search-noise sd** — see §3. It does not exist yet.

**Hardness is implemented LEXICOGRAPHICALLY, in one scalar, with a proof of separation.**

```
V(L) = SUM over the 13 axes of max(0, dir_g*(g(L)-g(armB)) / |g(armB)|)
     + max(0, ms(L) - MS_EDGE) / EPS                       # the speed leg
fitness(L) =  oxey_style(L)              if V(L) == 0      # FEASIBLE branch
              BIG + V(L)                 if V(L) >  0      # INFEASIBLE branch
BIG = 1e6
```

Why this and not the alternatives:

- **Not a summed penalty** (ARM G's failure, trap 51). The two branches are disjoint intervals:
  every feasible layout scores in `[oxey_min, oxey_max]` and every infeasible layout scores
  `>= BIG`. `oxey-style` over the entire real board is bounded by qwerty's **88.20** and the
  1-swap ball's **81.65** max / **1.63** min, and the most negative value any campaign layout
  reaches is arm A's **−12.49** — so the feasible branch cannot exceed ~1e2 while the
  infeasible branch starts at 1e6, a **4-order gap**. *There is no exchange rate between the
  objective and a constraint*, which is exactly the property ARM G lacked.
- **Not pure rejection** (`+inf` for infeasible). `V` is retained *inside* the infeasible
  branch so the search still has a gradient **toward** feasibility — necessary, because §2
  measures the feasible set as a needle (0 of 200 000 random layouts hold even 6 of the 13
  constraints).
- **Confirmed by construction, then tested.** Two independent layers, and the second does not
  share code with the first:
  1. the objective can never rank an infeasible layout above a feasible one (interval
     separation above, asserted numerically at run start);
  2. a **hard champion gate** recomputes (A13) and (Spd) on the returned champion **through
     the shipped `analyze` path** — not through `FastEval` — and exits **rc=1** if it is
     infeasible.
  **Planted-infeasible test (registered here, run before any result is read):** inject a
  known-infeasible layout as the top archive entry and require the gate to reject it (rc=1),
  then remove it and require rc=0.

**Search-side `EPS`, and why it cannot repeat ARM G's failure.** ARM G's search band edge was
**looser** than its verdict band edge by 0.0251 and every champion landed in the gap. The fix
here is structural, not a promise: **the baseline control runs FIRST and `EPS` is set to
`2*sd_H` from it, so the search band and the verdict band are THE SAME NUMBER, by
construction.** The control's champions are draws from the *baseline* objective and carry no
information about ARM H's objective or its result, so this is not tuning on the outcome — it is
the standing (POOL × REPLICATE-STRUCTURE × SCALE × STATISTIC) rule applied in the only order
that makes search and verdict agree. `sd_H`'s estimator is **fully specified in §3 with no
free parameter**, and **I pre-commit to no adjustment of `EPS` after any ARM H result is
read.**

**⚠ The one direction this can still bias, stated now: TIGHTER, i.e. toward EMPTY.** If
`2*sd_H` comes out small, the search never explores the ms region between the band edge and a
looser plausible band, so a layout living there is *missed*. That biases me toward reporting
EMPTY and **against** my own headline — the conservative direction, and the opposite of ARM
G's error. §2's flip threshold and §5's sensitivity table make the bias auditable rather than
merely named.

---

## 2. THE FEASIBLE SET IS NOT EMPTY — and I established that by ENUMERATION before searching

This is a **frozen-geometry** result: all **435** single transpositions of arm B, scored under
frozen gauge definitions. No search, no seed, no ARM H objective.

🟢 **Exactly 1 of 435 satisfies all 13 hard axis constraints.** Call it **BALL-1**:

```
arm B   flmpg-yuo,sntdcireahkxbwv'.jzq   ms 253.90057910352604   oxey-style 8.611045585392063
BALL-1  flmpg-yuo,sntcdireahkxbwv'.jzq   ms 253.96642626168640   oxey-style 7.577429131770819
                    ^^  (slots 13<->14 = d<->c)  +0.06584716        -1.033616453621244
```

- The swap is `d`↔`c` on slots 13 and 14 — **both left index, both row 2**: a *same-finger,
  same-row* exchange (verified through `Geometry.same_finger`, not asserted).
- Per-axis: **5 strictly better** (`comfort` −0.0316, `lsb` −0.5154, `lsb-dist` −1.0280,
  `sfb-dist` −0.0162, `sfs-dist` −0.1207), **8 exact ties**, **0 worse**.
- ⚠ **BALL-1 shares arm B's hand partition**, so `alt` and `imbalance` are ties **BY
  CONSTRUCTION** (ULTRAAUDIT-INTERIM), not earned. 210 of the 435 share the partition. I
  register now that these two cells will be reported as construction-ties in every count, and
  that per-pair **CONTESTED** counts — never a bare n/15 — are what I will publish.
- Its `oxey-style` gain is **1.0336**, which is **1.45e14 ×** my measured numerical floor
  (C6, 7.105e-15) and 1.0e9 × `TOL`. **Resolvable by an enormous margin.**

🟢 **0 of 435 are FASTER than arm B** (fastest neighbour is **+0.00944898**). So arm B is a
**strict 1-swap local minimum on speed** — an independent corroboration of the campaign's
most-reproduced result, obtained by exhaustive enumeration rather than by another search.

🔴 **AND THE CONSEQUENCE THAT DECIDES THIS ARM, REGISTERED BEFORE `sd_H` EXISTS.** BALL-1 is
13-axis-feasible and strictly better on `oxey-style`, but it sits **+0.0658 ms/char** from arm
B. Whether it is *speed-tied* therefore depends entirely on `sd_H`:

> **THE FLIP THRESHOLD IS `sd_H = 0.03292357908017607`** (= 0.06584715816035214 / 2).
> If `sd_H >= 0.032924`, BALL-1 is **in-band** and outcome 1 is available.
> If `sd_H <  0.032924`, BALL-1 is **out-of-band** and — absent any better find — the
> registered answer is **EMPTY FEASIBLE SET with SPEED as the binding constraint**, by
> `0.06584716 − 2*sd_H` ms/char.

Joint feasibility by candidate speed band, enumerated now (13 axes **and** the band):

| band edge (ms over arm B) | 0.0 | 0.02 | 0.05 | 0.0617 | 0.10 | 0.1234 | ∞ |
|---|---|---|---|---|---|---|---|
| neighbours inside the band | 0 | 2 | 7 | 8 | 14 | 18 | 435 |
| **…AND 13-axis feasible** | **0** | **0** | **0** | **0** | **1** | **1** | **1** |

So the 1-swap ball's answer flips between `0.0617` and `0.10`. Both of the two rulers this
campaign has ever measured — the borrowed **0.0617** (2sd = 0.1234) and ARM G's own
**0.049171** (2sd = 0.098342) — would admit BALL-1; a materially tighter `sd_H` would not.
**I am registering that fork, with its exact numeric threshold, before I can see which side I
land on.**

⚠ **What BALL-1 is NOT.** It is not an ARM H search result and I will never present it as one.
It is the *bar* my search must clear, and §4's F3 makes failing to clear it a registered
FAILURE.

**Two further prereg inputs, same enumeration family:**
- `roll` is violated by **all 12** non-armB reference layouts (the five other frozen 1M
  champions, arm-A, keybo-lsb, keybo-lsb+lm, flagship-c3, graphite, semimak) — and arm B is
  **best-of-six on `roll`, `sfs`, `sfs-dist`, `sr-roll`**. So the constraint set demands
  matching arm B where no sibling manages to.
- **No axis is violated ALONE** by any ball member (`n_violating_ONLY_this_axis` = 0 for all
  13), so no single axis can be named the binder from this pool — the binding-constraint
  report in §4 must come from the search's own archive, not from here.
- In a **200 000-layout random pool**, **0** hold even the six axes `oxey-style` *restates*
  (`sfb lsb scissor imbalance redir alt`), let alone all 13. **The feasible set is a needle
  around arm B**, which is why §3 runs a warm-start arm and predicts the cold arm finds
  nothing.

---

## 3. `sd_H` — my own ruler, its quadruple, and the runs

**`sd_H` = the sample sd (ddof=1) of champion `ms/char` across my own BASELINE-CONTROL seeds.**
Its quadruple, stated as the standing floor rule requires:

- **POOL** = my 5 ARM-H-family baseline-control champions (near-optimal, cold start, blend-v1);
- **REPLICATE-STRUCTURE** = independent cold-start search runs, one champion each;
- **SCALE** = raw ms/char;
- **STATISTIC** = sd, ddof=1, n=5.

**Neither ARM G's 0.049171 nor the borrowed 0.0617 is used in any verdict.** They appear only
in §5's sensitivity table, explicitly labelled as *other quadruples*, and no verdict changes
with them (that is what the table is for).

**Seeds — pre-registered formula, asserted DISJOINT from prior families.**
`seed(r) = 31337 + 104729*r`, r = 0..4 ⇒ **31337, 136066, 240795, 345524, 450253**. Asserted
in code to be disjoint from ARM G's `20260728 + 7919r` and the placebo's `900000 + 7919r`
families, so my draws are independent rather than a re-run.

**Arms — three, on the identical 5 seeds, identical budget:**

| arm | objective | why it is not optional |
|---|---|---|
| `baseline` | minimize ms/char (unmodified) | **measures `sd_H`**, and is the same-seed same-size placebo that makes any ARM H gain attributable to the OBJECTIVE rather than the draw (traps 17/32/34) |
| `armh-cold` | ARM H lexicographic, cold start (islands × 64 uniform random) | the **objective-vs-draw placebo** at matched initialization, and the test of P2 |
| `armh-warm` | ARM H lexicographic, warm start injecting arm B into every island | the arm that can actually reach the needle §2 measured |

**Warm start, declared and fail-loud (trap 10: "an optional warm start that finds nothing
degrades to a COLD run and still reports").** `armh-warm` injects arm B's permutation into
every island's initial population and **asserts at start-up** that (i) the injected layout is
present, and (ii) it evaluates as FEASIBLE with `V == 0`. If either fails the run exits
**rc=1** rather than silently running cold. **I state plainly that `armh-warm` is a
neighbourhood search around the incumbent, and I will not present anything it finds as a
cold-start discovery.**

**Budget — configuration-matched to the reference band.** 1 000 000 unique evals/seed,
`islands=20`, `epochs=12`, `overshoot=1.95`, `ga-share=0.6`, `polish-sweeps=40` — the identical
engine configuration as the 1M placebo and ARM G. Justified by SPEEDTIE-BUDGET-1: ~7.4M extra
evals bought **zero new territory** (the 10M champion set is a strict subset of the 1M set), so
1M is right and larger is the experiment already shown saturated.

**`unique_evals` reported ACHIEVED, never requested** — the engine stops on the epoch schedule,
so a run can fall short and still exit 0. **Pre-registered floor: a seed achieving < 80% of
1 000 000 unique evals is EXCLUDED from the primary n and reported as excluded**, with a
sensitivity analysis including it. **`.keys.npy` sidecars are RETAINED so `--resume` works**
(SPEEDTIE-BUDGET-1 deleted its 388 MB of sidecars and lost that ability).

**rc read from a SENTINEL, not from a callback** (trap 50: a `while pgrep` watcher died three
times in one session while the work completed fine). Every run writes its own rc file; the
result is gated on reading those files, and the completion callback is treated as
best-effort notification only.

**`n_runs` and `n_distinct` reported side by side, and Hamming BOTH over runs and over
distinct champions** (SPEEDTIE-BUDGET-1's traps 2 and 3 — keying a per-run collection on the
RESULT silently collapses n, and the two Hamming readings diverge sharply enough that either
alone supports the opposite verdict).

---

## 4. Verdicts — the three outcomes, and what makes them DISTINGUISHABLE in advance

Decided in this order. Every per-axis adjudication is on the **shipped `analyze`** path
(C3/C4 gate it), never on `FastEval` alone.

- **① COLLECTED** — a layout with **(A13) satisfied**, **(Spd) satisfied**, and
  `oxey_style < 8.611045585392063 − TOL`. Reported with: per-pair **CONTESTED** counts (ties
  named, `alt`/`imbalance` flagged as construction-ties where the hand partition is shared,
  `sfr` never counted), the **cluster-corrected** reading alongside the raw one (trap 39;
  `oxey-style` is R² = **0.9082** on its six restated components *in the fastest 2% of my own
  4000-layout random pool*, which I measured — trap 27 is live and I am not citing oxey as
  independent evidence alongside those six), and the **strict-win margin** against `TOL`.
  - **①a** if the layout comes from an **ARM H search**;
  - **①b** if the *only* qualifying layout is **BALL-1**, i.e. the collection is achieved by
    **enumeration** and the search added nothing. I register ①b as a distinct, weaker result
    now so I cannot later dress it as ①a.

- **② EMPTY FEASIBLE SET, demonstrated** — ≥ 3 seeds clear the 80% floor, **and** no layout in
  any champion, any run's top-50 archive, or the 435-member 1-swap ball satisfies (A13) **and**
  (Spd). Then I report the **BINDING constraint and by how much**: the axis (or the speed leg)
  whose relaxation would first admit a candidate, quantified as the minimum excess over all
  candidates that violate that leg alone. §2 already establishes that if this happens it will
  be the **SPEED** leg, by `0.06584716 − 2*sd_H` — and I register that prediction as P7.

- **③ FAILURE by my own condition** — any of:
  - **F1** fewer than 3 seeds clear the 80% `unique_evals` floor (below the registered n ≥ 3);
  - **F2** the champion gate rejects a returned champion, or the planted-infeasible test does
    not bite ⇒ my hardness construction is broken, so nothing I report is trustworthy;
  - **F3** **no ARM H search seed finds a 13-axis-feasible layout with `oxey_style <= 7.577429131770819`**
    — i.e. a **435-point exhaustive enumeration beat a 3 × 5 × 1M-eval search on its own home
    turf**. That is a failure of my *instrument*, not a fact about the world, and I will say so;
  - **F4** any control C1–C7 fails on re-run after the search, or the two paths disagree by
    more than 1e-9.

**FAILURE and EMPTY are distinguishable by construction:** ② is a statement about the *world*
(the constraints genuinely admit nothing at this band) and requires the search to have run
properly; ③ is a statement about *my instrument* (it did not run properly, or it lost to brute
force). F1/F2/F4 make ② unreportable — a broken run cannot demonstrate emptiness. F3 is the
one that separates them in the ambiguous case: if the search finds nothing but enumeration
found BALL-1, that is **F3 + ①b**, not ②.

**What I will NOT claim, pre-committed:**
- No adoption, no recommendation to adopt, no push, no CR, no edit to `PREREGISTRATIONS.md`.
- **No near-miss reported as a success.** A candidate violating (A13) or (Spd) by *any* amount
  above `TOL` is INFEASIBLE and is reported as such, however small the excess.
- **No constraint relaxed mid-run.** If I relax anything, that is a second arm with its own
  prereg, and I will say so rather than fold it in.
- No bare `n/15` win counts; no `sfr` in any count; `alt`/`imbalance` construction-ties named.
- No multiplier for `scissor`'s mispricing — direction only (SCISSORPRICE-1 unsettled).
- No claim that any gauge difference is **perceptible**. SPEEDTIE-1's caveat binds: the
  differences are **free**, not necessarily felt.
- No borrowed resolution floor in any verdict.

---

## 5. Predictions, registered now — they can fail, and two are self-adverse

| # | Prediction | Refuted by |
|---|---|---|
| **P1** | ≥ 3 of 5 `armh-warm` seeds return a champion satisfying (A13) | fewer than 3 do |
| **P2** | 🔴 *self-adverse:* `armh-cold` returns **ZERO** (A13)-feasible champions in 5 of 5 seeds (0 of 200 000 random layouts hold even 6 of 13 constraints) | any cold seed returns one |
| **P3** | the best ARM H **search** champion has `oxey_style` **strictly below BALL-1's 7.577429131770819** — i.e. the search beats the enumeration | none does (⇒ F3) |
| **P4** | 🔴 *self-adverse:* **no** ARM H champion is faster than arm B (`ms > 253.90057910352604` in all seeds) | any lands below |
| **P5** | `sd_H` lands **within 1.5×** of the borrowed 0.0617 in either direction. ⚠ *This is the OPPOSITE of ARM G's P5, which predicted ≥ 1.5× and was refuted at 1.255×.* I am predicting its data, not its guess | it lands outside 1.5× |
| **P6** | `2*sd_H >= 0.06584715816035214`, so **BALL-1 is in-band** | `sd_H < 0.032924` (⇒ ② with SPEED binding) |
| **P7** | if the verdict is ②, the binding constraint is **SPEED**, not any of the 13 axes | an axis binds instead |

**Sensitivity table I commit to publishing regardless of outcome** — the same verdict computed
under three *different quadruples*, so a reader can see exactly how much the ruler carries:

| ruler | value | 2× band | quadruple |
|---|---|---|---|
| **`sd_H` (PRIMARY)** | *unmeasured at commit time* | *unmeasured* | my 5 baseline champions, cold-start runs, raw ms/char, sd ddof=1 n=5 |
| `sd_G` (ARM G) | 0.049171 | 0.098342 | ARM G's 5 baseline champions, same structure, **different seed family** |
| borrowed (SPEEDTIE-1) | 0.0617 | 0.1234 | the 6-seed 1M placebo, **n=6, different seed family** |

---

## 6. Self-separation plan, committed before results exist

After the FIND pass I re-read my own output as a hostile stranger and, for each claim, name
(i) what would refute it; (ii) whether my check **shares a component** with its target (trap
45: a difference statistic cannot test a shared component; and the SELF-AUDIT SWEEP found two
"independent" controls that shared the component under test); (iii) whether any control ran
only **after** I used its result. I report what I **killed** of my own, not only what survived.

Four shared-component risks I can already name, with the handling registered now:

1. **`oxey-style` restates six of the axes I constrain** (R² = 0.9082 in-band, measured by me;
   trap 27 puts it at 0.9937 over the campaign's pool). So minimizing `oxey` while holding
   `sfb lsb scissor imbalance redir alt` is **partly self-cancelling by construction** — the
   objective is fighting its own constraints. This is a *structural* property of the arm, it
   predicts a small feasible gain, and I register it as an expectation rather than discovering
   it afterwards.
2. **`lsb` / `lsb-dist` are near-duplicates** (spearman 1.0000, sibling-measured) — one axis
   wearing two names. BALL-1 improves **both**, so its "5 strictly better" is at most **4
   independent** axes. Registered now, before I could be accused of noticing it late.
3. **My verdict path and my objective share `FastEval`.** Mitigation is C3/C4: every published
   number and the champion gate go through the **shipped `analyze`** path, and the two paths
   are pinned at 1.233e-14 with a mutation control proving the pin bites.
4. **`sd_H` sets both my search band and my verdict band.** That is deliberate (§1) and is what
   removes ARM G's failure mode, but it means the band is not independent of my control runs. It
   *is* independent of every ARM H result, which is the property that matters, and §5's
   sensitivity table shows what changes if the ruler changes.

---

## 7. Provenance of every borrowed number in this file

| Number | Value | Source | Re-derived by me? |
|---|---|---|---|
| arm B ms/char | 253.90057910352604 | ARM-B / SPEEDTIE-1 | 🟢 yes — FastEval absdiff **exactly 0.0**; shipped `analyze` 1.93e-12 |
| arm B's 14 gauges | — | ARM G's D-prereg-input | 🟢 all 14 re-derived from live code; **no value transcribed** |
| six frozen 1M champions | 6 layout strings | **the ORIGINAL artifact** `optevidence-1/search-noise-placebo.json` md5 `f5d78de67bf1c3c0f8e18a6b675942e0`, `runs.baseline[].layout` — **not ARM G's transcription of it** | 🟢 all six re-scored; ms sd re-derived as **0.06171827216720297** vs the artifact's stored 0.06171827216711913 |
| oxey-style spread across the six | 14.05× | SPEEDTIE-1 | 🟢 independently **14.051204647173282** |
| borrowed baseline sd | 0.0617 | same artifact, `bands.baseline.ms_per_char.sd` | 🟢 read from the artifact; **used in NO verdict** — sensitivity only |
| `sd_G` | 0.049171 | ARMG-1 | ⚠ quoted from the ledger; **used in NO verdict** — sensitivity only |
| incumbent layout STRINGS (arm-A, keybo-lsb, +lm, flagship-c3) | 4 strings | ARM G's `D-prereg-input.json` | 🟢 strings borrowed, **every gauge value re-derived**; graphite/semimak/qwerty from live `NAMED_LAYOUTS` |
| `alt`/`imbalance` hand-partition invariance | — | ULTRAAUDIT-INTERIM | 🟢 partition computed from live `Geometry`; I verified BALL-1 shares arm B's partition. (The *invariance theorem* itself I accept from the ledger and only ever use to **exclude** — the conservative direction.) |
| `sfr` permutation invariance | — | trap 23 / live `INVARIANT_GAUGES` | 🟢 read from live code (`LIVE_GAUGES` excludes it) |
| spearman(`lsb`,`lsb-dist`) = 1.0000 | — | sibling | ⚠ accepted; used only to *weaken* my own count (conservative) |
| engine config (islands 20 / epochs 12 / overshoot 1.95 / ga-share 0.6 / polish 40) | — | `search_placebo.py` + ARM G's runner | 🟢 read from the drivers, not a docstring |

**Every constant the objective uses is GENERATED from live code and re-asserted at run start**
(`armh_assert_constants()`, modelled on ARM G's `armg_assert_constants()`, which exists because
ARM G hand-transcribed 28 numbers and **all** of them were wrong by ~1e-5 while it hypothesized
a BLAS batch-shape cause and measured it — refuted by ~10 orders; the cause was its typing).
**The run REFUSES to start on drift.** GENERATE OR ASSERT, NEVER RETYPE.
