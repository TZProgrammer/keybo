# NECESSITY-POOL — is near-optimality NECESSARY for the cross-source instrument disagreement?

**Answer: YES for the cross-source leg at matched restriction — and the prior round's "not
excluded" upgrades to "positively supported."** The asymmetrically restricted random pool at the
archive's own `u_A/u_B` does **not** reproduce the archive: `archive − asym = +0.2303`,
CI `[+0.0865, +0.3742]`, `p = 0.0015`. **This refutes my own preregistered prediction**, which
expected asymmetric restriction to be largely sufficient with a CI including zero.

But the answer splits by leg, and that split is the report's main contribution:

| leg | archive | matched asymmetric random pool | is near-optimality necessary? |
|---|---|---|---|
| instrument-vs-instrument (cross) | **+0.2184** | **+0.0131** ± 0.0357 (R=12) | **YES** at matched restriction (p = 0.0015) |
| instrument-vs-itself (within) | **+0.9647** | **+0.6955** ± 0.0146 (R=12) | **YES** at matched restriction (+0.2723 gap, p = 0.00049 for the asym/sym contrast) |
| both legs jointly | +0.2184 / +0.9647 | reproduced at **3.7× the archive's `u_B`**, not at matched `u_B` | **NO** as an existence claim; **YES** as a claim at matched narrowness |

**A null was acceptable and this is not one.** Every number below is generated from the emitted
JSON by `tables.py`; no cell is hand-transcribed.

- **Branch** `necessity-pool`, **HEAD** `74ee9ac` (stated after the final commit via `git rev-parse`), base `e6a5b9e`.
- Worktree `/tmp/necesspool`; drivers + artifacts committed under `agent-artifacts/necessity/`.
- Nothing pushed, nothing merged, no layout adopted, no weight flipped, `PREREGISTRATIONS.md` untouched.
- **Preregistration committed at `eb2fb04` BEFORE any new score existed** — prediction, verdict and five named falsifiers.

---

## 1. Three defects in the brief I was given

I was told to verify every number. Three did not survive; the third is a method error I was
explicitly instructed to commit, and I declined and did the correct thing instead.

### 1.1 🔴 The "SLACK" is the SAME algebraic identity the ledger had just retracted

The ledger retracts `Spearman(rho, log C/D) = +0.999` as an identity, then promotes as *what
survives empirically*: "the closed form's SLACK is largest for the archive ALONE (+0.0634)
precisely because it is the only asymmetrically restricted pool." **The slack is that identity's
second-order term.** With `q = u_A/u_B` and `k = sd(C)/sd(D)`, exactly, for any pool:

```
r_Pearson = [(k² − 1)/(k² + 1)] · (1 + q²)/(2q)
SLACK     = [(k² − 1)/(k² + 1)] · ((1 + q²)/(2q) − 1)
```

Evaluated against all **13** published `A1_algebra` rows (see **T1**): max |predicted r − measured
r| = **4.441e-16**, max |predicted slack − reported slack| = **4.725e-16**. The `+0.0634` is
`(1+q²)/(2q) − 1` at `q = 0.249` times `(k²−1)/(k²+1)` — a deterministic function of two numbers
already in the table. **Trap 11/30 recurred one level up, inside the retraction of trap 11/30.**

*Why this changed my design, not just my report:* the second moments carry exactly **one**
empirical degree of freedom — `(k, q, scale) ↔ (r, u_A, u_B)` is 3 numbers ↔ 3 numbers. So a
Pearson-channel test is near-vacuous once `u_A` and `u_B` are matched, and the probe must live in
channels the identity does **not** determine: **Spearman** (rank structure) and the
**within-instrument** reliability. Both are reported for every cell.

### 1.2 🔴 "The archive is the ONLY asymmetric pool (~1.0 for every constructed pool)" is false

True only of `final.py`'s `boxmatch`/`curve` cells. `matched.py` had **already built eleven**
asymmetric random-lineage cells. Two straddle the archive and were never read as bearing on
necessity:

| already-run cell | u_A | u_B | q | cross rho |
|---|---|---|---|---|
| `jointband-match-archive-mid` | 0.0427 | 0.1559 | **0.2737** | **+0.0537** |
| `bandrandom-A-sd10` | 0.0999 | 0.4119 | **0.2425** | **+0.2205** |
| `archive-x400` (reference) | 0.0399 | 0.1605 | 0.2488 | +0.2184 |

`jointband-match-archive-mid` is a *good* two-sided match. The ledger's dismissal of "jointband"
quoted the `-full` cell (`u_A = 0.2205`, "5.2× the archive's") and generalized it to the
well-matched `-mid` cell. My new result agrees with `-mid` (+0.0537 vs my +0.0131) and shows
`bandrandom-A-sd10`'s +0.2205 is the coincidence: it matches the *ratio* at ~2.5× the *levels*.

### 1.3 🟡 The instruction to quote the "paired resolution ~0.17–0.24" is a units error — I did not follow it

The brief says to state the paired resolution `~0.17–0.24`, and *also* that a floor may be quoted
only if the quadruple (pool × replicate-structure × scale × statistic) matches. Those conflict.
Provenance check: `PREREGISTRATIONS.md:7983` ("paired resolution of **0.2222** (n=8 near-optimal
pool)"), `:7534` ("0.1688 / 0.1723 / 0.2400"), `:7279` ("resolves sub-**0.72 ms/char**") — all are
**ms/char floors on layout TIME**. My statistic is a **correlation**. The quadruple fails on its
fourth element, so importing that floor would be a units error dressed as rigour.

**Derived floor used instead:** the **replicate sd of rho across R = 12 independent construction
seeds** (**0.0357** asym, **0.0447** sym) plus the within-draw bootstrap CI at 8000 resamples. No
difference is called real unless it clears the replicate sd.

---

## 2. What I built, and that the construction actually worked

A **two-parameter** restriction, because the restriction has two sides — the exact defect that
made P4 unreadable ("one `u` for a two-sided restriction ⇒ its null means *no effect detected*,
never evidence *for* the null"). Selection is a 2-D box on `(y_A, y_B)`, both half-widths driven
by feedback on the **achieved** sd of the selected set, under-relaxed at exponent 0.7.

**F5, construction validity, gates everything and it PASSED:** achieved `u_A = 0.03981`,
`u_B = 0.16055`, `q = 0.24797 ± 0.0048` against the archive's `0.2488` — worst relative miss
**1.91% on `u_A`** and **1.92% on `u_B`**, over all 12 asym cells. All values reported are
**ACHIEVED**, measured from each pool's own scores; requested values are never quoted as results.

> **This also corrects the prior round's stated limit.** Its 2-D match "failed, landing at C =
> 0.553 when aiming for 0.085" because it fed back on box *geometry*. Feeding back on the
> *achieved statistic* converges in ~11 iterations. **The cell was always reachable; the earlier
> miss was a construction bug, not a structural limit.**

**Positive control before anything else (T0):** my from-scratch pipeline reproduces the published
cells at `|Δ| = 0.00e+00` on both rho anchors and `≤ 3.4e-05` on the within-seed means, archive
bank 2860, reference-bank moments identical to 6 dp. Note `evidence_scorer` is **not on main** —
it exists only on unmerged branches — so I re-implemented the two functions needed on top of
`surfaces.py`, which is **bit-identical** between `main` and `poolsweep`. Nothing was cherry-picked.

**Resolution-floor quadruple, matched to the archive:** pool size **400** ✓ · replicate structure
(same reference bank, same bootstrap protocol at 8000, same 3-seed within-instrument channel) ✓ ·
scale (`.native` frame asserted at load, ms/trigram, same reference moments) ✓ · statistic
(Spearman on ms/trigram) ✓. All four match, so the derived floor is quotable.

**Stated identification limit I cannot close:** **LEVEL is structurally unmatchable.** Archive
`mean_A = 254.83` vs random-bank `mean_A = 277.50` ms/trigram; no random permutation reaches
Pareto speed, so every random-lineage cell is centred on the random median (trap 16). I match
spread, size, replicate structure, scale and statistic; I do not match level, and say so.

---

## 3. The result

### 3.1 The primary cell — the archive is NOT reproduced (T2, T3)

| pool | ACHIEVED u_A | ACHIEVED u_B | ACHIEVED q | **cross** (inst-vs-inst) | **within** (inst-vs-itself) |
|---|---|---|---|---|---|
| `random-wide` | 0.9693 | 0.9711 | 0.9982 | **+0.7970** | **+0.9872** |
| `archive-x400` | 0.0399 | 0.1605 | 0.2488 | **+0.2184** | **+0.9647** |
| **`asym-match`** (R=12) | 0.0398 | 0.1606 | **0.2480** ± 0.0048 | **+0.0131** ± 0.0357 | **+0.6955** ± 0.0146 |
| `sym-match` (R=12) | 0.0800 | 0.0800 | 0.9997 | +0.0194 ± 0.0447 | +0.2345 ± 0.0239 |

- **`archive − asym` on cross = +0.2303, CI [+0.0865, +0.3742], p = 0.0015** (unpaired; disjoint
  layout universes and different lineage, so labelled unpaired). The archive sits **5.76
  asym-replicate-sds** above the asym mean on the cross leg and **18.50** on the within leg.
- Compare the prior round's properly matched cell: **+0.1106 [−0.0185, +0.2400], p = 0.098**. On
  the cell that also matches the **asymmetry**, the difference is **larger** and the CI **excludes
  zero**. Necessity for the cross leg moves from *not excluded* to *positively supported*.

### 3.2 🟢 The new finding: asymmetry is a WITHIN-instrument variable, not a cross-instrument one

The `asym` vs `sym` contrast is **paired** — same bank, same construction seed, identical
geometric-mean narrowness `u_geo = 0.0800`, asymmetry the only difference. Two tests, opposite
outcomes:

| paired test (R=12) | mean Δrho | replicate sd of Δ | Wilcoxon p |
|---|---|---|---|
| `asym − sym`, **CROSS** leg | **−0.0063** | 0.0607 | **0.7334** |
| `asym − sym`, **WITHIN** leg | **+0.4611** | 0.0257 | **0.00049** |

At fixed overall narrowness, restriction asymmetry **does not move cross-source agreement at all**
and **moves within-instrument reliability enormously**. The q-ladder (T4) shows why: within
tracks `u_B` monotonically across the full sweep (+0.9161 at `u_B` 0.3228 down to −0.4004 at
`u_B` 0.0203) while cross stays flat near zero throughout. **This is invisible to any single-`u`
treatment** — it is exactly the information P4 discarded when it fed `√(u_A u_B)`.

### 3.3 F4 does not survive replication — cross-rho is symmetric in log q (T7)

The single-draw ladder appeared to trigger F4 (|cross(1/4) − cross(4)| = 0.2630 > 0.20). Replicated
at 6 seeds per q it collapses: **0.0132** (q = 1/16 vs 16, MW p = 0.5887) and **0.0582** (q = 1/4
vs 4, MW p = 0.0649), against a pooled replicate sd of 0.042–0.056. The original 0.2630 was
construction noise — single ladder draws scatter by more than the effect (`ladder-q1` landed at
−0.1006 against a 12-seed sym mean of +0.0194). **So for the cross leg, asymmetry is a magnitude,
not a direction: which source is squeezed does not matter.** My §4.3 survives.

### 3.4 🔴 F2 TRIGGERED and REPLICATED — my own §4.2 is refuted as stated (T5, T8)

I predicted a random pool could not produce the archive's full two-legged signature. It can:

| arm (R=6) | u_B | u_seed_geo | cross | within |
|---|---|---|---|---|
| `c3-asym4x` | 0.5972 | 0.6012 | **+0.2457** ± 0.0311 | **+0.9660** ± 0.0028 |
| `c3-sym4x` | 0.3184 | 0.3283 | +0.3155 ± 0.0592 | +0.9103 ± 0.0047 |
| `archive-x400` | 0.1605 | 0.1617 | +0.2184 | +0.9647 |

A **random** pool reproduces both legs — within +0.9660 vs the archive's +0.9647, cross +0.2457
whose ±2 replicate-sd interval contains +0.2184 — **but only at 3.7× the archive's `u_B`.** So the
two-legged signature is **not unique to near-optimality**. What near-optimality does is produce it
**at the archive's narrowness**, where no random pool can (at matched `u_B` the random arm gives
+0.6955). My §4.2 is true at matched `u` and false at 4×`u`; I state it as refuted-as-written.

### 3.5 The within leg is not a spread artifact — and the quantitative bound (T6, T9)

Within-reliability is a correlation among COMMUNITY's **per-seed** refits, so it is attenuated by
per-seed spread, **not** by `u_B` (a seedmean quantity). I therefore built an arm matched on the
archive's own `u_seed_geo = 0.1617` rather than its `u_B`.

⚠️ **The confirmatory pass's bisection for this was mis-bracketed and I fixed it rather than
quoting it.** It searched `u_B ∈ [archive_u_B, 20×]`, but a random pool at `u_B = 0.1605` already
lands at `u_seed_geo = 0.1815` — *above* target — so the root lies *below* the bracket and the
search plateaued at a 12% miss. Reopened downward (B1, bracket verified): the root is
`u_B = 0.1379`, achieving `u_seed_geo = 0.1613` (miss −1.5%).

| arm | ACHIEVED u_B | ACHIEVED u_seed_geo | within | cross |
|---|---|---|---|---|
| `archive-x400` | 0.1605 | 0.1617 | **+0.9647** | **+0.2184** |
| B1 `u_seed`-matched random (R=4) | 0.1382 | 0.1613 | +0.6259 ± 0.0174 | +0.0645 ± 0.0201 |
| B2 `within`-matched random (R=4) | **0.5991** | 0.6035 | +0.9664 ± 0.0017 | +0.2916 ± 0.0234 |

- **B1:** at genuinely matched per-seed spread the archive leads by **+0.3388** on within and
  **+0.1540** on cross. (The uncorrected arm had *more* per-seed spread than the archive — hence
  *less* attenuation, favouring it — and still lost, so the correction only strengthens this.)
- **B2 — the bound the brief asked for:** a random pool at the archive's asymmetry needs
  `u_B = 0.5991` = **3.73× the archive's 0.1605** before COMMUNITY's own refits agree as well as
  they already do on the archive. **And its cross-source rho there is +0.2916, not the archive's
  +0.2184** — the two legs do not move together, which is the dissociation restated as a bound.

---

## 4. Scorecard against my own preregistration (`eb2fb04`)

| claim | prediction | outcome |
|---|---|---|
| §4.1 asymmetric restriction largely **sufficient** for the cross leg | rho ∈ [0.00, +0.16], difference CI includes 0 | 🔴 **REFUTED** — rho +0.0131 (in range) but CI **excludes** 0, p = 0.0015 |
| §4.2 asymmetric restriction **cannot** reproduce the within leg | < +0.80 | 🟠 **SPLIT** — holds at matched `u` (+0.6955); **refuted at 4×`u`** (+0.9660) |
| §4.3 ladder symmetric in log q | \|Δ\| < 0.15 | 🟢 **CONFIRMED** on replication (0.0132 / 0.0582) |
| F5 construction validity ≤ 10% both sides | — | 🟢 **PASS** (1.92%) |
| F1 asym ≥ +0.30 | — | not triggered (+0.0131) |
| F2 within ≥ +0.90 **and** cross ≤ +0.30 | — | 🔴 **TRIGGERED at 4×`u`**, not at matched `u` |
| F3 asym ≤ −0.10 | — | not triggered |
| F4 ladder asymmetric in log q | — | 🟢 not triggered (single-draw trigger was noise) |

**Two of my three substantive predictions failed.** The verdict below is the one the data
supports, not the one I registered.

---

## 5. Verdict

**Near-optimality IS necessary for the instrument disagreement as the campaign states it —
"the archive's cross-source rho collapses to +0.2184 while the instrument agrees with itself at
+0.9647, at the archive's restriction." Asymmetric restriction alone does not reproduce it.**

Three legs, three different answers, and the distinction matters:

1. **At matched restriction (the honest comparison), asymmetric restriction is NOT sufficient for
   either leg.** cross +0.0131 vs +0.2184 (p = 0.0015); within +0.6955 vs +0.9647 (+0.2723 gap,
   surviving a per-seed-matched control at +0.3388). This *upgrades* the prior round's "a ~0.1-in-rho
   near-optimality contribution is not excluded" to a measured **+0.2303 [+0.0865, +0.3742]**.
2. **Asymmetry is not the mechanism for the cross leg at all.** Paired, at fixed narrowness:
   p = 0.7334 on cross, p = 0.00049 on within. Asymmetry is a **within-instrument** variable. The
   ledger's reading of `u_A/u_B = 0.249` as the archive's cross-source signature is not supported —
   and the "SLACK" that was offered as its evidence is algebra (§1.1).
3. **The two-legged signature is not unique to near-optimality, only unreachable at the archive's
   narrowness by anything else.** A random pool reproduces it at 3.7× the archive's `u_B`. So the
   sharp claim that survives is *conditional on narrowness*, and it should be stated that way
   rather than as an existence claim about near-optimal pools.

**What this does NOT establish.** LEVEL remains unmatched (trap 16) — the archive is 22.7
ms/trigram faster than any random pool, so "near-optimality" here is confounded with "unreachably
fast", and a lineage-free pool *at the archive's speed* does not exist to be built. The within
leg rests on COMMUNITY's 3 per-seed refits; **AALTO ships none** (verified: `AALTO*seed*.npy` →
NONE), so there is no second independent pair and the two-source limit is structural, exactly as
the ledger says. All numbers are MODELLED ONLY on the `.native` frame, blend-v1, seed 0.

---

## 6. Artifacts

Committed under `agent-artifacts/necessity/` on branch `necessity-pool`:

| file | what |
|---|---|
| `PREREGISTRATION-necessity-pool.md` | prediction + 5 falsifiers, committed `eb2fb04` before any score |
| `nplib.py` | self-contained primitives (surface load, ms/trigram, both agreement channels) |
| `control.py` → `out/control.json` | positive control vs published cells (4/4 PASS) |
| `asym.py` → `out/asym-blend-seed0.json` | primary probe, 41 cells, R=12 paired asym/sym + q-ladder + level ladder |
| `confirm.py` → `out/confirm-blend-seed0.json` | C1 within-leg fairness, C2 replicated ladder (F4), C3 replicated 4× (F2), 56 cells |
| `bound.py` → `out/bound-blend-seed0.json` | B1 corrected `u_seed` match, B2 the 3.73× bound |
| `tables.py` → `out/TABLES.md` | every table above, generated from the JSON |
