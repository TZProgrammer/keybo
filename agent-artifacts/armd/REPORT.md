# ARM D — the evidence-weight search with `valid_domain` as a HARD CONSTRAINT

**Outcome: (iii), in its strongest form — but by a mechanism nobody predicted.**
Arm D's champion is **269.2762 ms/char: slower than qwerty.**

Corpus **blend-v1 (production default)**, `sha256(trigrams.txt) = 19806532ee3567f5…`, frame
**`.native`** (asserted from the weights JSON), **90 WPM**. Weights
`state/evidence-scorer/artifacts/arm-random400-native.json` (`COMMUNITY_BASE`, pool
`random-c30m-400`, n=400). **MODELLED ONLY** — every number is attribution of a *fitted* timing
surface, not measured typing. tau saturated, Phase-D cancelled. **No layout here is promoted or
adopted; that is the user's gate alone.** 🟢 = verified, 🟡 = read from source, 🟠 = inferred.

---

## 1. The board

| layout | ev CLAMP | ev EXTRAP | **ms/char** | vs best inc | normfloor | n_ood |
|---|---|---|---|---|---|---|
| **arm D** (domain-clamped) `jigoqhxvpmedayuctslrk'n-fzw.b,` | **−23.3157** | −23.5812 | **269.2762** | **+14.6455** | **−0.563179** | 3/14 |
| arm A (extrapolating evidence) | −18.6413 | **−45.4363** | 256.8466 | +2.2158 | +0.583619 | 10/14 |
| arm C (sign-constrained) | −18.5202 | −45.0664 | 256.0220 | +1.3912 | +0.568914 | 10/14 |
| arm B (baseline served) | −18.5790 | −41.3178 | **253.9006** | −0.7302 | +0.600450 | 10/14 |
| keybo-lsb *(best incumbent)* | −17.8939 | −37.4618 | 254.6307 | — | +0.726952 | 9/14 |
| keybo-lsb+lm | −17.8939 | −38.3090 | 254.6847 | +0.0539 | +0.730176 | 9/14 |
| lsb-sib | −18.4986 | −37.5827 | 254.7058 | +0.0750 | +0.739914 | 9/14 |
| archive-1846 | −18.3966 | −39.1339 | 254.7961 | +0.1653 | +0.745186 | 9/14 |
| archive-1843 | −18.3161 | −38.9624 | 254.8436 | +0.2129 | +0.744198 | 9/14 |
| flagship-c3 | −18.5402 | −38.9980 | 254.9761 | +0.3454 | +0.745096 | 9/14 |
| **qwerty30m** | −6.9764 | −6.9764 | **264.1389** | +9.5082 | 0.000000 | 0/14 |

🟢 All ms/char verified twice: through the fast evaluator and independently through the shipped
`keybo analyze --json` (269.2762 both ways, exact). ⚠ **The two evidence columns are different
rulers** — arm D optimized CLAMP, arms A/C optimized EXTRAP. Quoting one column across arms is the
error the parent's `domain_policy` field exists to prevent.

**Arm D is slower than qwerty by +5.1373 ms/char.** It is the worst layout on the board.

### Does the gap resolve? Yes, by 25–31×.
PAIRED resolution over the **named n=10 near-optimal pool** (4 champions + 6 incumbents,
excluding qwerty/graphite/semimak): median **0.1353**, conservative max **0.4964** ms/char.
Unpaired 0.4628 is the wrong ruler (trap 37). SS: layout 99.57%, seed 0.36%, residual 0.07%.

| pair | Δ ms/char | × conservative floor | resolves |
|---|---|---|---|
| arm D vs arm B | **+15.3756** | **30.98×** | 🟢 yes |
| arm D vs keybo-lsb | +14.6455 | 29.50× | 🟢 yes |
| arm D vs arm A | +12.4296 | 25.04× | 🟢 yes |
| arm D vs arm C | +13.2543 | 26.70× | 🟢 yes |
| arm D vs flagship-c3 | +14.3001 | 28.81× | 🟢 yes |

All 9 arm-D pairs resolve (also vs lsb-sib 29.35×, archive-1843 29.08×, archive-1846 29.17×,
keybo-lsb+lm 29.40×). This is not search noise: arm A's *entire* deficit was 2.9460 and
OPTEVIDENCE-1 measured the evidence-arm search-noise sd at 0.3440 — arm D's excess over arm A alone
is 36× that sd.

⚠ **On the brief's paired figure.** The brief specifies 0.2222 (n=8). My pool is n=10 (I added
`flagship-c3` and arm D itself), giving 0.4964. I used **my** larger, more conservative floor. Every
verdict here holds under either. The brief's "seed main effect is 78–83% of SS" is FLAGSHIP-1's iWeb
number, not this artifact's — here the seed is **0.36%**, because the pool spans 15 ms/char. Pairing
is still the correct instrument; it is just not doing much work when the layout effect is this large.

---

## 2. How many gauges is arm D out-of-domain on? **3 of 14** — and that is the surprise

Arm A: 10/14. Arm D: **3/14** (`comfort` 6.5110 vs floor 6.5236; `sfs` 6.6335 vs floor 6.7450;
`sfs-dist` 8.7386 vs floor 9.3687 — all three *marginally* below a floor). The brief anticipated
that a still-far-out-of-domain champion would mean broken wiring. The opposite happened, and the
wiring is verified sound:

🟢 **The clamp binds, exactly.** On arm D's own champion, through the same `ClampedEval` the search
used: pushing **any** of the 14 gauges 50 domain-widths past either edge changes the total by
**exactly 0.000e+00**. `all_bind = True`, worst |reward outside| = **0.000e+00**. This was my
pre-registered abort condition (P6) and it passed on all 14.

**But 11 of 14 gauges are strictly INSIDE their domains.** That is the important number, and it
falsifies the shared premise of my prediction *and* the sibling's warning.

---

## 3. 🔑 The mechanism: the clamp **relocated** the optimum, it did not flatten the objective

Everyone — my PREDICTION.md P11/P14, and the sibling's warning 2 — expected clamping to remove the
gradient outside the domains, leaving a flat objective and tie plateaus, with the champion drawn
arbitrarily from one. **That is not what happened.**

🟢 **Plateau census over the entire final population** (40 islands × 64 = 2560 slots, epoch 50):

| quantity | value |
|---|---|
| distinct layouts | 1730 |
| **distinct objective values** | **1730** |
| distinct values per distinct layout | **1.0000** |
| plateaus (≥2 layouts sharing a value to 12 s.f.) | **0** |
| champion's exact ties | **0** |

The clamped objective distinguishes **every single layout** in the final population. There is no
degeneracy at all. The search was **well-conditioned and confident**, and it converged to a layout
slower than qwerty.

**Why: because the clamped optimum is an interior point, not a boundary point.** My pre-run
reasoning (P5) was that 8 of 14 curves are minimized *at* a domain edge, so the clamped optimum
would sit on the boundary where the objective is flat. Wrong. Clamping changes *where the joint
optimum is*: with the out-of-domain rewards removed, the best reachable trade-off moved **into the
fitted interior**, where 11 of 14 curves still have live gradients steering the search.

🟢 **And it steered exactly where my pre-run headroom analysis said it would.** Before the run I
measured that 92.5% of the clamped headroom remaining from arm A's champion (6.8331 of 7.3897 units)
was *mechanism-WRONG* — collectable only by moving a gauge in the direction that makes a layout
slower. **All five of those gauges moved in the predicted direction:**

| gauge | arm A | arm D | Δ | predicted direction | mechanism |
|---|---|---|---|---|---|
| `oxey-style` | −12.4932 | 107.8723 | **+120.3655** | up | WRONG |
| `sfb-dist` | 1.6216 | 18.0999 | **+16.4783** | up | WRONG |
| `sfb` | 1.4093 | 12.2615 | **+10.8522** | up | WRONG |
| `scissor` | 0.1753 | 3.2447 | **+3.0694** | up | WRONG |
| `lsb-dist` | 2.2837 | 2.8836 | **+0.5999** | up | WRONG |

`sfb` went from 1.41% to **12.26%** same-finger bigrams and `scissor` from 0.18 to **3.24**. Those
are catastrophic in mechanism terms, and the clamped objective *paid* for them, because the fitted
curves for those gauges slope the wrong way inside their own fitted domains.

**So the pathology is not extrapolation, and not flatness. It is that the curves are mis-specified
in the region where they are supported.** Clamping is a faithful, correct implementation of "stay
inside the data" — and it makes the outcome *worse*, because the objective's interior is where the
sign errors live. This is the strongest possible form of (iii): the fitted curves carry almost no
*valid* in-domain signal about speed, and bounding them does not rescue that.

---

## 4. Which outcome? **(iii)**, and it also settles (ii)

- **(i) — arm D lands near the incumbents ⇒ extrapolation was the whole problem: 🔴 REFUTED.**
  +14.6455 ms/char from the best incumbent, 29.50× the resolution. Unbounded extrapolation was *a*
  defect and the clamp does remove it, but removing it made the search **worse, not better**.
- **(ii) — still well behind arm B ⇒ weights uninformative in the near-optimal band: 🟢 CONFIRMED,**
  and on much stronger evidence than arm C provided. The sibling's pre-registered rule (refutes
  ≤254.85, confirms ≥~255.5) resolves to **CONFIRMS** at 269.2762, far outside its ambiguous zone.
- **(iii) — the clamped objective carries almost no in-domain signal: 🟢 CONFIRMED, and it is the
  operative mechanism** — but *not* via "the objective is flat so the search wanders". Via
  "the objective is sharp, well-conditioned, and points somewhere bad." **A flat objective would
  have been the milder diagnosis:** a wandering search returns something mediocre, whereas a
  confident search optimizing a mis-signed interior returns something worse than qwerty.

**Direct answer to the user's question** ("what happens if we optimize with the evidence agent's
weights?"): with the domain enforced as the brief demands, you get a layout **slower than qwerty**.
Arm A's 256.85 was *flattered* by extrapolation — the unbounded objective happened to push two
correctly-signed gauges (`comfort`, `sr-roll`) into regions that correlate with real speed, which
partly masked the interior sign errors. Take the extrapolation away and the weights' true in-band
behaviour is exposed. 🟢

### Independent in-band rank test (my own pool, reproducing the sibling's shape)
3600 random 1–4-swap perturbations of the six incumbents — a pool selected by **neither** objective.
Instrument positive control ρ = **1.0000**.

| band | n | ρ raw | ρ **CLAMPED** |
|---|---|---|---|
| all | 3600 | +0.9017 | +0.5586 |
| ≤257.0 | 1010 | +0.5966 | +0.1237 |
| ≤256.0 | 590 | +0.3184 | +0.0416 |
| ≤255.5 | 305 | +0.1293 | **−0.0692** |
| ≤255.0 | 104 | +0.0809 | +0.0274 |

🟢 Reproduces the sibling's monotone decay on an independently constructed pool: pool-wide the
weights look informative (+0.56), and in the band a search actually operates in they are
**indistinguishable from zero or negative**. Clamping *lowers* ρ everywhere — it is not a fix.
(My ≤255.0 cell is +0.0274 where theirs was −0.0884; at n=104 both are ~0. I do not claim the sign
in that cell.)

---

## 5. Optimizing the ruler, and the normalized floor

🟢 **The cleanest optimizing-the-ruler demonstration in the campaign.** Arm D **wins** the clamped
evidence ruler it trained on by **4.67 units** over arm A (−23.3157 vs −18.6413) while being
**12.43 ms/char slower**. Winning its own objective and losing reality, in one pair.

On the 19-gauge frame it wins only **1–2 of 18** scored gauges against each incumbent (loses 16–17;
`sfr` excluded as a permutation invariant, trap 23). Its only wins are `sfs` and `sfs-dist` — and
those are precisely two of the three gauges it is *below the fitted floor* on. ⚠ Effective dof over
19 gauges is ~4–5, so raw win counts over-count independent evidence ~4× (trap 39); the direction
here is unambiguous regardless.

⚠ **P10 FAILED — the normalized floor is NEGATIVE: −0.563179.** Arms A/B/C were all positive
(+0.5836/+0.6005/+0.5689), which the brief flagged as falsifying the WSCISSOR-GEN-1 precedent. Arm D
**restores** that precedent: mean saved is **−1.6931%** (it *loses* to qwerty on the six-surface
frame). Ceiling re-derivation passed its frozen-iWeb positive control (worst diff 4.44e−14). 🟢

**Is the champion comfort-driven?** 🟢 Yes, as the brief anticipated. `comfort` carries **43.55%** of
the fitted attribution and is a hand-chosen taste table (`DEFAULT_COMFORT`, no fitted parameter —
trap 48), and arm D's champion sits at **6.5110**, pinned essentially exactly at its clamped floor of
6.5236 (P7 ✓). So the single largest term in this "evidence-based" objective is a rival's taste
table, saturated at its boundary, contributing nothing to discrimination among near-optimal layouts.

## 6. Admissibility (10-axis dominance frame, with the strict-win term)

⚠ **Frame correction (trap 20):** the brief says "12-axis"; `judgement.json.dominance_frame` has
**10** (`floor mean wfd genkey oxey1 oxey2 lsb scissor sfb sfs`), predicate `n_ge == 10 AND
n_strict ≥ 1` (trap 33). The brief's own quoted 3/10, 3/10, 1/10 match the 10-axis artifact, so this
is a transcription slip and the artifact wins. I used 10 axes.

| champion | dominator exists | best n_ge |
|---|---|---|
| **arm D** | **no** | **1 / 10** |
| arm A | no | 3 / 10 |
| arm C | no | 3 / 10 |
| arm B | no | 1 / 10 |

🟢 Arm D is **not admissible**, and is the *weakest* champion on the frame (P9 ✓). No arm produced a
dominator — unchanged.

---

## 7. Prediction scorecard — 11 of 16 correct, **5 failed, all reported**

Registered in `artifacts/PREDICTION.md` before the run (P1–P12) and mid-run before any ms/char was
computed (P13–P16, against the sibling's decision rule).

**Correct (11):** P1 (worse than arm B by >0.2222 ✓ +15.3756) · P4-direction · P6 (clamp binds, all
14, 0.000e+00) · P7 (comfort ≤6.5236 ✓ 6.5110) · P8 (ev_clamp in [−24.5,−21.0] ✓ −23.3157) · P9
(0 dominators, n_ge 1≤4) · P12 · P13 (in-band ρ ≤0) · P15 (wins clamped ruler, loses speed) · P16
(worse than the midpoint) · plus the headline **(iii)**.

**Failed (5), each informative:**
1. **P3 FAIL — I predicted arm D would beat arm A.** It is **12.43 ms/char worse**. I reasoned that
   removing a measured defect must help; it hurt. My own P16 argued the opposite (a search
   maximizing an anti-correlated ruler weakly selects for slowness) and P16 was the right one. I
   registered both and the pessimistic one won.
2. **P2 FAIL — predicted [254.5, 257.5]; actual 269.2762.** My mid-run addendum's 255.2–255.4 and
   the sibling's independent 255.3–256.3 were **both wrong by ~14 ms/char**. Two independent
   estimates agreeing is *not* evidence of correctness — we shared the false premise that a bounded
   objective must land in the band its data came from.
3. **P4 FAIL as stated — recovery is −421.9%, not >28%.** The clamp did not partially recover arm
   A's deficit; it multiplied it 5.2×. Arm C's 28% remains the only positive recovery.
4. **P5 FAIL — predicted n_ood ≥6; actual 3.** My boundary-optimum reasoning was backwards: the
   clamped optimum moved *into* the interior, not onto the edges.
5. **P11/P14 FAIL — predicted a plateau; there are zero.** The most load-bearing failure, because it
   **relocates the pathology** (§3). It also refutes the sibling's warning 2, which I had adopted.

---

## 8. What this changes, and what it does not

- 🟢 **Enforcing `valid_domain` is still correct** — an unbounded fitted objective is indefensible,
  and gate 1 confirms the clamp is sound and perturbs no supported level (587 in-domain levels
  bit-identical across all three policies). But **it is not a fix for a mis-specified curve set.**
  Bounding a wrong objective makes it *honestly* wrong. The parent's `SEARCH_DOMAIN_POLICY` should
  ship; nobody should expect it to make these weights usable.
- 🟢 **"The weights are uninformative in the near-optimal band" is CONFIRMED and needs no
  softening** — it needs *strengthening*. The brief noted the sibling retracted arm C as the
  warrant (arm C left extrapolation free, so its residual was circular). Arm D is the clean test,
  and it lands harder: the weights are not merely uninformative in-band, they are **actively
  anti-informative**, and the interior sign errors are the cause.
- 🟠 **A caution on the whole approach, inferred:** the two gauges arm A exploited by extrapolating
  (`comfort`, `sr-roll`) were *correctly* signed, and pushing them out-of-domain correlated with
  genuine speed. The unbounded objective was accidentally less bad than the bounded one. That is a
  warning about SHAP-fitted-curve objectives generally: per-gauge attribution from a collinear
  surrogate (VIF 12.8–119, trap 49) is uninterpretable per-gauge, so *any* separable sum of such
  curves can be exploited in whichever direction happens to be locally cheap.
- **Nothing here promotes or adopts any layout.** Arm D's champion is a diagnostic object.

## 9. Follow-up (from the sibling; noted, deliberately NOT run)

Under **archive**-fitted weights (`arm-archive400-native.json`), `keybo-lsb` is reportedly
out-of-domain on **0 of 14** gauges vs 9/14 under `random400` — the pool that failed as a *scorer*
has domains that actually **cover** the near-optimal band, and has never been tried as a *search*
objective. Given arm D's result that is the natural **arm E**, and better-posed than arm D: a clamp
only bounds a bad domain, whereas archive-fitted curves might not need clamping in-band at all.
🔴 I have verified none of its numbers. It needs its own pre-registration; my brief scopes me to arm D.

---

## Run integrity (all pre-registered abort conditions met)

| condition | result |
|---|---|
| Gate 1 (policy sound on the real fitted curves) | 🟢 1301 checks, 0 failures |
| Gate 2 (engine is arm A's; clamp wired in; resume bit-exact) | 🟢 33 checks, 0 failures |
| P6 (clamp binds on the champion) | 🟢 all 14 gauges, worst 0.000e+00 |
| ≥9.0M unique evals | 🟢 **10,099,380** (arm A: 9,434,590) |
| gauge computation bit-identical to arm A | 🟢 same `FastEval`; `ClampedCurve == evobj.Curve` in-domain |
| same seed / islands / overshoot / ga-share / polish-sweeps | 🟢 20260728 / 40 / 1.95 / 0.6 / 40 |
| per-epoch checkpointing (trap 7) | 🟢 50 epochs, budget reached early |
| `analyze` set-containment (trap 38, corrected) | 🟢 13 rows + 1 `--ref` extra, none dropped |
| six-surface ceiling positive control | 🟢 PASS (worst diff 4.44e−14) |

## Artifacts (all under `state/armd/artifacts/`)

| file | what |
|---|---|
| `PREDICTION.md` | pre-registration (P1–P12) + mid-run addendum (P13–P16, blind to ms/char) |
| `judgement.json` | full judgement: 13 layouts, 19-gauge frame, paired, dominance, clamp-binding, plateau census, in-band rank test |
| `runs/arm-domain.json` + `.ckpt.json` + `.keys.npy` + `.log` | the run, per-epoch checkpointed |
| `gate1-verify-policy.log` (rc=0) | 1301 checks on the **real** fitted curves |
| `gate2-engine.log` (rc=0) | positive control vs the frozen engine, clamp live, resume bit-exact |
| `pre-run-analysis.json`, `headroom.json` | the pre-run analysis the prediction rests on |
| `report.log`, `armd-rc.txt` (=0) | judge output + the run's rc sentinel |
| `drivers/` | `armd_obj.py` (the clamped search path), `search_armd.py`, `judge_armd.py`, `report_armd.py`, `verify_policy.py`, `gate2_engine.py`, `armd_load.py`, `headroom.py`, `predict_clamped.py` |

Committed on branch `domain-hard` in `/tmp/domainfix` (`3a3df7f` = the parent's plumbing, committed
verbatim before I touched anything; `e0b7a1b` = arm D's drivers + gates). Not pushed; no CR.

---

## ⚠ One finding the parent needs before shipping the policy plumbing

**`LossCurve.price(policy=...)` does NOT clamp a search.** `evobj.Curve.price` is a hand-rolled
vectorized reimplementation of the same arithmetic that never calls `LossCurve` (trap 28's habitat),
so adding the policy to `LossCurve` leaves every optimizer extrapolating. I had to add the clamp on
the vectorized path (`drivers/armd_obj.py::ClampedCurve`) and pin it against
`LossCurve.price(..., policy=CLAMP)` at exact float equality. **Had I trusted "the branch already
has the code", arm D would silently have been arm A** — and it would have "passed" gate 1, because
gate 1 tests the curve, not the search. Suggested follow-up for whoever owns this: make the
vectorized path call the validated one, or make `LossCurve` expose a vectorized `price` so there is
only one implementation to get right.
