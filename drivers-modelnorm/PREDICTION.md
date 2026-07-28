# MODELNORM-1 — PRE-REGISTERED PREDICTIONS

Written and committed **BEFORE** any step-2 anchor search or step-4 blend search was run.
Priority is verifiable from the git history of branch `modelnorm` in `/tmp/modelnorm`: this
file's commit precedes the commit carrying `runs/`.

Everything is on **corpus blend-v1**, the **`.native`** surface frame, `TRI_PS_FREQ_PRIOR`
family, at a **BAKED 90 WPM**. MODELLED ONLY — tau saturated at 1.0, Phase-D cancelled;
nothing here is a claim about realized typing speed.

## What is already measured (NOT predictions — evidence in hand before writing this)

* 🟢 `std − nat` is exactly independent of the third slot (max variation 1.14e-13) and
  **EXACTLY 0.0 for AALTO** ⇒ `.standardized` substitutes AALTO's bigram tensor into
  COMMUNITY and POOL. Frame is therefore `.native`, asserted in code.
* 🟢 The "0" anchor at n=100 vs n=1000 moves **< 1 SE** of the n=100 estimate
  (AALTO −0.979 SE = −1.70 % of span; COMMUNITY +0.128 SE; POOL −0.287 SE).
* 🟢 Effective number of independent models on a homogeneous n=10 000 random pool:
  **participation ratio 1.1672 of 3**, PC1 = 92.34 % of variance, Kaiser count 1.
  ρ(A,C)=0.8310, ρ(A,P)=0.8729, ρ(C,P)=0.9502.
* 🟢 qwerty30m normalizes to **0.50–0.62**, not ~0 — it is at the 0.00–0.20 percentile of a
  1000-layout random pool (z = −2.5 … −3.1). The brief's trap-3 assertion "qwerty30m must be
  ~0" is FALSE and is not usable as a direction guard.
* 🟢 Frozen ms/char re-derived via the shipped `keybo analyze --json`: arm B 253.9006,
  keybo-lsb 254.6307, flagship-c3 254.9761, arm A 256.8466, qwerty30m 264.1389.

## Predictions

### On the anchors

**P1.** Each per-model search's "1" anchor will be **materially faster than the best of the 8
candidate layouts** on that model — i.e. the per-model optimum beats every human/campaign
layout on its own surface. Quantitatively: ≥ 0.5 % further below the random-pool mean than
the best candidate is. *Rationale: the candidates were optimized against the SERVED
(standardized) objective or against community gauges, not against these native surfaces.*

**P2.** The two independent seeds for each model's "1" anchor will agree to within **0.10 % of
that model's anchor span**. If they do not, the normalization is not stable and I will say so
rather than proceed. *This is the trap-1 stability test.*

**P3.** The per-model searches will NOT all converge equally well. I predict the **spread of
(seed-A vs seed-B) disagreement across the three models will differ by at least 2×** between
the best- and worst-converged model — meaning the scheme's own failure mode (a
less-completely-optimized model gets a compressed scale) is present at some measurable level
even at campaign budget.

**P4.** The user's-scheme "1" anchor will be **faster (lower ms) than the ceiling-fraction
"1"** (best-of-a-fixed-set) on all three models, by **more than 1 %** of the anchor span.
*This is the specific improvement the design claims; I predict it is real and quantifiable.*

### On whether normalizing changes any ranking (deliverable B)

**P5.** Per-model rankings of the 8 candidates will be **completely unchanged** by
normalization. *Normalization is a per-model affine map with a positive scale; it is
rank-preserving within a model by construction. Any change here would be a bug, and this
prediction is a check on my own implementation, not a discovery.*

**P6.** The **equal-weight blend ranking WILL differ** from the raw-mean-of-three-surfaces
ranking on at least one adjacent pair of the 8 candidates. *Rationale: the raw mean is
dominated by COMMUNITY's wider span; re-weighting to equal normalized leverage must move
something.* I predict **at least 1 and at most 3** adjacent transpositions.

**P7.** Every ranking change P6 finds will be **BELOW the paired resolution floor** computed
on my own pool — i.e. the blend re-orders layouts the instrument cannot separate. *I predict
this explicitly because it is the honest expected outcome, and because the campaign's field
span (0.3454 ms/char over six incumbents) is under half the unpaired floor.*

### On the blend search (deliverable C)

**P8.** The equal-weight normalized-blend champion will be **SLOWER than arm B (253.9006
ms/char)** on the shipped served metric. *Rationale: arm B was optimized directly for that
metric; the blend champion is optimized for a 3-surface native-frame blend. Predicting my own
arm loses is the base rate here — arms A, C, D, E all lost to arm B.*

**P9.** Quantitatively: the blend champion will land in **[254.0, 257.5] ms/char**, i.e.
worse than arm B but better than arm A (256.8466) — and specifically **better than qwerty30m
(264.1389)** by a wide margin. *The failure mode I am guarding against is arm D's (269.28,
slower than qwerty); I predict the normalized blend does NOT do that, because unlike the
fitted loss curves these surfaces have no out-of-domain region to exploit — a 31³ lookup
table is defined everywhere the search can go.*

**P10.** The blend champion WILL beat arm B on **at least one of the three native surfaces**
(most likely COMMUNITY, whose span is widest and whose served-metric alignment is weakest).

### On the preference sweep (deliverable D)

**P11.** The (1,0,0) / (0,1,0) / (0,0,1) champions will be **three distinct layouts**, and
each will be the best layout found for its own model.

**P12.** Despite an effective model count of only ~1.17, the solo champions will **NOT be
near-identical**: I predict the pairwise Hamming distance between any two solo champions is
**≥ 8 of 30 positions**. *Rationale: high correlation across a WIDE random pool does not
imply agreement in the NARROW near-optimal band (trap 52 is exactly this distinction).*

**P13.** The (1,1,1) champion will **not equal any solo champion**, and its normalized score
on each model will be **within 0.05 of the max over the three solo champions' scores on that
model** — i.e. the equal blend buys near-simultaneous near-optima rather than a compromise
that is mediocre everywhere.

**P14.** The uneven weighting (I will use (2,1,1), AALTO-preferred) will produce a champion
**strictly between** the (1,0,0) and (1,1,1) champions in normalized AALTO score. *This is
the property that makes the weight readable as a preference; if it fails, the weight is not
behaving as one.*

### On admissibility (deliverable E)

**P15.** The blend champion will **NOT be a 10-axis dominator** of any incumbent (with the
strict-win term correctly required). *No arm has produced one; best `n_ge` so far is 3/10.*
I predict `best_n_ge ≤ 4` and `dominator_exists = False`.

**P16.** The blend champion's **normalized floor will be POSITIVE** (like arm E's +0.3986,
unlike arm D's −0.5632), because P9 says it is faster than qwerty on all surfaces.

### On the design itself

**P17.** I predict I will find **at least one further defect in the user's design beyond the
qwerty-anchor error already found** — and that it will be about the "0" anchor being a
*random-layout* mean: because random layouts are ~2.5–3 sd worse than qwerty, the [0,1] scale
spends most of its range on a region **no candidate ever occupies**, compressing all real
candidates into a narrow band near the top. I predict all 8 candidates will fall within
**a 0.25-wide window** of the normalized scale on every model.

**P18.** Consequently the *normalized* differences between candidates will be numerically
small, and I predict the equal-weight blend spread over the 8 candidates is
**< 0.30 on the 0–1 scale**, which I will report as a limitation of the anchoring rather than
as a finding about the layouts.

## Scoring rule

Each prediction is scored HELD / FAILED / UNTESTABLE against the artifacts, and **every
failure is reported in the callback and the report**. Prior arms pre-registered 8–16
predictions and the failures were the most informative parts; I expect several of these to
fail.
