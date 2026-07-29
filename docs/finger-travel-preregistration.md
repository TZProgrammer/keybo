# Pre-registration — `finger-travel` and `off-home` (the FT round, 2026-07-28)

**Status: PRE-REGISTERED. Written and committed BEFORE any layout was measured on either
metric.** The two definitions below, the partition rule, the denominator, and the numbered
predictions are fixed here so that "which layout wins" cannot have influenced them. If a
prediction is wrong, the record of it being wrong stays.

Two metrics are registered, because the user asked two separate questions on the same day and
they are **different quantities**:

1. **`finger-travel`** — "finger travel percentage… a finger that moves around more will have a
   higher percentage… the sum of all finger travel percentages would be 100%".
2. **`off-home`** — "off-home pinky usage. Pinky being used a lot is mostly fine, as long as it
   stays on the home row."

They are shipped as separate columns and are never summed together.

---

## 0. What this is NOT

**Both metrics are GEOMETRIC DESCRIPTORS, not times and not comfort claims.** A distance in key
units is not a millisecond and not a strain. This is registered up front because the campaign
already has a failure of exactly that shape: `bad_scissor`'s measured **+0.41 ms [+0.23, +0.55]**
effect had **bigram FREQUENCY explaining more variance than any geometric axis**, and its per-finger
split had to be relabelled "where the mass sits" rather than "which finger is strained" because
the causal claim was not identified on the sample.

So: any tie to predicted time in §5 is a **correlational** claim over a layout pool, stated with
its frequency control, and it does not license a mechanism.

The one legitimate physical anchor is that both metrics are functions of **key positions and
corpus frequency only** — no fitted model, no slowness weight, no preference constant enters the
headline definition. That is deliberate (see §1.4).

---

## 1. `finger-travel` — the definition

### 1.1 The four candidate definitions, and which one is registered

"Finger travel" is underdetermined. The candidates:

| id | definition | what it measures |
|---|---|---|
| (a) **static displacement** | `sum_keys freq(key) * dist(key, home(finger))` | how far off home a finger's keys SIT, weighted by use |
| (b) **path length between consecutive presses** | for consecutive presses of the same finger, `dist(k1, k2)` | how far the finger MOVES between its own presses |
| (c) **return-to-home** | `2 * dist(key, home)` per press | a finger that homes between every keystroke |
| (d) **hybrid / lag-resolved path** | per press, `dist(previous position of THAT finger, this key)` where the previous position is resolved from the corpus when observable and is `home` otherwise | the finger's actual traced path, to the depth the corpus can resolve |

**REGISTERED HEADLINE: (d), named `travel` — the lag-resolved path.** Reasons, in order:

1. The user's words are "**a finger that moves around more**". That is a statement about MOTION
   BETWEEN presses, which (a) and (c) do not measure — (a) is a static property of where keys sit,
   and (c) is (a) times two (see the proof in §1.5, which makes (c) a rescaling of (a), not an
   independent third option). Only (b)/(d) can distinguish "this finger uses three off-home keys
   and shuttles between them" from "this finger uses one off-home key repeatedly".
2. (b) alone is incomplete: it prices same-finger repeats but says nothing about the vastly more
   common case where a finger's press is *not* immediately preceded by its own press. (d) closes
   that by supplying an explicit, stated model for the unobserved predecessor.
3. **(a) is retained and shipped as a SEPARATE column** (`static`), both because it is the
   existing `per_finger_dislocation` quantity and because it is the sensitivity check the design
   question demands. Both are reported for every layout, always.

### 1.2 (d) stated exactly

Let `f` be a finger, `home(f)` its home position (`_FINGER_HOME`, already in
`keybo/scoring/utilization.py`). For an ordered bigram `xy` of corpus frequency `w`:

* if `x` and `y` are pressed by the **same finger** `f` (this includes index columns 1&2 and the
  K31 pinky columns 5&6), charge `f` with `w * dist(pos(x), pos(y))` — the finger's **observed**
  motion, from where it demonstrably was to where it goes.
* if they are pressed by **different fingers**, charge the finger of `y`, `f_y`, with
  `w * dist(home(f_y), pos(y))` — the **return-model** term: absent evidence of where `f_y` last
  was, it is modelled at home.

Summed over all corpus bigrams. Formally, with `B` the layout-restricted bigram set:

```
travel(f) = sum over xy in B of  w(xy) * [ same_finger(x,y) ? dist(pos x, pos y)
                                                            : (finger(y)==f) * dist(home f, pos y) ]
```

**Why lag-1 and not deeper.** The repo ships bigram, trigram and 1-skip tables and **no raw
text** (verified: `data/corpus/**` contains only n-gram count files). A finger's true unbounded
path is therefore not computable from the shipped corpus at all — the honest maximum resolution
is lag-1 from bigrams, extensible to lag-2 via trigrams. **The headline is lag-1**; a lag-2
sensitivity check (using trigrams to catch `a?a` same-finger returns that bigrams cannot see) is
registered as a reported variant, not as the headline, so that the headline needs only the
bigram table every other bigram gauge here already uses.

**This is a MODEL, and the modelled part is named.** The `different finger -> from home` branch is
an assumption, not an observation. It is exactly the assumption (c) makes universally, and it is
why (d) degenerates gracefully: as same-finger mass goes to zero, (d) approaches
`sum freq * dist(key,home)` = **(a)**, differing from (a) only in that (a) weights by
*letter* frequency and (d) by *second-of-bigram* frequency. That degeneracy is a feature —
it means (d) is a strict refinement of the existing quantity rather than a rival to it — and
§4 registers a test that pins it.

### 1.3 The partition and the denominator

`travel_share(f) = 100 * travel(f) / sum over all fingers travel(f)`.

* **Exact partition over 8 fingers by construction**: every charged unit of mass is attributed to
  exactly one finger (the finger of the bigram's SECOND character, or the shared finger), so the
  8 shares sum to 100 identically, not approximately.
* **Denominator = the total travel of the same 8 fingers on the same layout-restricted bigram
  set.** It is a self-normalizing metric: the denominator is the metric's own total, not a corpus
  mass. That is what makes the shares sum to 100% as the user requires.
* **Trap #9 note.** Because the denominator is the numerator's own sum, the space-inclusion
  choice that moves every `bad_scissor` share by ~1.497x **cannot** move a `travel` share the
  same way — it is a ratio of two quantities that both change. It CAN still change the shares,
  because space-touching bigrams contribute travel asymmetrically. Registered decision: **space
  is EXCLUDED from the travel corpus, and the thumb is NOT a ninth partition cell** (§1.6).
* **The absolute total is reported alongside the shares, always** (`travel_total`, in
  key-units x corpus-frequency). Registered explicitly because normalizing destroys the level:
  two layouts can have identical shares and very different total travel, which is precisely the
  `saved_vs_ref_pct` coverage artifact this ledger already registered. **A shares table without
  the level is a misleading table.**

### 1.4 NOT slowness-scaled

The headline `travel` is **pure distance**: no `DEFAULT_SLOWNESS`, no finger-strength weight.
Mixing distance with a per-finger cost weight makes the result neither a travel measure nor a
time measure, and this campaign has a registered failure of that exact shape (a "gauge" that
turned out to be a re-weighted restatement of other legs; `oxey-style` at R2=0.9937 on
{sfb,lsb,scissor,imbalance,redir,alt}).

A slowness-weighted variant is shipped as a **separate, separately-labelled column**
(`travel_slowness_weighted`), never as the headline, and it is explicitly flagged as a
preference.

### 1.5 Why (c) is not a third option — a proof, not an opinion

Under a strict return-to-home model every press is `home -> key -> home`, so a press of a key at
distance `d` costs `2d` and the per-finger total is `2 * sum freq * dist(key, home)` = exactly
`2 x (a)`. A positive scalar multiple **cannot change any share** (the 2 cancels in the ratio) and
cannot change any ranking. So (c) is (a) in different units, and the sensitivity check that
matters is (a) vs (d) — which is what is shipped. **Registered as a claim to be tested in code**
(§4), not asserted.

### 1.6 Thumb / space: EXCLUDED, and why

Space is excluded from the travel corpus and the thumb is not a ninth cell.

1. The thumb has **no home-position entry** and no travel semantics on this board: space is a
   fixed key at `(0,0)`, not an assignable slot, so `dist(home(thumb), space)` is identically 0
   for every layout. A ninth cell that is 0.0 for all layouts carries no information and would
   only dilute the eight shares that do.
2. Including space-touching bigrams *without* a thumb cell would be worse: the `x=space, y=letter`
   bigrams would charge the letter's finger a from-home term, which the same letter already gets
   from every other different-finger predecessor — it inflates the return-model branch with a
   third of the corpus for no added discrimination.
3. It matches the `kmstats`/`sfb`/`lsb`/`bad_scissor` convention (space-excluded), so the
   denominator convention is the one this repo's other bigram gauges already use.

**The eight cells are `L-pinky L-ring L-middle L-index R-index R-middle R-ring R-pinky`**, the
`bad_scissor.FINGER_ORDER` labels, so the two modules' columns line up.

---

## 2. `off-home` — the definition

### 2.1 The user's claim, stated as something falsifiable

> "Pinky being used a lot is mostly fine, as long as it stays on the home row."

Decomposed, this is **two** claims:

* **(C1) a measurement claim**: the interesting quantity is a finger's **off-home-row** usage
  share, not its total usage share.
* **(C2) a cost claim**: total pinky use does not hurt; off-home pinky use does.

**(C1) is a definition and is registered below. (C2) is an EMPIRICAL claim and is TESTED in §5,
not assumed.** Keeping them apart is the point: shipping the metric does not ship the claim.

### 2.2 Stated exactly

For finger `f` on layout `L`, over the layout-restricted, **space-excluded** *letter* frequency
distribution (each character of each corpus bigram contributes its bigram's frequency — the
existing `DislocationScorer._letter_freqs` construction):

```
usage(f)      = 100 * (mass on f's keys)                  / (total letter mass)
off_home(f)   = 100 * (mass on f's keys with row != 2)    / (total letter mass)
on_home(f)    = usage(f) - off_home(f)
off_frac(f)   = 100 * off_home(f) / usage(f)      # "% of this finger's own use that is off-home"
```

* **HOME ROW is `y == 2`** — the geometry's middle row, `_FINGER_HOME`'s row for all eight fingers.
* **`usage` is an exact partition of 100%** over the 8 fingers (space excluded), and `off_home`
  is an exact partition of the total off-home mass. Both are asserted (§4).
* **`off_frac` is NOT a partition and must never be summed** — it is a per-finger ratio with a
  different denominator per finger. Registered explicitly because summing it is the obvious
  mistake and it would produce a plausible-looking number in the 200-400 range.
* **PINKY = `abs(column) == 5`** on the 30-key board, **plus `abs(column) == 6`** if a K31
  geometry is passed (`_ABS_COLUMN_TO_FINGER` maps col 6 to the pinky). The metric asks the
  geometry, never a hardcoded column list.

### 2.3 Registered generalization

The metric is defined for **all eight fingers**, not just the pinky. The pinky is the user's
instance of it; whether the pinky is *special* or is one case of a broader "off-home use of a
weak finger" axis is an empirical question (§5.4) and is answered from the shipped
all-eight-fingers output, not by a separate pinky-only metric.

---

## 3. What the parent's first-cut numbers are, and what "reproduce" means

The parent supplied a first cut (blend-v1, letter-mass-weighted, pinky = `|col|==5`, home = row 2):

| layout | pinky tot | on-home | OFF-home | off % of pinky |
|---|---|---|---|---|
| arm B | 13.99 | 9.62 | 4.37 | 31.3% |
| BALL-1 | 13.99 | 9.62 | 4.37 | 31.3% |
| armH-hdln | 14.26 | 9.62 | 4.64 | 32.6% |
| keybo-lsb | 12.82 | 5.73 | 7.09 | 55.3% |
| graphite | 15.13 | 12.38 | 2.75 | 18.2% |
| semimak | 17.57 | 12.50 | 5.06 | 28.8% |
| qwerty30m | 10.82 | 7.62 | 3.21 | 29.6% |

**These are a HYPOTHESIS, not a fact, and they are re-derived independently here.** Registered in
advance: the numbers above are reproduced **only if** my independent implementation agrees to
within 0.01 pp on all 28 cells. Anything larger is a discrepancy to be explained in the report,
not averaged away. (Note `arm B` and `BALL-1` are printed identical — they differ by a single
`cd`->`dc` transposition on non-pinky columns, so identical pinky numbers are *expected*, and
their agreeing is a weak check that the column filter is right.)

---

## 4. Registered tests (written before the numbers)

Positive-controlled with `keybo.testkit` — `assert_module_under` (the editable-`.pth`
wrong-tree trap: this repo's venv resolves `keybo` to the SHARED clone unless
`PYTHONPATH=/tmp/travel/src` is set, verified today) and
`assert_harness_detects_a_fatal_mutant`.

1. **`travel` shares sum to 100.0** within float tolerance, for every registry layout.
2. **NOT trivially satisfied**: the test that would pass on a broken metric must fail. A
   normalization test alone passes for `travel(f) = 1` for all f. So: shares must be
   `assert_discriminating` across layouts, AND a mutant that charges the *wrong* finger must be
   caught.
3. **Unknown finger label RAISES, never silently appended** — the `bad_scissor._partition` D4
   failure, fixed today: a drifted `R-pinky` label printed 0.0000 while its real 0.4658 sat
   unprinted, and *every sum-the-values test still passed* because they sum `.values()` and never
   the printed columns. Both new partitions declare their key set and refuse anything else.
4. **`off_home + on_home == usage`** exactly, per finger; **`usage` sums to 100**; **`off_home`
   sums to the total off-home mass**.
5. **`off_frac` is guarded**: a test asserts it is NOT a partition (its sum is far from 100), so
   nobody later "fixes" it into one.
6. **(c) == 2x(a) in shares** (§1.5): the return-to-home model's shares are bit-equal to the
   static model's shares. If this fails, §1.5's proof is wrong.
7. **Degeneracy** (§1.2): with same-finger travel forced off, `travel` shares equal the
   second-of-bigram-weighted static shares.
8. **Space/thumb**: no ninth cell; and a K31 geometry puts column 6 on the pinky.

---

## 5. PREDICTIONS — registered before measuring

Numbered so each can be scored right or wrong. Confidence tags are this repo's.

**P1 🟠 INFERRED — `travel` will be HIGHLY correlated with `sfb-dist`, |r| > 0.8.** Mechanism:
the same-finger branch of (d) is *literally* a same-finger distance sum, which is what `sfb-dist`
is. If this holds, the honest report is that the headline `travel` **total** is close to a
restatement of an existing gauge, and its value lies in the per-finger SPLIT, not the total.

**P2 🟠 INFERRED — the per-finger travel SHARES will discriminate pairs the frame ties by
construction.** `alt` and `imbalance` are hand-partition invariants and `sfr` is a permutation
invariant, so any pair of layouts related by a within-hand permutation ties on all three. Travel
shares depend on WHICH KEY within the hand, so they should separate such pairs. This is the
high-value case, and P2 is the prediction most worth being wrong about.

**P3 🟡 HIGH — `off_home(pinky)` will NOT be ~1.0 correlated with any of the 15 gauges.** No
existing gauge is row-restricted per finger; the closest are `sfb-dist`/`lsb-dist` (pair
distances, not per-finger row occupancy) and `comfort` (a bigram-weighted preference sum).

**P4 🟠 INFERRED — keybo-lsb will be the worst layout in the field on `off_home(pinky)` and will
NOT be worst on predicted ms/char.** This is the parent's reversal claim, and registering it as a
prediction is what makes confirming it a check rather than an echo. **If it holds, it does NOT
by itself validate the user's claim (C2)** — a metric can disagree with the time model because
the metric is measuring something the time model prices differently, or because it is wrong.

**P5 🔴 UNCERTAIN — (C2) will NOT be cleanly supported.** Specifically: in a regression of
predicted ms/char on {pinky-total, pinky-off-home} over a layout pool, I predict off-home carries
more signal than total, BUT that both will be swamped once bigram-frequency structure is
controlled — because that is exactly what happened to `bad_scissor`. Registered as the null I
expect to fail to reject, so that "we found an effect" has to clear a bar I set beforehand.

**P6 🟠 INFERRED — the pinky is NOT special.** If off-home matters, off-home-ring will behave
similarly and the real axis is "off-home use of a weak finger". If instead the pinky separates
from the ring, that is a stronger and more surprising result.

**P7 🟡 HIGH — `travel` is movable by a 1-swap but only slightly.** A greedy 1-swap probe will
find travel-total reductions of a few percent, and the swaps it wants will be
`move a high-frequency letter onto a home key`. Registered prediction: the probe will show
travel is **optimizable but should NOT be optimized** — it is a descriptor whose minimizer is
a degenerate "all frequent letters on home row" board that the time model does not prefer,
the same "optimizing the ruler" failure WSCISSOR-GEN-1 registered.

---

## 6. Stopping rule / what would make this a negative result

**"This metric is redundant with X" is a COMPLETE and valuable answer and will be reported as
the headline if that is what the numbers say.** Concretely, the report must state plainly:

* if `travel_total` is R2 > 0.95 on the existing gauge frame, it is a restatement and its total
  should not become a 16th gauge;
* if `off_home(pinky)` is |r| > 0.95 with any single existing gauge, likewise;
* if the frequency control in §5 kills the ms/char relationship, then (C2) is **unsupported on
  this evidence** and must be reported as unsupported, not as "promising".

No value will be manufactured for either metric.
