# Why is A faster than B? — `keybo compare` (COMPARE-1)

The sibling of `keybo analyze`. `analyze` **scores** boards on every gauge; `compare`
**explains** the ms/char speed gap between exactly two of them, as a signed per-feature
budget that sums back to the gap — and, for every feature, **what each board's value of
that feature actually is.**

```
$ keybo compare flagship-c3 graphite

COMPARE  flagship-c3 -> graphite   corpus=default  wpm=90
  channel: both

RECONCILIATION (checked before any interpretation)
  ms/char shipped card()  flagship-c3: 254.9761   graphite: 258.1696   <- what `keybo analyze` prints
  gap (b-a; +ve = flagship-c3 faster) : +3.1934 ms/char
    T2 bigram channel    : +0.9981  (31.3%)  <- decomposed
    Tcond trigram channel: +2.1953  (68.7%)  <- decomposed
  DECOMPOSED SHARE: 100.0% of the gap    undecomposed: +0.0000 ms/char
  RECONCILES: True
  EXTERNAL GAUGE TIE: OK

  BLOCK CONTRIBUTIONS (primary) to gap_t2 (+0.9981 ms/char)
    block           ms/char    share      favours   top column     flagship…  graphite
    ROW             +0.7136    71.5%  flagship-c3   bottom            0.0770    0.1190
    GEOMETRY        +0.3081    30.9%  flagship-c3   dx                4.3023    4.5003
    WPM             -0.0922    -9.2%     graphite   wpm              90.0000   90.0000 [NO-DIFF]
    ...
```

Read that top row as: *the bottom row accounts for +0.71 ms/char of flagship-c3's advantage,
and flagship-c3 gets it by doing **less** bottom-row work — 7.7% of weighted keystrokes
against graphite's 11.9%.*

## The estimand

**Frequency-weighted LMDI attribution of the ms/char gap between two layouts, on the fitted
time surface.** Precisely:

* the quantity decomposed is `card(B).ms_per_char - card(A).ms_per_char` — the *same* number
  `keybo analyze` prints, tied to it externally on every run;
* it is split first into the gauge's own two terms, `gap_t2 + gap_tcond`, each decomposed on
  **its own frame** (20 bigram columns over `n²` position pairs; 46 trigram columns over `n³`
  triples) and under **its own corpus weight** (`w2`, the trigram table's first-two-character
  marginal, vs `w3`, the trigram frequency directly);
* within each channel the ms conversion is the **log-mean Divisia (LMDI)** weight, which is an
  *algebraic identity*, not a linearization — so there is no approximation error to report and
  the measured residual is pure float64 rounding (~1e-16).

`mean_a` / `mean_b` are a **second, different quantity** beside the attribution: the
corpus-frequency-weighted mean of the feature column itself on each board, under the same
weight the gap uses. They are per-**board** (no pair-specific weight enters), and they are
*not* attributions.

## The exactness guarantee, and its two independent halves

A default run prints **both** residual families, and this is not redundancy:

| family | what it validates | measured |
|---|---|--:|
| **INTERNAL** — contributions sum to each cell's ms change, and to the channel gap | the **arithmetic** | ~1e-16 |
| **EXTERNAL** — the decomposed gap equals the independently-shipped `card()` gauge | the **choice of quantity** | ~1e-7 |

The internal family alone is **not sufficient, and this was measured, not assumed.** Under a
wrong corpus weighting both sides of the sums-back identity share the weight table, so the
identity still closes at ~1e-16 while decomposing the *wrong quantity* — and on the registered
pair the analogous trigram error additionally **inverts** which block leads. So:

> **If the external tie exceeds 1e-3 ms/char, `compare` prints no attribution table at all.**
> The tables are *suppressed*, not annotated, and the exit code is non-zero.

Both negative controls stay reachable, and each breaks a *different* family — which is the
evidence that neither is redundant:

```
keybo compare A B --control bigram-table    # T2 weighted by bigrams.txt   -> EXTERNAL fails
keybo compare A B --control tcond-marginal  # Tcond weighted by the marginal-> EXTERNAL fails
keybo compare A B --control shuffle         # SHAP deltas permuted         -> INTERNAL fails
```

A control that *reconciles* means the identity has become vacuous, so `--control` inverts the
exit code: `rc=0` means the control correctly failed.

## The five ways this output can mislead — and what the tool does about each

Every item was **measured** in SHAPDIFF-1 / SHAPDIFF-TCOND / FM4, not imagined.

### 1. Per-column credit is not unique → blocks are the default

TreeSHAP's split of credit across **correlated** columns is one of many valid splits. The
symptom is on the face of the report: `wpm` is a *constant* column at a fixed scoring WPM, yet
carries **-0.0922 ms/char**. The trigram frame is structurally worse — `bg1_*` and `bg2_*` are
the same 19 placement features on two overlapping key pairs.

So the **block table is the default** and the per-column table is `--columns` opt-in. A block
sum is invariant to redistribution *within* the block. It is **not** invariant to leakage
*between* blocks, which is why the next item exists.

### 2. Coupled columns → the leakage flags, and the joint

Two flags, both computed from the numbers:

* **`COUPLED`** — `bg1_X` and `bg2_X` carry **opposite-signed** credit for the same physical
  property. Measured: `bg1_bottom` **-0.2337** against `bg2_bottom` **+0.7382**. Neither column
  stands alone; the report prints the **JOINT** (`+0.5045`), which does not depend on how the
  credit was divided. Seven properties fire on the flagship-c3/graphite pair.
* **`NO-DIFF`** — the two boards do not differ in the feature *at all* (`mean_a == mean_b`) yet
  it still carries credit, so the credit is necessarily an interaction artifact. Measured:
  `wpm`. **This flag is only computable because the value columns exist** — without them the
  report cannot tell "B does less of it" from "B does exactly as much of it".

### 3. Channels do not add per feature → the refusal

The T2 `bottom` (23.3% of the gap) and the Tcond `bg2_bottom` (23.1%) are the same physical
property on two different frames, each already carrying its own channel's full share. They are
**not** 46.5%. The report names all 19 doubled properties with a worked example, and the
library **raises** rather than returning a cross-channel total:

```python
diff.total_for_property("bottom")   # ValueError: REFUSED: ... MUST NOT be summed
```

`gap_total` is the only cross-channel total that is well defined.

### 4. Magnitudes carry the model's calibration error → ranking, not ms

The per-fold calibration slope reaches **1.407** on the bigram surface (qwerty fold) and
**0.7304** on the trigram one (dvorak fold), so an ms/char magnitude can be off by tens of
percent. **Orderings are affine-invariant and therefore safe**; read the ranking and treat the
ms figures as scaled. And every number is a contribution to *this fitted surface's prediction*,
never a biomechanical claim — this surface prices **long travel as cheaper**, so a positive
`dx` is a fact about the model's pricing, not about distance being good.

### 5. A column's name is also a gauge's name → the honest display name

Four served columns share a name with a gauge `keybo analyze` reports, **while measuring
something else** — so a reader who moves between the two tables draws a false inference. This is
the one that already did damage: the SHAPDIFF-1 read-through treated the `lateral` column as the
lateral-*stretch* measure, and it is not one.

Each verdict is a measurement over the **full** enumeration of `ROW_STAGGERED_30` — all 900
ordered position pairs, all 27,000 ordered triples — against the gauge's *own* code path:

| served column | same-named gauge | verdict | printed as |
|---|---|---|---|
| `scissor` | `scissor` (oxey, comfort) | **EQUAL** — 0 disagreements, both call `classify.is_scissor` | `scissor` (unchanged) |
| `lsb` | `lsb` (keymeow, \|dx\| ≥ 2) | **DIFFERENT** — column fires 32, gauge 24; strict superset, the 8 extra are exactly the dx = 1.75 pairs | `lsb_dx1p5` |
| `redirect` | `redirect` / `redir` | **DIFFERENT** — column fires 4320, both gauges 2808; the 1512 extra are exactly the same-finger-constituent firings | `redirect_ungated` |
| `bad_redirect` | `bad_redirects_total` | **DIFFERENT** — column fires 864, gauge 540 | `bad_redirect_ungated` |
| `lateral` | `lat-span` | **DIFFERENT** — a landing-KEY one-hot (invariant in the first key) vs a graded PAIRWISE stretch (symmetric); 126 pairs fire one and not the other, each way | `landing_off_home` |

The display names are checked against the other frames too, not just the gauges: `interp.1` (branch
`interpframe`) serves a real column called `off_home_column` that counts **both** keys (0/1/2) where
this one is a 0/1 landing-key one-hot — they disagree on 180 of 900 pairs — so the obvious name
would have manufactured a *fresh* collision on merge. Hence `landing_off_home`.

`scissor` keeps its name deliberately: a shared name that is *truthful* is informative, and
annotating it would train readers to ignore the annotation. So would renaming
`redirect`/`bad_redirect` if they matched — and a prior reading held that they did, citing
`analysis/redirects.py`'s exhaustive equality. **That equality is between the two GAUGES**
(`kmstats._is_redirect` and `community._v1_pattern`: 2808 both, 0 exactly-one) and neither side
of it is the model's column. The gauge's predicate does already have a column name here —
`redirect_sfgated`/`bad_redirect_sfgated` match it exactly on all 27,000 triples — which is what
makes `_ungated` the accurate name for the served pair rather than an invented one.

**The rename is in the READER only; the served schema is untouched.** That is not a shortcut, it
is the correct fix, and the alternative was measured: `models/base.py`'s load guard compares
`feature_version` alone and never `feature_names`, so renaming a schema column would *not*
invalidate the six `data/models/k31` artifacts — but it would desync their sidecars from the
schema, and `_shap_tables` refuses a model whose stored names differ from the schema list. A
schema rename therefore makes `keybo compare` **raise instead of report**, while `shap-report`
(which takes its names from the *sidecar*) would silently keep printing the old ones. Fixing the
display layer keeps every published number reproducible; `keybo.analysis.effect_curves` set the
same precedent when it renamed its own copies of `inwards`/`outwards` to `outer_high`/`outer_low`.

## What it fundamentally cannot do

**It cannot attribute any part of the gap that is not in the model's features at all.** Neither
frame carries a hand-identity channel, so *"board B overloads one hand"* is unaskable; the
bigram frame carries no direction-of-travel channel either (`inwards`/`outwards` are
swap-invariant; the trigram frame's `redirect_ungated`/`bad_redirect_ungated` — served as
`redirect`/`bad_redirect` — are the one order-aware signal).
Such a difference is **priced inside the features that are served**, not reported as a
remainder — so the absence of a "hand balance" row is not evidence that hand balance did
nothing.

Two smaller limits worth stating:

* An LMDI ms attribution is a property of the **A→B comparison**, not of either board alone
  (the weight is pair-specific). A per-board budget is well defined only in log space, which is
  what `log_a`/`log_b` carry.
* The external tie is to the *shipped* `card()` gauge. It cannot detect an error in `card()`
  itself. The claim is that the decomposition decomposes **the gauge's own quantity** — not
  that the gauge is right.

## CLI

```
keybo compare <layout_a> <layout_b> [options]
```

Registry names (`graphite`) or raw 30-character row-major strings, exactly as `analyze` accepts.
The sign convention is fixed and printed: **positive means `layout_a` is faster.**

| flag | effect |
|---|---|
| `--channel {t2,tcond,both}` | which gauge term to decompose (default `both` — the only setting that can reach 100%; a single channel names its undecomposed remainder) |
| `--columns` | also print the subordinate per-column table (off by default; see §1) |
| `--top N` | with `--columns`, show the N largest columns and **name** how many were withheld and their total |
| `--corpus NAME` | corpus for the frequency weighting (`--corpus iweb` reproduces the campaign's frozen boards) |
| `--target-wpm F` | scoring WPM (default 90) |
| `--control {bigram-table,tcond-marginal,shuffle}` | run a negative control; all must fail |
| `--json PATH` | write the complete result as JSON — **never truncated by `--top`**, and it carries the caveats, the leakage flags and the mean columns |

## Neighbours — which tool answers which question

| command | unit | question |
|---|---|---|
| `keybo analyze` | one board per row | how fast is each of these, on every gauge? |
| **`keybo compare`** | **one feature per row** | **why is A faster than B, and which board does more of it?** |
| `keybo layout-diff` | one n-gram per row | which corpus *strings* got slower? |
| `keybo shap-report` | one feature per row, one board | what does the model use? |

## Library

```python
from keybo.analysis.shap_diff import shap_diff, format_report

diff = shap_diff(LAYOUT_A, LAYOUT_B, name_a="flagship-c3", name_b="graphite")
assert diff.reconciles()          # check BEFORE reading any feature
assert diff.gauge_tie_ok()        # the external family specifically

for block in diff.t2.blocks():                    # the PRIMARY table
    print(block.block, block.ms_per_char, block.leading, block.flag)

for c in diff.tcond.ranked():                     # subordinate detail
    print(c.feature, c.ms_per_char, c.mean_a, c.mean_b, c.flag)

diff.tcond.leakage()              # {'bg1_bottom': 'COUPLED', 'wpm': 'NO-DIFF', ...}
diff.tcond.joint("bottom")        # +0.5045 — the number that survives the bg1/bg2 split
diff.cross_channel_properties()   # the 19 names that must not be summed across channels
diff.to_dict()                    # the JSON payload, caveats included
```

The module keeps the name `keybo.analysis.shap_diff` after COMPARE-1 renamed the *command*: the
SHAPDIFF-1/-TCOND ledger entries, committed artifacts and tests all name it, and renaming it
would cost that audit trail for no user-facing gain.

## Provenance

Registered in `PREREGISTRATIONS.md`: **SHAPDIFF-1** (the bigram channel, the LMDI construction,
the two bar families and the two negative controls), **SHAPDIFF-TCOND** (the conditioned-trigram
channel — the 68.7% the bigram frame was structurally blind to — and the block partition),
**COMPARE-1** (the rename, the feature-value columns, the honesty layer), and **COMPARE-1
ADDENDUM 1** (which corrects one of COMPARE-1's own bars: `wpm`'s mean cannot be *exactly*
`target_wpm`, because a weighted sum of a constant over ~30k cells accumulates float64
rounding — the bar was mis-specified, not the code).
