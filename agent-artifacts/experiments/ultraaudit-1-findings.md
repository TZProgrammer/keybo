# ULTRAAUDIT-1 — the label-vs-referent audit: findings (FINDING commit, no code change)

Deliberate reproduction of the mechanism that found all eight of this campaign's methodological defects:
**a fresh reader with a different purpose.** Self-review is 0-for-8 here, so "read more carefully" is a
refuted strategy. This round hunted ONE signature — *a name doing load-bearing work with no check that it
matches its referent* — plus the asymmetry that hides it: **a defect that flatters your reference point
survives** (`wfd` sat wrong for a whole campaign because it spared qwerty).

Method: 9 independent finders over disjoint surfaces (the 14 never-audited gauges, the corpus tables, the
frozen gates mutation-tested, surface provenance pairwise, the CLI output contracts), each finding then put to
a **3-lens adversarial panel** (REPRODUCE / ALREADY-KNOWN-OR-INTENDED / AUTHORITY) prompted to refute and
defaulting to refuted when uncertain, majority-refute kills. Findings below are the survivors, plus a parent
pass on an independent evidence path. **22 of 47 panel verdicts were refutations** — the panel is doing work.

Every number here was produced by a command that was actually run. No fix is applied in this commit.

---

## F1 — `keybo optimize` still loads the WRONG skipgram table: ALLGAUGE-1's fix was INCOMPLETE

**Verdict: WRONG.** Blast radius: rank 3 (a shipped search objective silently disagrees with the gauge that
reports on it). Found independently by **three** agents and the parent.

ALLGAUGE-1 (2026-07-26) is registered as fixing "a SHIPPED CORPUS-FILE BUG": `analyze` loaded
`data/corpus/1-skip.txt` while every frozen board loads `1-skip31.txt`, the true trigram marginalization
(`skip(a,c) = sum_b tri(a,b,c)`). That fix landed in `analyze` **and nowhere else.** Both search-objective
branches still hardcode the wrong file:

- `src/keybo/cli/optimize.py:122` — the `--comfort-weight` branch
- `src/keybo/cli/optimize.py:142` — the `--oxey-weight` branch

`keybo.data.corpus.PRODUCTION_SKIPGRAMS` exists, is set to `1-skip31.txt`, and is ignored here. Worse,
`corpus.py:65-68` keeps `1-skip.txt` in `REQUIRED_TABLES` **because** "different call sites load different
ones" — the divergence is documented rather than removed. So the campaign fixed the *reporting* path and left
the *search* path — the one that actually produced the layouts — on the unreproducible pass.

**Reproducer** (from the repo root):

```bash
uv run --no-sync python - <<'PY'
from keybo.data.corpus import load_frequencies
from keybo.scoring.comfort import ComfortBigramScorer
from keybo.scoring.oxey import OxeyStyleScorer
from keybo.layout import Layout
from keybo.geometry import ROW_STAGGERED_30
from keybo.layouts import NAMED_LAYOUTS
bi  = load_frequencies("data/corpus/bigrams.txt")
tri = load_frequencies("data/corpus/trigrams.txt")
s1  = load_frequencies("data/corpus/1-skip.txt")     # what optimize loads
s31 = load_frequencies("data/corpus/1-skip31.txt")   # what analyze + every frozen board loads
lays = dict(NAMED_LAYOUTS)
lays.update({"keybo-lsb":    "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
             "flagship-c3":  "pyou'vgdnmheai.cstrlkjz,-wfbxq",
             "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
             "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq"})
o1, o31 = OxeyStyleScorer(bi, s1, tri), OxeyStyleScorer(bi, s31, tri)
for n, l in lays.items():
    lay = Layout(l, ROW_STAGGERED_30)
    x, y = o1.fitness(lay), o31.fitness(lay)
    print(f"{n:<13} oxey(1-skip)={x:11.6f}  oxey(1-skip31)={y:11.6f}  d={100*(y-x)/abs(x):+7.3f}%")
PY
```

**Observed** (iWeb, where the two files genuinely differ — 3474 vs 4087 keys):

| layout | d% oxey-style | d% comfort |
|---|---|---|
| **qwerty** | **+0.083** | **−0.001** |
| dvorak | +0.263 | +0.027 |
| colemak | +0.281 | −0.005 |
| graphite | +0.343 | +0.042 |
| semimak | +2.044 | +0.065 |
| flagship-c3 | +1.774 | +0.101 |
| **keybo-lsb** | **+4.327** | **+0.163** |
| **keybo-lsb+lm** | **+4.426** | **+0.168** |
| **archive-1843** | **+4.571** | **+0.167** |

**This is the qwerty-flattering asymmetry, quantified: ~52× larger on the campaign's own optimized layouts
than on the reference layout.** A spot-check on qwerty — the natural thing to do — reads as 0.08% noise.

Two conditions kept it invisible:
1. **On `blend-v1` (the default corpus since CORPUS-SWAP-1) the two files are BYTE-IDENTICAL** — md5
   `449590934b1bb50b7e9e1ca7e05140dd` for both, confirmed independently by `analyze --json`'s per-table
   sha256 (`corpus_provenance.sha256` gives the same digest for `1-skip.txt` and `1-skip31.txt`). So no value
   assertion at the default corpus can fail.
2. **`tests/cli/test_optimize_fastpath.py:139` PINNED the bug.** It wrote *only* `1-skip.txt` and asserted the
   loader read it (`assert captured == {"de": 7}`). A test that writes one convention and asserts it was read
   cannot distinguish "loaded the right table" from "loaded the only table" — this is TOOLING-TRAPS #13
   ("report both" shipping a bug as a convention) in test form.

Ranking impact over the 9-layout registry: **none** for either gauge, which is why the verdict is `WRONG`
about the objective but the downstream numbers do not move. Any future search run with a non-zero
`--comfort-weight`/`--oxey-weight` on `--corpus iweb` optimizes a different objective than the one reported.

---

## F2 — `alt` and `imbalance` are HAND-PARTITION INVARIANTS: ties by construction on the pairs the campaign adjudicates

**Verdict: UNSUPPORTED.** Blast radius: rank 2 — **~14 registered denominators move and TWO registered
majority verdicts flip.** Found independently by a gauge finder and the parent; the triage pass then found a
third vector neither had.

TOOLING-TRAPS #23 / ledger line 7232 registered `sfr` as a permutation invariant and corrected the frame to
"18 gauges, not 19". **That correction is incomplete by two axes.** `alt` and `imbalance` are functions of the
left/right **character partition** only:

- `kmstats._trigram_value("alt", a, b, c)` reads only `a.hand`, `b.hand`, `c.hand`.
- `oxey.pattern_shares`'s `imbalance` reads only `hand_load[-1]` and `hand_load[1]`.

So any two layouts that put the same characters on the same *hand* — regardless of which key — score
identically. **This is worse than `sfr`'s case, not the same:** `sfr` is a global constant, so it ties every
pair and its correction is a pure denominator fix. `alt`/`imbalance` tie only pairs that *share a partition*,
which makes them **layout-set-dependent** ties that read as genuine agreement.

**Reproducer** (mechanism — shuffle within hands, charset and hand-assignment held fixed):

```bash
uv run --no-sync python - <<'PY'
import random
from keybo.analysis.kmstats import KmStats
from keybo.data.corpus import load_frequencies
from keybo.layouts import NAMED_LAYOUTS
k = KmStats(load_frequencies("data/corpus/bigrams.txt"),
            load_frequencies("data/corpus/1-skip31.txt"),
            load_frequencies("data/corpus/trigrams.txt"))
def within_hand_shuffle(s, rng):
    rows = [list(s[i*10:(i+1)*10]) for i in range(3)]
    left  = [c for r in rows for c in r[:5]]
    right = [c for r in rows for c in r[5:]]
    rng.shuffle(left); rng.shuffle(right)
    out, li, ri = [], 0, 0
    for _ in range(3):
        out += left[li:li+5]; li += 5
        out += right[ri:ri+5]; ri += 5
    return "".join(out)
rng = random.Random(1)
seen = {g: set() for g in ("sfr","alt","imbalance","sfb","lsb","roll","redir")}
for _ in range(30):
    s = k.stats(within_hand_shuffle(NAMED_LAYOUTS["qwerty"], rng))
    for g in seen:
        if g in s: seen[g].add(round(s[g], 10))
for g, v in seen.items():
    print(f"{g:<10} distinct values over 30 within-hand shuffles: {len(v):3d}"
          f"{'   <== INVARIANT' if len(v) == 1 else ''}")
PY
```

**Observed:** `sfr` 1, `alt` 1 — **invariant**; `sfb` 30, `lsb` 30, `roll` 30, `redir` 30 (`imbalance` is
computed by `oxey`, verified separately: 1 distinct value over 40 within-hand shuffles, vs 40/40 under full
permutation).

**It hits exactly the pairs the campaign compares.** The 15 registry layouts fall into 11 distinct partitions,
and **four share one: `keybo-lsb`, `keybo-lsb+lm`, `flagship-c3`, `archive-1843`.** End-to-end via
`keybo analyze keybo-lsb keybo-lsb+lm flagship-c3 archive-1843 archive-1846 graphite --json`:

```
keybo-lsb     imbalance = 2.077879     archive-1846  imbalance = 2.053553
keybo-lsb+lm  imbalance = 2.077879     graphite      imbalance = 2.495865
flagship-c3   imbalance = 2.077879     qwerty        imbalance = 16.093881
archive-1843  imbalance = 2.077879
```

Exact-tie census over the 15-gauge frame on iWeb:

| pair | tied cells | which |
|---|---|---|
| **keybo-lsb vs keybo-lsb+lm** | **9 of 15** | sfr, sfb, sfs, lsb, lsb-dist, alt, roll, redir, imbalance |
| flagship-c3 vs archive-1843 | 3 of 15 | sfr, alt, imbalance |
| keybo-lsb vs flagship-c3 | 3 of 15 | sfr, alt, imbalance |
| archive-1843 vs archive-1846 | 1 of 15 | sfr |
| flagship-c3 vs graphite | 1 of 15 | sfr |

The 9-of-15 pair is the one **LMSCISSOR-1** adjudicated. And note the incumbent pool is **5-of-5 distinct on
`alt`** — so the degeneracy is invisible on exactly the layouts a sanity check would use.

**Downstream (established by the triage pass re-deriving off the frozen artifacts, not by inference):** all 18
registered numerators in CORPUS-BLEND-1 re-derive **exactly** off
`state/keybo-optimization/artifacts/build-corpus/board_iweb_vs_blend.json` — so the numerators are right and
only denominators move. Two registered **sub-majority** verdicts become **majorities**:

- `PREREGISTRATIONS.md:6342` CORPUS-BLEND-1 `keybo-lsb+lm 7/15` (iWeb) → 7 of 12 contested = majority.
- `PREREGISTRATIONS.md:6341`/`:6777` NO-ANCHOR-1 `archive-1843 7/15` (blend-v1-no-anchor) → 7 of 12 = majority.
  This is the ledger's headline flagship "INVERSION".

Also downstream: `RESELECT-90-110`'s "EROSION not inversion" (9/15 against a 7.5 threshold), GEOMEAN-1's
"17 of 45 field-best re-derived EXACTLY" (8 of its 45 cells are ties), and `blend-v1/PROVENANCE.md:185`'s
`alt archive-1846 -> keybo-lsb` "winner change", which is a stable-sort tie-break over a **4-way
hex-identical tie** (`0x1.693fa324d32c9p+5`).

**A separate defect surfaced on the same path** (trap #33 recurring in a different artifact):
`SELECT-MAXIMIN-1`'s registered `keybo-lsb 8 of 45 field-best` is **0 strict wins** — all 8 are ties credited
by `board_iweb_vs_blend.py:101-105`, a plain stable sort with no strict-win term.

---

## F3 — a partial `--surface-dir` override yields a silently MIXED surface frame

**Verdict: UNSUPPORTED.** Blast radius: rank 4 (latent — the vendored dir is complete today, so nothing
shipped is affected). Found independently by the surface finder and the parent.

`surfaces._resolve` is **first-hit-wins per surface NAME**, and the override dir is checked before the
vendored one. A dir holding *some* of a family's three surfaces therefore produces a frame assembled from two
different sources while every report still labels it as one family. Demonstration — put `AALTO_BASE`'s array
on disk under the *name* `AALTO_TRI_PS_FREQ_PRIOR`:

```bash
T=$(mktemp -d); SD=/local/home/zegertho/agent/state/keybo-selmethod/artifacts/old-new-layout-comparison/tri_frequency_old_new_surfaces
cp $SD/AALTO_BASE.standardized.npy $T/AALTO_TRI_PS_FREQ_PRIOR.standardized.npy
uv run --no-sync python -m keybo analyze keybo-lsb --surface-dir $T --json | python3 -c \
  "import json,sys; d=json.load(sys.stdin)['rows']['keybo-lsb']['model_scores']; print(d['family']); print({k:v['fit'] for k,v in d['surfaces'].items()})"
uv run --no-sync python -m keybo analyze keybo-lsb --json | python3 -c \
  "import json,sys; d=json.load(sys.stdin)['rows']['keybo-lsb']['model_scores']; print({k:v['fit'] for k,v in d['surfaces'].items()})"
rm -rf $T
```

**Observed:** the AALTO fit changes (`225894995238.7975` override vs `223980183688.9508` vendored) while
COMMUNITY and POOL stay vendored and unchanged — and all three are still reported under
`family: TRI_PS_FREQ_PRIOR`. `model_scores` carries no `path`/`dir`/`source`/`sha` key, and the per-surface
cells are only `[surface, fit, saved_vs_ref_pct]`, so **the resolved path is never recorded.**

The asymmetry is the point: `corpus_identity()` emits a per-table **sha256** into
`corpus_provenance` precisely so "a *modified* table cannot masquerade as a known corpus". Surfaces get the
name only. Two sibling findings on the same resolver, from the surface finder:

- `--model-family FREQ_PRIOR` reports `available: True, reason: None` with a **2-of-3** panel, because
  `AALTO_FREQ_PRIOR` does not exist; the missing pool is never disclosed.
- An unresolvable `KEYBO_SURFACE_DIR` is **silently ignored**, while the identical `--surface-dir` typo is a
  hard error.

---

## F4 — `load_frequencies` silently drops rows four ways

**Verdict: WRONG (latent).** Blast radius: rank 4 — verified NOT live.

`src/keybo/data/corpus.py:160-183` has three `continue` branches and a dict assignment, all silent:

```bash
uv run --no-sync python - <<'PY'
import tempfile
from keybo.data.corpus import load_frequencies
for name, body in {"duplicate key": "th\t100\nth\t7\n", "float count": "th\t1.5\nhe\t7\n",
                   "no tab": "th 100\nhe\t7\n", "empty field": "th\t\nhe\t7\n"}.items():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(body); p = f.name
    print(f"{name:<15} -> {load_frequencies(p)}")
PY
```

**Observed:** `duplicate key -> {'th': 7}` (the 100 is **overwritten**), `float count -> {'he': 7}`,
`no tab -> {'he': 7}`, `empty field -> {'he': 7}`. Exit 0, no warning, no count assertion — the same shape as
the `analyze` dropped-row bug (trap #38), one layer down.

**Correctly graded latent, and this is the load-bearing check:** all 8 shipped tables were probed against
every branch — `data/corpus/{bigrams,trigrams,1-skip,1-skip31}.txt` and the `blend-v1` four — and every one
has 0 duplicates, 0 no-tab lines, 0 non-integer counts, with `loaded == lines` exactly (4054, 102676, 3474,
4087, 4081, 114920, 4094, 4094). **No shipped number is affected.** A hand-built or externally-supplied
corpus directory would silently mis-score.

---

## Honest negatives — recorded so they are not re-litigated

These cost real compute and are as much of the deliverable as the findings. Each closes a hypothesis.

1. **Surface provenance: NO hidden identities. This remit item is CLOSED.** All 28 pairs compared in **both**
   the `.native` and `.standardized` frames: **0 bit-identical, 0 affine.** Closest pair
   POOL_FREQ_PRIOR vs POOL_TRI_PS_FREQ_PRIOR (maxabs 72.4, k=0.98993, residual 0.197 of B's range). The
   finder adds the layout-level check: rho over the 11-layout registry is 0.309 (AALTO,COMMUNITY), 0.691
   (AALTO,POOL), 0.809 (COMMUNITY,POOL) — **genuinely different rankers.** The label "three independent model
   surfaces" HOLDS. (Reproduced independently by two agents on different code paths.)
2. **`alt`/`roll`/`redir` ARE mutually exclusive** over all 27,000 triples — roll 32.00%, alt 25.00%, redir
   10.40%, none 28.60%, roll+sr-roll 4.00% — and `sr-roll` is a strict subset of `roll` as documented. No
   partition bug.
3. **blend-v1's skipgram marginal gap is ROUNDING, not a convention divergence.** 2854 of 4094 keys disagree
   with `sum_b tri(a,b,c)`, but maxabs = **11** and sum|diff|/1e9 = **4.7e-6**; both totals are exactly 1e9.
   iWeb's `1-skip31.txt` matches the marginalization **exactly** (0 of 4087 disagree). The manifest's
   `skipgram_convention` claim survives.
4. **`sum_c tri(a,b,c) != bi(a,b)`** on both corpora (3250–3682 of ~4054 keys, maxrel 1.0) — but the manifest
   never claims that identity and the tables are independently-derived counts. Recorded because it looks
   alarming and will tempt the next finder.
5. **`comfort`'s whole-corpus denominator is DISCLOSED, so it is not trap #9.** It divides by
   `bigram_mass = sum(bigrams.values())` while every kmstats share uses a layout-restricted total — a
   ~1.58–1.60× ratio that is **layout-dependent** (qwerty 0.625806 vs the apostrophe charsets 0.632044, a
   0.997% relative difference). `analyze.py:316-318` states this verbatim: *"this denominator differs from
   every other gauge's here — stated, not hidden"*, and it is the frozen board's convention. Disclosure
   matches behaviour → not a defect.
6. **Frozen-gate coverage — the weak version only.** `tests/analysis/test_kmstats.py` pins **qwerty alone**
   and passes 4/4 under a live, provably qwerty-sparing mutation (drop apostrophe/hyphen bigrams: qwerty
   `lsb` identical at 3.024213, keybo-lsb 0.758275 → 0.697080). **But `test_analyze_allgauge.py` and
   `test_kan1_parity.py` DO bite** (4 failed). Defense-in-depth holds; the unit gate alone does not.
7. **`a is b` in kmstats is sound.** All 30 `_KEYS` are distinct objects with distinct `(finger, x, row)`
   signatures, so no `==`-but-not-`is` collision exists.
8. **The `NA` sentinel does not leak into JSON as a string.** 230 leaves, 13 unavailable cells, **all `null`,
   zero `"N/A"` strings** — a consumer cannot average a string by accident.
9. **`lsb` correctly excludes index-to-index pairs.** Columns 3/4 and 5/6 share a finger, so `kind_diff == 0`
   and they can never be an `lsb` — correct, because such a pair is a same-finger bigram (`sfb`), not a
   lateral stretch.
10. **Community gauges' different corpus IS disclosed** — top-level
    `corpus_invariant: "community, community_primed (vendored corpora)"`.
11. **Corpus provenance hashing is genuinely solid** — `corpus_identity` emits a per-table sha256, so a
    modified table cannot masquerade as a known corpus. (This is what F3 shows surfaces lack.)

---

## What this says about the base rate

The user asked: *how many have we not caught?* The honest answer from this round is **the population is still
being sampled, not exhausted** — 9 finders on surfaces chosen because nobody had looked at them returned
findings at roughly 1.7 per finder, with no sign of a decaying rate. The mechanism that works is confirmed
again: **every finding here came from an agent that did not write the code**, and the two strongest were found
by *multiple* independent agents who could not see each other's work.

The one general check that would have caught the most of these, and the cheapest to add:
**for every gauge/quantity, assert what it is INVARIANT to, on a NON-reference layout.** F2 is exactly that
check missing; F1 is invisible on qwerty by a factor of 52; the `test_kmstats.py` gap is a gate that pins only
qwerty. A frozen board that pins one reference layout is not a positive control on a metric — it is a positive
control on that layout.
