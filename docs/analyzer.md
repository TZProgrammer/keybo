# The keybo analyzer — `keybo analyze` (KAN-1)

Charter b330ab4 (+ deviation 6ace715). One command, one corpus, every gauge —
the analyzer the campaign wished it had, and the thing no community tool does:
its primary metric is **predicted typing time from measured keystrokes**, not
hand-tuned weights.

```
$ keybo analyze keybo-c30m keybo-lsb semimak graphite --ref qwerty30m

== predicted typing time (measured-keystroke surface; ref = qwerty30m) ==
layout        ms/char  saved%  coverage%
qwerty30m      262.43   +0.00       92.5
keybo-c30m     253.17   +3.53       92.5
keybo-lsb      253.21   +3.51       92.5
semimak        255.74   +2.55       92.5
graphite       256.19   +2.38       92.5
...
```

## What it reports

1. **Predicted typing time** — total predicted ms over the corpus on the
   K31-trained production surfaces (bigram REG-LOLO + conditioned trigram,
   seed-averaged, LOLO-gated on >100M real keystrokes), at `--target-wpm`
   (default 90). Reported as ms/char, percent time saved vs `--ref`, and the
   layout's corpus **coverage** (n-grams typable on its charset) so a charset
   that ducks corpus mass is visible. `--attribution` breaks the total into
   per-finger shares and the costliest bigrams.
2. **Community scores** — exact ports of genkey `Score`, oxeylyzer-1 and
   oxeylyzer-2 (+ its weighted-finger-distance term), each on its own native
   corpus (vendored) because those numbers are only meaningful in their home
   convention. Not approximations: integer-exact for both oxeylyzers,
   float-exact for genkey, pinned by golden tests.
3. **keymeow-class stats** — sfr/sfb/sfs (+distances), lsb (+distance), alt,
   roll, sr-roll, redir, computed natively on the shared analyzer corpus with
   keymeow's exact geometry and layout-restricted denominators.

`--json` emits the whole board as one JSON object for scripting.

## Why it surpasses the tools it combines

* **Grounded primary metric.** genkey/oxeylyzer/keymeow rank layouts by
  opinion-weighted heuristics; their weights have never been fit to a human.
  The analyzer's headline is a measured-time model (the repo's whole training
  and gating pipeline stands behind it), with the heuristics kept as the
  secondary, community-comparable view.
* **One corpus.** Cross-tool comparisons today mix four different corpora; the
  campaign repeatedly traced "score gaps" to corpus artifacts (see the K31→K30M
  reversal, docs/layout-k30m.md). Here the time surface and the keymeow-class
  stats share one corpus, and the tool ports are labeled with their native ones.
* **Attribution.** No community tool answers "which finger, which bigrams cost
  the time?" — the surface does, per layout, in milliseconds.
* **Exactness as a feature.** The ports are parity-gated (tests below), so a
  keybo number can be quoted next to a genkey/oxeylyzer number without an
  asterisk.

## Parity gates (tests/analysis/test_kan1_parity.py + tests/cli/test_analyze.py)

| gate | claim | bar |
|---|---|---|
| G1 | genkey port == binary-gated campaign values | exact (8 layouts) |
| G2 | oxeylyzer-1/-2 ports == repl-gated campaign values | integer-exact (8 layouts) |
| G3 | native keymeow stats == kmrun on the same corpus | ≤0.02pp/stat (5 layouts × 11 stats; measured worst 0.0004pp) |
| G4 | time surface == P17 campaign board | rel err ≤1e-6 (measured ≤7e-15) |

The vendored surfaces and tool corpora live in `data/models/k31/` and
`data/community/vendored/` (gzipped, versioned), so a fresh clone reproduces the
flagship board with the one command above — closing the reproducibility gap
where every campaign number previously lived in un-versioned scratch space.

## Limits (honest edges)

* Time numbers are **model predictions**, LOLO-gated but never yet confirmed by
  a human learning an optimized layout (the repo's standing biggest gap).
* The surface's training data covers the C30M/classic 30-key charsets at ANSI
  row-stagger; other geometries and charsets are out of scope (the coverage
  column shows what the corpus subset sees).
* genkey/oxeylyzer scores use those tools' own English corpora by design;
  changing the shared corpus does not move them.
