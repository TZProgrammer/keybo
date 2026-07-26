# `blend-v1-no-anchor` — the fully reproducible blend

The anchor-free sibling of [`blend-v1`](../blend-v1/PROVENANCE.md). Same generator
(`keybo build-corpus`), same sources, same conventions, same declared total — with the one
non-reproducible component **removed**:

```bash
keybo build-corpus --no-anchor --out data/corpus/blend-v1-no-anchor
```

`blend-v1` gives 50% of its weight to the `iweb-anchor` register: licensed,
non-redistributable iWeb counts with no extraction script ever committed, consumed as an
opaque trust anchor. Its *identity* is hash-verifiable; its *derivation* is not. This variant
drops it, so **every byte of this corpus is regenerable from named local sources**.

**These files ADD a corpus; they do not replace one.** `data/corpus/{bigrams,trigrams,1-skip,
1-skip31}.txt` and `data/corpus/blend-v1/` are untouched — both are pinned by SHA-256 in
`tests/data/test_no_anchor_corpus.py`, which fails loudly and names the file if either is
overwritten in place.

---

## 1. Honesty statement

| | |
|---|---|
| **Reproducible** | **Everything.** All 4 sources, 100% of the weight. `manifest.json` records `reproducible_without_anchor: true`. |
| **NOT reproducible** | Nothing — that is the point of this variant. |

Verified, not asserted: rebuilding into a scratch directory produced **byte-identical**
`bigrams.txt`, `trigrams.txt`, `1-skip.txt`, `1-skip31.txt` **and** `manifest.json`.

Two limits carry over from `blend-v1` and are restated rather than buried:

* **Source scale.** ~40 MB of text across three registers. These are *registers*, not a
  general-English sample. Dropping the anchor removes the only large, sampled,
  general-English component — so this corpus is **more reproducible and less
  representative** than `blend-v1`. Both properties matter; neither is a strict improvement.
* **Machine dependence.** `python-stdlib` is located through the running interpreter and
  `man-pages` through `/usr/share/man`, so both depend on the host. `manifest.json` records
  each resolved `root` plus `built_with_python`, which is what makes an identical rerun
  checkable rather than assumed.

### A third limit, specific to this build

The `repo-markdown` register is a walk of the repo's own `*.md`, so it depends on the **tree
state at build time**. This corpus was built from a pinned `git archive` export of `ff793cb`
— the commit that added `blend-v1` — so the two corpora differ *only* by the anchor.

That pinning exposed a discrepancy in `blend-v1` itself, recorded here because it is a
provenance fact about the comparison baseline:

| source | this build | `blend-v1`'s manifest | match? |
|---|---:|---:|---|
| `python-stdlib` | `ac37ab97…` | `ac37ab97…` | ✅ identical |
| `man-pages-man1+man8` | `7317f159…` | `7317f159…` | ✅ identical |
| `repo-latex` | `f46a6e33…` | `f46a6e33…` | ✅ identical |
| `repo-markdown` | 717,176 B | 717,013 B | ❌ **+163 B** |

Every host-dependent source reproduces exactly; only repo prose differs, by 163 B over the
same 44 files — each of which is byte-identical to the clone that produced `blend-v1`. So
`blend-v1`'s tables were built mid-commit, from a tree whose prose was 163 B smaller, and
that tree is not recoverable from git.

**Measured, not waved away** (`keybo-e2e/bound_prose_drift.py`): 163 B is 0.023% of the prose
register, and the quantity actually at risk is whether a *dominance verdict* could flip.

| bound | max abs axis delta | dominance flips |
|---|---:|---:|
| committed `blend-v1` vs rebuilt-from-pristine (arm A / B) | 1.45e-4 / 3.16e-4 | **0 / 0** |
| this corpus vs one built from a deliberately drifted tree (arm A / B) | 7.41e-3 / 1.91e-2 | **0 / 0** |

The drifted tree carries **+53,201 B of prose over 45 files — 326× the unexplained gap** — and
still flips zero dominance decisions across 5 incumbents × 7 reported layouts × 2 arms. The
163 B is therefore bracketed from above by a much larger perturbation that is itself inert.

---

## 2. Sources

Identical to `blend-v1` minus the anchor. Hashes cover the *raw* concatenated source bytes in
sorted-unit order.

| source | register | files | raw bytes | sha256 (first 16) |
|---|---|---:|---:|---|
| `repo-markdown` | prose | 44 | 717,176 | `a075c0a717e21c56…` |
| `repo-latex` | prose | 12 | 103,901 | `f46a6e336abc23b2…` |
| `python-stdlib` | code | 633 | 6,923,454 | `ac37ab97557a36a1…` |
| `man-pages-man1+man8` | reference | 2,661 | 32,237,026 | `7317f159fab99428…` |

Resolved roots: repo prose/LaTeX from the pinned `ff793cb` export; `python-stdlib` from
`sysconfig.get_paths()["stdlib"]` (CPython 3.13.14); `man-pages` from `/usr/share/man`.
Extraction rules are unchanged — see [`blend-v1/PROVENANCE.md` §2](../blend-v1/PROVENANCE.md).

---

## 3. Weights — and how the renormalization was done

The anchor's 0.50 is **not** redistributed by hand. `blend_tables` renormalizes over the
registers actually present (`weight / weight_mass`), which is the same code path that makes
`--weights` safe, so naming an absent register cannot silently steal mass.

| register | declared | **effective here** | = declared ÷ 0.50 |
|---|---:|---:|---|
| `anchor` | 0.50 | **dropped** | — |
| `prose` | 0.25 | **0.50** | 0.25 / 0.50 |
| `code` | 0.15 | **0.30** | 0.15 / 0.50 |
| `reference` | 0.10 | **0.20** | 0.10 / 0.50 |
| | | **sum 1.00 exactly** | |

Each survivor's share is exactly **doubled**, because the surviving mass was exactly 0.50.
The register *ratios* are therefore identical to `blend-v1`'s — 5 : 3 : 2 — so this variant is
not a different editorial judgment about the local registers; it is `blend-v1` with the anchor
removed and nothing else re-decided. `manifest.json` records both `weights_declared` (the
unmodified defaults) and `weights_effective` (the renormalized shares).

Consequence worth stating plainly: prose+reference are 0.70 of this corpus and `code` is 0.30,
against 0.35 and 0.15 in `blend-v1`. Every register's *relative* pull doubles, so the technical
registers speak twice as loudly here as they do in `blend-v1`.

---

## 4. Format and totals

Every table sums to **exactly 1,000,000,000**, so `count / 1e9` sums to 1 and `count / 1e7`
sums to 100.

| table | files | types | total |
|---|---|---:|---:|
| bigrams | `bigrams.txt` | 3,896 | 1,000,000,000 |
| trigrams | `trigrams.txt` | 62,372 | 1,000,000,000 |
| skipgrams | `1-skip.txt`, `1-skip31.txt` | 4,077 | 1,000,000,000 |

Fewer types than `blend-v1` (4,081 / 114,920 / 4,094) — expected: the anchor contributed a
long tail of general-English n-grams that no local register covers.

Charset, case preservation, space-as-a-real-character, the marginalized skipgram convention
(`skip(a,c) = Σ_b tri(a,b,c)`, emitted under both production filenames), determinism, and the
integer-total rationale are all unchanged from
[`blend-v1/PROVENANCE.md` §4](../blend-v1/PROVENANCE.md).

Output hashes:

| file | sha256 |
|---|---|
| `bigrams.txt` | `366078b0d7b128a4837cae86c5e84a05402fa0eca8fb2dfa256f1a6dde7e6835` |
| `trigrams.txt` | `3111ebc2cdf81e5d5685e23b506b4b898e11719dde54a0d4cc7fa552e9f5b666` |
| `1-skip.txt` = `1-skip31.txt` | `ee8cad21db3e9085e6c2c85d481a9387d8e17ac2735a13c188d25e094fe7e05e` |

---

## 5. What this corpus does to the layout board

This is the variant where the board moves **most**, which is why it is the honest stress test
rather than a lesser fallback. Re-measured on this artifact
(`keybo-e2e/board_three_corpora.py` — the earlier figures were computed against a `/tmp` build
that no longer exists):

| corpus | gauges reordered | pairwise inversions | winners changed |
|---|---:|---:|---:|
| `blend-v1` | 8/15 | 23/315 (7.3%) | 3 |
| **`blend-v1-no-anchor`** | **11/15** | **63/315 (20.0%)** | **9** |

Corpus-sensitive axes the closure-3 flagship wins, per incumbent:

| incumbent | iWeb | `blend-v1` | `blend-v1-no-anchor` |
|---|---:|---:|---:|
| keybo-lsb | 9/15 | 10/15 | 11/15 |
| keybo-lsb+lm | 7/15 | 8/15 | 10/15 |
| lsb-sib | 11/15 | 11/15 | 11/15 |
| archive-1843 | 10/15 | 9/15 | **7/15** |
| archive-1846 | 11/15 | 10/15 | 9/15 |
| qwerty | 14/15 | 14/15 | 14/15 |

The flagship **loses its majority against archive-1843** (7/15) under the fully reproducible
corpus, while getting *stronger* against both keybo-lsb variants — the blend does not
uniformly favour or disfavour it. The iWeb and `blend-v1` columns reproduce the previously
reported values exactly, which is what licenses trusting the third column.

**Corpus-invariant by construction** — `genkey`, `oxeylyzer-1`, `oxeylyzer-2`, `wfd`. Each
reads its own vendored corpus and `community_suite(pinned)` takes no corpus argument, so they
cannot move under any blend; measured spread across all three corpora is **exactly 0.0**. A
"no-anchor" column for them would be fabricated.

**Not evaluated**: the measured speed surface (`analysis/timecard.py`) — `models/` is empty and
a speed surface is a model fit, not a corpus reweighting. A scope boundary, not a result.

---

## 6. How to regenerate

```bash
# this corpus (from the pinned tree the committed artifact was built from)
git archive ff793cb | tar -x -C /tmp/pristine-ff793cb
keybo build-corpus --no-anchor --repo /tmp/pristine-ff793cb \
    --out data/corpus/blend-v1-no-anchor

# verify it is byte-identical and additive
uv run --extra dev pytest tests/data/test_no_anchor_corpus.py
```

Tests: `tests/data/test_no_anchor_corpus.py` (15 tests) covers the additive-only guards (both
corpora pinned by hash), the exact declared totals, the renormalized weights and the
doubling identity, the manifest's reproducibility fields, per-source hashes, the shared
skipgram content, a round trip through the production `load_frequencies`, and row ordering.
The swap guard is itself negative-controlled: simulating an in-place overwrite of both
production `bigrams.txt` and `blend-v1/trigrams.txt` fails 3 tests and names each file.
