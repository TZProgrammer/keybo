# `data/community/vendored/` — provenance, licences, and what is *not* reproducible

> Committed 2026-07-31 on branch `vendored-provenance-record` (WIRE-1/VENDPROV-1),
> closing the KAN-1 DEVIATION deliverable "with provenance notes" (registered 2026-07-13,
> delivered 18 days later). The oxeylyzer-2 licence question remains OPEN and user-gated —
> see NOTICE. An earlier draft header here said "NOT COMMITTED"; committing locally is
> reversible and was never the gated act — pushing/publishing is.

The four community-tool corpora that the `genkey` / `oxeylyzer-1` / `oxeylyzer-2` gauges score on,
plus the keymeow-format corpus used by the G3 parity gate. All four were added by a single commit,
[`ec18356`](#) (2026-07-13, the KAN-1 analyzer commit), and none has been modified since.

This file exists because KAN-1 registered it and then did not write it: the KAN-1 DEVIATION note
(`PREREGISTRATIONS.md:4451,4457-4459`) states that this tool data was *"vendored gzipped under
data/community/vendored/ **with provenance notes**."* No notes were written. This closes that gap.

---

## 1. The honesty statement — read this first

| | |
|---|---|
| **Reproducible** | **All four, byte-exactly** — payload *and* gzip container. Verified 2026-07-31 by re-running the generators and re-compressing, then comparing sha256 (not by inspection). |
| **NOT reproducible** | Nothing here is irreproducible *at this layer*. But `genkey-keybo` and `keymeow-keybo` are built from `data/corpus/*.txt` — the licensed **iWeb** tables — so they **inherit iWeb's unverifiable derivation** exactly as `blend-v1`'s anchor does. And the two oxeylyzer corpora are **retrievable but not derivable**: verbatim upstream blobs, whose own source text is not published. |

Three limits, stated rather than buried:

* **The generators live outside the repo.** `build_genkey_corpus.py` and `build_keybo_corpus.py`
  are in `~/gk-parity/` — an **unversioned scratch directory**. They are the only reason two of
  these files are reproducible, and they are one `rm -rf` from gone. Durable payload copies do
  survive in `state/gk-parity/artifacts/corpora/`.
* **Identity is not pinned by any hash in this repo.** The sha256s below are recorded here for the
  first time. Nothing asserts them (contrast `data/corpus`, where `corpus.py:133` insists *"`corpus`
  is the label, `sha256` is the fact"*). A swap is nonetheless *caught* — indirectly — by the
  KAN-1 goldens; see §6.
* **These corpora are iWeb-derived, while the analyzer's default corpus is `blend-v1`.** That is by
  design (community scores are only meaningful on their tools' native conventions, so they are
  corpus-*invariant* and ignore `--corpus`), but it means the community columns and the speed
  columns do not share a corpus.

---

## 2. What each file is

| file | `.gz` bytes | payload bytes | schema | contents |
|---|---:|---:|---|---|
| `genkey-keybo.json.gz` | 265,360 | 1,349,933 | genkey `TextData` (`text.go`) | `letters` 32, `bigrams` 1,088, `trigrams` 30,631, `toptrigrams` 30,631, `skipgrams` 961, `TotalBigrams` 341,073,092, `Total` 428,099,828 |
| `keymeow-keybo.json.gz` | 131,123 | 360,637 | keycat `Corpus` ingest JSON | `charset` 33, `bigrams` 1,084, `skipgrams` 961, `trigrams` 28,808, `chars` 33 |
| `oxeylyzer1-english.json.gz` | 343,444 | 2,095,574 | oxeylyzer-1 `language_data` | `chars` 44, `bigrams` 1,932, `skipgrams`/`2`/`3` 1,935/1,936/1,936, `trigrams` 57,680, 6 `*_total` fields |
| `oxeylyzer2-english.json.gz` | 373,364 | 2,301,639 | oxeylyzer-2 `data` | `chars` 46, `bigrams` 2,103, `skipgrams` 2,114, `trigrams` 67,090, 4 `*_total` fields |

**Value convention differs by tool.** genkey and keymeow files hold **integer counts**; the
oxeylyzer files hold **normalized shares** (`th` = 2.9818734595657292 in o1, 1.979115585922148 in
o2), which `_load_freq_matrix` (`community.py:134`) multiplies by the declared `*_total` to recover
the integer arithmetic the binaries use.

### Who reads what

| file | production consumers | test consumers |
|---|---|---|
| `oxeylyzer1-english.json.gz` | `community.py:396` (`Oxeylyzer1.__init__`) **and** `select.py:134` (`behavior_stats`) | via `community_suite` |
| `oxeylyzer2-english.json.gz` | `community.py:228` (`Oxeylyzer2.__init__`) | via `community_suite` |
| `genkey-keybo.json.gz` | `community.py:491` (`Genkey.__init__`) | via `community_suite` |
| `keymeow-keybo.json.gz` | **none** | `tests/analysis/test_kan1_parity.py:57` (gate G3) only |

⚠ **`keymeow-keybo.json.gz` has no production caller.** `kmstats.py` computes keymeow-*class* stats
natively on whatever corpus it is given; this file exists so gate G3 can compare `kmstats` against
`kmrun` **on an identical corpus** — the first G3 attempt compared against kmrun-on-shai-iweb and
failed at 0.38 pp, which was the *corpus delta*, not a port error (`PREREGISTRATIONS.md:4467-4471`).
It is a **test fixture stored in a data directory**; treat it as such.

⚠ **Some payload fields are never read**: genkey's `toptrigrams`, `Total`, `TotalBigrams`;
oxeylyzer-2's `trigrams` (67,090 entries — most of that file); keymeow's `charset` and `chars`.
They are faithful tool-format dumps, kept whole rather than trimmed.

---

## 3. Sources and how each was produced

### 3.1 `genkey-keybo.json.gz` and `keymeow-keybo.json.gz` — **our corpus, their formats**

These are **not** upstream data. They are keybo's own n-gram tables re-expressed in each tool's
input schema, so that genkey and keymeow could be run **on our corpus** (GK-PARITY,
`PREREGISTRATIONS.md:2618`, 2026-07-11). Verification: the `th` bigram is `10712957` in both
payloads *and* in keybo's own case-folded `data/corpus/bigrams.txt`; and `corpora/keybo.json` is
**untracked** in genkey's git tree (upstream genkey ships only `shai-iweb.json` and `tr.json`).

| | |
|---|---|
| generator | `~/gk-parity/build_genkey_corpus.py` → `~/gk-parity/genkey/corpora/keybo.json` |
| generator | `~/gk-parity/build_keybo_corpus.py` → `~/gk-parity/keybo_corpus.json` |
| input | `data/corpus/{bigrams,trigrams,1-skip}.txt` (the **iWeb** tables — *not* `blend-v1`) |
| durable copies | `state/gk-parity/artifacts/corpora/{genkey_keybo_corpus,keycat_keybo_corpus}.json` |

**Extraction rules** (from the generators, which document their own conventions):

| file | rule |
|---|---|
| `genkey-keybo` | Lowercase; apply genkey's `CharSubstitutions` (`? → /`, `: → ;`, `_ → -`, `" → '`); keep only `ValidChars` = `a-z , . / ; - '` + space. `letters` summed from bigram second-characters excluding space. `TotalBigrams` **excludes space-containing bigrams** (genkey's own convention); `Total` = sum of letters. `toptrigrams` = trigrams sorted by descending count. |
| `keymeow-keybo` | Fold case; 33-char keycat charset = `a-z` + `space , . / ' ; -`; keep bigrams/skipgrams of length 2 and trigrams of length 3 whose every character is in the charset. `chars` populated from bigram second-character marginals — the generator notes this is only used by Monogram metrics and `total_char_count`, **neither of which any reported metric uses**. |

⚠ Both derive `letters`/`chars` from **bigram marginals**, not from a true unigram pass. The
generators say so explicitly. It is an approximation, and it matters only for genkey's `fspeed`
denominator (`total`) and its `index_imbalance_pct` — both of which are *ratios* over the same
approximated mass, and both are float-exact against the binary-gated goldens (gate G1).

### 3.2 `oxeylyzer1-english.json.gz` and `oxeylyzer2-english.json.gz` — **verbatim upstream blobs**

Unmodified third-party data files, copied from pinned checkouts and gzipped. Each is byte-identical
to the blob at its upstream commit:

```bash
git -C ~/gk-parity/oxeylyzer   show d015a169:oxeylyzer-core/static/language_data/english.json  # → 12fe4dbc…
git -C ~/gk-parity/oxeylyzer-2 show 52b271a3:data/english.json                                 # → bff9e2f0…
```

`oxeylyzer1-english.json.gz` was taken from the installed data dir
`~/.local/share/oxeylyzer/static/language_data/english.json` (identical to the repo copy).

⚠ **`oxeylyzer1-english.json.gz` internally carries `"name": "shai"`, not `"english"`.** Upstream ships *both* files — `oxeylyzer-core/static/language_data/english.json` and `static/language_data/shai.json`, NOT siblings in one directory — `english.json` (`char_total` 449,763,627) and `shai.json` (`char_total` 449,763,611) —
different files, both self-identifying as `shai`, differing by ~16 characters of total mass. The
filename asserts a provenance the payload does not confirm; the sha256 in §4 is the fact.

### 3.3 Upstream versions

| tool | upstream | commit | date | licence |
|---|---|---|---|---|
| genkey | `github.com/semilin/genkey` | `f1f41733931c2339d4ed161a5ace5f03412b282e` | 2024-07-03 | GPL-3.0 |
| keymeow | `github.com/semilin/keymeow` | `a8e95912e5b2022369276c4557040088cb12e25b` | 2024-12-06 | GPL-3.0 |
| oxeylyzer-1 | `github.com/o-x-e-y/oxeylyzer` | `d015a1692d3768e88e23ae4e9ae271a0f06807dc` (tag `v0.2.0`) | 2026-06-18 | Apache-2.0 |
| oxeylyzer-2 | `github.com/o-x-e-y/oxeylyzer-2` | `52b271a3eb747ed1cf62ccd5721190db9ff27a38` | 2026-02-12 | ⚠ **none declared** |

`community.py:6` pins the genkey port to `generate.go Score @ f1f4173` — the same commit.

---

## 4. Identity — sha256

Recorded so the *identity* of these bytes is verifiable even though their derivation partly is not.

| file | `.gz` sha256 | payload sha256 (uncompressed) |
|---|---|---|
| `genkey-keybo.json.gz` | `21cab37d26f8b4ab8cc26544302d852ac794041057b338c2e4bb93294bb469e1` | `a78a5998ca4ad340292d930ac64ede7f4d59aea947677dca8a1a1d09510287d8` |
| `keymeow-keybo.json.gz` | `0d070f5794a20077032b6f1674f6fb7162f1b2335189de8ad2bc85cce3d05356` | `936d7a9173846cd84fcde147ffc1f2bc5ed56159984c1204dffd4d4a19394cc7` |
| `oxeylyzer1-english.json.gz` | `42844251345fca404ff8aa421faf166b56dbc852eec475f9e32ad606ed2cf581` | `12fe4dbcd7d42a176df03ddffc4cffa99a60d8cfdbdbc3d21eec46af28b3f537` |
| `oxeylyzer2-english.json.gz` | `c7bcfd6f4c11749db4fabf76b9d5be74af8f902d670d3505fe2c6a2f9115bad5` | `bff9e2f088999364a9590590cdf9f21935b672dd2e74a8890aca752f7e39af97` |

Each `.gz` header also records the **original filename and source mtime** (`gzip -n` was *not*
used), which is independent provenance:

| file | FNAME in header | MTIME in header |
|---|---|---|
| `genkey-keybo.json.gz` | `keybo.json` | 2026-07-11T22:13:43Z |
| `keymeow-keybo.json.gz` | `keybo_corpus.json` | 2026-07-11T22:12:18Z |
| `oxeylyzer1-english.json.gz` | `english.json` | 2026-07-13T03:40:06Z |
| `oxeylyzer2-english.json.gz` | `english.json` | 2026-07-13T03:02:14Z |

---

## 5. Licences and attribution — ⚠ open items

**The four files split two ways, and only two carry third-party content.**

* `genkey-keybo.json.gz`, `keymeow-keybo.json.gz` — **our data in their formats.** What is borrowed
  is the *schema*. The genkey/keymeow projects are GPL-3.0, but no genkey or keymeow *content* is
  redistributed here, so their copyleft is not engaged by these files. Naming the upstream tools is
  a clarity courtesy, which this document provides.
* `oxeylyzer1-english.json.gz`, `oxeylyzer2-english.json.gz` — **genuine third-party data**,
  redistributed verbatim. These are the ones with obligations:

| file | upstream licence | status |
|---|---|---|
| `oxeylyzer1-english.json.gz` | **Apache-2.0** (`LICENCE.md` in the oxeylyzer repo) | Redistribution is permitted, but Apache-2.0 §4(a)/(b)/(d) require notice retention and attribution. **No `NOTICE`, licence copy, or upstream reference currently accompanies this file.** ⚠ Needs a decision. |
| `oxeylyzer2-english.json.gz` | ⚠ **NONE** | No `LICENSE`/`COPYING` anywhere in the oxeylyzer-2 tree, and no `license =` field in **any** of its six `Cargo.toml` files. Default copyright ⇒ *all rights reserved*; **there is no grant on record permitting redistribution of this file.** ⚠ Needs a decision. |

Two further facts for whoever resolves this:

* **This repo declares no licence of its own** (no `LICENSE` file, no `license` field in
  `pyproject.toml`), so its redistribution posture is undeclared.
* **GK-PARITY scoped the tools as *"clone + build … READ-ONLY LOCAL USE"*** (`:2626`). Committing
  their data into the repo goes beyond what that registration contemplated.

*This section is a factual record of licence texts read from the local pinned checkouts on
2026-07-31, not legal advice. Upstream may have changed since; re-check before acting.*

---

## 6. What depends on these bytes

**Gauges** (`analyze.py:387-418`): `genkey`, `oxey1`, `oxey2`, `wfd`, and the primed variants
`genkey_primed`, `oxey1_primed`, `oxey2_primed`, plus `wfd_legacy_reconciliation`. Via
`select.py:134`: `bad_redirect_pct` and `sk{1,2,3}_{samekey,sftravel}_pct`.

**Registered claims**: KAN-1 gates G1/G2/G3 (`PREREGISTRATIONS.md:4467`); GK-PARITY's board and the
FSPEED τ=0.611 flag (`:2642`); OXL2-GAUGE's exact oxeylyzer-2 board (`:3890`); the P14/P14b
"3 of 4 exact tools" bars (`:4046`, `:4088`); and the corpus-invariance claim in
`blend-v1/PROVENANCE.md:216-221`.

**How a swap would be caught today.** No hash is asserted anywhere, but the KAN-1 goldens
(`tests/analysis/golden_kan1.json`, 8 layouts, float/integer-exact) pin these corpora *indirectly*.
Measured 2026-07-31 by substituting a plausible sibling file from each upstream directory:

| substituted file | plausible substitute | outcome |
|---|---|---|
| `oxeylyzer1-english.json.gz` | `shai.json` | caught, 8/8 layouts (G2) |
| `oxeylyzer2-english.json.gz` | `english_no_space.json` | caught, 8/8 layouts (G2) |
| `genkey-keybo.json.gz` | `shai-iweb.json` | caught, 8/8 layouts (G1) |
| `keymeow-keybo.json.gz` | genkey keybo n-grams / `shai-iweb` n-grams | caught, 18 / 27 assertions (worst 0.0976 / 0.7562 pp vs a 0.02 pp bar) (G3) |

So identity is *enforced* but not *declared*. Recording the sha256s above is what makes it
declared — and it is what disambiguates the `english`/`shai` hazard in §3.2, which no score-based
check can name.

⚠ Two traps for anyone re-running that experiment: `community_suite` is `@lru_cache(maxsize=4)`
(`community.py:559`), so a swap test **must** call `cache_clear()` or it silently compares cached
baselines; and `@pytest.mark.slow` does **not** deselect gate G3 — `pyproject.toml` `addopts` is
just `-q`, so a plain `pytest` runs it.

---

## 7. How to regenerate

```bash
# 1. genkey + keymeow corpora — from this repo's committed iWeb tables.
#    (Both scripts currently live in the UNVERSIONED ~/gk-parity/; they hardcode their
#     output paths and CORPUS_DIR=<repo>/data/corpus.)
python ~/gk-parity/build_genkey_corpus.py   # -> ~/gk-parity/genkey/corpora/keybo.json
python ~/gk-parity/build_keybo_corpus.py    # -> ~/gk-parity/keybo_corpus.json

# 2. oxeylyzer corpora — re-fetch the verbatim upstream blobs at the pinned commits.
git clone https://github.com/o-x-e-y/oxeylyzer   && git -C oxeylyzer   checkout d015a169
git clone https://github.com/o-x-e-y/oxeylyzer-2 && git -C oxeylyzer-2 checkout 52b271a3
#   sources: oxeylyzer/oxeylyzer-core/static/language_data/english.json
#            oxeylyzer-2/data/english.json

# 3. Re-vendor. `-9` and the ORIGINAL FNAME+MTIME are what matter; `-c` is NOT required.
#    CORRECTED 2026-07-31 (prov-check): an earlier revision of this file claimed `gzip -9 <file>`
#    "differs by one byte". That is FALSE for all four — `cmp -l` reports 0 differing bytes,
#    because gzip stores only the BASENAME in FNAME, so the redirected and in-place forms agree.
#    The negative control was the one claim never run. What DOES differ by exactly one byte:
#      * `gzip -9` vs plain `gzip`  -> the XFL byte at offset 9 (0x02 vs 0x00)
#      * a 1-second mtime delta     -> offset 5
#    The observation was real and got misattributed to the missing `-c` instead of a dropped `-9`.
#    ⚠ TOOL-SPECIFIC: verified with gzip(1) 1.12. A Python `gzip.GzipFile(compresslevel=9)`
#    reconstruction does NOT reproduce these bytes even with FNAME, MTIME and the OS byte
#    restored, because zlib's deflate output differs from gzip(1)'s. A reader reaching for
#    Python — this repo's own language — would wrongly conclude the files are unreproducible.
cp <source> <FNAME-from-the-table-in-§4>
touch -d "<MTIME-from-§4, ISO-8601 — NO `@` prefix; `@` demands a UNIX epoch,
#          which this record does not publish. Or read it straight from the gz
#          header, offset 4-7 little-endian.>" <FNAME>
gzip -9 -c <FNAME> > data/community/vendored/<name>.json.gz
```

Verified 2026-07-31: this reproduces **all four `.gz` files byte-for-byte** (sha256 match, §4). The
genkey/keymeow re-runs match because their inputs have not moved — `data/corpus/{bigrams,trigrams,
1-skip}.txt` have zero commits since `ec18356` and identical hashes.

Tests that exercise these files: `tests/analysis/test_kan1_parity.py` (gates G1/G2/G3),
`tests/analysis/test_community_primes.py`, `tests/analysis/test_community_wfd_legacy_board.py`,
`tests/analysis/test_select.py`, `tests/cli/test_analyze_corpus_swap.py:150` (bit-identity across
corpora).

## Generator rescue (added 2026-07-31, VENDPROV-1)

The two generator scripts (`build_genkey_corpus.py`, `build_keybo_corpus.py`) live in
`~/gk-parity/`, which is **not a git repository** — verified, so they were one `rm -rf` from
unrecoverable while being the only way to rebuild two of these four files. They are now copied,
with `SHA256SUMS.txt`, to:

    /local/home/zegertho/agent/state/keybo-optimization/artifacts/gk-parity-generators/

⚠ Those generators read `data/corpus/*.txt` — the **licensed iWeb tables, not blend-v1** — so the
two files they build inherit iWeb's unverifiable derivation, exactly the trust-anchor status
`data/corpus/blend-v1/PROVENANCE.md` records for itself. Byte-exact reproduction was confirmed by
re-running them 20 days after the original vendoring.

## Why this file exists

`PREREGISTRATIONS.md`'s KAN-1 DEVIATION (2026-07-13) registered this data as vendored "with
provenance notes". **The notes were never written** — a registered-deliverable miss, not a
convention gap, and it stood for 18 days. Identity was *enforced* the whole time (substituting a
plausible sibling from any upstream is caught 8-of-8 by the KAN-1 parity goldens) but never
*declared*: no digest was recorded anywhere. `manifest.json` beside this file now declares them.


> Recipe note (2026-07-31): `touch -d` takes ISO-8601 directly; the `@` prefix demands a UNIX
> epoch, which this record does not publish — a literal run of the earlier `@`-form returned
> `invalid date format`. A reproducer can also read the epoch straight from the gz header
> (offset 4-7, little-endian).
