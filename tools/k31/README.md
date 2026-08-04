# `tools/k31` — regenerating the K31 stroke tables

**Read this before you try to reproduce anything that trains.** These two scripts produce the
stroke tables every trained surface in this project is fitted on. They are committed because
the tables themselves are **1.1 GB and not in git**, and because the obvious substitute is wrong.

## ⚠ `keybo process-data` does NOT reproduce these tables

The CLI has no flag for either thing K31 requires:

1. **31-char maps** — each layout gains its physical ANSI quote-slot char at position `(6, 2)`:
   qwerty `'`, dvorak `-`, azerty `ù`, qwertz `ä`.
2. **BUF2-BOTH windowing** — a keystroke only enters a window after 2 contiguous correct
   predecessors (`BUF_K = 2`), with 5 s caps and the conditioned-trigram timing convention
   (the trigram sample records `t3 - t2`, not `t3 - t1`).

Run `keybo process-data` instead and you get a **different, entirely plausible-looking table**
— and every published number silently stops being comparable. That is the failure mode this
directory exists to prevent.

## Reproducing (~8.5 min for both tables, deterministic)

```bash
# 1. Get the raw public dump (~20 GB extracted). Source URL is in keybo/data/download.py.
keybo fetch-data --out-dir dataset
#    -> dataset/Keystrokes/files/{*_keystrokes.txt, metadata_participants.txt}

# 2. Extract BOTH tables in one pass. 62,095 qualifying files; 504 s measured.
K31_FILES_DIR=dataset/Keystrokes/files python tools/k31/k31_extract.py
#    -> bistrokes31_v1.tsv (582 MB) + tristrokes31_cond_v1.tsv (545 MB)

# 3. VERIFY you got the pinned artifacts, byte-for-byte:
sha256sum bistrokes31_v1.tsv tristrokes31_cond_v1.tsv
#    bistrokes31_v1.tsv        0f2663ad6ed42aa5...
#    tristrokes31_cond_v1.tsv  46c6c3b1cc8919ad...
```

Those two prefixes are the values **preregistered in `PREREGISTRATIONS.md`** (search
`0f2663ad`) and re-confirmed against the live files on 2026-08-04. ⚠ **The ledger records
`sha256`, not `md5`** — hashing with the wrong algorithm produces a "mismatch" that means
nothing (this cost one confused round-trip; don't repeat it).

If `KEYBO_SRC` is set it is prepended to `sys.path` — only needed when `keybo` is not
installed (`pip install -e .`). `K31_BI_OUT` / `K31_TRI_OUT` override the output paths.

### Determinism

Byte-identical output is expected on a re-run: no RNG, no parallelism, no timestamps in the
output, and files are iterated in `sorted(os.listdir(...))` order. Verify with the sha256s
above rather than assuming it.

## Retraining the production surfaces (`k31_train.py`)

```bash
python tools/k31/k31_train.py       # expects bistrokes31_v1.tsv in cwd
```
Runs the registered LOLO gate, then — only on PASS — trains and saves
`bigram_reg31_seed{0,1,2}` + `trigram_cond31_seed{0,1,2}`. **~44 min** (2611 s measured).

**You usually do not need this.** The fitted surfaces are already vendored in `data/models/k31/`
(1.6 MB, in git), so `keybo analyze`, `keybo compare`, `keybo frame-collapse` and every layout
comparison work off a fresh clone with **no data and no retraining**.

## What `runs/` holds

The original run logs, as cited by the K31 gate entries in `PREREGISTRATIONS.md`:

| log | evidence it carries |
|---|---|
| `runs/k31_extract.log` | Stage B: 2202 rows = 2111 plain + 91 quote-slot; 29,532,228 bigram + 27,672,132 cond-trigram occurrences; 504 s |
| `runs/k31_train.log` | Stage D: bigram LOLO **taus [1.0, 1.0], rho/ceiling 1.0135** vs baseline 1.0236 → PASS; trigram sanity 0.9892 vs 0.9928 |

Stage B's registered gate was that v5's rows be a subset with the delta explained by
quote-slot windows: **2111 plain rows reproduced v5 exactly, 0 delta**, since quote chars were
previously off-layout window-drops.

## Table shape (surprising, worth stating)

`bistrokes31_v1.tsv` is **2202 lines** but 582 MB — one line per `(layout, position-pair,
bigram)` with every raw observation inline as `(wpm, ms, participant_id, 0)` tuples, ~93k
fields on the first line alone. So it is not a big table; it is a small table of large rows.
Line-oriented tools will behave unexpectedly.

Four layouts only: azerty, dvorak, qwerty, qwertz — that is what makes LOLO 4-fold.
