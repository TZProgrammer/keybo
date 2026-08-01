# Artifacts index — normopt (NORMOPT-1)

All artifacts are CPU-local (no hardware/pod involved — this is a pure-CPU optimizer campaign on
`/local/home/zegertho/repos/keybo`). Durable location for everything below is the git branch plus
this state dir; nothing lives only in a temp worktree.

**Durable code + full results:** branch `normopt-layouts` @ `184536f`, dir `drivers-normopt/`
in the shared clone `/local/home/zegertho/repos/keybo` (objects live in the shared clone, so the
`/tmp/normopt` worktree can be removed without loss).

**Recover the branch after this workspace is destroyed:**
```
git -C /local/home/zegertho/repos/keybo worktree add --detach \
    $(git -C /local/home/zegertho/repos/keybo rev-parse normopt-layouts) /tmp/normopt-recover
```
(`--detach` per HANDOFF-1 DEFECT 1: the naive `worktree add <path> <branch>` form fails while any
worktree still holds the branch. The BRANCH NAME is the durable handle; resolve the SHA with
`git rev-parse`, do not trust a SHA quoted in prose — HANDOFF-1 DEFECT 2.)

---

## Sweep — NORMOPT-1, 4 arms × 10 search seeds = 40 optimizer runs

| field | value |
|---|---|
| **What** | shipped `keybo optimize` under 4 objectives; which LAYOUTS each produces |
| **When** | 2026-08-01 |
| **Base** | `main` @ `96e6138` (main moved to `bccc136` mid-run — 3 commits, `PREREGISTRATIONS.md`-only from my branch's perspective) |
| **Arms** | A = shipped `--ngram bigram`; A2 = the REPORTED gauge (parity 1.2e−14); B = normgauge `registered (c)` (0.5411/0.3977/0.0612); C = normgauge 50/50 |
| **Held at defaults** | `--alpha 0.999`, `--max-outer` unset, 2-opt polish ON, `--attempts 1` (per brief: search hyperparams at DEFAULTS — `searchparams` owns that axis) |
| **Seeds** | 0–9, SAME across arms (paired) |
| **Cost** | arms A/B/C 30 runs in 6.5 min wall; arm A2 10 runs in 58 s |
| **Model** | `data/models/k31/bigram_reg31_seed0` (gz-inflated to `/tmp/normopt-scratch/models/`; `XGBoostTypingModel.load` uses `.with_suffix('.meta.json')` so it cannot read a `.gz` path) |
| **Corpus** | `blend-v1` (default), sha256 `19806532…` — matches the anchors' recorded provenance |
| **Anchors** | `drivers-normgauge/anchors.json` (tracked on main) |
| **VERDICT** | vs the CORRECTED control A2, normgauge winners are **WORSE on ms/char**: B−A2 **+1.472439**, C−A2 **+1.544987** (10.9×/11.4× the 0.135 model-seed floor; 1.67×/1.75× the 0.883 search-seed floor). Symmetrically A2 is worse on the blends by 0.031494 / 0.038506. **Bootstrap CIs include 0 → point-estimate reversal, not significance.** |
| **Biggest finding** | **the RULER, not the objective**: `spearman(bigram-table ms/char, analyze-trigram ms/char) = +0.246` (p=0.085, n=50), 42.7% of pairs discordant. Fixing it is worth −1.558 ms/char (11.5× floor, CI excludes 0) — larger than the whole normgauge effect. Independently corroborated by sibling `SEARCHPARAMS-1` (their spearman 0.6715). |
| **Task 4** | 0/40 exact reproductions; but arm C reaches **Hamming 4** and arm B **Hamming 5** from `ng:droppool-best` (a 4-cycle of `{a,g,u,.}`), while BOTH controls never get within 15 of any field board. |

### Files in this dir (236 KB total — pointers + small JSON only, no bulk data)

| file | what |
|---|---|
| `A-s0..9.json` (10) | arm A raw `keybo optimize --out` results |
| `B-s0..9.json` (10) | arm B raw results (incl. the recorded `normalized_gauges` / `blend_higher_is_better` / frame caveat the CLI writes) |
| `C-s0..9.json` (10) | arm C raw results |
| `armA2.json` | arm A2's 10 layouts + ms/char (produced by `drivers-normopt/armA2.py`, parity-gated) |
| `verdict.json` | the 30 A/B/C layouts + 20 field boards, every ruler, + winners + floor |
| `final.json` | the 4-arm comparison incl. A2; `winners` map |
| `crossscore.json` | bigram-table ms/char for the same set (the OTHER ruler — kept to document the 2.14× scale trap) |
| `sgdist.json` | per-run sg_dist (computed from `geometry.distance`; the shipped gauge is NOT on main — see report §7) |
| `names.json` | run-id → layout and field-name → layout maps |

### Drivers (committed on the branch, `drivers-normopt/`)

`recon.py` `recon2.py` (TASK 1 gate + 3 bit-exact reconciliations) · `sweep.sh` (arms A/B/C) ·
`armA2.py` (the corrected control, with its parity gate) · `crossscore.py` `analyze_all.py`
`verdict.py` `rulers2.py` `deep.py` (bootstrap) · `rulerdis.py` (the ruler-disagreement measurement) ·
`task34.py` `sgdist.py` `near.py` `finger.py` · `final.py` · `sweepA2.sh` (the trigram arm that did
NOT complete — kept so the omission is auditable) · `PREREGISTRATION.md` · `runs/` (all result JSON
incl. `analyze-all.json`, the 51-row shipped-`analyze` output).

### Negative / partial results — named, not silently dropped

- **`keybo optimize --ngram trigram` (arm A′): NOT RUN.** A `--max-outer 200` *capped* run exceeded
  550 s under host thrash (load avg peaked ~600 on 192 cores from concurrent siblings) vs 18 s for
  arms B/C, because `TableTrigramScorer` ships but is wired into no CLI path. Arm A2 substitutes for
  it by targeting the reported gauge directly. Driver `sweepA2.sh` retained.
- **`sg_dist` gauge unavailable on main.** `SGDIST-SHIP-1` (`6516348`) is a `PREREGISTRATIONS.md`-only
  commit; the code (`bbc2332`) is on unmerged branch `sgdist-ship`. Quantity computed directly and
  labelled 🟡 rather than claimed as the shipped gauge.
- **No significance claim** on B−A2 / C−A2: both bootstrap CIs include 0 at n=10 seeds.

### Thread-limit note (host thrash, 2026-08-01)

Everything after the first sweep ran with
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2`.
`PYTHONPATH` was pinned to `/tmp/normopt/src` for **every** driver invocation — the venv's editable
install points at `/local/home/zegertho/repos/keybo/src`, which follows whatever branch that SHARED
checkout is on, so an unpinned driver silently imports another agent's branch. Verified pinned:
`model_norm.__file__ == /tmp/normopt/src/keybo/scoring/model_norm.py` and
`production_corpus_dir(None) == /tmp/normopt/data/corpus/blend-v1`.
