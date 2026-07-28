# Profiles & artifacts index — arme (ARM E)

All paths under `/local/home/zegertho/agent/state/arme/artifacts/` unless absolute. All numbers
**MODELLED ONLY** (fitted-surface attribution, not measured typing). Corpus **blend-v1**
(`md5(trigrams.txt) = c5066fa7bcc46dea1ecbc987fb465b4a`, `sha256` starts `19806532ee3567f5`),
frame **`.native`**, **90 WPM**.

## Run — ARM E search
| field | value |
|---|---|
| **verdict** | **258.1803 ms/char** — outcome **E3** (≥256.9), at its bottom edge |
| champion | `ou-qdbpmlsaiehvgctnr.,y'kfwjzx` |
| objective | `ev_CLAMP` **−2.690226** (archive curves, `SEARCH_DOMAIN_POLICY=CLAMP`) |
| unique evals | **10,017,839** (arm D 10,099,380; arm A 9,434,590) |
| epochs / islands / seed | 49 of 55 cap / 40 / 20260728 |
| overshoot / ga-share / polish-sweeps | 1.95 / 0.6 / 40 — identical to arms A/B/D |
| weights | `state/evidence-scorer/artifacts/arm-archive400-native.json` (pool `archive-400`, n=400) |
| pricing path | `LossCurve.price_many(..., policy=CLAMP)` — validated, cf5f731 |
| elapsed | 543.7 s |
| rc sentinel | `arme-rc.txt` = **0** 🟢 |
| durable location | `runs/arm-archive.json` + `.ckpt.json` + `.keys.npy` (77 MB) + `.log`; committed copy of the JSON at `arm-e` `29af7d7` |

## Post-hoc audit runs (2026-07-28)
| run | verdict | durable location |
|---|---|---|
| **second-seed noise probe** (seed **20260729**, else identical) | 🔴 **VERDICT-CHANGING.** champion `,qkbw'juzxastgphnieromdfc.v-yl`, ev −2.677732, **ms/char 267.6096**, n_ood 9/14, 10,084,782 unique evals, rc=0. Objectives 0.46% apart but **9.4293 ms/char apart** (2/30 shared positions) ⇒ arm E's search spread is ~9.43, so every per-pair Δ was quoted against the paired TIMING floor (0.4964), the wrong ruler. Arm-level conclusion survives; specific gap sizes do not. | `runs/arm-archive-seed2.json` + `.ckpt.json` + `.keys.npy` + `.log`; sentinel `seed2-rc.txt` = **0** |
| **fixed-`price_many` re-score** | 🟢 arm E reproduces **bit-identically** under the parent's `79cb175`: ev_clamp **−2.690225544692558** = the frozen artifact, ms/char **258.1803** unchanged, board ordering + argmin identical, worst diff 4.441e-16. Fixed version is **0/14** shape-dependent (defective: **14/14** on a 101-level grid) with worst \|price_many−price\| = **0.000e+00** | computed inline; the extracted module is scratch at `/tmp/arme-fixcheck/evidence_scorer_fixed.py` (reproducible via `git show 79cb175:src/keybo/analysis/evidence_scorer.py`) |

⚠ **`79cb175` is on branch `domain-hard`, NOT yet an ancestor of `main`** (verified with
`git merge-base --is-ancestor`). My worktree is at `cf5f731`+3, i.e. **pre-fix** — so every number in
`judgement.json` was produced by the *defective* `price_many`, and the re-score above is what
licenses citing them.

## Gates (both PASS — sentinels verified, not inferred)
| file | rc | what |
|---|---|---|
| `gate1-policy.log` / `gate1-policy.json` | `gate1-rc.txt` = **0** | 113 checks, 0 failures. `price_many` vs scalar vs arm D's frozen `ClampedCurve` on the real archive curves; mutation control + sensitivity floor |
| `gate2-engine.log` / `gate2-engine.json` | `gate2-rc.txt` = **0** | 28 checks, 0 failures. Positive controls on arm A **and** arm D vs the frozen drivers; gauges bitwise identical across workers; resume bit-exact on the COUNT; P6 = 0.000e+00 |

## Judgement
| file | rc | what |
|---|---|---|
| `judgement.json` | `report-rc.txt` = **0** | 14 layouts, 4 rulers kept separate, paired resolution, 19-gauge frame, dominance, clamp-binding, plateau census, in-band rank test, champion drivers + moves. Asserts all 15 cited top-level keys present (trap 19) |
| `report.log` | — | the printable board |

## Pre-registration & pre-run analysis
| file | what |
|---|---|
| `PREDICTION.md` | 16 predictions P1–P16 + explicit premise + 5 abort conditions. **Committed at `414f2a6` BEFORE the search ran** — provably prior |
| `prerun-arme.json` / `.log` | domain comparison vs random400 (6 of 14 fully disjoint), curve extremes, headroom split by mechanism, CLAMP-freeze census, in-band ρ |

## Drivers (`drivers/`, all committed on branch `arm-e`)
`arme_obj.py` (the objective — `ValidatedClampedEval`) · `arme_load.py` (round-trip-asserted curve
loader) · `gate1_policy.py` · `gate2_engine.py` · `prerun_arme.py` · `search_arme.py` (arm D's
engine + `--arm archive`) · `judge_arme.py` · `report_arme.py` · `run_arme.sh`

## Independent verification
`258.1803` reproduced through the shipped `keybo analyze --json` (`/tmp/arme-cli.json`, corpus
blend-v1, 90 WPM, skipgram `1-skip31.txt`), exact to 4 dp against the fast evaluator.
Set-containment asserted, **not** row count — `analyze` legitimately added its `--ref` row
(`qwertyuiopasdfghjkl;zxcvbnm,./`), so 2 requested → 3 returned, 0 missing (trap 38).

## Git
Branch **`arm-e`** in worktree `/tmp/arme` (own branch off `cf5f731`; `domain-hard` was already
checked out at `/tmp/domainfix` so it could not be reused).
`414f2a6` = pre-registration + gates (before the run) · `29af7d7` = the result + drivers.
**Not pushed, no CR** — per the brief's scope.

## Scratch (NOT durable — recorded so nothing is cited from it)
`/tmp/arme-cli.json`, `/tmp/arme-detail.txt`, `/tmp/arme-gauges.txt`, `/tmp/arme-paired.txt`,
`/tmp/arme-clusters.txt`, `/tmp/arme-ext.json`, `/tmp/arme-gate2-*` — all reproducible from the
committed drivers; every number quoted in `report.md` also lives in `judgement.json`.
