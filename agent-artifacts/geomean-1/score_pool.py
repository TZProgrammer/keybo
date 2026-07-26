"""GEOMEAN-1 step 1 — score a LARGE layout pool on the 19-gauge frame, 3 corpora.

Why a large pool and not the 7 incumbents: the correlation structure of the gauge frame
is a property of the *gauge space*, and 7 points cannot estimate a 19x19 rank-correlation
matrix (and 4 of the 7 are near-duplicates of one another by construction). The pool is
the NSGA-II 12-axis Pareto archive plus the known candidates
(`artifacts/frontier_map.json`: 2860 + 45 = 2865 unique C30M layouts), plus a block of
uniformly random C30M permutations so the matrix is not conditioned only on
already-optimized layouts (an optimized-only sample understates correlation among the
axes the optimizer traded off, and overstates it among the ones it did not touch).

FRAME, stated once (trap 13 — never stitch a row across two conventions):
  * 15 corpus-sensitive gauges: KmStats' 11 + oxey.pattern_shares' scissor/imbalance +
    oxey.fitness + comfort/full-bigram-mass. Identical code path to
    `keybo analyze --json`'s `gauges` block and to noanchor-1's board driver, which is
    why my qwerty30m positive control reproduces `board_three_corpora.json` to <=5e-6.
  * 4 corpus-invariant gauges: genkey, oxeylyzer1, oxeylyzer2, wfd. Each scores on its
    own vendored corpus and takes no corpus argument, so they are computed ONCE.
  * wfd convention: `o2.wfd` (the COMPONENTS wfd, via the validating `_dof_arrays`). This is
    the only CORRECT wfd. There is no second convention to choose between: the sibling
    `wfd-frames` agent found, and I verified independently, that
    `Oxeylyzer2.wfd_apostrophe_pinned` (community.py:205) is a BUG, not a frame -- it
    hand-rolls its index arrays, bypasses `_dof_arrays`' permutation check, and never assigns
    `;` a position, so `;` keeps its `np.zeros` default and lands on dof 0 (top-left, left
    pinky), EVICTING the character that belongs there while the vacated `'` dof is refilled by
    index 0. My own check on keybo-lsb: the effective 31-slot board is
    ";yuo,vgdnlhiea.cstrm'kj-zqfwbxq" -- `q` on TWO keys, `p` absent, 30 distinct chars of 31,
    `dof_of_char` not a permutation (dof 0 doubly occupied, dof 25 empty).
    The buggy value is still recorded as `wfd_legacy_nonperm` so the size of the error is
    auditable, but it is NOT an input to any aggregate and no row mixes the two.
    Upstream root cause: noanchor-1/drivers/oxey_ports.py:255-264 `perm_arrays`. Poisoned
    artifacts (do not reuse their wfd): wscissor-allgauge.json, hunt-*-norm.json,
    wider-dominance-*, closure3-*, gen-on-blend/*, wscissor-armb-1/*. CLEAN:
    noanchor-1/board_three_corpora.json (verified: its `corpus_invariant.*.wfd` equals this
    column exactly for all 6 incumbents).

Every layout in the pool is C30M (verified), so every cell is scorable; no N/A handling.
MODELED/gauge only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

CLONE = Path("/local/home/zegertho/repos/keybo")
NOANCHOR_CLONE = Path("/local/home/zegertho/agent/state/noanchor/keybo")
ART = Path("/local/home/zegertho/agent/state/keybo-optimization/artifacts")
OUT = Path("/local/home/zegertho/agent/state/geomean/artifacts/geomean-1")
sys.path.insert(0, str(CLONE / "src"))

C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"

#: The 15 corpus-sensitive gauges, in the frozen board's order.
SENSITIVE = (
    "sfr", "sfb", "sfs", "sfb-dist", "sfs-dist", "lsb", "lsb-dist",
    "alt", "roll", "sr-roll", "redir", "scissor", "imbalance", "oxey-style", "comfort",
)
#: The 4 corpus-invariant gauges.
INVARIANT = ("genkey", "oxey1", "oxey2", "wfd")

#: corpus label -> directory. iWeb + blend-v1 live in the main clone; blend-v1-no-anchor
#: exists ONLY in the noanchor state clone (verified md5 876ae3c3... on its trigrams).
CORPORA = {
    "iweb": CLONE / "data" / "corpus",
    "blend": CLONE / "data" / "corpus" / "blend-v1",
    "noanchor": NOANCHOR_CLONE / "data" / "corpus" / "blend-v1-no-anchor",
}

#: The campaign's 7 board layouts (board_three_corpora.json `layouts`), verbatim.
BOARD_LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "qwerty": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}

_STATE: dict = {}


def _init(corpus_dir: str) -> None:
    """Per-worker one-time setup: load the corpus and build the scorers."""
    from keybo.analysis.kmstats import KmStats
    from keybo.data.corpus import load_frequencies
    from keybo.scoring.comfort import ComfortBigramScorer
    from keybo.scoring.oxey import OxeyStyleScorer

    d = Path(corpus_dir)
    bi = load_frequencies(str(d / "bigrams.txt"))
    sk = load_frequencies(str(d / "1-skip31.txt"))
    tri = load_frequencies(str(d / "trigrams.txt"))
    _STATE["km"] = KmStats(bi, sk, tri)
    _STATE["oxey"] = OxeyStyleScorer(bi, sk, tri)
    _STATE["comfort"] = ComfortBigramScorer(bi, skipgram_freqs=sk)
    _STATE["bigram_mass"] = sum(bi.values())


def _score_sensitive(lay: str) -> list[float]:
    from keybo.geometry import ROW_STAGGERED_30
    from keybo.layout import Layout

    layout = Layout(lay, ROW_STAGGERED_30)
    g = dict(_STATE["km"].stats(lay))
    shares = _STATE["oxey"].pattern_shares(layout)
    g["scissor"] = shares["scissor"]
    g["imbalance"] = shares["imbalance"]
    g["oxey-style"] = _STATE["oxey"].fitness(layout)
    g["comfort"] = _STATE["comfort"].fitness(layout) / _STATE["bigram_mass"]
    return [g[k] for k in SENSITIVE]


def _pool_layouts() -> tuple[list[str], dict[str, str]]:
    """(pool layout strings, name -> layout for the named board layouts)."""
    import random

    fm = json.loads((ART / "frontier_map.json").read_text())
    seen: dict[str, None] = {}
    for entry in fm["archive"] + fm["known_candidates"]:
        seen.setdefault(entry["layout"], None)
    archive = list(seen)
    assert all(set(x) == set(C30M) for x in archive), "non-C30M layout in the archive"

    # A random block so the matrix is not conditioned only on optimized layouts.
    rnd = random.Random(20260726)
    rand = []
    have = set(archive)
    while len(rand) < 1500:
        cand = "".join(rnd.sample(C30M, 30))
        if cand not in have:
            have.add(cand)
            rand.append(cand)

    named = dict(BOARD_LAYOUTS)
    pool = archive + rand + [v for v in named.values() if v not in have]
    return pool, named


def main() -> int:
    import multiprocessing as mp

    from keybo.analysis.community import community_suite, pinned_char

    pool_layouts, named = _pool_layouts()
    n_arch = len(json.loads((ART / "frontier_map.json").read_text())["archive"])
    print(f"pool: {len(pool_layouts)} layouts (archive-derived + 1500 random + named)", flush=True)

    # ---- corpus-INVARIANT gauges, computed once.
    gk, v1, o2 = community_suite(pinned_char(C30M))

    # GUARD: the wfd bug this driver was corrected for was invisible because the buggy path
    # boarded a NON-PERMUTATION and still returned a plausible number. Assert the property
    # directly on the board the correct path builds, so a regression cannot pass silently.
    #
    # Kept as an inline assert on purpose. The sibling `wfd-frames` agent shipped a better
    # version of this same predicate as a PUBLIC `keybo.analysis.community.check_dof_permutation`
    # (it reports both halves of the damage: keys with no character, keys with more than one) --
    # but that lives on their local, UNPUSHED branch and is NOT in `main`, so importing it here
    # would make this driver fail to run against the committed tree. Switch to the import once it
    # lands on main. Their other correction is adopted: the guard belongs on the CORRECT path and
    # on new code, NEVER inside `wfd_legacy_board()` -- asserting there would make every frozen
    # artifact's wfd unreproducible, which is that method's entire reason to exist.
    from keybo.analysis.community import _dof_arrays  # the validating path

    _probe = BOARD_LAYOUTS["keybo-lsb"]
    _char_at, _dof_of = _dof_arrays(_probe, list(o2.chars))
    assert sorted(_dof_of.tolist()) == list(range(len(o2.chars))), (
        "wfd board is not a permutation — the o2.wfd path is no longer safe"
    )
    print("wfd permutation guard: OK (correct path boards a true 31-slot permutation)", flush=True)
    inv: dict[str, dict[str, float]] = {}
    for i, lay in enumerate(pool_layouts):
        inv[lay] = {
            "genkey": float(gk.score(lay)),
            "oxey1": float(v1.score(lay)),
            "oxey2": float(o2.score(lay)),
            "wfd": float(o2.wfd(lay)),
            "wfd_legacy_nonperm": float(o2.wfd_apostrophe_pinned(lay)),
        }
        if i % 1000 == 0:
            print(f"  invariant {i}/{len(pool_layouts)}", flush=True)
    print("invariant gauges done", flush=True)

    sens: dict[str, dict[str, list[float]]] = {}
    nproc = min(48, (os.cpu_count() or 8) - 2)
    for corpus, directory in CORPORA.items():
        assert (directory / "bigrams.txt").exists(), f"missing corpus {directory}"
        with mp.Pool(nproc, initializer=_init, initargs=(str(directory),)) as p:
            rows = p.map(_score_sensitive, pool_layouts, chunksize=8)
        sens[corpus] = dict(zip(pool_layouts, rows, strict=True))
        print(f"corpus {corpus}: {len(rows)} layouts scored", flush=True)

    payload = {
        "purpose": "GEOMEAN-1: 19-gauge frame over a large layout pool, 3 corpora",
        "frame": {
            "sensitive": list(SENSITIVE),
            "invariant": list(INVARIANT),
            "wfd_convention": "o2.wfd (components wfd, via the validating _dof_arrays) — the only CORRECT one",
            "wfd_rejected_column": (
                "wfd_legacy_nonperm (o2.wfd_apostrophe_pinned) is a BUG, not a convention: it "
                "boards a NON-PERMUTATION (';' unassigned -> dof 0, evicting a char; one char on "
                "two keys). Recorded for auditability; NOT an aggregate input; never mixed into a row."
            ),
            "comfort_denominator": "FULL corpus bigram mass (board_three_corpora convention)",
            "scissor_denominator": "oxey.pattern_shares — layout-restricted bigram mass",
        },
        "corpora": {k: str(v) for k, v in CORPORA.items()},
        "pool": {
            "n_total": len(pool_layouts),
            "n_archive_derived": n_arch + 45,
            "n_random": 1500,
            "random_seed": 20260726,
        },
        "named": named,
        "invariant": inv,
        "sensitive": sens,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pool_gauges.json").write_text(json.dumps(payload) + "\n")
    print(f"wrote {OUT / 'pool_gauges.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
