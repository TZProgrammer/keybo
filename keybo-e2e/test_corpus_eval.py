"""GEN-ON-BLEND positive controls — must ALL pass before any verdict is trusted.

Prereg §6. The gates:
  1. iWeb ARM-A reproduces the FROZEN board / closure3-verdict incumbent axes (<1e-9).
  2. Ceilings re-derive from the reference population, and the iWeb re-derivation
     reproduces the established frozen constant (<1e-9).
  3. The fast kmstats/scissor path == the slow ground-truth path, under BOTH the keymeow
     and the corpus tabling (a fast path verified on only one tabling is not verified).
  4. The frozen wider-dominance dominators still come out 10/10 under iWeb ARM-A, and
     keybo-lsb / keybo-lsb+lm still resist. Machinery that cannot reproduce the known
     positive result cannot be trusted on a new corpus.
  5. The corpus swap actually changes the objective (catches a silent no-op swap), and
     the invariant axes really are invariant.
  6. saved_batch == saved; normfloor_batch == normfloor; non-permutations rejected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corpus_eval as CE  # noqa: E402

FROZEN_BOARD = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/replicate-gen/gauge-board.json"
)
CLOSURE3_VERDICT = Path(
    "/local/home/zegertho/agent/state/keybo-optimization/artifacts/"
    "closure3-generator/closure3-verdict.json"
)
#: The three dominators the frozen wider-dominance verdict reports, with their targets.
FROZEN_DOMINATORS = {
    "pyou,vgdnlheai.cstmrk'zj-wfbqx": "lsb-sib",
    "pyou'vgdnmheai.cstrlkjz,-wfbxq": "archive-1846",
    "uyo,.fdnsleiatkpchmrq-xg'bwvzj": "archive-1843",
}
#: The layout the frozen verdict reports as dominating THREE incumbents at once.
FROZEN_TRIPLE = "uyog.bdnsleiat,pchmrz-'kjfwvxq"
RESISTERS = ("keybo-lsb", "keybo-lsb+lm")

TOL = 1e-9


@pytest.fixture(scope="module")
def iweb_a() -> CE.ArmBoard:
    return CE.ArmBoard(corpus="iweb", arm="A")


@pytest.fixture(scope="module")
def blend_b() -> CE.ArmBoard:
    return CE.ArmBoard(corpus="blend", arm="B")


# --- gate 1: reproduce the frozen board ------------------------------------
def test_iweb_arm_a_reproduces_frozen_verdict_axes(iweb_a: CE.ArmBoard) -> None:
    """iWeb ARM-A must reproduce the frozen closure3-verdict incumbent axes on the 8
    axes the verdict stores in a corpus-comparable form, plus the RAW floor/mean."""
    with open(CLOSURE3_VERDICT) as fh:
        frozen = json.load(fh)["incumbent_axes"]
    max_err = 0.0
    worst = None
    for name, row in frozen.items():
        got = iweb_a.axes(row["layout"], floor_kind="raw")
        for axis in CE.AXES:
            err = abs(got[axis] - row[axis])
            # wfd/oxey are O(1e13); compare relatively there, absolutely elsewhere.
            scale = max(1.0, abs(row[axis]))
            rel = err / scale
            if rel > max_err:
                max_err, worst = rel, f"{name}.{axis} got={got[axis]!r} want={row[axis]!r}"
    assert max_err < TOL, f"max rel err {max_err:.3e} at {worst}"
    print(f"gate1 frozen-verdict axes: max rel|err|={max_err:.3e} OK")


def test_iweb_arm_a_reproduces_frozen_gauge_board(iweb_a: CE.ArmBoard) -> None:
    """Independent second frozen artifact: replicate-gen/gauge-board.json rows."""
    with open(FROZEN_BOARD) as fh:
        board = json.load(fh)["rows"]
    max_err = 0.0
    worst = None
    checked = 0
    for lay, row in board.items():
        if len(lay) != 30 or set(lay) != set(CE.C30M):
            continue
        got = iweb_a.axes(lay, floor_kind="raw")
        want = {
            "floor": row["six_surface"]["floor_saved_pct"],
            "mean": row["six_surface"]["mean_saved_pct"],
            "lsb": row["diagnostics"]["kmstats"]["lsb"],
            "sfb": row["diagnostics"]["kmstats"]["sfb"],
            "sfs": row["diagnostics"]["kmstats"]["sfs"],
            "scissor": row["diagnostics"]["tb_objective_axes"]["scissor"],
        }
        checked += 1
        for axis, value in want.items():
            err = abs(got[axis] - value)
            if err > max_err:
                max_err, worst = err, f"{lay}.{axis} got={got[axis]!r} want={value!r}"
    assert checked >= 5, f"only {checked} board rows checked"
    assert max_err < TOL, f"max|err| {max_err:.3e} at {worst}"
    print(f"gate1 frozen gauge-board ({checked} rows): max|err|={max_err:.3e} OK")


# --- gate 2: ceilings ------------------------------------------------------
def test_iweb_ceilings_reproduce_frozen_constant() -> None:
    """Re-deriving the ceilings on iWeb must reproduce the established frozen constant,
    proving the re-derivation is canonical and not a new convention."""
    six = CE.SixSurface("iweb")
    max_err = 0.0
    for surface, frozen in CE.FROZEN_IWEB_CEILINGS.items():
        max_err = max(max_err, abs(six.ceiling_map[surface] - frozen))
    assert max_err < TOL, f"iWeb ceilings differ from frozen: max|err|={max_err:.3e}"
    print(f"gate2 iWeb ceilings vs frozen: max|err|={max_err:.3e} OK")


def test_blend_ceilings_differ_and_are_positive() -> None:
    """Blend ceilings must be re-derived (not the iWeb constants) — reusing iWeb ceilings
    under blend weights would make the 'normalized floor' a two-corpus hybrid."""
    six = CE.SixSurface("blend")
    assert all(v > 0 for v in six.ceiling_map.values())
    diffs = [abs(six.ceiling_map[s] - CE.FROZEN_IWEB_CEILINGS[s]) for s in CE.SURFACES]
    assert max(diffs) > 1e-6, "blend ceilings identical to iWeb — corpus swap was a no-op"
    print(f"gate2 blend ceilings re-derived; max|delta vs iWeb|={max(diffs):.4f} OK")


# --- gate 3: fast path == slow path, under BOTH tablings -------------------
@pytest.mark.parametrize("corpus,arm", [("iweb", "A"), ("iweb", "B"), ("blend", "B")])
def test_kmstats_fast_matches_slow_all_tablings(corpus: str, arm: str) -> None:
    """The bilinear kmstats form must match KmStats.stats for the tabling in use.
    Verifying only the keymeow tabling would leave ARM-B's corpus tabling unverified."""
    from keybo.analysis.kmstats import KmStats

    board = CE.ArmBoard(corpus=corpus, arm=arm)
    if arm == "A":
        bi, sk, tri = CE.keymeow_tables()
    else:
        bi, sk, tri = CE.corpus_tables(corpus)
    slow = KmStats(bi, sk, tri)
    rng = np.random.default_rng(11)
    layouts = list(CE.INCUMBENTS.values()) + [
        "".join(rng.permutation(list(CE.C30M))) for _ in range(8)
    ]
    max_err = 0.0
    for lay in layouts:
        fast = CE.kmstats_fast(CE.perm_of(lay), board.km)
        want = slow.stats(lay)
        for key in ("sfb", "sfs", "lsb"):
            max_err = max(max_err, abs(fast[key] - want[key]))
    assert max_err < TOL, f"{corpus}/{arm}: kmstats fast vs slow max|err|={max_err:.3e}"
    print(f"gate3 kmstats fast==slow [{corpus}/arm-{arm}]: max|err|={max_err:.3e} OK")


@pytest.mark.parametrize("corpus", ["iweb", "blend"])
def test_scissor_fast_matches_slow(corpus: str) -> None:
    """tb_scissor_fast must match ComfortObjective.values()['scissor']."""
    board = CE.ArmBoard(corpus=corpus, arm="A")
    rng = np.random.default_rng(12)
    layouts = list(CE.INCUMBENTS.values()) + [
        "".join(rng.permutation(list(CE.C30M))) for _ in range(8)
    ]
    max_err = 0.0
    for lay in layouts:
        fast = CE.tb_scissor_fast(CE.perm_of(lay), board.comfort)
        slow = float(board.comfort.values(lay)["scissor"])
        max_err = max(max_err, abs(fast - slow))
    assert max_err < TOL, f"{corpus}: scissor fast vs slow max|err|={max_err:.3e}"
    print(f"gate3 scissor fast==slow [{corpus}]: max|err|={max_err:.3e} OK")


@pytest.mark.parametrize("corpus,arm", [("iweb", "A"), ("blend", "B")])
def test_axes_fast_matches_axes_slow(corpus: str, arm: str) -> None:
    """The whole 10-axis board: fast path vs the slow ground-truth path."""
    board = CE.ArmBoard(corpus=corpus, arm=arm)
    max_err = 0.0
    worst = None
    for lay in CE.INCUMBENTS.values():
        fast = board.axes(lay)
        slow = board.axes_slow(lay)
        for axis in CE.AXES:
            err = abs(fast[axis] - slow[axis]) / max(1.0, abs(slow[axis]))
            if err > max_err:
                max_err, worst = err, f"{lay}.{axis}"
    assert max_err < TOL, f"{corpus}/{arm}: axes fast vs slow max rel|err|={max_err:.3e} at {worst}"
    print(f"gate3 axes fast==slow [{corpus}/arm-{arm}]: max rel|err|={max_err:.3e} OK")


# --- gate 4: reproduce the frozen wider-dominance VERDICT ------------------
def test_frozen_dominators_still_dominate_under_iweb_arm_a(iweb_a: CE.ArmBoard) -> None:
    """The known-positive control: each frozen dominator must be 10/10 vs its target."""
    inc = iweb_a.incumbent_axes(floor_kind="norm")
    for lay, target in FROZEN_DOMINATORS.items():
        cand = iweb_a.axes(lay, floor_kind="norm")
        is_dom, n_ge, n_gt = CE.dominates(cand, inc[target])
        assert is_dom, f"{lay} no longer dominates {target}: n_ge={n_ge}/10 n_gt={n_gt}"
    print(f"gate4 {len(FROZEN_DOMINATORS)} frozen dominators reproduce 10/10 OK")


def test_frozen_triple_dominates_three(iweb_a: CE.ArmBoard) -> None:
    """The frozen verdict's strongest positive: one layout dominating three incumbents."""
    inc = iweb_a.incumbent_axes(floor_kind="norm")
    cand = iweb_a.axes(FROZEN_TRIPLE, floor_kind="norm")
    beaten = {n for n in inc if CE.dominates(cand, inc[n])[0]}
    assert beaten == {"lsb-sib", "archive-1843", "archive-1846"}, f"got {beaten}"
    print(f"gate4 frozen triple dominates {sorted(beaten)} OK")


def test_resisters_are_not_dominated_by_frozen_candidates(iweb_a: CE.ArmBoard) -> None:
    """keybo-lsb / keybo-lsb+lm must still resist every frozen candidate."""
    inc = iweb_a.incumbent_axes(floor_kind="norm")
    for lay in list(FROZEN_DOMINATORS) + [FROZEN_TRIPLE]:
        cand = iweb_a.axes(lay, floor_kind="norm")
        for resister in RESISTERS:
            is_dom, n_ge, _ = CE.dominates(cand, inc[resister])
            assert not is_dom, f"{lay} unexpectedly dominates {resister}"
            assert n_ge <= 8, f"{lay} reaches n_ge={n_ge}/10 vs {resister} (frozen max is 8)"
    print("gate4 resisters still resist all frozen candidates (n_ge<=8) OK")


def test_incumbents_are_mutually_nondominated(iweb_a: CE.ArmBoard) -> None:
    """The structural pre-search fact the frozen report rests on."""
    inc = iweb_a.incumbent_axes(floor_kind="norm")
    for a in inc:
        for b in inc:
            if a != b:
                assert not CE.dominates(inc[a], inc[b])[0], f"{a} dominates {b}"
    print("gate4 the 5 incumbents are mutually non-dominated OK")


# --- gate 5: the corpus swap is real; invariant axes are invariant ---------
def test_corpus_swap_actually_changes_the_objective(
    iweb_a: CE.ArmBoard, blend_b: CE.ArmBoard
) -> None:
    """Catches a silent no-op swap (loading the same table twice)."""
    moved = {}
    for name, lay in CE.INCUMBENTS.items():
        a = iweb_a.axes(lay)
        b = blend_b.axes(lay)
        moved[name] = {ax: abs(a[ax] - b[ax]) for ax in CE.AXES}
    for axis in ("floor", "mean", "scissor", "sfb", "sfs", "lsb"):
        biggest = max(moved[n][axis] for n in moved)
        assert biggest > 1e-6, f"axis {axis} did not move under the corpus swap"
    print("gate5 corpus swap moves floor/mean/scissor/lsb/sfb/sfs OK")


def test_invariant_axes_are_invariant(iweb_a: CE.ArmBoard, blend_b: CE.ArmBoard) -> None:
    """wfd/genkey/oxey1/oxey2 CANNOT move — they take no corpus argument (prereg §3).
    This is a structural claim the whole two-arm design rests on, so it is asserted."""
    max_err = 0.0
    for lay in CE.INCUMBENTS.values():
        a = iweb_a.axes(lay)
        b = blend_b.axes(lay)
        for axis in CE.INVARIANT_AXES:
            max_err = max(max_err, abs(a[axis] - b[axis]))
    assert max_err == 0.0, f"an 'invariant' axis moved: max|err|={max_err:.3e}"
    print("gate5 wfd/genkey/oxey1/oxey2 bit-identical across corpora OK")


def test_arm_a_and_arm_b_differ_only_in_kmstats(blend_b: CE.ArmBoard) -> None:
    """ARM-A vs ARM-B on the SAME corpus must differ on exactly lsb/sfb/sfs."""
    blend_a = CE.ArmBoard(corpus="blend", arm="A")
    for lay in CE.INCUMBENTS.values():
        a = blend_a.axes(lay)
        b = blend_b.axes(lay)
        for axis in ("floor", "mean", "scissor", *CE.INVARIANT_AXES):
            assert a[axis] == b[axis], f"{axis} differs between arms on the same corpus"
        assert any(abs(a[ax] - b[ax]) > 1e-6 for ax in ("lsb", "sfb", "sfs"))
    print("gate5 arms differ on exactly lsb/sfb/sfs OK")


# --- gate 6: vectorization + input validation ------------------------------
@pytest.mark.parametrize("corpus", ["iweb", "blend"])
def test_saved_batch_matches_scalar(corpus: str) -> None:
    six = CE.SixSurface(corpus)
    rng = np.random.default_rng(7)
    perms = np.array([CE.perm_of("".join(rng.permutation(list(CE.C30M)))) for _ in range(24)])
    batch = six.saved_batch(perms)
    scalar = np.array([six.saved(p) for p in perms])
    err = float(np.max(np.abs(batch - scalar)))
    assert err < 1e-11, f"{corpus}: saved_batch != saved, max|err|={err:.3e}"
    nf_batch = six.normfloor_batch(perms)
    nf_scalar = np.array([six.normfloor(p) for p in perms])
    nf_err = float(np.max(np.abs(nf_batch - nf_scalar)))
    assert nf_err < 1e-11, f"{corpus}: normfloor_batch != normfloor, max|err|={nf_err:.3e}"
    print(f"gate6 saved/normfloor batch==scalar [{corpus}]: {err:.2e}/{nf_err:.2e} OK")


def test_evaluate_batch_matches_axes(blend_b: CE.ArmBoard) -> None:
    """The EA's 6 in-loop objectives must equal the board's own axes."""
    rng = np.random.default_rng(19)
    layouts = ["".join(rng.permutation(list(CE.C30M))) for _ in range(12)]
    movables = np.array([CE.perm_of(lay)[:30] for lay in layouts])
    objs = blend_b.evaluate_batch(movables)
    max_err = 0.0
    for i, lay in enumerate(layouts):
        ax = blend_b.axes(lay, floor_kind="norm")
        want = [-ax["floor"], -ax["mean"], ax["scissor"], ax["lsb"], ax["sfb"], ax["sfs"]]
        max_err = max(max_err, float(np.max(np.abs(objs[i] - np.array(want)))))
    assert max_err < 1e-10, f"evaluate_batch != axes, max|err|={max_err:.3e}"
    print(f"gate6 evaluate_batch==axes: max|err|={max_err:.3e} OK")


def test_non_permutation_rejected() -> None:
    for bad in ("qwerty", CE.C30M[:-1] + "q", CE.C30M + "x"):
        with pytest.raises(ValueError):
            CE.perm_of(bad)
    print("gate6 non-permutations rejected OK")


def test_space_is_pinned() -> None:
    """Space must always land on slot 30 (the pinned slot)."""
    rng = np.random.default_rng(23)
    for _ in range(5):
        perm = CE.perm_of("".join(rng.permutation(list(CE.C30M))))
        assert perm[30] == CE.SPACE
        assert sorted(perm[:30].tolist()) == list(range(30))
    print("gate6 space pinned at slot 30 OK")


def test_movable_to_layout_roundtrip() -> None:
    rng = np.random.default_rng(29)
    for _ in range(20):
        lay = "".join(rng.permutation(list(CE.C30M)))
        assert CE.movable_to_layout(CE.perm_of(lay)[:30]) == lay
    print("gate6 movable_to_layout roundtrip OK")
