"""The legacy `wfd` board is CORRUPT, not a second convention — characterised exactly.

`wfd_apostrophe_pinned` was documented as "the other wfd convention": ``'`` pinned on the
quote slot instead of the layout's own quote character. It is not a convention. It evaluates
a board that is not a permutation of the 31 keys: ``;`` is never assigned a position, so it
keeps its ``np.zeros`` default and lands on **dof 0** (top-left, left pinky), evicting the
character that genuinely sits there; the dof that ``'`` vacated is then filled by index 0
(``q``), so ``q`` is typed on two keys and one letter is absent from the board entirely.

These tests pin the corruption itself, so the legacy number stays reproducible (the frozen
dominance boards were computed on it) while nothing can mistake it for a modelling choice.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.analysis.community import (
    APOS_DOF,
    N31,
    SLOT2DOF,
    check_dof_permutation,
    community_suite,
    legacy_board_of,
)

C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"
CLASSIC = "qwertyuiopasdfghjkl;zxcvbnm,./"

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
    "graphite": "bldwz'foujnrtsgyhaeixqmcvkp,.-",
    "qwerty30m": C30M,
}


@pytest.fixture(scope="module")
def oxey2():
    _genkey, _v1, o2 = community_suite(";")
    return o2


# --- the corruption, stated exactly -------------------------------------------------------


@pytest.mark.parametrize("label", sorted(set(LAYOUTS) - {"qwerty30m"}))
def test_the_legacy_board_is_not_a_permutation(label):
    """8 of the 9 campaign layouts: a letter is deleted and ``q`` is duplicated."""
    board = legacy_board_of(LAYOUTS[label])
    assert len(board) == N31
    assert len(set(board)) < N31, "expected a duplicated character"
    assert board.count("q") == 2, "the dof ' vacated is refilled by index 0 == 'q'"
    assert board[0] == ";", "; is never assigned a dof, so it lands on dof 0"
    assert LAYOUTS[label][0] not in board, "the character on slot 0 is evicted entirely"


def test_qwerty_is_the_one_layout_the_bug_barely_touches(oxey2):
    """qwerty's slot-0 character IS ``q``, so its legacy board degenerates to a valid
    permutation — which is why the legacy/correct gap is 0.08% for qwerty and 1-7%
    everywhere else, and why a qwerty-referenced direction check could not catch this."""
    board = legacy_board_of(C30M)
    assert len(set(board)) == N31, "qwerty's legacy board IS a permutation (still wrong)"
    assert board[0] == ";"
    gap = abs(oxey2.wfd_legacy_board(C30M) - oxey2.wfd(C30M)) / abs(oxey2.wfd(C30M))
    assert gap < 0.001, "qwerty's gap is ~0.08% — the bug hides behind the reference layout"
    worst = max(
        abs(oxey2.wfd_legacy_board(lay) - oxey2.wfd(lay)) / abs(oxey2.wfd(lay))
        for lab, lay in LAYOUTS.items()
        if lab != "qwerty30m"
    )
    assert worst > 0.01, "every other layout moves by >1%"


def test_the_legacy_board_is_a_bug_in_every_regime_it_accepts(oxey2):
    """The guard admits only C30M layouts, and C30M is exactly the corrupt regime: the
    hand-rolled mapping is only sound when the pinned character IS ``'``."""
    with pytest.raises(ValueError, match="needs '"):
        oxey2.wfd_legacy_board(CLASSIC)


# --- the legacy number stays reproducible (the frozen boards were computed on it) ---------

#: FROZEN: wscissor-allgauge.json -> invariant_direction_derivation.reference_scores[*].wfd,
#: flagship-compare's board wfd axis, and every hunt's `best_axes.wfd`. Produced by the
#: campaign's `oxey_ports.perm_arrays`, whose bug this method reproduces bit-for-bit.
LEGACY_BOARD_WFD = {
    "keybo-lsb": -15082741528300,
    "lsb-sib": -18677349618200,
    "archive-1843": -20397087463100,
    "archive-1846": -17308029826700,
    "keybo-lsb+lm": -15079957839700,
    "qwerty30m": -65690928179200,
}

#: FROZEN: board-blend-reselect.json / board_three_corpora.json -> corpus_invariant[*].wfd.
#: The CORRECT board: the layout's own `pinned_char` on the quote slot.
CORRECT_WFD = {
    "keybo-lsb": -16213995653000,
    "keybo-lsb+lm": -16198743046400,
    "lsb-sib": -17974982692100,
    "archive-1843": -20928900614900,
    "archive-1846": -18252238492800,
    "flagship-c3": -17469561624900,
    "qwerty30m": -65746277057400,
}


@pytest.mark.parametrize(("label", "expected"), sorted(LEGACY_BOARD_WFD.items()))
def test_legacy_board_wfd_still_reproduces_the_frozen_dominance_boards(oxey2, label, expected):
    assert oxey2.wfd_legacy_board(LAYOUTS[label]) == expected


@pytest.mark.parametrize(("label", "expected"), sorted(CORRECT_WFD.items()))
def test_correct_wfd_reproduces_the_boards_that_used_the_valid_permutation(oxey2, label, expected):
    assert oxey2.wfd(LAYOUTS[label]) == expected


# --- the delta decomposes EXACTLY, over three dofs ----------------------------------------


@pytest.mark.parametrize("label", sorted(LAYOUTS))
def test_the_delta_decomposes_exactly_over_the_three_corrupted_dofs(oxey2, label):
    """wfd is a sum over same-finger dof pairs of (geometry x finger weight) x (char-pair
    weight), so it is exactly additive over dof pairs. The legacy/correct delta therefore
    attributes, to the last integer, to pairs touching the three dofs the bug disturbs —
    and pairs touching none of them contribute exactly zero."""
    lay30 = LAYOUTS[label]
    correct = np.array([oxey2.chars.index(c) for c in oxey2_board(oxey2, lay30)])
    legacy = np.array([oxey2.chars.index(c) for c in legacy_board_of(lay30)])
    weight = oxey2.SFW + oxey2.SFW.T
    changed = {d for d in range(N31) if correct[d] != legacy[d]}
    assert len(changed) == 3, "; onto dof0, ' onto the quote slot, q into the vacated dof"

    attributed = 0
    for i, j, dist in zip(oxey2.SF_I, oxey2.SF_J, oxey2.SF_D, strict=True):
        term = int(dist) * (int(weight[legacy[i], legacy[j]]) - int(weight[correct[i], correct[j]]))
        if i in changed or j in changed:
            attributed += term
        else:
            assert term == 0, "a pair touching no corrupted dof must contribute nothing"

    assert attributed == oxey2.wfd_legacy_board(lay30) - oxey2.wfd(lay30)


def oxey2_board(o2, lay30: str) -> str:
    """The CORRECT 31-key board: the layout's own pinned character on the quote slot."""
    index = {character: position for position, character in enumerate(o2.chars)}
    dof_of_char = np.empty(N31, dtype=np.int64)
    for slot, character in enumerate(lay30):
        dof_of_char[index[character]] = SLOT2DOF[slot]
    dof_of_char[index[o2.chars[30]]] = APOS_DOF
    char_at_dof = np.empty(N31, dtype=np.int64)
    char_at_dof[dof_of_char] = np.arange(N31)
    return "".join(o2.chars[i] for i in char_at_dof)


def test_the_two_boards_genuinely_disagree(oxey2):
    """If these ever coincide, the labelling has become pointless — say so loudly."""
    for label, lay30 in LAYOUTS.items():
        assert oxey2.wfd(lay30) != oxey2.wfd_legacy_board(lay30), label


def test_the_old_name_still_works_and_warns(oxey2):
    """`wfd_apostrophe_pinned` is the campaign-era name; keep it working (artifacts and
    sibling drivers call it) but make it announce that it is a bug, not a convention."""
    with pytest.deprecated_call(match="not a convention"):
        legacy = oxey2.wfd_apostrophe_pinned(LAYOUTS["keybo-lsb"])
    assert legacy == LEGACY_BOARD_WFD["keybo-lsb"]


# --- the guard whose absence WAS the bug --------------------------------------------------


def test_the_permutation_guard_catches_the_legacy_mapping(oxey2):
    """`check_dof_permutation` is the one line that would have prevented all of this.

    Fed the legacy construction it must name both halves of the damage: the key left with
    no character, and the key handed two.
    """
    index = {character: position for position, character in enumerate(oxey2.chars)}
    dof_of_char = np.zeros(N31, dtype=np.int64)
    for slot, character in enumerate(LAYOUTS["keybo-lsb"]):
        dof_of_char[index[character]] = SLOT2DOF[slot]
    dof_of_char[index["'"]] = APOS_DOF

    with pytest.raises(ValueError, match="not a permutation of the 31 keys") as excinfo:
        check_dof_permutation(dof_of_char)
    message = str(excinfo.value)
    assert "keys with no character [25]" in message, "the dof ' vacated"
    assert "keys with more than one [0]" in message, "; collides with the slot-0 character"


def test_the_permutation_guard_passes_the_correct_mapping(oxey2):
    """And it must not fire on a valid board, or it would break every correct call."""
    index = {character: position for position, character in enumerate(oxey2.chars)}
    dof_of_char = np.empty(N31, dtype=np.int64)
    for slot, character in enumerate(LAYOUTS["keybo-lsb"]):
        dof_of_char[index[character]] = SLOT2DOF[slot]
    dof_of_char[index[";"]] = APOS_DOF
    assert check_dof_permutation(dof_of_char) is dof_of_char


def test_the_legacy_board_does_not_assert_so_artifacts_stay_reconcilable(oxey2):
    """`wfd_legacy_board` must NOT apply the guard: its purpose is to evaluate the broken
    board, so asserting would make every frozen artifact unreconcilable. The guard belongs
    on the correct path and on new code, not here."""
    assert oxey2.wfd_legacy_board(LAYOUTS["keybo-lsb"]) == LEGACY_BOARD_WFD["keybo-lsb"]
