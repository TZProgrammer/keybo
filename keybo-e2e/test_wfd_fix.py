"""Pin the corrected wfd accessor: EXACT agreement with the validated path, and the guard bites.

Trap 1 discipline: each assertion must be able to fail. ``test_guard_bites_on_hand_rolled_map``
reconstructs the campaign's exact hand-rolled mapping and asserts ``check_dof_permutation``
rejects it — that is the check whose absence IS the bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from keybo.analysis.community import APOS_DOF, N31, SLOT2DOF, legacy_board_of

from wfd_fix import C30M_CHARS31, CorrectedWfd, assert_c30m_permutation, check_dof_permutation

# The five campaign incumbents plus qwerty30m (the layout the bug spares).
LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "qwerty30m": "qwertyuiopasdfghjkl'zxcvbnm,.-",
}


@pytest.fixture(scope="module")
def W() -> CorrectedWfd:
    return CorrectedWfd()


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_wfd_is_exactly_the_validated_component(W: CorrectedWfd, name: str) -> None:
    """Zero-reuse check: our contracted sum == Oxeylyzer2.wfd (which goes through
    components()/_dof_arrays) at EXACTLY 0.0 relative error."""
    lay = LAYOUTS[name]
    ours = W.wfd(lay)
    theirs = float(W.o2.wfd(lay))
    assert ours == theirs, f"{name}: {ours!r} != {theirs!r}"


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_board_is_a_valid_permutation(W: CorrectedWfd, name: str) -> None:
    board = W.board_of(LAYOUTS[name])
    assert len(board) == N31
    assert sorted(board) == sorted(C30M_CHARS31), f"{name}: board {board!r} is not a permutation"


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_legacy_board_differs_and_is_broken_except_qwerty(W: CorrectedWfd, name: str) -> None:
    """The legacy board is a non-permutation for every layout whose slot-0 char is not `q`."""
    lay = LAYOUTS[name]
    legacy = legacy_board_of(lay)
    is_perm = sorted(legacy) == sorted(C30M_CHARS31)
    assert is_perm == (lay[0] == "q"), f"{name}: legacy perm={is_perm} but slot0={lay[0]!r}"
    if not is_perm:
        assert W.wfd(lay) != W.wfd_legacy(lay), f"{name}: corrected == legacy on a broken board"


def test_guard_bites_on_the_campaigns_hand_rolled_map() -> None:
    """Rebuild `oxey_ports.perm_arrays` verbatim and assert the guard rejects it.

    This is the mutation test for the fix: if `check_dof_permutation` were a no-op, this
    test would pass silently, so it asserts the raise AND the message names both halves
    of the damage.
    """
    lay30 = LAYOUTS["keybo-lsb"]
    index = {c: k for k, c in enumerate(C30M_CHARS31)}
    dof_of_char = np.zeros(N31, dtype=np.int64)  # <- the bug: zeros, not empty+assign-all
    for slot, ch in enumerate(lay30):
        dof_of_char[index[ch]] = SLOT2DOF[slot]
    dof_of_char[index["'"]] = APOS_DOF
    with pytest.raises(ValueError, match="not a permutation of the 31 keys"):
        check_dof_permutation(dof_of_char)


def test_wfd_raises_on_a_non_permutation_layout(W: CorrectedWfd) -> None:
    with pytest.raises(ValueError):
        W.wfd("pyuo,vgdnlhiea.cstrmkj-z'fwbxp")  # p twice, q missing


def test_assert_c30m_permutation_bites() -> None:
    assert assert_c30m_permutation(LAYOUTS["keybo-lsb"]) == LAYOUTS["keybo-lsb"]
    with pytest.raises(ValueError, match="not a C30M permutation"):
        assert_c30m_permutation("pyuo,vgdnlhiea.cstrmkj-z'fwbxp")
    with pytest.raises(ValueError, match="not a C30M permutation"):
        assert_c30m_permutation("abc")


def test_qwerty_is_the_minimum_under_both_boards(W: CorrectedWfd) -> None:
    """Direction DERIVED, never assumed (trap 5): qwerty is worst => wfd is HIGHER-better."""
    corrected = {n: W.wfd(lay) for n, lay in LAYOUTS.items()}
    legacy = {n: W.wfd_legacy(lay) for n, lay in LAYOUTS.items()}
    assert min(corrected, key=corrected.get) == "qwerty30m"
    assert min(legacy, key=legacy.get) == "qwerty30m"
