"""REHUNT — the CORRECTED wfd accessor for the campaign drivers, plus its validation.

WHY THIS FILE EXISTS
--------------------
The campaign's ``oxey_ports.perm_arrays`` hand-rolls the character->key index arrays and so
bypasses ``keybo.analysis.community._dof_arrays``'s permutation validation. For a C30M layout
it never assigns ``;`` a position, so ``;`` keeps its ``np.zeros`` default and lands on dof 0
(left pinky); the character genuinely on slot 0 is evicted and another is duplicated. Every
frozen ``best_axes.wfd`` in every hunt artifact is that number, taken on a board that cannot
exist. See ``community.Oxeylyzer2.wfd_legacy_board``.

WHAT THIS PROVIDES
------------------
``wfd_corrected(lay30)``  — the only correct wfd. Routes the index construction through the
    VALIDATED ``community._dof_arrays`` (which calls the public ``check_dof_permutation``), so
    a malformed board raises instead of scoring. Computes only the same-finger term, so it is
    not paying for ``components()``'s stretch term in an SA inner loop.
``wfd_legacy(lay30)``     — ``community.Oxeylyzer2.wfd_legacy_board``: the frozen-artifact
    number, kept ONLY as the positive control that we are reproducing the campaign's frame
    before we correct it. Never used to rank or gate.
``assert_c30m_permutation(lay30)`` — cheap board-level guard for the search loop's reported
    layouts (the dof-level guard lives in ``_dof_arrays``).

Trap 28 discipline: this does NOT reimplement the constructor. ``_dof_arrays`` builds and
validates the arrays; this module only contracts the same-finger sum over them, and
``test_wfd_fix.py`` pins it at EXACTLY 0.0 relative error against ``Oxeylyzer2.wfd``.

MODELED/gauge only. Held-layout tau saturated at 1.0; Phase-D cancelled. No realized-speed claim.
"""

from __future__ import annotations

import numpy as np

from keybo.analysis.community import (
    APOS_DOF as APOS_DOF_REF,
)
from keybo.analysis.community import (
    SLOT2DOF as SLOT2DOF_REF,
)
from keybo.analysis.community import (
    C30M_CHARS,
    N31,
    Oxeylyzer2,
    _dof_arrays,
    check_dof_permutation,
    community_suite,
)

#: the C30M board's 31 characters: the 30 movable ones plus the pinned ``;``.
C30M_CHARS31 = list(C30M_CHARS) + [";"]

__all__ = [
    "C30M_CHARS31",
    "CorrectedWfd",
    "assert_c30m_permutation",
    "check_dof_permutation",
]


def assert_c30m_permutation(lay30: str) -> str:
    """Raise unless ``lay30`` is a permutation of the 30 movable C30M characters."""
    if len(lay30) != 30 or set(lay30) != set(C30M_CHARS):
        missing = sorted(set(C30M_CHARS) - set(lay30))
        extra = sorted(set(lay30) - set(C30M_CHARS))
        raise ValueError(
            f"not a C30M permutation: {lay30!r} (len={len(lay30)}, "
            f"distinct={len(set(lay30))}, missing={missing}, extra={extra})"
        )
    return lay30


class CorrectedWfd:
    """wfd on the VALID board, plus the legacy board for the positive control only.

    ``self.o2`` is the repo's validated :class:`~keybo.analysis.community.Oxeylyzer2` over the
    C30M character universe — the same object the CLEAN artifacts
    (``board_three_corpora``, ``all-gauge-table``, ``comm-pool-board``) were computed with.
    """

    def __init__(self) -> None:
        _genkey, _oxey1, o2 = community_suite(";")
        if not isinstance(o2, Oxeylyzer2):  # pragma: no cover - contract check
            raise TypeError(f"community_suite(';') did not return an Oxeylyzer2: {type(o2)}")
        if list(o2.chars) != C30M_CHARS31:  # pragma: no cover - contract check
            raise ValueError(f"unexpected character universe: {''.join(o2.chars)!r}")
        self.o2 = o2
        # cached views so the inner loop is pure numpy fancy-indexing
        self._SFW = o2.SFW
        self._SF_I = o2.SF_I
        self._SF_J = o2.SF_J
        self._SF_D = o2.SF_D

    # -- the only correct wfd ------------------------------------------------
    def wfd(self, lay30: str) -> float:
        """Same-finger weighted-distance term on the VALID 31-key board.

        The layout's 30 characters are given, so the single left-over character (``;`` for a
        C30M board) has exactly one place to go — the quote slot. Own-pin is FORCED, not
        chosen. ``_dof_arrays`` validates the mapping, so a malformed board raises.
        """
        char_at_dof, _ = _dof_arrays(lay30, self.o2.chars)
        a = char_at_dof[self._SF_I]
        b = char_at_dof[self._SF_J]
        return float(((self._SFW[a, b] + self._SFW[b, a]) * self._SF_D).sum())

    # -- the frozen-artifact number, for the positive control ONLY -----------
    def wfd_legacy(self, lay30: str) -> float:
        """The corrupt-board number in every frozen hunt artifact. Positive control only."""
        return float(self.o2.wfd_legacy_board(lay30))

    # -- ZERO-REUSE slow reference (verification only) -----------------------
    def wfd_slow_reference(self, lay30: str) -> float:
        """wfd from a FRESH Oxeylyzer2 via an explicit Python loop — zero fast-path reuse.

        Shares no cached array, no vectorized contraction and no object with :meth:`wfd`: it
        builds its own scorer, its own ``{char: dof}`` map (validated), and sums the
        same-finger pairs one at a time. Used only to verify a REPORTED layout; prior campaign
        rounds required max relative error EXACTLY 0.0 on this comparison.
        """
        assert_c30m_permutation(lay30)
        fresh = Oxeylyzer2(list(C30M_CHARS31))
        index = {c: k for k, c in enumerate(fresh.chars)}
        dof_of_char = np.full(N31, -1, dtype=np.int64)
        for slot, character in enumerate(lay30):
            dof_of_char[index[character]] = SLOT2DOF_REF[slot]
        leftover = [c for c in fresh.chars if c not in set(lay30)]
        if len(leftover) != 1:  # pragma: no cover - contract check
            raise ValueError(f"expected exactly one left-over character, got {leftover}")
        dof_of_char[index[leftover[0]]] = APOS_DOF_REF
        check_dof_permutation(dof_of_char)
        char_at_dof = [None] * N31
        for char_index, dof in enumerate(dof_of_char.tolist()):
            if char_at_dof[dof] is not None:  # pragma: no cover - guard above makes it dead
                raise ValueError(f"two characters on dof {dof}")
            char_at_dof[dof] = char_index
        total = 0
        for i, j, d in zip(
            fresh.SF_I.tolist(), fresh.SF_J.tolist(), fresh.SF_D.tolist(), strict=True
        ):
            a, b = char_at_dof[i], char_at_dof[j]
            total += (int(fresh.SFW[a, b]) + int(fresh.SFW[b, a])) * int(d)
        return float(total)

    # -- diagnostics ---------------------------------------------------------
    def dof_of_char(self, lay30: str) -> np.ndarray:
        _char_at_dof, dof = _dof_arrays(lay30, self.o2.chars)
        return check_dof_permutation(dof)

    def board_of(self, lay30: str) -> str:
        """The valid 31-key board as a string, indexed by dof."""
        char_at_dof, _ = _dof_arrays(lay30, self.o2.chars)
        if len(char_at_dof) != N31:  # pragma: no cover - contract check
            raise ValueError("char_at_dof is not 31 long")
        return "".join(self.o2.chars[i] for i in char_at_dof)
