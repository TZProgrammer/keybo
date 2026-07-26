"""The campaign's TWO wfd conventions, both pinned against the board that uses each.

`wfd` names two different numbers in this campaign's artifacts, differing by which
character sits on the 31st (quote) key. They disagree by ~1-7%, which is exactly the size
of a plausible real movement — so a comparison stitched across the two frames looks like a
finding. Both are pinned here, each against the frozen board that produced it, so neither
can drift into the other.
"""

from __future__ import annotations

import pytest

from keybo.analysis.community import community_suite

C30M = "qwertyuiopasdfghjkl'zxcvbnm,.-"

#: FROZEN: wscissor-allgauge.json -> invariant_direction_derivation.reference_scores[*].wfd
#: and flagship-compare's board wfd axis. Produced by `oxey_ports.O2Port.wfd`, which pins
#: ``'`` on the quote slot unconditionally.
BOARD_WFD = {
    "keybo-lsb": -15082741528300,
    "lsb-sib": -18677349618200,
    "archive-1843": -20397087463100,
    "archive-1846": -17308029826700,
    "keybo-lsb+lm": -15079957839700,
    "qwerty": -65690928179200,
}

#: FROZEN: board-blend-reselect.json -> corpus_invariant[*].wfd. Produced by
#: `Oxeylyzer2.components()['wfd']`, which pins the layout's own `pinned_char`.
COMPONENTS_WFD = {
    "flagship-c3": -17469561624900,
}

LAYOUTS = {
    "keybo-lsb": "pyuo,vgdnlhiea.cstrmkj-z'fwbxq",
    "lsb-sib": "fyou,vgdnlheaikcstrmzj'.-pwbxq",
    "archive-1843": "pyou,vgdnmheai.cstlrjz'k-fwbxq",
    "archive-1846": "pyou,vgdnmheai.cstrlkq'z-fbwjx",
    "keybo-lsb+lm": "pyuo,vgdnmhiea.cstrlkj-z'fwbxq",
    "qwerty": C30M,
    "flagship-c3": "pyou'vgdnmheai.cstrlkjz,-wfbxq",
}


@pytest.fixture(scope="module")
def oxey2():
    _genkey, _v1, o2 = community_suite(";")
    return o2


@pytest.mark.parametrize(("label", "expected"), sorted(BOARD_WFD.items()))
def test_apostrophe_pinned_wfd_reproduces_the_dominance_boards(oxey2, label, expected):
    assert oxey2.wfd_apostrophe_pinned(LAYOUTS[label]) == expected


@pytest.mark.parametrize(("label", "expected"), sorted(COMPONENTS_WFD.items()))
def test_components_wfd_reproduces_the_reselect_board(oxey2, label, expected):
    assert oxey2.wfd(LAYOUTS[label]) == expected


def test_the_two_frames_genuinely_disagree(oxey2):
    """If these ever coincide, the labelling below has become pointless — say so loudly."""
    for label, lay30 in LAYOUTS.items():
        components = oxey2.wfd(lay30)
        board = oxey2.wfd_apostrophe_pinned(lay30)
        assert components != board, f"{label}: the two wfd frames unexpectedly agree"
        # and the gap is big enough to be mistaken for a real movement
        assert abs(components - board) / abs(components) > 0.0008, label


def test_apostrophe_pinned_wfd_refuses_a_layout_without_an_apostrophe(oxey2):
    classic = "qwertyuiopasdfghjkl;zxcvbnm,./"
    with pytest.raises(ValueError, match="needs '"):
        oxey2.wfd_apostrophe_pinned(classic)
