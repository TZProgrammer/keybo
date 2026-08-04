"""GATEFOLDS-1 — PACE INVARIANCE is a per-frame property, and it is what the high-wpm gate sees.

The mechanism this file pins: for a fixed position pair, ``interp.1`` and ``hybrid-B`` produce a
feature vector that is **bit-identical across the whole wpm range**, while the served frame and the
11-column ``interp-wpm`` variant do not. That matters because the high-wpm non-regression gate
scores a **within-bucket Spearman** (``keybo.training.validate._per_bucket_rho``): a frame whose
vector cannot vary with pace emits ONE fixed ranking of pairs and replays it against every bucket's
own observed ordering, so it cannot track any cross-pace re-ordering. The gate's high-wpm refusal
of both frames is a statement about that, and NOT about interpretability, ordinals, or resolution.

⚠ EVERY assertion here is exercised at BOTH polarities. An invariance test that only ever ran on
invariant frames would pass just as well against a featurizer that ignored ``wpm`` entirely — the
subject would be unable to vary, and the assertion would be vacuous (GATEFOLDS-1 invariant 5). So
each test is parametrized over the invariant frames AND the pace-adaptive ones, asserting opposite
outcomes.

⚠ AND THE RANK CHECK IS THE HONEST ONE. ``TypingModel.to_ms`` multiplies a LOGRAT prediction by
``12000/wpm``, so predicted MILLISECONDS differ across buckets even for a wpm-invariant frame. A
naive value comparison would therefore report "not invariant" and miss the mechanism completely.
Spearman is invariant to that positive monotone rescale — which is precisely why the gate cannot
see the pace factor either.
"""

import numpy as np
import pytest

from keybo.features import BIGRAM_FEATURE_NAMES, bigram_features_from_positions
from keybo.features.ngram import REPLACEMENT_FRAME_FLAGS, replacement_frame
from keybo.geometry import ROW_STAGGERED_31

G = ROW_STAGGERED_31

#: The bucket MIDPOINTS ``validate()`` feeds the model at the registered config
#: (wpm_lo=40, wpm_hi=140, bucket_width=20) — ``Cell.wpm`` is ``bucket + bucket_width / 2``.
BUCKET_MIDPOINTS = (50.0, 70.0, 90.0, 110.0, 130.0)

#: Frames with NO ``wpm`` column: the vector cannot depend on pace.
PACE_INVARIANT_FLAGS = (True, "hybridb")
#: Frames that DO carry a ``wpm`` column, so the vector moves with pace.
PACE_ADAPTIVE_FLAGS = ("wpm",)

# A spread of pairs: cross-hand, same-hand-different-finger, same-finger, and a row-skip.
PAIRS = (
    ((1, 1), (-3, 0)),
    ((1, 1), (3, 0)),
    ((2, 1), (2, -1)),
    ((6, 1), (1, -1)),
    ((-1, 0), (-1, 0)),
)


def _vectors_across_pace(build, pair):
    """The frame's vector for ``pair`` at every bucket midpoint, as one 2-D array."""
    return np.vstack(
        [np.asarray(build(G, pair, wpm=m), dtype=np.float64) for m in BUCKET_MIDPOINTS]
    )


def _served(g, positions, wpm):
    return bigram_features_from_positions(g, positions, wpm=wpm)


def _replacement(flag):
    build = replacement_frame(flag)[0]

    def f(g, positions, wpm):
        return build(g, positions, wpm=wpm)

    return f


# --- the property itself, at BOTH polarities --------------------------------------------


@pytest.mark.parametrize("flag", PACE_INVARIANT_FLAGS)
@pytest.mark.parametrize("pair", PAIRS)
def test_wpm_free_frames_are_bit_identical_across_the_whole_pace_range(flag, pair):
    """interp.1 / hybrid-B: the vector does not move at all between 50 and 130 wpm."""
    mats = _vectors_across_pace(_replacement(flag), pair)
    spread = mats.max(axis=0) - mats.min(axis=0)
    assert not spread.any(), (
        f"{flag!r} is supposed to carry no pace channel, but columns "
        f"{sorted(np.nonzero(spread)[0].tolist())} vary across {BUCKET_MIDPOINTS}"
    )


@pytest.mark.parametrize("pair", PAIRS)
def test_the_served_frame_does_vary_with_pace(pair):
    """The OPPOSITE polarity — without this the invariance test above is vacuous."""
    mats = _vectors_across_pace(_served, pair)
    spread = mats.max(axis=0) - mats.min(axis=0)
    assert spread.any(), "the served frame must carry pace; it appears not to"
    # and it must be the wpm column doing it, not something else
    assert np.nonzero(spread)[0].tolist() == [BIGRAM_FEATURE_NAMES.index("wpm")]


@pytest.mark.parametrize("flag", PACE_ADAPTIVE_FLAGS)
@pytest.mark.parametrize("pair", PAIRS)
def test_the_interp_wpm_variant_does_vary_with_pace(flag, pair):
    """interp-wpm restored the column, so its vector MUST move — the second polarity."""
    names = list(replacement_frame(flag)[1])
    mats = _vectors_across_pace(_replacement(flag), pair)
    spread = mats.max(axis=0) - mats.min(axis=0)
    assert spread.any(), f"{flag!r} carries a wpm column but its vector does not move with pace"
    assert [names[i] for i in np.nonzero(spread)[0]] == ["wpm"]


def test_pace_invariance_is_exactly_the_absence_of_a_wpm_column():
    """The property is DERIVABLE from the schema, so it cannot silently drift from the frame.

    Covers every registered replacement frame plus the served one, so a NEW frame added to the
    registry is classified by this test rather than escaping it.
    """
    for flag in REPLACEMENT_FRAME_FLAGS:
        build, names, *_ = replacement_frame(flag)
        mats = _vectors_across_pace(_replacement(flag), PAIRS[0])
        varies = bool((mats.max(axis=0) - mats.min(axis=0)).any())
        assert varies == ("wpm" in names), (
            f"{flag!r}: vector-varies-with-pace is {varies} but 'wpm' in names is "
            f"{'wpm' in names} — the two must agree"
        )
    mats = _vectors_across_pace(_served, PAIRS[0])
    assert bool((mats.max(axis=0) - mats.min(axis=0)).any()) is ("wpm" in BIGRAM_FEATURE_NAMES)


# --- the CONSEQUENCE the gate actually measures -----------------------------------------


@pytest.mark.parametrize("flag", PACE_INVARIANT_FLAGS)
def test_a_pace_invariant_frame_cannot_reorder_pairs_across_buckets(flag):
    """The gate-relevant consequence, on the FRAME rather than on a trained model.

    Any predictor is a function of the feature vector alone, so if two pairs' vectors are
    unchanged between buckets then ANY model's ordering of them is unchanged too — no training
    required to establish it. Asserted here as the rank-vector identity a within-bucket Spearman
    would consume.
    """
    build = _replacement(flag)
    per_bucket = [
        np.vstack([np.asarray(build(G, p, wpm=m), dtype=np.float64) for p in PAIRS])
        for m in BUCKET_MIDPOINTS
    ]
    for m, X in zip(BUCKET_MIDPOINTS[1:], per_bucket[1:], strict=True):
        assert np.array_equal(X, per_bucket[0]), (
            f"{flag!r}: the whole design matrix must be identical at {m} wpm and "
            f"{BUCKET_MIDPOINTS[0]} wpm for the no-reordering argument to hold"
        )


def test_the_served_frame_can_reorder_pairs_across_buckets():
    """The opposite polarity for the reordering claim: the served design matrix DOES move."""
    per_bucket = [
        np.vstack([np.asarray(_served(G, p, wpm=m), dtype=np.float64) for p in PAIRS])
        for m in BUCKET_MIDPOINTS
    ]
    assert not np.array_equal(per_bucket[-1], per_bucket[0])


def test_to_ms_pace_factor_is_rank_preserving_within_a_bucket():
    """Why the gate cannot see the ONE thing an invariant frame still varies by.

    ``to_ms`` applies ``exp(pred) * 12000 / wpm``. Within a bucket every cell shares one ``wpm``,
    so the map is a strictly increasing function of ``pred`` — order-preserving, hence invisible to
    a within-bucket Spearman. Demonstrated on values rather than asserted in prose.
    """
    rng = np.random.default_rng(20260804)
    pred = rng.normal(size=64)
    for wpm in BUCKET_MIDPOINTS:
        ms = np.exp(pred) * 12000.0 / wpm
        assert np.array_equal(np.argsort(np.argsort(ms)), np.argsort(np.argsort(pred))), (
            f"the LOGRAT->ms conversion reordered cells at {wpm} wpm, which would make the "
            f"per-bucket rho depend on the pace factor"
        )
    # ...and the VALUES do change, so this test is not asserting a no-op.
    assert not np.allclose(
        np.exp(pred) * 12000.0 / BUCKET_MIDPOINTS[0],
        np.exp(pred) * 12000.0 / BUCKET_MIDPOINTS[-1],
    )
