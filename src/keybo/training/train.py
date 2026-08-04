"""Train typing-time models from stroke data.

Feature vectors are built with the SAME pipeline used for scoring (via the
``*_from_positions`` entry points), using the physical positions recorded in the data. This
is the guarantee against train/serve skew: there is exactly one feature computation, and a
model's metadata records the ``FEATURE_VERSION`` it was trained under.

Each stroke row contributes one training example per WPM group: the target is the IQR-mean
of that group's durations, and the WPM enters as a feature so a single model spans the range.

Two corrections found by the real-data LOLO harness (2026-07-04/05, arm R1W — see
``agent-artifacts/OQ1-frequency-feature.md``) are built in for bigram training:

- **Additive practice term.** Frequent bigrams are fast partly because they are practiced —
  an effect that is *approximately*, but measurably NOT exactly, layout-independent (see the
  caveat below). Left unmodeled, the geometry model absorbs it (frequent bigrams' qwerty
  positions look "fast" — omitted-variable bias); modeled as a raw freq
  feature it becomes a per-position memorization key (98.7% of the data is qwerty). The fix:
  fit ``time = g(geometry, wpm) + b(bigram)`` by backfitting — b is the shrunk per-bigram
  mean residual, g refits on the residualized target ``y − b̂``. b is keyed by bigram
  identity, so it cancels exactly in layout comparisons; its only job is cleaning g's
  training target. Measured: pooled out-of-sample layout tau +0.667 → +1.0, and including b
  cuts held-out magnitude error ~64% (wmae 28.74 → 9.12, better on 12/12 fold×seed cells and
  4/4 folds) while moving the calibration slope to ~1.0 (pooled 0.948 → 0.999).

  ⚠ **"Layout-independent" is an approximation, and the deviation is measured.** Fitting b on
  qwerty-only vs non-qwerty-only data gives a disattenuated correlation of **0.6682** between
  the two estimates — below the 0.80 bar a pre-registered test set for "layout-independent" —
  at **1.249× matched-noise rms**. So the *effect* carries a real layout-specific component.
  Two things this does NOT undermine, both measured: the identity-keyed **cancellation** below
  (structural, exact), and the attribution itself — b encodes almost exactly the frequency
  dependence an independent geometry-differenced measurement licenses (ratio **1.0614**,
  CI95 [0.976, 1.166]), and it is not absorbing geometry (R²(b ~ served geometry) = −0.015
  out-of-fold). Why the layout-specific part exists is OPEN; the leading untested reading is
  that practice attaches to motor sequences as much as to letter pairs, which would make the
  residual real rather than a defect. Deciding it needs a 5th training layout, because the
  ngram→geometry bijection rules out a within-layout instrument.
- **Layout balance weights.** Inverse-layout-share example weights (capped) stop the 98.7%
  qwerty majority from dominating the fit. Measured: composes with the practice term
  (rho/ceiling .928 → .931).

Both are on by default and controllable (``practice_term=False`` / ``layout_weights=False``).
The fitted practice term is stored in the model metadata (``extra["practice_term"]``) for
inspection; scoring deliberately ignores it — and the reason is STRUCTURAL, not the
(approximate) layout-independence above. Because b is keyed by NGRAM IDENTITY, its
frequency-weighted total depends only on the corpus and the charset: measured bit-identical
(spread 0.0) across all ten campaign boards, across 200 random permutations of a fixed
charset, and at both bigram and trigram level. The optimizer's only moves — 2-opt swap and
3-opt triple permutation — are charset-PRESERVING, so b is a constant offset over the entire
reachable search space and cannot change any ranking a search can reach.
⚠ It does NOT cancel across DIFFERENT charsets: the residual is 4.45e-3 log units (bigram) /
1.80e-3 (trigram) ≈ 1 ms/char, which is material for cross-charset magnitude claims (e.g.
qwerty-vs-tuned percentages) even though it is inert for ranking within a charset.

A third correction (T-REL, 2026-07-10): models train in the **LOGRAT target space** —
``log(ms * wpm / 12000)``, time as a log-multiple of the typist's session-mean keystroke.
The ms label carries the typist-pace scale, so every geometry leaf must re-learn the wpm
hyperbola; pre-factoring it out cut cross-layout wmae 37% (bigram) with the rare-ngram
guards held (an additive DIFF control moved nothing ⇒ the multiplicative scale structure
is the mechanism), and the conditioned-trigram A/B reproduced it (wmae −30.7%, every
guard improved). ``predict()`` therefore returns log-ratios for these models; consumers
convert back via ``TypingModel.predict_ms`` / ``to_ms``. The practice term is backfit in
the training space, so for LOGRAT models the stored b values are log-scale.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from keybo.data.strokes import StrokeRow, iqr_average
from keybo.features import (
    bigram_features_from_positions,
    trigram_features_from_positions,
)
from keybo.features.ngram import REPLACEMENT_FRAME_FLAGS, replacement_frame
from keybo.features.schema import (
    BIGRAM_DIRECTION_FEATURE_NAMES,
    BIGRAM_FEATURE_NAMES,
    BIGRAM_KITCHENSINK_FEATURE_NAMES,
    FEATURE_VERSION,
    FEATURE_VERSION_DIRECTION,
    FEATURE_VERSION_KITCHENSINK,
    TRIGRAM_DIRECTION_FEATURE_NAMES,
    TRIGRAM_FEATURE_NAMES,
    TRIGRAM_KITCHENSINK_FEATURE_NAMES,
)
from keybo.geometry import ROW_STAGGERED_30, Geometry
from keybo.models.base import ModelMetadata
from keybo.models.xgboost_model import XGBoostTypingModel

#: practice-term shrinkage: b(ngram) = sum(w·resid) / (sum(w) + K). Pre-registered at 100
#: raw samples; the LOLO conclusion is robust across k ∈ [10, 1000] (audit 2026-07-05).
PRACTICE_SHRINKAGE_K = 100.0
#: backfit iterations (b and g alternate); 2 sufficed in every measured arm.
PRACTICE_BACKFIT_ITERS = 2
#: cap on inverse-layout-share weights, so a 64-participant layout can't dominate.
LAYOUT_WEIGHT_CAP = 50.0

#: known target spaces. MS: IQR-mean of raw durations. LOGRAT (the adopted bigram
#: space, T-REL 2026-07-10, -37% cross-layout wmae): PER-SAMPLE log-ratios robustly
#: averaged (a trimmed geometric mean — PACE-2 ANCHOR-PS, 2026-07-10, -1.6% over
#: log-of-mean: multiplicative noise wants log-space aggregation).
_TARGET_SPACES = ("MS", "LOGRAT")


def normalize_target_space(target_space: str) -> str:
    """Upper-case ``target_space`` and reject anything outside :data:`_TARGET_SPACES`.

    The single gate for every entry point that accepts a target space. It exists because the
    check used to live only in ``_train``: the public :func:`build_training_matrix` forwarded
    its raw string to ``_build_matrix_full``, and ``_group_target`` compared it to ``"LOGRAT"``
    exactly, so ``"BOGUS"`` *and* ``"lograt"`` both fell through to MS targets with no error —
    the caller named the right space and got the other one (CBTESTS-1).

    Case is NORMALIZED rather than rejected, because case-insensitivity is already this
    codebase's convention for this field in both other places that read it: ``_train`` has
    always applied ``.upper()``, and :attr:`keybo.models.base.TypingModel.target_space`
    upper-cases the value it reads back out of a model sidecar (pinned by
    ``test_target_space_reads_sidecar_case_insensitively``). Making the matrix builder the one
    case-SENSITIVE reader would trade a silent wrong answer for a spurious failure on a string
    the rest of the pipeline accepts.
    """
    normalized = str(target_space).upper()
    if normalized not in _TARGET_SPACES:
        raise ValueError(f"unknown target_space {target_space!r} (known: {sorted(_TARGET_SPACES)})")
    return normalized


def _group_target(durations: list[int], wpm: int, target_space: str) -> float:
    """The per-(row, wpm-group) training target in the given space.

    LOGRAT uses the GROUP-MEAN construction (log of the IQR-mean duration): the
    per-sample alternative was adopted on PACE-2's plain-extraction frame (-1.6%) but
    FAILED replication on the production v5 frame (+0.4%, PS-V5 2026-07-11) — the win
    was frame-specific (BUF2-BOTH cleaning already removes the tail the per-sample
    robustness bought); reverted per the registered rule (0cb4b9d).

    Every space is dispatched EXPLICITLY: this used to be ``if LOGRAT ... else MS``, so any
    string that reached here unrecognized silently became an MS target. The callers all
    normalize now, so the final ``raise`` is unreachable by design — which is the point. It
    turns "a space was added to ``_TARGET_SPACES`` and wired only into ``_train``" from a
    plausible-looking wrong matrix into an error at the row that produces it.
    """
    if target_space == "LOGRAT":
        w = max(float(wpm), 1.0)
        return float(np.log(max(iqr_average(durations), 1.0) * w / 12000.0))
    if target_space == "MS":
        return iqr_average(durations)
    raise ValueError(
        f"unknown target_space {target_space!r} reached _group_target "
        f"(known: {sorted(_TARGET_SPACES)}); callers must normalize via normalize_target_space"
    )


def _rows_to_examples(
    row: StrokeRow,
    geometry: Geometry,
    ngram: str,
    target_space: str = "MS",
    direction: bool = False,
    kitchensink: bool = False,
    interp: bool = False,
):
    """Yield (feature_vector, target) per WPM group in a stroke row.

    ``direction=False`` (the default) builds the served frame byte for byte. ``direction=True``
    builds the widened order-aware frame (:data:`BIGRAM_DIRECTION_FEATURE_NAMES` /
    :data:`TRIGRAM_DIRECTION_FEATURE_NAMES`) — the same switch the feature pipeline exposes,
    threaded here so a model can be TRAINED on the wider frame, not only scored on it.

    ``kitchensink=True`` builds the KITCHEN-SINK frame (the widened frame plus the twelve
    external-project channels; :data:`BIGRAM_KITCHENSINK_FEATURE_NAMES` /
    :data:`TRIGRAM_KITCHENSINK_FEATURE_NAMES`). It implies ``direction``, so the three trainable
    frames are narrow / widened / kitchen-sink and each has its own version stamp.

    ``interp`` selects a REPLACEMENT basis rather than a widening, so it is mutually exclusive
    with both flags above and is bigram-only. ``True`` = INTERPFRAME-1's 10-column
    interpretability frame, ``"wpm"`` = its 11-column pace-adapting variant, ``"hybridb"`` =
    HYBRIDB-1's 18-column frame (interp.1's ordinals + the served row/finger one-hots). The
    ``interp``-to-frame mapping lives in ONE place, :func:`keybo.features.ngram.replacement_frame`.

    ⚠ Neither ``True`` nor ``"hybridb"`` has a ``wpm`` column, so the WPM group still selects the
    TARGET (each group is a separate example at its own observed pace) but no longer appears as an
    input — which is exactly the constant-column artifact those frames exist to remove, and equally
    exactly why a model on either cannot span a WPM range.
    """
    by_wpm: dict[int, list[int]] = defaultdict(list)
    for wpm, duration, _pid, _hold in row.samples:
        by_wpm[wpm].append(duration)

    for wpm, durations in by_wpm.items():
        target = _group_target(durations, wpm, target_space)
        if interp:
            # `interp` is a STRING-OR-TRUE flag, not a plain bool. Resolved through the ONE
            # registry in keybo.features.ngram so the builder here and the name list / monotone
            # tuple / version stamp chosen in `_fit_model` cannot disagree, and so an unknown
            # flag raises rather than silently selecting the 10-column frame -- two of these
            # frames differ by a single column, so a wrong pick reads as a plausible number.
            builder = replacement_frame(interp)[0]
            vec = builder(geometry, row.positions, wpm=wpm)
        elif ngram == "bigram":
            vec = bigram_features_from_positions(
                geometry, row.positions, wpm=wpm, direction=direction, kitchensink=kitchensink
            )
        else:
            vec = trigram_features_from_positions(
                geometry, row.positions, wpm=wpm, direction=direction, kitchensink=kitchensink
            )
        yield vec, target, len(durations)


def build_training_matrix(
    rows: list[StrokeRow],
    ngram: str,
    target_wpm: float,
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    target_space: str = "MS",
    with_layouts: bool = False,
    direction: bool = False,
    kitchensink: bool = False,
) -> tuple[np.ndarray, ...]:
    """Turn stroke rows into (X, y) using the shared feature pipeline.

    ``target_wpm`` is unused for the matrix itself (WPM is taken per-sample) but kept in the
    signature so callers pass their intended scoring WPM explicitly; it is recorded in model
    metadata by the ``train_*`` helpers. ``progress`` shows a tqdm bar over the stroke rows
    (feature building is the visible-latency stage on a real-sized table).

    ``target_space`` was previously accepted only by the private ``_build_matrix_full``, so a
    caller of THIS function could not ask for anything but ``"MS"`` — while every shipped k31
    model is ``"LOGRAT"`` (verified: all six ``data/models/k31/*.meta.json.gz``). The default
    stays ``"MS"`` so existing callers are unaffected, but it is now a *choice* rather than a
    hardwired mismatch (KAGGLE-1 FINAL, ledger ``cf6ee07``).

    ``target_space`` is validated and case-normalized HERE, on the public boundary, so an
    unknown space raises instead of silently producing MS targets and ``"lograt"`` means what
    it says (CBTESTS-1; see :func:`normalize_target_space` for why case normalizes rather than
    raises). Before that gate this function did no checking at all — the raw string went to
    ``_group_target``, which tested ``== "LOGRAT"`` exactly.

    ``with_layouts=True`` additionally returns the per-example layout label, which is exactly
    what a grouped cross-validation needs for its ``groups`` argument. The labels were always
    computed here and simply discarded.
    """
    X, y, _ngrams, layouts, _n = _build_matrix_full(
        rows,
        ngram=ngram,
        geometry=geometry,
        progress=progress,
        target_space=normalize_target_space(target_space),
        direction=direction,
        kitchensink=kitchensink,
    )
    if with_layouts:
        return X, y, layouts
    return X, y


def _build_matrix_full(
    rows,
    ngram,
    geometry,
    progress=False,
    target_space="MS",
    direction=False,
    kitchensink=False,
    interp=False,
):
    """(X, y, example ngram ids, example layouts, example raw-sample counts).

    ``y`` is already in ``target_space`` (per-sample aggregation for LOGRAT).
    """
    iterator = rows
    if progress:
        from tqdm import tqdm

        iterator = tqdm(rows, desc="building features", unit="row", leave=False)
    features: list[np.ndarray] = []
    targets: list[float] = []
    ngrams: list[str] = []
    layouts: list[str] = []
    counts: list[float] = []
    for row in iterator:
        for vec, target, n in _rows_to_examples(
            row, geometry, ngram, target_space, direction, kitchensink, interp
        ):
            features.append(vec)
            targets.append(target)
            ngrams.append(row.ngram)
            layouts.append(row.layout)
            counts.append(float(n))
    if not features:
        return (
            np.empty((0, 0)),
            np.empty((0,)),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype=object),
            np.empty((0,)),
        )
    return (
        np.vstack(features),
        np.array(targets, dtype=np.float64),
        np.array(ngrams, dtype=object),
        np.array(layouts, dtype=object),
        np.array(counts, dtype=np.float64),
    )


def layout_balance_weights(layouts: np.ndarray, cap: float = LAYOUT_WEIGHT_CAP) -> np.ndarray:
    """Inverse-layout-share example weights, capped, normalized to mean 1."""
    share: dict[str, float] = defaultdict(float)
    for la in layouts:
        share[la] += 1.0
    total = float(len(layouts))
    w = np.array([min(cap, total / (len(share) * share[la])) for la in layouts])
    return w / w.mean()


def fit_practice_term(
    ngrams: np.ndarray,
    residuals: np.ndarray,
    counts: np.ndarray,
    k: float = PRACTICE_SHRINKAGE_K,
) -> dict[str, float]:
    """Shrunk per-ngram mean residual: b = Σ(count·resid) / (Σcount + k).

    ``counts`` are raw-sample counts per example, so a bigram seen 10,000 times gets its
    full residual while a bigram seen 5 times is shrunk hard toward 0 (no practice claim
    from noise).
    """
    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    for ng, r, c in zip(ngrams, residuals, counts, strict=True):
        num[ng] += c * r
        den[ng] += c
    return {ng: num[ng] / (den[ng] + k) for ng in num}


def _train(
    rows,
    ngram,
    target_wpm,
    wpm_range,
    geometry,
    progress=False,
    practice_term=True,
    layout_weights=True,
    target_space="MS",
    calibration=False,
    direction=False,
    kitchensink=False,
    interp=False,
    monotone=True,
    **params,
) -> XGBoostTypingModel:
    target_space = normalize_target_space(target_space)
    if interp is not False and interp not in REPLACEMENT_FRAME_FLAGS:
        raise ValueError(
            f"interp must be False (the served frame) or one of "
            f"{list(REPLACEMENT_FRAME_FLAGS)!r}: True = INTERPFRAME-1's 10-column frame, "
            f"'wpm' = its 11-column pace-adapting variant, 'hybridb' = HYBRIDB-1's 18-column "
            f"frame; got {interp!r}"
        )
    if interp:
        # REFUSED, not silently ignored. `interp` REPLACES the served columns rather than widening
        # them, so every combination below would produce a frame whose stamp lies about its
        # columns -- and the stamp is the only thing standing between a model and being scored on
        # the wrong matrix.
        if direction or kitchensink:
            raise ValueError(
                f"interp={interp!r} selects a REPLACEMENT basis "
                f"({replacement_frame(interp)[4]}), so it cannot be combined with "
                f"direction=/kitchensink=, which WIDEN the served frame"
            )
        if ngram != "bigram":
            raise ValueError(
                f"interp={interp!r} ({replacement_frame(interp)[4]}) is a bigram-only frame; "
                f"got ngram={ngram!r}"
            )

    # Targets are built directly in the model's target space (per-sample log aggregation
    # for LOGRAT — PACE-2 ANCHOR-PS).
    X, y, ngrams, layouts, counts = _build_matrix_full(
        rows,
        ngram=ngram,
        geometry=geometry,
        progress=progress,
        target_space=target_space,
        direction=direction,
        kitchensink=kitchensink,
        interp=interp,
    )
    # The version stamp and the name list move TOGETHER with the frame: a widened model records
    # FEATURE_VERSION_DIRECTION so it can never load where a served model is expected (base.py
    # hard-errors on a mismatch), and it carries the widened name list so importances stay
    # labelled. Stamping the narrow version on a wide frame is exactly the silent train/serve
    # skew DIRECTION-1 refused to create. The kitchen-sink frame is the third population and gets
    # the third stamp on the same principle — and it is checked FIRST because it implies
    # ``direction``, so an `if direction` test would otherwise claim a kitchen-sink model.
    if interp:
        # Checked FIRST for the same reason kitchensink is checked before direction: the frames are
        # mutually exclusive and the first matching branch wins, so the most specific goes first.
        # (The refusals above already make an illegal combination unreachable; this ordering is
        # what keeps that true if a later flag is added.)
        #
        # The name list, the stamp AND the constraint tuple come from ONE registry lookup, so
        # they can never disagree — a model stamped ``interp.1`` while carrying the 11-column
        # constraint tuple would be exactly the train/serve skew the stamp exists to prevent, and
        # it is the same lookup ``_rows_to_examples`` used to pick the builder.
        _, names, constraints, stamp, _tag = replacement_frame(interp)
        # The monotone constraints ride WITH the frame, because they are part of what the frame
        # CLAIMS: each sign is the mechanism its column name asserts (see BIGRAM_INTERP_MONOTONE).
        # A caller can turn them off to price the constraint itself (INTERPFRAME-1 §5d) -- and
        # `monotone=False` must then be visible in the artifact, or two models with different
        # constraint sets would be indistinguishable after saving.
        #
        # ⚠ hybrid-B's tuple is PARTIAL: its ten interp columns are constrained and its eight
        # added one-hots carry 0 (see BIGRAM_HYBRIDB_MONOTONE for why, and for the registered
        # consequence that MONOFRAC cannot reach 1.0 on that frame). xgboost reads 0 as
        # "unconstrained", so a partial tuple is expressed rather than special-cased here.
        if monotone:
            # xgboost maps the tuple to columns POSITIONALLY, so a length mismatch would silently
            # constrain the wrong columns rather than raising.
            assert len(constraints) == len(names), "one constraint per column"
            params = {**params, "monotone_constraints": tuple(constraints)}
    elif kitchensink:
        names = (
            BIGRAM_KITCHENSINK_FEATURE_NAMES
            if ngram == "bigram"
            else TRIGRAM_KITCHENSINK_FEATURE_NAMES
        )
        stamp = FEATURE_VERSION_KITCHENSINK
    elif direction:
        names = (
            BIGRAM_DIRECTION_FEATURE_NAMES if ngram == "bigram" else TRIGRAM_DIRECTION_FEATURE_NAMES
        )
        stamp = FEATURE_VERSION_DIRECTION
    else:
        names = BIGRAM_FEATURE_NAMES if ngram == "bigram" else TRIGRAM_FEATURE_NAMES
        stamp = FEATURE_VERSION
    metadata = ModelMetadata(
        feature_version=stamp,
        feature_names=names,
        wpm_range=wpm_range,
        ngram=ngram,
    )
    # The interp frame has no ``wpm`` column, so ``names.index("wpm")`` would raise. Only the
    # first-finger calibration branch below reads ``wpm_col``, and that branch is unreachable for
    # this frame (it needs the served bigram columns), so the guard is the honest form -- rather
    # than inventing a wpm vector the frame deliberately does not carry.
    if len(y) and "wpm" in names:
        wpm_col = np.maximum(X[:, names.index("wpm")], 1.0)

    # First-finger calibration (PINKY-FIT): OFF by default since CAL-REMOVE (2026-07-12).
    # The measured effect (PINKY-GAP +27ms qwerty matched pairs) stands as a finding, but
    # the seam contributed ~0 speed (ARM-NOCAL +3.90% vs +3.95%, LOLO identical), its
    # evidence is single-population, and the community ring_first estimate sign-flips
    # (D5-U2). Serving still reads deltas from any older sidecar that carries them —
    # backward compatible. Opt back in with calibration=True (bigram + LOGRAT only).
    fitted_deltas: dict[str, float] = {}
    if calibration and ngram == "bigram" and target_space == "LOGRAT" and len(y):
        from keybo.training.calibration import (
            delta_log,
            finger_class,
            fit_first_finger_deltas,
        )

        fitted_deltas = fit_first_finger_deltas(rows, geometry)
        if fitted_deltas:
            # one class per row (positions are row-constant); expand to the example grid
            row_cls = {id(r): finger_class(geometry, *r.positions) for r in rows}
            adj = np.zeros(len(y))
            i = 0
            for row in rows:
                n_groups = len({s[0] for s in row.samples})
                cls = row_cls[id(row)]
                if cls is not None:
                    for j in range(i, i + n_groups):
                        adj[j] = delta_log(cls, wpm_col[j], fitted_deltas)
                i += n_groups
            y = y - adj

    weights = layout_balance_weights(layouts) if layout_weights and len(y) else None

    def fit(target):
        model = XGBoostTypingModel(metadata, **params)
        model._regressor.fit(X, target, sample_weight=weights)
        model._fitted = True
        return model

    from keybo.training.calibration import CALIBRATION_VERSION

    calibration_tag = (
        {
            "version": CALIBRATION_VERSION,
            "deltas_ms": {k: round(float(v), 3) for k, v in fitted_deltas.items()},
        }
        if fitted_deltas
        else None
    )

    # Recorded in the artifact because the constraint set is NOT recoverable from the saved
    # booster: xgboost bakes the constraints into the tree structure at fit time and does not
    # serialize the parameter, so two models trained with and without them would be
    # indistinguishable after `save()` -- and "was this constrained?" is the whole question
    # INTERPFRAME-1 §5 asks. Written as the resolved value actually passed to xgboost, not as the
    # `monotone` flag, so the record cannot drift from the fit.
    frame_tag = (
        {
            "frame": replacement_frame(interp)[4],
            "monotone_constraints": list(params.get("monotone_constraints") or ()),
        }
        if interp
        else None
    )

    if not practice_term or not len(y):
        model = fit(y)
        model.metadata.extra["training"] = {
            "target_space": target_space,
            "practice_term": None,
            "layout_weights": bool(weights is not None),
            "calibration": calibration_tag,
            "interp_frame": frame_tag,
        }
        return model

    # Backfit: b absorbs the shrunk per-ngram residual mean; g refits on y - b. Runs in
    # the target space, so a LOGRAT model's b values are log-ratios (stored at higher
    # precision — rounding log-scale values to 3 decimals would destroy them).
    model = fit(y)
    bmap: dict[str, float] = {}
    for _ in range(PRACTICE_BACKFIT_ITERS):
        bmap = fit_practice_term(ngrams, y - model.predict(X), counts)
        bvec = np.array([bmap.get(ng, 0.0) for ng in ngrams])
        model = fit(y - bvec)
    b_digits = 3 if target_space == "MS" else 6
    model.metadata.extra["training"] = {
        "target_space": target_space,
        "practice_term": {
            "shrinkage_k": PRACTICE_SHRINKAGE_K,
            "backfit_iters": PRACTICE_BACKFIT_ITERS,
            "n_ngrams": len(bmap),
            "values": {ng: round(float(v), b_digits) for ng, v in bmap.items()},
        },
        "layout_weights": bool(weights is not None),
        "calibration": calibration_tag,
        "interp_frame": frame_tag,
    }
    return model


def train_bigram_model(
    rows: list[StrokeRow],
    target_wpm: float,
    wpm_range: tuple[int, int] = (60, 120),
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    practice_term: bool = True,
    layout_weights: bool = True,
    target_space: str = "LOGRAT",
    calibration: bool = False,
    direction: bool = False,
    kitchensink: bool = False,
    interp: bool = False,
    monotone: bool = True,
    **params,
) -> XGBoostTypingModel:
    """Fit a bigram typing-time model from bistroke rows (R1W + LOGRAT recipe).

    ``calibration`` defaults OFF per CAL-REMOVE (2026-07-12): the first-finger seam was
    speed-neutral (+3.90% vs +3.95%, LOLO identical) with single-population evidence and
    mixed community transfer. The estimator (``keybo.training.calibration``) remains
    available as a measurement tool; older sidecars with deltas still serve correctly.

    ``direction=True`` trains on the widened order-aware frame and stamps
    ``FEATURE_VERSION_DIRECTION``; the default reproduces the served frame exactly.

    ``kitchensink=True`` trains on the KITCHEN-SINK frame (the widened frame plus the twelve
    external-project channels) and stamps ``FEATURE_VERSION_KITCHENSINK``. It implies
    ``direction`` and takes precedence over it, so the three model populations stay disjoint.

    ``interp`` trains on a REPLACEMENT basis and stamps that frame's own version: ``True`` ->
    INTERPFRAME-1's 10-column frame / ``FEATURE_VERSION_INTERP``, ``"wpm"`` -> its 11-column
    pace-adapting variant, ``"hybridb"`` -> HYBRIDB-1's 18-column frame /
    ``FEATURE_VERSION_HYBRIDB``. Unlike ``direction``/``kitchensink`` these REPLACE the served
    columns rather than widening them, so combining them with either is REFUSED, and each applies
    its own monotone tuple unless ``monotone=False`` (which exists to price the constraint itself
    -- INTERPFRAME-1 §5d -- and is recorded in the artifact).

    ⚠ ``interp=1`` selects the ``True`` frame, because ``1 == True`` in Python. Pre-existing
    behaviour of this string-or-bool flag, noted rather than special-cased: the only values a
    caller writes are ``False``/``True``/``"wpm"``/``"hybridb"``, and anything outside that set
    that is not ``1`` raises.

    ``progress`` is consumed here (feature-build bar), never forwarded into ``**params`` --
    XGBoost silently ignores unknown keyword params, so a leak would be invisible.
    """
    return _train(
        rows,
        "bigram",
        target_wpm,
        wpm_range,
        geometry,
        progress=progress,
        practice_term=practice_term,
        layout_weights=layout_weights,
        target_space=target_space,
        calibration=calibration,
        direction=direction,
        kitchensink=kitchensink,
        interp=interp,
        monotone=monotone,
        **params,
    )


def train_trigram_model(
    rows: list[StrokeRow],
    target_wpm: float,
    wpm_range: tuple[int, int] = (60, 120),
    geometry: Geometry = ROW_STAGGERED_30,
    progress: bool = False,
    practice_term: bool = True,
    layout_weights: bool = True,
    target_space: str = "LOGRAT",
    direction: bool = False,
    kitchensink: bool = False,
    **params,
) -> XGBoostTypingModel:
    """Fit a trigram typing-time model from tristroke rows. See train_bigram_model.

    LOGRAT by default per the conditioned-trigram A/B (2026-07-10): wmae −30.7% with
    umae/dec3/taus all improved — the bigram mechanism carries.

    ``direction=True`` trains on the widened order-aware frame and stamps
    ``FEATURE_VERSION_DIRECTION``; the default reproduces the served frame exactly.

    ``kitchensink=True`` trains on the KITCHEN-SINK frame (the widened frame plus the twelve
    external-project channels) and stamps ``FEATURE_VERSION_KITCHENSINK``. It implies
    ``direction`` and takes precedence over it, so the three model populations stay disjoint.
    """
    return _train(
        rows,
        "trigram",
        target_wpm,
        wpm_range,
        geometry,
        progress=progress,
        practice_term=practice_term,
        layout_weights=layout_weights,
        target_space=target_space,
        direction=direction,
        kitchensink=kitchensink,
        **params,
    )
