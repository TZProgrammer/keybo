"""Load fitted `LossCurve`s back out of an evidence-weights JSON.

`LossCurve.to_dict` exists but there is no `from_dict` — the artifact is write-only, and both
consumers so far (`evobj.load_weights`, my report) re-read the raw dict fields by hand. That is
exactly the habitat trap 28 describes, and the field names are NOT the dataclass field names
(`weight_ms_per_unit` -> `weight`, `weight_ci95` -> `weight_ci`, `mean_abs_shap_ms` ->
`mean_abs_shap`), so a hand-rolled reader gets a `KeyError` at best and a silently wrong curve at
worst.

So: ONE loader, and it asserts `to_dict(from_dict(d)) == d` per curve. If the serializer ever
gains or renames a field, this raises instead of quietly dropping it.
"""

from __future__ import annotations

import json
from pathlib import Path

from keybo.analysis.evidence_scorer import LIVE_GAUGES, LossCurve


def curve_from_dict(d: dict) -> LossCurve:
    curve = LossCurve(
        metric=d["metric"],
        form=d["form"],
        coeffs=[float(c) for c in d["coeffs"]],
        knot=None if d["knot"] is None else float(d["knot"]),
        domain=(float(d["valid_domain"][0]), float(d["valid_domain"][1])),
        observed_range=(float(d["observed_range"][0]), float(d["observed_range"][1])),
        weight=float(d["weight_ms_per_unit"]),
        weight_ci=(float(d["weight_ci95"][0]), float(d["weight_ci95"][1])),
        r2=float(d["r2"]),
        r2_linear=float(d["r2_linear"]),
        mean_abs_shap=float(d["mean_abs_shap_ms"]),
        shap_share_pct=float(d["shap_share_pct"]),
    )
    # Round-trip guard: every key the serializer writes must come back identical, so a schema
    # change cannot be silently dropped on the floor here.
    round_tripped = curve.to_dict()
    extra = set(d) - set(round_tripped)
    known_extra = {"expected_sign", "sign_plausible"}  # added by the sign audit, not curve state
    unexpected = extra - known_extra
    if unexpected:
        raise AssertionError(f"{d['metric']}: weights JSON has fields the loader drops: {unexpected}")
    for key, value in round_tripped.items():
        if d[key] != value:
            raise AssertionError(f"{d['metric']}.{key}: round-trip {value!r} != source {d[key]!r}")
    return curve


def load_curves(path: str | Path) -> dict[str, LossCurve]:
    """metric -> LossCurve, asserted to cover exactly `LIVE_GAUGES`."""
    blob = json.load(open(path))
    weights = blob["weights"] if "weights" in blob else blob
    out = {g["metric"]: curve_from_dict(g) for g in weights["weights"]}
    if set(out) != set(LIVE_GAUGES):
        raise AssertionError(f"weights cover {sorted(out)} but LIVE_GAUGES is {sorted(LIVE_GAUGES)}")
    return out


def load_meta(path: str | Path) -> dict:
    blob = json.load(open(path))
    weights = blob["weights"] if "weights" in blob else blob
    return {k: v for k, v in weights.items() if k != "weights"}
