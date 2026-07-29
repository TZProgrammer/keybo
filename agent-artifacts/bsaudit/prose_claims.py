"""Every remaining QUANTITATIVE claim in the bad_scissor docstrings, checked against behaviour.

The table rows are already checked by census_predicate.py. This driver takes the claims made in
PROSE — in the module docstring and in the individual method docstrings — and measures each.
These are the ones a reader acts on and no test asserts.

Claims checked:
  1. ``by_cell``: "The dy2 subtotal ... is under a tenth of the priced mass"
  2. module: "96.6 % of the flagged mass has bottom key ``c`` or ``x``"
  3. module: the oxey denominator's ~1.497x magnitude (direction checked separately)
  4. module: "``Layout.has_key(" ")`` is True" and "Space is in no bad-scissor pair
     (``hand(0) == 0``)"
  5. ``_kind``'s robustness: is a thumb/space position reachable into ``_kind``, or does the
     guard order protect it? (an IndexError waiting on a guard reorder)
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from keybo.analysis import bad_scissor as BS
from keybo.cli.analyze import _EXTRA_NAMED, _shared_corpora, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.testkit import assert_module_under

ROOT = Path("/tmp/bsaudit")
assert_module_under("keybo", ROOT)
assert_module_under("keybo.analysis.bad_scissor", ROOT)
REGISTRY = {k: v for k, v in {**NAMED_LAYOUTS, **_EXTRA_NAMED}.items() if len(v) == 30}


def main() -> int:
    bigrams, _sk, _tri = _shared_corpora(production_corpus_dir("iweb"))
    scorer = BS.BadScissor(bigrams)
    failures, notes = [], {}

    # --- claim 1: dy2 is "under a tenth of the priced mass" ------------------------------
    doc_by_cell = BS.BadScissor.by_cell.__doc__ or ""
    print("=== CLAIM 1 — by_cell: 'the dy2 subtotal is under a tenth of the priced mass' ===")
    print(f"  (docstring says: {' '.join(doc_by_cell.split())[:150]!r})")
    over = {}
    rows = {}
    for label, lay in sorted(REGISTRY.items()):
        L = Layout(lay, G)
        share = scorer.share(L)
        cells = scorer.by_cell(L)
        dy2 = sum(v for k, v in cells.items() if k.endswith("dy2"))
        frac = dy2 / share if share else 0.0
        rows[label] = {"share": share, "dy2": dy2, "dy2_frac_of_share": frac}
        flag = "  OVER A TENTH" if frac >= 0.10 else ""
        print(f"  {label:16s} dy2={dy2:8.5f} share={share:9.5f} "
              f"dy2/share={100 * frac:6.3f}%{flag}")
        if frac >= 0.10:
            over[label] = frac
    if over:
        failures.append(
            f"'dy2 under a tenth' is FALSE on {sorted(over)} "
            f"({', '.join(f'{k} {100 * v:.3f}%' for k, v in over.items())})")
    notes["dy2_over_a_tenth_on"] = over
    print(f"  => claim holds on {len(rows) - len(over)}/{len(rows)} registry layouts; "
          f"violated on {sorted(over) or 'none'}")

    # --- claim 2: "96.6 % of the flagged mass has bottom key c or x" ---------------------
    print("\n=== CLAIM 2 — module: '96.6 % of the flagged mass has bottom key c or x' ===")
    m = re.search(r"\*\*([\d.]+)\s*% of the flagged mass has bottom\s*\n?key ``(\w)`` or ``(\w)``",
                  BS.__doc__ or "")
    claimed_pct, k1, k2 = (float(m.group(1)), m.group(2), m.group(3)) if m else (96.6, "c", "x")
    print(f"  parsed from docstring: {claimed_pct}% on bottom keys {k1!r}/{k2!r}")
    per_layout = {}
    for label, lay in sorted(REGISTRY.items()):
        L = Layout(lay, G)
        flagged = 0.0
        on_kx = 0.0
        for bg, freq in bigrams.items():
            if len(bg) != 2 or " " in bg or not all(L.has_key(c) for c in bg):
                continue
            a, b = L.pos(bg[0]), L.pos(bg[1])
            if not BS.bad_scissor(G, a, b):
                continue
            flagged += freq
            bottom = bg[0] if a[1] < b[1] else bg[1]
            if bottom in (k1, k2):
                on_kx += freq
        per_layout[label] = 100.0 * on_kx / flagged if flagged else 0.0
    for label, pct in sorted(per_layout.items(), key=lambda kv: -kv[1]):
        print(f"  {label:16s} {pct:7.3f}% of flagged mass on bottom {k1}/{k2}")
    qwerty_pct = per_layout["qwerty"]
    print(f"  => on qwerty (the layout the claim is about): {qwerty_pct:.3f}% "
          f"vs claimed {claimed_pct}%")
    notes["c_or_x_pct_per_layout"] = per_layout
    notes["c_or_x_claimed"] = claimed_pct
    if abs(qwerty_pct - claimed_pct) > 0.5:
        failures.append(f"'{claimed_pct}% bottom key c/x' — qwerty measures {qwerty_pct:.3f}%")
    # the claim is stated unqualified; check whether it generalizes
    generalizes = all(p >= 90.0 for p in per_layout.values())
    print(f"  claim is stated UNQUALIFIED; >=90% on every registry layout? {generalizes} "
          f"(min {min(per_layout.values()):.3f}% on "
          f"{min(per_layout, key=per_layout.__getitem__)})")
    notes["c_or_x_generalizes_over_registry"] = generalizes

    # --- claim 3: has_key(" ") is True; space is in no bad-scissor pair -------------------
    print("\n=== CLAIM 3 — 'Layout.has_key(\" \") is True' and space is in no bad-scissor pair ===")
    L = Layout(REGISTRY["qwerty"], G)
    has_space = L.has_key(" ")
    print(f"  Layout.has_key(' ')      = {has_space}")
    print(f"  layout.pos(' ')          = {L.pos(' ')}")
    print(f"  geometry.hand(0)         = {G.hand(0)}")
    space_pairs = [(c, " ") for c in L.chars] + [(" ", c) for c in L.chars]
    space_flagged = [(x, y) for x, y in space_pairs
                     if BS.bad_scissor(G, L.pos(x), L.pos(y))]
    print(f"  space-touching pairs flagged by the predicate: {len(space_flagged)} "
          f"(of {len(space_pairs)})")
    if not has_space:
        failures.append("docstring: has_key(' ') is claimed True but is False")
    if space_flagged:
        failures.append(f"docstring: space claimed in no bad-scissor pair, found {len(space_flagged)}")
    notes["space_pairs_flagged"] = len(space_flagged)

    # --- claim 5: is _kind reachable with a thumb position? ------------------------------
    print("\n=== CLAIM 5 — _kind on the space/thumb position (guard-order robustness) ===")
    print(f"  geometry.finger(0)       = {G.finger(0)}  value={G.finger(0).value!r}")
    try:
        k = BS._kind(G, 0)
        print(f"  _kind(G, 0)              = {k!r}  (no exception)")
        kind_raises = False
    except Exception as e:
        print(f"  _kind(G, 0) RAISES       {type(e).__name__}: {e}")
        kind_raises = True
    # Is it reachable through the public predicate? same_hand must short-circuit first.
    reached = []
    for pos in ((0, 0),):
        for slot in sorted(G.slots):
            for a, b in ((pos, slot), (slot, pos)):
                try:
                    BS.bad_scissor(G, a, b)
                    BS.bad_scissor_finger(G, a, b)
                    BS.bad_scissor_cell(G, a, b)
                except Exception as e:
                    reached.append((a, b, f"{type(e).__name__}: {e}"))
    print(f"  public API called on space pairs: {len(reached)} exceptions "
          f"(guard order {'PROTECTS' if not reached else 'FAILS TO PROTECT'} _kind)")
    notes["_kind_raises_on_thumb"] = kind_raises
    notes["public_api_exceptions_on_space_pairs"] = reached[:5]
    if kind_raises and not reached:
        print("  => LATENT ONLY: _kind cannot handle a thumb, but same_hand() short-circuits "
              "first, so no public path reaches it. A guard reorder would expose an IndexError.")

    # --- the 1.497x MAGNITUDE (direction is in denominator_direction.py) ------------------
    print("\n=== CLAIM 3b — the ~1.497x magnitude ===")
    mags = []
    for label, lay in sorted(REGISTRY.items()):
        Ly = Layout(lay, G)
        c, w = scorer.share(Ly), scorer.share(Ly, exclude_space=False)
        mags.append(c / w)
    claimed = re.findall(r"1\.49\d+", BS.__doc__ or "")
    print(f"  docstring magnitudes mentioned: {claimed or '~1.497 (prose)'}")
    print(f"  measured ratio range over 15 layouts: {min(mags):.6f}..{max(mags):.6f}")
    print(f"  => magnitude claim ~1.497 is {'CORRECT' if 1.49 < min(mags) and max(mags) < 1.51 else 'WRONG'}")
    notes["magnitude_range"] = [min(mags), max(mags)]

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} PROSE CLAIM(S) DO NOT MATCH BEHAVIOUR:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("every prose claim checked matches behaviour")

    p = ROOT / "agent-artifacts/bsaudit/prose_claims.json"
    p.write_text(json.dumps({"failures": failures, "notes": notes, "dy2": rows}, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
