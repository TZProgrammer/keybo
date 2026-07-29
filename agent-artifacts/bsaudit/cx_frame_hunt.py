"""HOSTILE CHECK on the '96.6 % of the flagged mass has bottom key c or x' claim.

My first measurement said qwerty is 7.559%, not 96.6%. Before reporting that as a defect I
must find the frame in which 96.6% IS true — "the number is wrong" and "the number is right
but its scope is undisclosed" are very different findings, and the second is the one the
ledger's own lesson (A WRONG CONSTANT ATTACHED TO A TRUE CONCLUSION) predicts.

So enumerate the plausible readings of "flagged mass" x "bottom key" and measure each. If one
lands on 96.6%, the claim is TRUE and the defect is a missing scope qualifier. If none does,
the constant itself is unsupported by anything I can compute here — and I say exactly that,
including that the Aalto keystroke frame (where BADSCISSOR-1 fit it) is NOT in this repo, so
I cannot rule it out.
"""

from __future__ import annotations

import json
from pathlib import Path

from keybo.analysis import bad_scissor as BS
from keybo.cli.analyze import _EXTRA_NAMED, _shared_corpora, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30 as G
from keybo.layout import Layout
from keybo.layouts import NAMED_LAYOUTS
from keybo.testkit import assert_module_under

ROOT = Path("/tmp/bsaudit")
assert_module_under("keybo", ROOT)
REGISTRY = {k: v for k, v in {**NAMED_LAYOUTS, **_EXTRA_NAMED}.items() if len(v) == 30}
TARGET = 96.6


def flagged_rows(bigrams, layout):
    """Every flagged bigram with the facts each interpretation needs."""
    out = []
    for bg, freq in bigrams.items():
        if len(bg) != 2 or " " in bg or not all(layout.has_key(c) for c in bg):
            continue
        a, b = layout.pos(bg[0]), layout.pos(bg[1])
        if not BS.bad_scissor(G, a, b):
            continue
        lower_char = bg[0] if a[1] < b[1] else bg[1]
        lower_row = min(a[1], b[1])
        out.append({"bg": bg, "freq": freq, "lower_char": lower_char,
                    "lower_row": lower_row, "dy": abs(a[1] - b[1])})
    return out


def main() -> int:
    corpora = {}
    for name in ("iweb", "blend-v1"):
        try:
            corpora[name] = _shared_corpora(production_corpus_dir(name))[0]
        except Exception as e:
            print(f"  (corpus {name!r} unavailable: {type(e).__name__}: {e})")
    print(f"=== corpora available: {sorted(corpora)} ===\n")

    interps = {
        "any-dy, lower key in {c,x}":
            lambda r: r["lower_char"] in ("c", "x"),
        "any-dy, lower key in {c,x,z}":
            lambda r: r["lower_char"] in ("c", "x", "z"),
        "any-dy, lower key on BOTTOM ROW and in {c,x}":
            lambda r: r["lower_row"] == 1 and r["lower_char"] in ("c", "x"),
        "dy==2 only, lower key in {c,x}":
            lambda r: r["dy"] == 2 and r["lower_char"] in ("c", "x"),
        "dy==2 only, lower key in {c,x,z}":
            lambda r: r["dy"] == 2 and r["lower_char"] in ("c", "x", "z"),
        "dy==2 only, lower key on BOTTOM ROW":
            lambda r: r["dy"] == 2 and r["lower_row"] == 1,
    }
    # For the dy-restricted readings the denominator is the dy-restricted flagged mass.
    denoms = {
        "any-dy, lower key in {c,x}": lambda r: True,
        "any-dy, lower key in {c,x,z}": lambda r: True,
        "any-dy, lower key on BOTTOM ROW and in {c,x}": lambda r: True,
        "dy==2 only, lower key in {c,x}": lambda r: r["dy"] == 2,
        "dy==2 only, lower key in {c,x,z}": lambda r: r["dy"] == 2,
        "dy==2 only, lower key on BOTTOM ROW": lambda r: r["dy"] == 2,
    }

    results = {}
    hits = []
    for cname, bigrams in corpora.items():
        for lname in ("qwerty", "qwerty30m"):
            rows = flagged_rows(bigrams, Layout(REGISTRY[lname], G))
            print(f"=== {cname} · {lname} · {len(rows)} flagged bigram identities ===")
            for iname, pred in interps.items():
                num = sum(r["freq"] for r in rows if pred(r))
                den = sum(r["freq"] for r in rows if denoms[iname](r))
                pct = 100.0 * num / den if den else 0.0
                near = abs(pct - TARGET) <= 0.5
                mark = "  <<< MATCHES 96.6%" if near else ""
                print(f"  {iname:48s} {pct:7.3f}%{mark}")
                results[f"{cname}|{lname}|{iname}"] = pct
                if near:
                    hits.append((cname, lname, iname, pct))
            print()

    print("=" * 78)
    if hits:
        print("THE CLAIM IS TRUE IN THESE FRAMES (so the defect is a MISSING SCOPE, not a "
              "wrong number):")
        for cname, lname, iname, pct in hits:
            print(f"  {pct:.3f}%  corpus={cname} layout={lname}  reading={iname}")
    else:
        print("NO computable frame lands on 96.6%. Closest per reading (over all "
              "corpus x layout):")
        by_reading = {}
        for k, v in results.items():
            _c, _l, i = k.split("|")
            if i not in by_reading or abs(v - TARGET) < abs(by_reading[i][1] - TARGET):
                by_reading[i] = (k, v)
        for i, (k, v) in by_reading.items():
            print(f"  {v:7.3f}%  {k}")
        print("\n  ⚠ SCOPE LIMIT I STATE EXPLICITLY: BADSCISSOR-1 fit this on the AALTO")
        print("    keystroke frame, which is NOT in this repo (data/ has only the derived")
        print("    surfaces, no raw TSV). So I CANNOT rule out that 96.6% is exact on that")
        print("    frame. What I can say is that no reading over the SHIPPED corpora and the")
        print("    SHIPPED predicate reproduces it, and the docstring attaches the number to")
        print("    'the flagged mass' with no frame named at all.")

    p = ROOT / "agent-artifacts/bsaudit/cx_frame_hunt.json"
    p.write_text(json.dumps({"target": TARGET, "results": results,
                             "matching_frames": hits}, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
