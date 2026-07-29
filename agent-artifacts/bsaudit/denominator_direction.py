"""Does the DENOMINATOR disclosure match the behaviour? Measured, not read.

The module docstring says choosing the oxey (space-including) denominator "inflates every
share by a plausible ~1.497x constant", and names
``test_the_space_including_denominator_would_inflate_every_share_by_about_1_497x`` as the
pinning test. The suite's actual test is named ``..._moves_...`` and its own docstring says
the spec wording "has the direction backwards".

So: measure the direction on all 15 registry layouts through the SHIPPED accessor, and check
which of the two documents the shipped code agrees with. The ledger already registered the
DIRECTION as backwards (LANDED entry / ALLGAUGE-1's self-correction) — what is NOT settled,
and what this driver is for, is whether the MODULE DOCSTRING still asserts the refuted
direction to a production reader.
"""

from __future__ import annotations

import json
from pathlib import Path

from keybo.analysis.bad_scissor import BadScissor
from keybo.cli.analyze import _shared_corpora, production_corpus_dir
from keybo.geometry import ROW_STAGGERED_30
from keybo.layout import Layout
from keybo.testkit import assert_discriminating, assert_module_under, assert_operands_computed

ROOT = Path("/tmp/bsaudit")
assert_module_under("keybo", ROOT)
assert_module_under("keybo.analysis.bad_scissor", ROOT)


def registry_layouts() -> dict[str, str]:
    """The shipped layout registry, read from the CLI's own sources of truth.

    GENERATE, don't retype: this is exactly the union ``analyze`` itself resolves names
    against (``keybo.layouts.NAMED_LAYOUTS`` plus the CLI's ``_EXTRA_NAMED``), so the sweep
    covers whatever the shipped registry contains rather than a hand-copied list.
    """
    from keybo.cli.analyze import _EXTRA_NAMED
    from keybo.layouts import NAMED_LAYOUTS

    return {**NAMED_LAYOUTS, **_EXTRA_NAMED}


def main() -> int:
    corpus_dir = production_corpus_dir("iweb")
    bigrams, _sk, _tri = _shared_corpora(corpus_dir)
    scorer = BadScissor(bigrams)

    # POSITIVE CONTROL FIRST, before any ratio is used: space must actually be present in
    # the corpus mass, else "space-excluded vs space-including" is a distinction with no
    # measurable difference and every ratio below would be 1.0 for a trivial reason.
    total = sum(bigrams.values())
    space_mass = sum(f for bg, f in bigrams.items() if " " in bg and len(bg) == 2)
    print("=== POSITIVE CONTROL ===")
    print(f"  corpus bigram mass            = {total}")
    print(f"  space-touching mass           = {space_mass}  ({100.0 * space_mass / total:.4f}%)")
    assert space_mass > 0, "no space-touching bigrams: the two denominators are identical"
    print("  PASS: space-touching mass is nonzero, so the two denominators DIFFER\n")

    rows = {}
    for label, lay in sorted(registry_layouts().items()):
        if len(lay) != 30:
            rows[label] = {"skipped": f"len={len(lay)}"}
            continue
        layout = Layout(lay, ROW_STAGGERED_30)
        correct = scorer.share(layout)                        # space-EXCLUDED (shipped)
        wrong = scorer.share(layout, exclude_space=False)      # oxey convention
        assert_operands_computed([correct, wrong], f"{label} shares")
        rows[label] = {
            "share_space_excluded_SHIPPED": correct,
            "share_space_including_oxey": wrong,
            "correct_over_wrong": correct / wrong if wrong else float("nan"),
            "oxey_denominator_DEFLATES": wrong < correct,
        }

    scored = {k: v for k, v in rows.items() if "skipped" not in v}
    assert_discriminating([v["share_space_excluded_SHIPPED"] for v in scored.values()],
                          "shipped shares across layouts")

    print("=== DIRECTION, on every registry layout (space-excluded = SHIPPED) ===")
    print(f"  {'layout':16s} {'excluded':>10s} {'including':>10s} {'exc/inc':>9s}  oxey deflates?")
    for label, v in scored.items():
        print(f"  {label:16s} {v['share_space_excluded_SHIPPED']:10.5f} "
              f"{v['share_space_including_oxey']:10.5f} {v['correct_over_wrong']:9.5f}  "
              f"{v['oxey_denominator_DEFLATES']}")

    ratios = [v["correct_over_wrong"] for v in scored.values()]
    all_deflate = all(v["oxey_denominator_DEFLATES"] for v in scored.values())
    print(f"\n  ratio range: {min(ratios):.6f} .. {max(ratios):.6f}  (n={len(ratios)})")
    print(f"  oxey denominator DEFLATES on every layout: {all_deflate}")
    print(f"  => the module docstring's word 'inflates' is "
          f"{'WRONG (backwards)' if all_deflate else 'right'}")

    # And the named pinning test: does it exist?
    import re

    from keybo.analysis import bad_scissor as BS
    named = re.findall(r"``(test_[a-z0-9_]+)``", BS.__doc__ or "")
    testsrc = (ROOT / "tests/analysis/test_bad_scissor.py").read_text()
    actual = re.findall(r"^def (test_\w+)", testsrc, re.M)
    print("\n=== DOES THE NAMED PINNING TEST EXIST? ===")
    missing = []
    for n in named:
        ok = n in actual
        print(f"  {'OK  ' if ok else 'FAIL'} {n}")
        if not ok:
            missing.append(n)
    print(f"  the suite's actual denominator test: "
          f"{[a for a in actual if 'denominator' in a and '497' in a]}")

    out = {
        "corpus": str(corpus_dir),
        "space_touching_pct_of_bigram_mass": 100.0 * space_mass / total,
        "ratio_min": min(ratios), "ratio_max": max(ratios), "n_layouts": len(ratios),
        "oxey_deflates_on_every_layout": all_deflate,
        "docstring_named_tests": named,
        "docstring_named_tests_missing": missing,
        "suite_actual_test_names": actual,
        "per_layout": rows,
    }
    p = ROOT / "agent-artifacts/bsaudit/denominator_direction.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
