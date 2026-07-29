"""Triage the 16 mutation survivors: EQUIVALENT (unkillable) vs a real TEST GAP.

A survivor is only a test-coverage defect if the mutant CHANGES BEHAVIOUR. If the mutated
program is semantically identical to the original, no test could ever kill it and counting it
against the suite is a false accusation. So classify each survivor by DIFFING BEHAVIOUR
directly over the exhaustive pair space and over real corpus shares.

Three survivor classes to separate:
  * the 7 ``f(g, a, b) -> f(g, b, a)`` argument-order swaps,
  * the 5 ``FINGER_ORDER`` label mutants (L-ring/L-middle/R-middle/R-ring/R-pinky) — note
    L-pinky, L-index and R-index were CAUGHT, so the label set is only partly pinned,
  * the ``_weak_and_strong`` hand-boundary constant ``weak_x < 0`` -> ``weak_x < 1`` and the
    three ``_check_geometry`` message strings.
"""

from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path

from keybo.testkit import assert_module_under

ROOT = Path("/tmp/bsaudit")
assert_module_under("keybo", ROOT)

TARGET = ROOT / "src/keybo/analysis/bad_scissor.py"


def behaviour_fingerprint(mod) -> dict:
    """Everything observable about the module, over the exhaustive pair space + a corpus."""
    from keybo.cli.analyze import _EXTRA_NAMED, _shared_corpora, production_corpus_dir
    from keybo.geometry import ROW_STAGGERED_30 as G
    from keybo.layout import Layout
    from keybo.layouts import NAMED_LAYOUTS

    slots = sorted(G.slots)
    pred, finger, cell = [], [], []
    for a, b in itertools.product(slots, slots):
        pred.append(mod.bad_scissor(G, a, b))
        finger.append(mod.bad_scissor_finger(G, a, b))
        cell.append(mod.bad_scissor_cell(G, a, b))

    bigrams, _sk, _tri = _shared_corpora(production_corpus_dir("iweb"))
    scorer = mod.BadScissor(bigrams)
    shares, byfinger, bycell = {}, {}, {}
    for label, lay in sorted({**NAMED_LAYOUTS, **_EXTRA_NAMED}.items()):
        if len(lay) != 30:
            continue
        L = Layout(lay, G)
        shares[label] = scorer.share(L)
        byfinger[label] = scorer.by_finger(L)
        bycell[label] = scorer.by_cell(L)
    return {
        "predicate": pred, "finger": finger, "cell": cell,
        "shares": shares, "by_finger": byfinger, "by_cell": bycell,
        "FINGER_ORDER": list(mod.FINGER_ORDER),
        "ATTRIBUTION_RULE": mod.ATTRIBUTION_RULE,
    }


def load_variant(source: str, name: str):
    """Import a source string as a fresh module (so the original stays untouched on disk)."""
    import sys
    import types
    mod = types.ModuleType(name)
    mod.__file__ = str(TARGET)  # keeps any relative expectations sane
    sys.modules[name] = mod
    exec(compile(source, str(TARGET), "exec"), mod.__dict__)
    return mod


def apply_mutant(src: str, kind: str, lineno: int, before: str) -> str | None:
    """Re-apply one mutant, located by (kind, lineno, unparsed-text) rather than a line number
    alone — a bare line number is the citation form that rots."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != lineno:
            continue
        try:
            if ast.unparse(node) != before:
                continue
        except Exception:
            continue
        if kind == "swap_args" and isinstance(node, ast.Call) and len(node.args) >= 2:
            node.args[-1], node.args[-2] = node.args[-2], node.args[-1]
        elif kind == "const" and isinstance(node, ast.Constant):
            v = node.value
            node.value = (not v) if isinstance(v, bool) else (
                v + 1 if isinstance(v, (int, float)) else v + "X")
        elif kind == "cmp" and isinstance(node, ast.Compare):
            swaps = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.GtE: ast.Lt,
                     ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.In: ast.NotIn, ast.NotIn: ast.In,
                     ast.Is: ast.IsNot, ast.IsNot: ast.Is}
            op = type(node.ops[0])
            if op not in swaps:
                return None
            node.ops = [swaps[op]()]
        else:
            return None
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    return None


def main() -> int:
    original = TARGET.read_text()
    sweep = json.loads((ROOT / "agent-artifacts/bsaudit/mutation_sweep.json").read_text())
    survivors = [r for r in sweep["results"] if r["verdict"] == "SURVIVED"]

    base_mod = load_variant(original, "_bs_base")
    base = behaviour_fingerprint(base_mod)

    # POSITIVE CONTROL on the differ itself, BEFORE using it: a mutant known to change
    # behaviour must be reported as DIFFERENT. Without this, "no diff" is uninformative.
    ctl_src = apply_mutant(original, "cmp", None, None) if False else None
    ctl_tree = ast.parse(original)
    for n in ast.walk(ctl_tree):
        if isinstance(n, ast.FunctionDef) and n.name == "bad_scissor":
            n.body = [n.body[0], ast.Return(value=ast.Constant(value=False))]
    ast.fix_missing_locations(ctl_tree)
    ctl_src = ast.unparse(ctl_tree)
    ctl = behaviour_fingerprint(load_variant(ctl_src, "_bs_ctl"))
    ctl_differs = ctl["predicate"] != base["predicate"]
    print("=== POSITIVE CONTROL on the behaviour differ (runs BEFORE any verdict) ===")
    print(f"  'bad_scissor -> return False' detected as behaviour-changing: {ctl_differs}")
    assert ctl_differs, "the differ cannot see a fatal change; its EQUIVALENT verdicts are junk"
    print("  PASS\n")

    out = []
    print("=== SURVIVOR TRIAGE ===")
    for i, r in enumerate(survivors, 1):
        src = apply_mutant(original, r["kind"], r["line"], r["before"])
        if src is None:
            out.append({**r, "class": "COULD-NOT-REAPPLY"})
            print(f"  [{i:2d}] ?? could not re-apply {r['kind']} L{r['line']} {r['before'][:50]}")
            continue
        try:
            mod = load_variant(src, f"_bs_mut{i}")
            fp = behaviour_fingerprint(mod)
        except Exception as e:  # a mutant that cannot even import changes behaviour
            out.append({**r, "class": "IMPORT-ERROR", "detail": repr(e)[:200]})
            print(f"  [{i:2d}] IMPORT-ERROR {r['kind']} L{r['line']}")
            continue

        diffs = {k: (fp[k] != base[k]) for k in
                 ("predicate", "finger", "cell", "shares", "by_finger", "by_cell",
                  "FINGER_ORDER", "ATTRIBUTION_RULE")}
        observable = [k for k, v in diffs.items() if v]
        # A FINGER_ORDER-only diff changes the *reported dict keys*, which IS observable
        # output even though every share value is unchanged.
        cls = "EQUIVALENT" if not observable else "TEST-GAP"
        out.append({**r, "class": cls, "observable_diffs": observable})
        print(f"  [{i:2d}] {cls:11s} {r['kind']:10s} L{r['line']:<4d} "
              f"{r['before'][:52]!r} -> {r['after'][:40]!r}")
        if observable:
            print(f"       observable: {observable}")
            if "FINGER_ORDER" in observable:
                print(f"       keys base    : {base['FINGER_ORDER']}")
                print(f"       keys mutant  : {fp['FINGER_ORDER']}")
            if "by_finger" in observable:
                bl = next(iter(fp["by_finger"]))
                print(f"       by_finger[{bl}] base   = {base['by_finger'][bl]}")
                print(f"       by_finger[{bl}] mutant = {fp['by_finger'][bl]}")

    eq = sum(1 for r in out if r["class"] == "EQUIVALENT")
    gap = sum(1 for r in out if r["class"] == "TEST-GAP")
    print(f"\n=== {eq} EQUIVALENT (unkillable) · {gap} REAL TEST GAP ===")
    for r in out:
        if r["class"] == "TEST-GAP":
            print(f"  GAP {r['kind']:10s} L{r['line']:<4d} {r['before'][:60]!r} "
                  f"-> {r['after'][:50]!r}  diffs={r['observable_diffs']}")

    assert TARGET.read_text() == original, "the target file was modified — it must not be"
    p = ROOT / "agent-artifacts/bsaudit/triage_survivors.json"
    p.write_text(json.dumps({"equivalent": eq, "test_gap": gap, "survivors": out}, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
