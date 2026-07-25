"""Build a MULTI-SOURCE n-gram frequency blend — the reproducible corpus generator.

Closes the gap the audit named (GAP-CORPUS-1): ``data/corpus/*.txt`` is a single-source
iWeb import with **no generator in the history**, so the tables cannot be regenerated, and
the stated requirement — *"a frequency list which sums to 1/100 and uses many different
corpuses"* — was never met. This module is the missing generator.

What is and is NOT reproducible — the load-bearing honesty note
--------------------------------------------------------------
iWeb is licensed and non-redistributable and **no extraction script was ever kept**, so the
committed iWeb counts *cannot* be regenerated from source text. They are therefore treated
here as ONE weighted input **anchor**: an opaque, unverifiable-but-documented component,
consumed as derived counts. Every OTHER component is built by this module from a named
local source with a documented extraction rule, byte count and SHA-256. So the blend is
reproducible **except for the anchor**, whose hash is recorded so at least its *identity* is
verifiable even though its *derivation* is not. Set ``--no-anchor`` for a fully reproducible
(anchor-free) blend.

Conventions pinned against the production loader and gauges
-----------------------------------------------------------
* **Format**: ``ngram<TAB>count``, one per line, count an integer — ``load_frequencies``
  parses with ``int()`` and *silently skips* anything else, so fractional "sums to 1"
  tables would load as empty. See ``DECLARED_TOTAL``.
* **Space is a real character** and n-grams are taken verbatim (``"e "`` != ``"e"``).
* **Charset** = the production charset (:data:`CORPUS_CHARSET`): space + 52 ASCII letters +
  ``" ' , - . / : ; < > ?`` — the ANSI 30-key universe plus its shifted forms. **Case is
  preserved**, because the iWeb tables preserve it (95% of their bigram mass is lowercase,
  so case carries real signal). A lowercase-folded component could not be summed with the
  anchor, so every component uses this same universe.
* **Skipgrams are derived by marginalization**: ``skip(a,c) = sum_b tri(a,b,c)``. Verified
  byte-exact against the committed ``1-skip31.txt`` (4087 entries, 0 mismatching keys), so
  this reproduces the production convention rather than inventing one. (``1-skip.txt`` is a
  *different*, unreproducible pass; both names are emitted so the blend drops into either
  loader path.)
* **Determinism**: no randomness anywhere — sources are walked in sorted order, ties in the
  apportionment break on ``(-remainder, ngram)``, and output rows sort by ``(-count, ngram)``.
  Re-running on the same inputs is byte-identical. There is no seed because nothing samples.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import platform
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: The production corpus charset, verified against ``data/corpus/*.txt`` (64 characters:
#: space + 52 ASCII letters + the 11 ANSI punctuation/shifted characters). Case-preserving.
CORPUS_CHARSET: frozenset[str] = frozenset(" \"',-./:;<>?" + string.ascii_letters)

#: Declared per-table total. Every emitted table sums to EXACTLY this, so ``count / 1e9``
#: sums to 1 and ``count / 1e7`` sums to 100 — the "sums to 1/100" requirement — while the
#: file itself stays integral, which ``load_frequencies``' ``int()`` parse requires.
#: (Ratio gauges normalize internally anyway — ``kmstats.stats`` divides by layout-restricted
#: totals, ``oxey.pattern_shares`` by its own corpus mass — so the scale is a *declaration*,
#: not a change in meaning: it cannot move any ranking.)
DECLARED_TOTAL: int = 1_000_000_000

#: Table kind -> emitted filenames. ``1-skip.txt`` and ``1-skip31.txt`` get the same content
#: (the marginalized skipgrams) so the blend is a drop-in wherever either name is loaded.
TABLE_FILENAMES: dict[str, tuple[str, ...]] = {
    "bigrams": ("bigrams.txt",),
    "trigrams": ("trigrams.txt",),
    "skipgrams": ("1-skip.txt", "1-skip31.txt"),
}


# --------------------------------------------------------------------------- text handling


def normalize(text: str) -> str:
    """Map text onto the corpus charset, collapsing everything else to a single space.

    Out-of-charset characters become word boundaries rather than being deleted, so
    ``"foo(bar)"`` yields the bigram ``"o "`` and not the false bigram ``"ob"``. Runs of
    whitespace collapse to one space, matching a single inter-word gap.
    """
    kept = "".join(ch if ch in CORPUS_CHARSET else " " for ch in text)
    return re.sub(r" +", " ", kept)


def strip_markdown(s: str) -> str:
    """Markdown -> prose: drop fenced/inline code, URLs, and table/emphasis/link markup."""
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"~~~.*?~~~", " ", s, flags=re.S)
    s = re.sub(r"`[^`]*`", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"^\s*\|.*$", " ", s, flags=re.M)  # table rows are data, not prose
    s = re.sub(r"[|#*_>\[\]()]", " ", s)
    return s


def strip_latex(s: str) -> str:
    """LaTeX -> prose: drop comments, control sequences, and math/markup punctuation."""
    s = re.sub(r"(?<!\\)%.*", " ", s)
    s = re.sub(
        r"\\begin\{(equation|align|tabular|figure|lstlisting)\*?\}.*?\\end\{\1\*?\}",
        " ",
        s,
        flags=re.S,
    )
    s = re.sub(r"\\[a-zA-Z@]+\*?", " ", s)
    s = re.sub(r"[{}$&_^~\\]", " ", s)
    return s


def strip_python(s: str) -> str:
    """Python -> code: drop docstrings and comments, keeping the identifier-heavy code.

    The prose inside a Python file belongs to the *prose* register; removing it is what
    makes this component a genuinely distinct CODE register rather than a prose/code mix.
    """
    s = re.sub(r'"""(?:.|\n)*?"""', " ", s)
    s = re.sub(r"'''(?:.|\n)*?'''", " ", s)
    s = re.sub(r"(?m)#.*$", " ", s)
    return s


def strip_roff(s: str) -> str:
    """man/roff -> prose: drop comments, requests, and inline font/size escapes."""
    s = re.sub(r"(?m)^\.\\\".*$", " ", s)  # .\" comments
    s = re.sub(r"(?m)^[.'][a-zA-Z0-9]*.*$", lambda m: " " + _roff_request_text(m.group(0)), s)
    s = re.sub(r"\\f[BIRP]|\\f\(..|\\s[-+]?\d+|\\[&|,)/^%{}]", " ", s)
    s = re.sub(r"\\\(.{2}|\\\[[^\]]*\]", " ", s)
    s = re.sub(r"\\-", "-", s)
    return s


def _roff_request_text(line: str) -> str:
    """Keep the argument text of prose-bearing roff requests, drop the request itself."""
    parts = line.split(None, 1)
    if len(parts) < 2:
        return " "
    request = parts[0].lstrip(".'")
    # .SH/.SS/.TP/.IP/.B/.I/.BR... carry visible words; .TH/.de/.nr/.if carry metadata.
    if request in {"TH", "de", "nr", "if", "ie", "el", "ds", '\\"', ""}:
        return " "
    return parts[1]


# --------------------------------------------------------------------------- counting


def count_ngrams(text: str) -> tuple[Counter[str], Counter[str]]:
    """Sliding-window bigram and trigram counts over ``text`` (already normalized).

    Returns ``(bigrams, trigrams)``. Skipgrams are NOT counted here: they are derived from
    the trigrams by :func:`marginalize_skipgrams`, which is the production convention.
    """
    bigrams: Counter[str] = Counter()
    trigrams: Counter[str] = Counter()
    for i in range(len(text) - 1):
        bigrams[text[i : i + 2]] += 1
    for i in range(len(text) - 2):
        trigrams[text[i : i + 3]] += 1
    return bigrams, trigrams


def marginalize_skipgrams(trigrams: dict[str, int] | Counter[str]) -> Counter[str]:
    """``skip(a, c) = sum_b tri(a, b, c)`` — the committed ``1-skip31.txt`` convention."""
    skip: Counter[str] = Counter()
    for gram, count in trigrams.items():
        if len(gram) == 3:
            skip[gram[0] + gram[2]] += count
    return skip


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class Source:
    """One named corpus component: where its text comes from and how it is extracted.

    ``register`` groups sources that play the same linguistic role (``prose``, ``code``,
    ``reference``, ``anchor``); weights are declared per register in :data:`DEFAULT_WEIGHTS`.
    """

    name: str
    register: str
    extraction: str
    reader: Callable[[], Iterator[tuple[str, str]]] = field(repr=False)
    #: Resolved filesystem root the reader walked. Recorded in the manifest because a
    #: source like the Python stdlib is located via the *running interpreter*, so the
    #: path (and thus the text) is not reproducible unless the resolved root is pinned.
    root: Path | None = None
    optional: bool = False

    def collect(self) -> tuple[str, dict]:
        """Read every unit, returning ``(normalized_text, stats)`` with bytes and SHA-256.

        The hash is over the *raw* concatenated source bytes in sorted-unit order, so it
        identifies the inputs independently of any extraction-rule change.
        """
        digest = hashlib.sha256()
        raw_bytes = 0
        units = 0
        chunks: list[str] = []
        for label, raw in self.reader():
            digest.update(label.encode("utf-8"))
            digest.update(b"\0")
            encoded = raw.encode("utf-8", "replace")
            digest.update(encoded)
            raw_bytes += len(encoded)
            units += 1
            chunks.append(raw)
        text = normalize(" ".join(chunks))
        return text, {
            "root": str(self.root) if self.root else None,
            "units": units,
            "raw_bytes": raw_bytes,
            "sha256": digest.hexdigest(),
            "normalized_chars": len(text),
        }


def _read_files(
    paths: Iterable[Path], transform: Callable[[str], str], root: Path | None = None
) -> Iterator[tuple[str, str]]:
    """Yield ``(label, transformed_text)`` for each readable path, in sorted order."""
    for path in sorted(paths):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        label = str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name
        yield label, transform(raw)


def _read_gzipped(
    paths: Iterable[Path], transform: Callable[[str], str]
) -> Iterator[tuple[str, str]]:
    for path in sorted(paths):
        try:
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                raw = handle.read()
        except (OSError, EOFError, gzip.BadGzipFile):
            continue
        yield path.name, transform(raw)


#: Directories never walked for repo sources. These hold *transient, untracked* files — a
#: virtualenv's vendored ``LICENSE.md``, pytest's cache README, build output — whose presence
#: depends on whether anyone has run the tests yet. Including them made the build
#: non-reproducible (a rerun after ``pytest`` created ``.venv/`` picked up 16 extra files and
#: changed every count) AND contaminated the "repo prose" register with third-party licence
#: boilerplate. Caught by the determinism check, which is why that check exists.
_REPO_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "dist",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".tox",
        ".eggs",
        "site-packages",
    }
)


def _repo_files(repo: Path, pattern: str) -> list[Path]:
    """Version-controlled repo files matching ``pattern``, excluding transient directories."""
    return [
        p
        for p in repo.rglob(pattern)
        if not (_REPO_SKIP_DIRS & set(p.relative_to(repo).parts))
        and "data/corpus" not in p.relative_to(repo).as_posix()
    ]


def repo_prose_source(repo: Path) -> Source:
    """Register ``prose``: this repo's own Markdown — technical/scientific English."""

    def reader() -> Iterator[tuple[str, str]]:
        yield from _read_files(_repo_files(repo, "*.md"), strip_markdown, repo)

    return Source(
        name="repo-markdown",
        register="prose",
        extraction=(
            "all *.md tracked under the repo, sorted by path, excluding data/corpus and the "
            f"transient directories {sorted(_REPO_SKIP_DIRS)} (whose contents depend on "
            "whether the tests have been run); fenced+inline code, URLs, table rows and "
            "emphasis/link markup stripped (strip_markdown), then normalized onto the charset"
        ),
        reader=reader,
        root=repo,
    )


def repo_latex_source(repo: Path) -> Source:
    """Register ``prose``: the paper/poster LaTeX — formal academic English."""

    def reader() -> Iterator[tuple[str, str]]:
        yield from _read_files(_repo_files(repo, "*.tex"), strip_latex, repo)

    return Source(
        name="repo-latex",
        register="prose",
        extraction=(
            "all *.tex tracked under the repo, sorted by path, excluding the transient "
            "directories listed for repo-markdown; comments, math/tabular/figure environments "
            "and control sequences stripped (strip_latex), then normalized"
        ),
        reader=reader,
        root=repo,
    )


def python_stdlib_source(stdlib: Path) -> Source:
    """Register ``code``: the Python standard library — identifier-heavy source text."""

    def reader() -> Iterator[tuple[str, str]]:
        paths = [
            p
            for p in stdlib.rglob("*.py")
            if "site-packages" not in p.parts and "test" not in p.parts and "tests" not in p.parts
        ]
        yield from _read_files(paths, strip_python, stdlib)

    return Source(
        name="python-stdlib",
        register="code",
        extraction=(
            "all *.py under the CPython stdlib root excluding site-packages/ and test "
            "directories, sorted by path; docstrings and comments stripped (strip_python, "
            "so the prose inside code lands in the prose register, not here), then normalized"
        ),
        reader=reader,
        root=stdlib,
    )


def man_pages_source(man_root: Path, sections: Sequence[str] = ("man1", "man8")) -> Source:
    """Register ``reference``: man pages — terse imperative reference English."""

    def reader() -> Iterator[tuple[str, str]]:
        paths = [p for section in sections for p in (man_root / section).glob("*.gz")]
        yield from _read_gzipped(paths, strip_roff)

    return Source(
        name=f"man-pages-{'+'.join(sections)}",
        register="reference",
        extraction=(
            f"gzipped man pages in {'/'.join(sections)}, sorted by name; roff comments, "
            "metadata requests and font/size escapes stripped (strip_roff) while "
            "prose-bearing request arguments are kept, then normalized"
        ),
        reader=reader,
        root=man_root,
        optional=True,
    )


#: Register -> share of the blend. **Prose-dominant with a declared technical share**, so
#: the blend answers the requirement ("many different corpuses") without pretending a
#: keyboard for English prose should be optimized on code. Rationale per register:
#:
#: * ``anchor`` 0.50 — the iWeb component. Still the plurality, because it is the only
#:   component built from a *large, sampled, general-English* corpus (hundreds of millions
#:   of n-grams from the open web); the locally-built components are small and specialized.
#:   Halving it from 1.00 is the actual change: no single source can now decide a ranking.
#: * ``prose`` 0.25 — locally-built English prose (repo Markdown + LaTeX). Reproducible,
#:   and the register the layout is genuinely for.
#: * ``code`` 0.15 — the programmer register the audit predicted would diverge materially
#:   (and did: alt-code inverted rankings). A real but minority share: a general layout
#:   should feel this pull without being governed by it.
#: * ``reference`` 0.10 — terse command/reference English, a third distinct register that
#:   is neither flowing prose nor code.
DEFAULT_WEIGHTS: dict[str, float] = {
    "anchor": 0.50,
    "prose": 0.25,
    "code": 0.15,
    "reference": 0.10,
}


# --------------------------------------------------------------------------- blending


def apportion(shares: dict[str, float], total: int) -> dict[str, int]:
    """Round real-valued ``shares`` to integers summing to EXACTLY ``total``.

    Largest-remainder (Hamilton) apportionment: floor everything, then hand the leftover
    units to the largest fractional remainders, breaking ties on the n-gram string so the
    result is deterministic. Zero-share entries are dropped (a zero count is not a datum).
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")
    mass = sum(shares.values())
    if mass <= 0:
        return {}
    scaled = {k: v * total / mass for k, v in shares.items() if v > 0}
    floors = {k: int(v) for k, v in scaled.items()}
    leftover = total - sum(floors.values())
    if leftover:
        order = sorted(scaled, key=lambda k: (-(scaled[k] - floors[k]), k))
        for key in order[:leftover]:
            floors[key] += 1
    return {k: v for k, v in floors.items() if v > 0}


def blend_tables(
    components: dict[str, dict[str, int]],
    weights: dict[str, float],
    total: int = DECLARED_TOTAL,
) -> dict[str, int]:
    """Combine per-component count tables into one table summing to exactly ``total``.

    Each component is first converted to a **probability distribution over its own mass**,
    so a component's influence is set by its declared weight and not by how much text it
    happened to contain — the property that makes the weights meaningful. Weights are
    renormalized over the components actually present.
    """
    present = {name: w for name, w in weights.items() if name in components and w > 0}
    if not present:
        raise ValueError(f"no weighted component present: have {sorted(components)}")
    weight_mass = sum(present.values())
    shares: dict[str, float] = {}
    for name, weight in present.items():
        table = components[name]
        component_mass = sum(table.values())
        if component_mass <= 0:
            continue
        factor = (weight / weight_mass) / component_mass
        for gram, count in table.items():
            shares[gram] = shares.get(gram, 0.0) + count * factor
    return apportion(shares, total)


def write_table(path: Path, table: dict[str, int]) -> None:
    """Write ``ngram<TAB>count`` sorted by descending count then n-gram (deterministic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for gram, count in sorted(table.items(), key=lambda kv: (-kv[1], kv[0])):
            handle.write(f"{gram}\t{count}\n")


def load_anchor(corpus_dir: Path) -> dict[str, dict[str, int]]:
    """Load the committed iWeb tables as the ``anchor`` component.

    The anchor's skipgrams are re-derived from its own trigrams (not read from
    ``1-skip.txt``) so the blend is internally consistent under one skipgram convention —
    the same marginalization every other component uses.
    """
    from keybo.data.corpus import load_frequencies

    bigrams = load_frequencies(str(corpus_dir / "bigrams.txt"))
    trigrams = load_frequencies(str(corpus_dir / "trigrams.txt"))
    return {
        "bigrams": bigrams,
        "trigrams": trigrams,
        "skipgrams": dict(marginalize_skipgrams(trigrams)),
    }


def anchor_manifest_entry(corpus_dir: Path) -> dict:
    """Provenance for the anchor: hashes of the files, and its honesty statement."""
    files = {}
    for name in ("bigrams.txt", "trigrams.txt"):
        path = corpus_dir / name
        data = path.read_bytes()
        files[name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return {
        "name": "iweb-anchor",
        "register": "anchor",
        "extraction": (
            "NOT REPRODUCIBLE — consumed as committed derived counts. The iWeb corpus "
            "(Davies 2018) is licensed and non-redistributable and no extraction script "
            "was ever committed, so these counts cannot be regenerated from source text. "
            "Their identity is pinned by the sha256 below; their derivation is unverifiable. "
            "Skipgrams re-derived from trigrams.txt by marginalization for consistency."
        ),
        "reproducible": False,
        "files": files,
    }


@dataclass
class BuildResult:
    """Everything a build produced: the tables and the manifest describing them."""

    tables: dict[str, dict[str, int]]
    manifest: dict


def build_blend(
    sources: Sequence[Source],
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    anchor_dir: Path | None = None,
    total: int = DECLARED_TOTAL,
) -> BuildResult:
    """Build the blended tables plus a manifest of sources, hashes and weights.

    Sources sharing a ``register`` are pooled *within* that register in proportion to their
    own n-gram mass, then registers are combined at the declared weights.
    """
    per_source: dict[str, dict[str, dict[str, int]]] = {}
    source_entries: list[dict] = []
    for source in sources:
        text, stats = source.collect()
        bigrams, trigrams = count_ngrams(text)
        if not bigrams:
            if source.optional:
                continue
            raise ValueError(f"source {source.name!r} produced no n-grams")
        per_source[source.name] = {
            "bigrams": dict(bigrams),
            "trigrams": dict(trigrams),
            "skipgrams": dict(marginalize_skipgrams(trigrams)),
        }
        source_entries.append(
            {
                "name": source.name,
                "register": source.register,
                "extraction": source.extraction,
                "reproducible": True,
                **stats,
                "ngrams": {
                    "bigram_types": len(bigrams),
                    "bigram_tokens": sum(bigrams.values()),
                    "trigram_types": len(trigrams),
                    "trigram_tokens": sum(trigrams.values()),
                },
            }
        )

    # Pool each register's sources by their own mass, giving one table per register.
    registers: dict[str, dict[str, dict[str, int]]] = {}
    for source in sources:
        if source.name not in per_source:
            continue
        target = registers.setdefault(source.register, {})
        for kind, table in per_source[source.name].items():
            merged = target.setdefault(kind, {})
            for gram, count in table.items():
                merged[gram] = merged.get(gram, 0) + count

    if anchor_dir is not None:
        registers["anchor"] = load_anchor(anchor_dir)
        source_entries.append(anchor_manifest_entry(anchor_dir))

    tables = {
        kind: blend_tables({r: t[kind] for r, t in registers.items() if kind in t}, weights, total)
        for kind in ("bigrams", "trigrams", "skipgrams")
    }

    effective = {r: w for r, w in weights.items() if r in registers and w > 0}
    weight_mass = sum(effective.values())
    manifest = {
        "schema": "keybo-corpus-manifest/1",
        # The stdlib source is located via the RUNNING interpreter, so an identical rerun
        # needs the same version — recorded here rather than left implicit.
        "built_with_python": platform.python_version(),
        "declared_total": total,
        "charset": "".join(sorted(CORPUS_CHARSET)),
        "charset_size": len(CORPUS_CHARSET),
        "case_preserved": True,
        "skipgram_convention": "skip(a,c) = sum_b trigram(a,b,c) (matches committed 1-skip31.txt)",
        "reproducible_without_anchor": "anchor" not in registers,
        "weights_declared": dict(weights),
        "weights_effective": {r: w / weight_mass for r, w in effective.items()},
        "sources": source_entries,
        "outputs": {
            kind: {
                "files": list(TABLE_FILENAMES[kind]),
                "types": len(table),
                "total": sum(table.values()),
            }
            for kind, table in tables.items()
        },
    }
    return BuildResult(tables=tables, manifest=manifest)


def default_sources(
    repo: Path, stdlib: Path | None = None, man_root: Path | None = None
) -> list[Source]:
    """The default source set: repo prose + LaTeX, Python stdlib, and man pages if present."""
    import sysconfig

    stdlib = stdlib or Path(sysconfig.get_paths()["stdlib"])
    sources = [repo_prose_source(repo), repo_latex_source(repo), python_stdlib_source(stdlib)]
    man_root = man_root or Path("/usr/share/man")
    if man_root.is_dir():
        sources.append(man_pages_source(man_root))
    return sources


def write_build(result: BuildResult, out_dir: Path) -> list[Path]:
    """Write every table (under all of its production names) plus ``manifest.json``."""
    written = []
    for kind, table in result.tables.items():
        for filename in TABLE_FILENAMES[kind]:
            path = out_dir / filename
            write_table(path, table)
            written.append(path)
    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(result.manifest, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    written.append(manifest_path)
    return written
