"""Load n-gram frequency tables (the English corpus counts used to weight the objective).

File format is one ``ngram<TAB>count`` per line. The n-gram may contain a space (the space
key), so the key is taken verbatim from before the tab and is NOT stripped of surrounding
whitespace — ``"e "`` (e then space) is a different bigram from ``"e"``.

Fixes bug #4: loaders take explicit file paths and a missing path raises, rather than the
old code's silent-empty behavior when a literal placeholder string was passed by mistake.

**Which corpus is production (CORPUS-SWAP-1).** :func:`production_corpus_dir` is the single
answer; it used to be hardcoded at eight call sites. The default is now ``blend-v1`` — a
four-register blend (iWeb anchor 0.50 / prose 0.25 / code 0.15 / reference 0.10) rather than
iWeb alone. Three properties make that swap safe, and each is enforced here:

* **iWeb stays reachable by name.** Every frozen board in the campaign was computed on
  iWeb, so ``KEYBO_CORPUS=iweb`` (or ``--corpus iweb``) must keep reproducing them. A
  default that made those numbers unreachable would have destroyed the audit trail.
* **A wrong name fails loudly.** There is no silent fallback to the default: an unknown
  name, or a directory missing a table, raises ``SystemExit`` naming what it looked for.
  Scoring the wrong corpus quietly is the failure this whole module exists to prevent.
* **The resolved corpus is identifiable.** :func:`corpus_identity` returns the name, the
  path, a sha256 per table and the manifest's declared total, for embedding in report
  output. The hash is what makes it a fact rather than a label: a *modified* table cannot
  masquerade as a known corpus. Names alone are how two corpora get stitched into one
  table.

⚠ **The two skipgram tables are not interchangeable in iWeb.** ``1-skip31.txt`` IS the
trigram marginalization ``skip(a,c) = sum_b tri(a,b,c)`` and is the table every frozen board
was computed on; iWeb's ``1-skip.txt`` is a different, unreproducible pass (3474 vs 4087
keys). ``blend-v1`` writes the marginalization under BOTH names, so it drops into either
loader path — but an iWeb-vs-blend comparison must still pin ``1-skip31.txt`` at both ends
or it confounds the corpus change with a skipgram-convention change.

⚠ **Swapping the corpus does not re-fit anything.** The measured-keystroke surface and the
three fitted model surfaces are baked artifacts (90 WPM); changing the corpus changes the
frequency *weighting* of the objective, not the timing model underneath it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

#: Environment variable naming the corpus to score on (a known name or a directory path).
CORPUS_ENV_VAR = "KEYBO_CORPUS"

#: The single-source iWeb tables (Davies 2018): licensed, non-redistributable, and NOT
#: regenerable — no extraction script was ever committed. Kept as the anchor register of
#: every blend and as the name that reproduces the campaign's frozen boards.
IWEB = "iweb"

#: The production corpus as of CORPUS-SWAP-1. See ``data/corpus/blend-v1/PROVENANCE.md``.
PRODUCTION_DEFAULT = "blend-v1"

#: Known corpus name -> directory, relative to the repo root. A name not listed here is
#: still usable as an explicit path; the registry exists so the common cases are one word.
_CORPUS_DIRS: dict[str, tuple[str, ...]] = {
    IWEB: ("data", "corpus"),
    PRODUCTION_DEFAULT: ("data", "corpus", "blend-v1"),
}

#: Every table a corpus directory must provide to be usable as production. ``1-skip.txt``
#: and ``1-skip31.txt`` are BOTH required: different call sites load different ones, and a
#: directory that supplies only one silently changes the skipgram convention per gauge.
REQUIRED_TABLES: tuple[str, ...] = ("bigrams.txt", "trigrams.txt", "1-skip.txt", "1-skip31.txt")

#: The skipgram table the campaign's frozen gauge boards were computed on.
PRODUCTION_SKIPGRAMS = "1-skip31.txt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def known_corpora() -> tuple[str, ...]:
    """The corpus names that resolve without a path (sorted)."""
    return tuple(sorted(_CORPUS_DIRS))


def resolve_corpus_dir(name_or_path: str) -> Path:
    """Resolve a corpus name or directory path, or exit with a message naming the options.

    Fails loudly on both failure modes — an unknown name and a directory missing a table —
    because the alternative (falling back to the default) scores the wrong corpus while
    reporting success.
    """
    if name_or_path in _CORPUS_DIRS:
        resolved = _repo_root().joinpath(*_CORPUS_DIRS[name_or_path])
    else:
        candidate = Path(name_or_path).expanduser()
        if not candidate.is_dir():
            raise SystemExit(
                f"unknown corpus {name_or_path!r}: not one of "
                f"{', '.join(known_corpora())} and not an existing directory"
            )
        resolved = candidate

    missing = [table for table in REQUIRED_TABLES if not (resolved / table).is_file()]
    if missing:
        raise SystemExit(
            f"corpus directory {str(resolved)!r} is missing {', '.join(missing)}; a corpus "
            f"must provide all of {', '.join(REQUIRED_TABLES)}"
        )
    return resolved


def production_corpus_dir(override: str | None = None) -> Path:
    """The corpus directory to score on: explicit argument, else env var, else the default.

    The precedence matters: a CLI flag must beat an inherited ``KEYBO_CORPUS`` export, or a
    stale shell environment silently overrides what the user typed.
    """
    selected = override or os.environ.get(CORPUS_ENV_VAR) or PRODUCTION_DEFAULT
    return resolve_corpus_dir(selected)


def corpus_name_for(directory: Path | str) -> str:
    """The registry name of a corpus directory, or ``"custom"`` for an unregistered one."""
    resolved = Path(directory).resolve()
    for name, parts in _CORPUS_DIRS.items():
        if _repo_root().joinpath(*parts).resolve() == resolved:
            return name
    return "custom"


def corpus_identity(directory: Path | str) -> dict:
    """A provenance block naming and hashing the corpus, for embedding in report output.

    Reports carry this so a number can always be traced to the tables that produced it —
    ``corpus`` is the label, ``sha256`` is the fact. Cost is ~1 ms for all four tables.
    ``declared_total`` comes from the corpus's own ``manifest.json`` and is ``None`` for a
    corpus that ships none (iWeb), never a total borrowed from a different corpus.
    """
    resolved = Path(directory)
    digests = {}
    for table in REQUIRED_TABLES:
        path = resolved / table
        if path.is_file():
            digests[table] = hashlib.sha256(path.read_bytes()).hexdigest()

    declared_total = None
    manifest_path = resolved / "manifest.json"
    if manifest_path.is_file():
        try:
            declared_total = json.loads(manifest_path.read_text())["declared_total"]
        except (json.JSONDecodeError, KeyError, OSError):
            declared_total = None

    return {
        "corpus": corpus_name_for(resolved),
        "path": resolved.as_posix(),
        "sha256": digests,
        "declared_total": declared_total,
        "skipgram_table": PRODUCTION_SKIPGRAMS,
    }


def load_frequencies(path: str) -> dict[str, int]:
    """Load one ``ngram<TAB>count`` frequency file into a dict.

    Raises ``FileNotFoundError`` if the path does not exist (so a wrong path fails loudly
    instead of yielding an empty table).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"frequency file not found: {path}")

    freqs: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            ngram, _, count = line.partition("\t")
            count = count.strip()
            if not ngram or not count:
                continue
            try:
                freqs[ngram] = int(count)
            except ValueError:
                continue
    return freqs


@dataclass(frozen=True)
class Corpus:
    """The three frequency tables the objective and models draw on."""

    trigrams: dict[str, int]
    bigrams: dict[str, int]
    skipgrams: dict[str, int]


def load_corpus(trigrams: str, bigrams: str, skipgrams: str) -> Corpus:
    """Load all three frequency tables from their explicit paths."""
    return Corpus(
        trigrams=load_frequencies(trigrams),
        bigrams=load_frequencies(bigrams),
        skipgrams=load_frequencies(skipgrams),
    )
