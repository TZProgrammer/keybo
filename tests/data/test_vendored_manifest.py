"""The vendored community data's identity must be DECLARED, not merely enforced.

VENDPROV-1: substituting a plausible sibling from any upstream is CAUGHT 8/8 by the KAN-1 parity
goldens — but no digest was recorded anywhere, so a swap could be detected yet never NAMED (one
payload internally self-identifies as "shai", and upstream ships two variants differing by ~16 in
char_total). The manifest declares the digests; this test binds the bytes to the declaration.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

VENDORED = Path(__file__).resolve().parents[2] / "data" / "community" / "vendored"


def _manifest() -> dict:
    return json.loads((VENDORED / "manifest.json").read_text())


def test_every_vendored_file_is_declared_and_every_declaration_has_a_file() -> None:
    on_disk = {p.name for p in VENDORED.glob("*.json.gz")}
    declared = set(_manifest()["files"])
    assert on_disk == declared, f"disk vs manifest drift: {on_disk ^ declared}"


def test_container_and_payload_digests_match_the_declaration() -> None:
    for name, meta in _manifest()["files"].items():
        raw = (VENDORED / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == meta["sha256_container"], name
        assert len(raw) == meta["bytes"], name
        payload = gzip.decompress(raw)
        assert hashlib.sha256(payload).hexdigest() == meta["sha256_payload"], name


def test_the_shai_ambiguity_is_pinned_by_payload_not_by_internal_name() -> None:
    """The oxey1 payload says name="shai" though the bytes are english.json's — the exact case
    where an internal label cannot identify the file and only the digest can."""
    raw = (VENDORED / "oxeylyzer1-english.json.gz").read_bytes()
    payload = json.loads(gzip.decompress(raw))
    internal = payload.get("name") or payload.get("language")
    declared = _manifest()["files"]["oxeylyzer1-english.json.gz"]["sha256_payload"]
    assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == declared
    # If upstream ever renames the payload, this assert documents WHY we do not key on it.
    assert internal == "shai", (
        f"internal name changed ({internal!r}) — re-verify the vendored source"
    )


def test_the_provenance_and_notice_files_exist_beside_the_data() -> None:
    """KAN-1's registered deliverable: 'vendored ... with provenance notes'."""
    assert (VENDORED / "PROVENANCE.md").is_file()
    assert (VENDORED / "NOTICE").is_file()
