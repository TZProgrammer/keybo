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
import re
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


#: Every repo the manifest cites must be spelled the SAME WAY in all three documents. This exists
#: because it did not hold: NOTICE and manifest.json said `github.com/O-X-L/oxeylyzer` while the pinned
#: checkouts' real remotes (and PROVENANCE.md) say `github.com/o-x-e-y/` — missing an E, not a case
#: variant — so the ONE document written to discharge Apache-2.0 attribution named the WRONG
#: ATTRIBUTEE. A fresh reader found it by setting `upstream` to `github.com/EVIL/typosquat`, and the
#: suite PASSED. `upstream` was declared and unenforced; these tests enforce it.
_DOCS = ("PROVENANCE.md", "NOTICE", "manifest.json")


def _text(name: str) -> str:
    return (VENDORED / name).read_text()


def test_the_provenance_and_notice_files_have_CONTENT_not_just_a_name() -> None:
    """`is_file()` checks the NAME, not the THING — truncating either file to empty used to PASS.

    That is the campaign's whole signature (a name diverging from the thing it names) reproduced inside
    the test written to bind a registered deliverable. KAN-1 registered this data as vendored "with
    provenance notes"; a zero-byte file satisfies `is_file()` and discharges nothing.
    """
    for name in ("PROVENANCE.md", "NOTICE"):
        path = VENDORED / name
        assert path.is_file(), name
        body = path.read_text()
        assert len(body) > 500, f"{name} is {len(body)} bytes — too short to be a provenance record"
        assert body.count("\n") > 10, f"{name} has no structure"


def test_every_declared_upstream_is_spelled_IDENTICALLY_across_all_three_documents() -> None:
    """The typosquat test. A repo cited in the manifest must appear verbatim in PROVENANCE.md.

    Scoped to the manifest's own declarations rather than a hardcoded list, so adding a vendored file
    cannot silently escape the check.
    """
    import json

    declared = {m["upstream"] for m in json.loads(_text("manifest.json"))["files"].values()}
    prov = _text("PROVENANCE.md")
    missing = sorted(u for u in declared if u not in prov)
    assert not missing, f"manifest cites upstreams absent from PROVENANCE.md: {missing}"


def test_the_third_party_upstreams_are_attributed_in_NOTICE_by_the_same_name() -> None:
    """NOTICE is the Apache-2.0 artifact: the third-party repos must be named there, spelled the same."""
    import json

    files = json.loads(_text("manifest.json"))["files"]
    third_party = {
        name: meta["upstream"]
        for name, meta in files.items()
        if "third-party" in meta.get("content", "")
    }
    assert third_party, "no third-party entries declared — the licence risk would be invisible"
    notice = _text("NOTICE")
    for name, upstream in sorted(third_party.items()):
        # WORD-BOUNDARY match, not `in`. A plain substring test cannot detect a dropped attribution
        # here, because "github.com/o-x-e-y/oxeylyzer" is a PREFIX of ".../oxeylyzer-2": deleting the
        # oxeylyzer-1 line entirely still left the oxeylyzer-2 line satisfying it, and that mutant
        # SURVIVED my first cut of this test. It is the same name-vs-thing shape as the defect the test
        # exists to catch — a check that matches something adjacent to what it means.
        pattern = re.compile(rf"{re.escape(upstream)}(?![\w./-])")
        assert pattern.search(notice), f"{name}: NOTICE does not name {upstream} as a distinct repo"
        assert name in notice, f"{name} is third-party but unnamed in NOTICE"


def test_each_declared_pin_appears_in_PROVENANCE_so_a_silent_re_pin_cannot_pass() -> None:
    """A pin is only a pin if the record and the manifest agree on it."""
    import json

    prov = _text("PROVENANCE.md")
    for name, meta in json.loads(_text("manifest.json"))["files"].items():
        pin = meta["pin"]
        # PROVENANCE.md may carry the full 40-char sha where the manifest carries a prefix.
        assert pin[:7] in prov, f"{name}: pin {pin} not found in PROVENANCE.md"


def test_the_unlicensed_file_is_flagged_as_such_in_BOTH_the_manifest_and_NOTICE() -> None:
    """oxeylyzer-2 has no licence anywhere upstream, so it has no redistribution grant on record.

    That is the one finding a reader must not miss, and a licence field silently changed to something
    reassuring is exactly the edit this test exists to catch.
    """
    import json

    files = json.loads(_text("manifest.json"))["files"]
    unlicensed = [n for n, m in files.items() if "NONE" in m["licence"].upper()]
    assert unlicensed == ["oxeylyzer2-english.json.gz"], f"licence flags changed: {unlicensed}"
    notice = _text("NOTICE")
    assert "NONE FOUND" in notice, "NOTICE no longer records the missing licence"
    assert "all-rights-reserved" in notice.lower(), "NOTICE no longer states the consequence"


def test_the_attribution_requirement_is_stated_where_it_applies() -> None:
    """Apache-2.0 permits redistribution but REQUIRES attribution; that has to be asserted somewhere."""
    import json

    files = json.loads(_text("manifest.json"))["files"]
    apache = [n for n, m in files.items() if "Apache" in m["licence"]]
    assert apache == ["oxeylyzer1-english.json.gz"], apache
    assert "Apache" in _text("NOTICE"), "NOTICE must name the licence it discharges"
