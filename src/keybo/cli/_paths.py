"""Shared CLI output-path validation.

Fail FAST: every workflow validates (and creates) its output location BEFORE the expensive
stage. Real incident (2026-07-04): `keybo train --output models/bigram.json` trained for
hours, then died in XGBoost's C++ writer because `models/` didn't exist — the README's own
example on a fresh clone (the directory is gitignored). Library-level saves also auto-mkdir
as a second layer, but the CLI check is what protects the user's time.
"""

from __future__ import annotations

import os


def ensure_writable_output(path: str, flag: str = "--output") -> None:
    """Create the output's parent directory now, or exit with a clear message.

    Raises ``SystemExit`` naming the flag when the parent can't be created (e.g. a path
    component is an existing regular file) or the target itself is a directory.
    """
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except (NotADirectoryError, FileExistsError, PermissionError, OSError) as e:
            raise SystemExit(
                f"unusable output path for {flag}: {path!r} — cannot create parent "
                f"directory {parent!r} ({e.__class__.__name__}: {e})"
            ) from e
    if os.path.isdir(path):
        raise SystemExit(f"unusable output path for {flag}: {path!r} is a directory")
