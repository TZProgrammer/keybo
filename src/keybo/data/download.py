"""Download and extract the public 136M Keystrokes dataset (Aalto University).

The raw keystroke dump is what :mod:`keybo.data.keystrokes` (``keybo process-data``) consumes
to build the bistroke/tristroke training tables. It is ~1.5 GB, so the download is resumable
(HTTP range request) and idempotent: an already-complete file is left alone, and an already-
extracted tree is not re-extracted.

Source: Dhakal, Feit, Kristensson, Oulasvirta, "Observations on Typing from 136 Million
Keystrokes" (CHI 2018). https://userinterfaces.aalto.fi/136Mkeystrokes/

Note: this fetches the *keystroke* dataset only. The n-gram frequency files under
``data/corpus/`` derive from the iWeb corpus (licensed, not freely downloadable) and are
committed in the repo already, so there is nothing to fetch for them.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile

KEYSTROKES_URL = "https://userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip"

_CHUNK = 1 << 20  # 1 MiB


def _remote_size(url: str) -> int | None:
    """Total size of the remote file in bytes, or None if the server won't say."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted, https)
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None


def download_file(url: str, dest: str, show_progress: bool = True) -> None:
    """Download ``url`` to ``dest``, resuming a partial file and skipping a complete one.

    Uses a byte-range request to continue an interrupted download. If ``dest`` already
    matches the remote size, it is left untouched.
    """
    total = _remote_size(url)
    have = os.path.getsize(dest) if os.path.exists(dest) else 0

    if total is not None and have == total:
        return  # already complete — nothing to do
    if total is not None and have > total:
        have = 0  # local file is larger than remote (stale/corrupt) — start over

    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted, https)
        # If we asked for a range but the server sent the whole file (200, not 206),
        # ignore our partial progress and overwrite from the start.
        resuming = have > 0 and resp.status == 206
        mode = "ab" if resuming else "wb"
        initial = have if resuming else 0
        progress = _make_progress(total, initial, show_progress)
        with open(dest, mode) as f:
            while chunk := resp.read(_CHUNK):
                f.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))
    if progress is not None:
        progress.close()


def _make_progress(total: int | None, initial: int, show: bool):
    if not show:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(
        total=total,
        initial=initial,
        unit="B",
        unit_scale=True,
        desc="Keystrokes.zip",
    )


def _find_files_dir(root: str) -> str | None:
    """Locate the ``.../files`` directory (holding *_keystrokes.txt) under ``root``."""
    for dirpath, dirnames, _ in os.walk(root):
        if os.path.basename(dirpath) == "files" and "metadata_participants.txt" in os.listdir(
            dirpath
        ):
            return dirpath
        # Common shape: <root>/Keystrokes/files
        if "files" in dirnames and os.path.basename(dirpath) == "Keystrokes":
            candidate = os.path.join(dirpath, "files")
            if os.path.exists(os.path.join(candidate, "metadata_participants.txt")):
                return candidate
    return None


def extract_keystrokes(zip_path: str, out_dir: str, progress: bool = False) -> str:
    """Extract the archive into ``out_dir`` and return the path to its ``files`` directory.

    The real archive holds ~168k member files (minutes of otherwise-silent work), so with
    ``progress`` a tqdm bar tracks members extracted. Per-member ``ZipFile.extract`` performs
    the same path sanitization as ``extractall`` (which just loops over it internally).
    """
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        members = z.infolist()
        iterator = members
        if progress:
            from tqdm import tqdm

            iterator = tqdm(members, desc="extracting", unit="file", leave=False)
        for member in iterator:
            z.extract(member, out_dir)
    files_dir = _find_files_dir(out_dir)
    if files_dir is None:
        raise FileNotFoundError(
            f"extracted archive under {out_dir} but found no 'files' dir with "
            "metadata_participants.txt"
        )
    return files_dir


def fetch_keystrokes(
    out_dir: str = "dataset",
    url: str = KEYSTROKES_URL,
    force: bool = False,
    show_progress: bool = True,
) -> str:
    """Download and extract the keystroke dataset into ``out_dir``.

    Returns the path to the ``files`` directory that ``keybo process-data --files-dir``
    expects. Idempotent: skips extraction when an extracted tree is already present (unless
    ``force``). Only the download itself resumes; a complete download is not refetched.
    """
    if not force:
        existing = _find_files_dir(out_dir)
        if existing is not None:
            return existing

    zip_path = os.path.join(out_dir, "Keystrokes.zip")
    download_file(url, zip_path, show_progress=show_progress)
    return extract_keystrokes(zip_path, out_dir, progress=show_progress)
