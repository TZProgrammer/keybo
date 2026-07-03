"""Tests for the dataset downloader.

These never touch the network or the real 1.5 GB archive. A local threaded HTTP server
serves a tiny fixture zip (with byte-range support), so we can exercise the real download +
resume + extract code paths deterministically and fast.
"""

import http.server
import io
import socketserver
import threading
import zipfile
from pathlib import Path

import pytest

from keybo.data.download import (
    KEYSTROKES_URL,
    download_file,
    extract_keystrokes,
    fetch_keystrokes,
)


def _make_keystrokes_zip() -> bytes:
    """A minimal archive mirroring the real one's layout: Keystrokes/files/..."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Keystrokes/files/1_keystrokes.txt", "PARTICIPANT_ID\tLETTER\n1\ta\n")
        z.writestr(
            "Keystrokes/files/metadata_participants.txt", "PARTICIPANT_ID\tLAYOUT\n1\tqwerty\n"
        )
    return buf.getvalue()


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    payload = b""  # set per-server

    def log_message(self, *args):  # silence
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            body = self.payload[start:]
            self.send_response(206)
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(self.payload)


@pytest.fixture
def zip_server():
    payload = _make_keystrokes_zip()
    handler = type("H", (_RangeHandler,), {"payload": payload})
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/Keystrokes.zip", payload
    finally:
        server.shutdown()
        server.server_close()


def test_url_points_at_aalto():
    assert KEYSTROKES_URL.startswith("https://")
    assert "aalto" in KEYSTROKES_URL
    assert KEYSTROKES_URL.endswith(".zip")


def test_download_file_fetches_full_payload(tmp_path, zip_server):
    url, payload = zip_server
    dest = tmp_path / "Keystrokes.zip"
    download_file(url, str(dest), show_progress=False)
    assert dest.read_bytes() == payload


def test_download_is_idempotent_when_already_complete(tmp_path, zip_server):
    url, payload = zip_server
    dest = tmp_path / "Keystrokes.zip"
    dest.write_bytes(payload)  # already fully downloaded
    mtime_before = dest.stat().st_mtime
    download_file(url, str(dest), show_progress=False)
    # A complete file is left untouched (no re-download).
    assert dest.read_bytes() == payload
    assert dest.stat().st_mtime == mtime_before


def test_download_resumes_from_partial(tmp_path, zip_server):
    url, payload = zip_server
    dest = tmp_path / "Keystrokes.zip"
    dest.write_bytes(payload[:10])  # a partial download
    download_file(url, str(dest), show_progress=False)
    assert dest.read_bytes() == payload


def test_extract_unpacks_keystrokes_tree(tmp_path):
    zip_path = tmp_path / "Keystrokes.zip"
    zip_path.write_bytes(_make_keystrokes_zip())
    files_dir = extract_keystrokes(str(zip_path), str(tmp_path / "out"))
    files_dir = Path(files_dir)
    assert files_dir.is_dir()
    assert files_dir.name == "files"
    assert (files_dir / "1_keystrokes.txt").exists()
    assert (files_dir / "metadata_participants.txt").exists()


def test_fetch_keystrokes_end_to_end(tmp_path, zip_server):
    url, _ = zip_server
    files_dir = fetch_keystrokes(str(tmp_path / "dataset"), url=url, show_progress=False)
    files_dir = Path(files_dir)
    assert (files_dir / "metadata_participants.txt").exists()
    assert (files_dir / "1_keystrokes.txt").exists()


def test_fetch_skips_extract_when_already_present(tmp_path, zip_server):
    url, _ = zip_server
    out = tmp_path / "dataset"
    first = Path(fetch_keystrokes(str(out), url=url, show_progress=False))
    # Drop a marker; a second fetch must not wipe/re-extract over it.
    marker = first / "MARKER"
    marker.write_text("x")
    second = Path(fetch_keystrokes(str(out), url=url, show_progress=False))
    assert second == first
    assert marker.exists()
