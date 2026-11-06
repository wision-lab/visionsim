"""Static file server with HTTP Range support, for previewing the landing page.

`python3 -m http.server` ignores Range headers, which makes <video> elements
non-seekable in Chromium (seekable === [0,0]) until the entire file is buffered.
That silently breaks any scroll-scrubbed video: `currentTime = x` is discarded
and the element keeps painting frame 0.

Usage:
    python3 site/tools/serve-range.py [port] [directory]

Defaults to serving the site/ directory it lives in.
"""

from __future__ import annotations

import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        header = self.headers.get("Range")
        if not header:
            return super().send_head()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        size = os.path.getsize(path)
        match = RANGE_RE.fullmatch(header.strip())
        if not match:
            return super().send_head()

        start_s, end_s = match.groups()
        if start_s == "":
            start = max(0, size - int(end_s))
            end = size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        # Returned to BaseHTTPRequestHandler.copyfile, which closes it -- the stdlib
        # contract, so the file must outlive this function.
        f = open(path, "rb")  # noqa: SIM115
        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _LimitedFile(f, end - start + 1)

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


class _LimitedFile:
    """File wrapper that stops reading after n bytes."""

    def __init__(self, f, n):
        self._f, self._left = f, n

    def read(self, amount=-1):
        if self._left <= 0:
            return b""
        if amount < 0 or amount > self._left:
            amount = self._left
        data = self._f.read(amount)
        self._left -= len(data)
        return data

    def close(self):
        self._f.close()


def main() -> None:
    from pathlib import Path

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    directory = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).resolve().parent.parent)
    handler = partial(RangeHandler, directory=directory)
    print(f"serving {directory} on http://0.0.0.0:{port}/ (Range enabled)")
    ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()


if __name__ == "__main__":
    main()
