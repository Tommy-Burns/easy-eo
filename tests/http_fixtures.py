"""A range-capable HTTP server for the remote-raster tests.

GDAL reads a COG over HTTP with range requests, so testing that path needs a
server that answers them — ``http.server``'s own handler does not, and GDAL
refuses a server that cannot ("Range downloading not supported by this
server!").

**The server runs in a separate process, and has to.** rasterio holds the GIL
while GDAL blocks on curl, so a server thread in the same interpreter never
gets to answer, and the read deadlocks. The child writes one line per request
to a log file, which is how a test sees what was fetched.

Nothing here reaches outside the machine: the child binds ``127.0.0.1`` on a
port the operating system picks.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import os
import socketserver
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote


class RangeRequestHandler(http.server.BaseHTTPRequestHandler):
    """Serve files from a directory, honouring ``Range`` requests."""

    protocol_version = "HTTP/1.1"
    directory = "."
    log_path = ""

    def log_message(self, *args):  # noqa: D102 - silence the default stderr log
        pass

    def do_HEAD(self):  # noqa: D102, N802 - http.server's naming
        self._serve(body=False)

    def do_GET(self):  # noqa: D102, N802 - http.server's naming
        self._serve(body=True)

    def _resolve(self) -> Path | None:
        """Return the file this request names, or None if it names no file.

        The request path is attacker-controlled by construction, so it is
        never used to *build* a filesystem path — only to pick one out of the
        directory's own listing by exact name. A path cannot escape a
        directory it was never joined to, and ``..`` or an encoded separator
        simply matches no entry. Only files directly in the directory are
        served, which is all the fixtures need.
        """
        requested = unquote(self.path.split("?")[0].split("#")[0]).lstrip("/")
        try:
            entries = list(Path(self.directory).iterdir())
        except OSError:
            return None
        for entry in entries:
            if entry.name == requested and entry.is_file():
                return entry
        return None

    def _serve(self, *, body: bool) -> None:
        path = self._resolve()
        if path is None:
            # GDAL probes for sidecars (.aux.xml, .ovr, ...); 404 is the
            # answer, and so is a path that points outside the directory.
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self._record(status=404, served=0)
            return

        size = path.stat().st_size
        start, stop, status = 0, size - 1, 200
        header = self.headers.get("Range")
        if header and header.startswith("bytes="):
            first, _, last = header[len("bytes=") :].partition("-")
            start = int(first) if first else 0
            stop = min(int(last), size - 1) if last else size - 1
            status = 206

        length = stop - start + 1
        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{stop}/{size}")
        self.end_headers()

        if not body:
            self._record(status=status, served=0)
            return
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(length)
        self._record(status=status, served=len(data))
        # GDAL closes a connection once it has the bytes it wanted.
        with contextlib.suppress(BrokenPipeError):
            self.wfile.write(data)

    def _record(self, *, status: int, served: int) -> None:
        entry = {
            "method": self.command,
            "path": self.path,
            "range": self.headers.get("Range"),
            "status": status,
            "served": served,
        }
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(json.dumps(entry) + "\n")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def _main() -> None:
    """Run the server; the child process's entry point."""
    directory, log_path, port_path = sys.argv[1:4]
    handler = type(
        "BoundHandler",
        (RangeRequestHandler,),
        {"directory": directory, "log_path": log_path},
    )
    with _Server(("127.0.0.1", 0), handler) as httpd:
        # Written to a temporary file and moved into place, so the parent can
        # never read the file while it is still empty — which on Windows it
        # did, producing a URL with no port in it at all.
        port_file = Path(port_path)
        pending = port_file.with_suffix(".pending")
        pending.write_text(str(httpd.server_address[1]), encoding="utf-8")
        os.replace(pending, port_file)
        httpd.serve_forever()


class ServedDirectory:
    """A directory being served over HTTP, and the requests made to it."""

    def __init__(self, base_url: str, log_path: Path) -> None:
        self.base_url = base_url
        self.log_path = log_path

    def url(self, name: str) -> str:
        """Return the URL of one file in the served directory."""
        return f"{self.base_url}/{name}"

    def requests(self) -> list[dict]:
        """Return every request served so far, oldest first."""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line]

    def bytes_served(self) -> int:
        """Return the total number of file bytes sent to clients."""
        return sum(entry["served"] for entry in self.requests())

    def reset(self) -> None:
        """Forget the requests served so far."""
        self.log_path.write_text("", encoding="utf-8")


@contextmanager
def serve_directory(directory, tmp_path):
    """Serve ``directory`` over HTTP from a child process, for the block's duration.

    Parameters
    ----------
    directory : path-like
        Directory whose files are served.
    tmp_path : pathlib.Path
        Scratch directory for the request log and the port handshake file.

    Yields
    ------
    ServedDirectory
        Handle giving each file's URL and the requests served.
    """
    log_path = Path(tmp_path) / "requests.log"
    port_path = Path(tmp_path) / "port.txt"
    log_path.write_text("", encoding="utf-8")
    port_path.unlink(missing_ok=True)

    child = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(directory),
            str(log_path),
            str(port_path),
        ]
    )
    try:
        deadline = time.monotonic() + 30
        port = ""
        while not port.isdigit():
            # The port, not merely the file: a file that exists but is not yet
            # readable would otherwise yield "http://127.0.0.1:/scene.tif".
            port = port_path.read_text(encoding="utf-8").strip() if port_path.exists() else ""
            if port.isdigit():
                break
            if child.poll() is not None:
                raise RuntimeError(f"the test HTTP server exited with {child.returncode}")
            if time.monotonic() > deadline:
                raise RuntimeError("the test HTTP server did not report a port")
            time.sleep(0.02)
        yield ServedDirectory(f"http://127.0.0.1:{port}", log_path)
    finally:
        child.terminate()
        child.wait(timeout=30)


if __name__ == "__main__":
    _main()
