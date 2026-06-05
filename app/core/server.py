# -*- coding: utf-8 -*-
"""Local HTTP server for web UI assets, media playback and generated files."""

from __future__ import annotations

import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, unquote, urlparse


class _State:
    web_root: str | None = None
    media_path: str | None = None
    data_root: str | None = None
    token = "autocon"


def _ctype(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send_file(self, path: str, head_only: bool = False) -> None:
        if not os.path.isfile(path):
            self.send_error(404)
            return
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            try:
                start_raw, end_raw = rng[6:].split("-", 1)
                start = int(start_raw) if start_raw else 0
                end = int(end_raw) if end_raw else size - 1
            except ValueError:
                start, end = 0, size - 1
            start = max(0, start)
            end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", _ctype(path))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            if not head_only:
                self._pipe(path, start, length)
            return

        self.send_response(200)
        self.send_header("Content-Type", _ctype(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self._pipe(path, 0, size)

    def _pipe(self, path: str, start: int, length: int) -> None:
        try:
            with open(path, "rb") as file:
                file.seek(start)
                remaining = length
                while remaining > 0:
                    data = file.read(min(256 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_media(self, head_only: bool = False) -> None:
        parts = self.path.split("/")
        if len(parts) < 3 or unquote(parts[2]) != _State.token:
            self.send_error(404)
            return
        if not _State.media_path:
            self.send_error(404)
            return
        self._send_file(_State.media_path, head_only)

    def _serve_local(self, head_only: bool = False) -> None:
        from urllib.parse import parse_qs

        query = parse_qs(urlparse(self.path).query)
        raw = (query.get("p") or [""])[0]
        path = os.path.normpath(raw)
        allowed = []
        if _State.data_root:
            allowed.append(os.path.normpath(_State.data_root))
        if not path or not any(path.startswith(root) for root in allowed):
            self.send_error(404)
            return
        self._send_file(path, head_only)

    def _serve_static(self, head_only: bool = False) -> None:
        root = _State.web_root
        if not root:
            self.send_error(404)
            return
        rel = unquote(urlparse(self.path).path).lstrip("/")
        if rel in ("", "index.html"):
            rel = "index.html"
        target = os.path.normpath(os.path.join(root, rel))
        if not target.startswith(os.path.normpath(root)):
            self.send_error(404)
            return
        self._send_file(target, head_only)

    def _route(self, head_only: bool = False) -> None:
        if self.path.startswith("/media/"):
            self._serve_media(head_only)
        elif self.path.startswith("/local"):
            self._serve_local(head_only)
        else:
            self._serve_static(head_only)

    def do_GET(self) -> None:
        self._route(False)

    def do_HEAD(self) -> None:
        self._route(True)


class MediaServer:
    def __init__(self) -> None:
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    def start(self, web_root=None) -> int:
        if web_root:
            _State.web_root = str(web_root)
        if self._httpd:
            return self.port
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.port

    def set_data_root(self, root) -> None:
        _State.data_root = str(root)

    def base_url(self) -> str:
        self.start()
        return f"http://127.0.0.1:{self.port}"

    def index_url(self) -> str:
        return self.base_url() + "/index.html"

    def serve(self, path: str) -> str:
        self.start()
        _State.media_path = path
        return f"{self.base_url()}/media/{_State.token}"

    def local_url(self, path: str) -> str:
        return f"{self.base_url()}/local?p={quote(str(path))}"

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd = None
