"""Read-only loopback preview; no build, upload, mutation or directory listings."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .build import validate_preview


class PreviewHandler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(404,'Directory listings are disabled')
        return None

    def send_head(self):
        # Refuse traversal and symlink escapes, even after the initial site check.
        path = unquote(urlsplit(self.path).path)
        parts = path.split('/')
        if '\\' in path or any(part in ('.','..') or part.startswith('.') for part in parts if part):
            self.send_error(404,'Unknown preview resource');return None
        target = Path(self.directory)
        for part in parts:
            if part:
                target = target/part
                if target.is_symlink():
                    self.send_error(404,'Unknown preview resource');return None
        return super().send_head()

    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if '/artifacts/' in self.path:
            self.send_header('Content-Disposition','attachment')
        self.send_header('Cache-Control','no-store')
        super().end_headers()


def serve(root: Path, output: Path, port=8000):
    validate_preview(root,output)
    handler = partial(PreviewHandler,directory=str(output.resolve()))
    with ThreadingHTTPServer(('127.0.0.1',port),handler) as server:
        print(f'Local preview: http://127.0.0.1:{server.server_port}/ (Ctrl-C to stop)',flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
