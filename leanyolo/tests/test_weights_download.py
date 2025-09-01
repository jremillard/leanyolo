import os
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import torch

from leanyolo.utils.weights import WeightsEntry


def _sha256_of_file(path: str):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class _HTTPServer:
    def __init__(self, root: str):
        handler = partial(SimpleHTTPRequestHandler, directory=root)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        try:
            self.httpd.shutdown()
        finally:
            self.thread.join(timeout=2)


def test_downloads_and_loads_state_dict(tmp_path):
    # Prepare a small, valid state_dict served over HTTP
    src_dir = tmp_path / "src"
    cache_dir = tmp_path / "cache"
    src_dir.mkdir()
    cache_dir.mkdir()

    sd = {"layer.weight": torch.tensor([1.0, 2.0, 3.0])}
    src_path = src_dir / "file.pt"
    torch.save(sd, src_path)
    sha = _sha256_of_file(str(src_path))

    server = _HTTPServer(str(src_dir)).start()
    url = f"http://127.0.0.1:{server.port}/file.pt"

    try:
        entry = WeightsEntry(name="test", url=url, filename="file.pt", sha256=sha)
        out = entry.get_state_dict(cache_dir=str(cache_dir))
        assert set(out.keys()) == set(sd.keys())
        assert torch.equal(out["layer.weight"], sd["layer.weight"])  # round-trip exact

        cached = cache_dir / "file.pt"
        assert cached.exists()
        assert _sha256_of_file(str(cached)) == sha
    finally:
        server.stop()


def test_uses_env_dir_if_present(tmp_path, monkeypatch):
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    sd = {"x": torch.tensor([42])}
    env_path = env_dir / "yolov10s.pt"
    torch.save(sd, env_path)

    monkeypatch.setenv("LEANYOLO_WEIGHTS_DIR", str(env_dir))

    entry = WeightsEntry(name="yolov10s", url=None, filename="yolov10s.pt")
    out = entry.get_state_dict()
    assert torch.equal(out["x"], sd["x"])


def test_redownload_on_hash_mismatch(tmp_path):
    src_dir = tmp_path / "src"
    cache_dir = tmp_path / "cache"
    src_dir.mkdir()
    cache_dir.mkdir()

    sd = {"p": torch.ones(2, 2)}
    src_path = src_dir / "file.pt"
    torch.save(sd, src_path)
    sha = _sha256_of_file(str(src_path))

    # Create a corrupted cache file
    bad = cache_dir / "file.pt"
    with open(bad, "wb") as f:
        f.write(b"corrupted")

    server = _HTTPServer(str(src_dir)).start()
    url = f"http://127.0.0.1:{server.port}/file.pt"
    try:
        entry = WeightsEntry(name="test", url=url, filename="file.pt", sha256=sha)
        out = entry.get_state_dict(cache_dir=str(cache_dir))
        # File should have been re-downloaded and now match hash
        assert _sha256_of_file(str(cache_dir / "file.pt")) == sha
        assert torch.equal(out["p"], sd["p"])
    finally:
        server.stop()
