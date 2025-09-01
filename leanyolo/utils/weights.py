from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import hashlib
import os
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

import torch
import re
import sys
import types
from torch.serialization import add_safe_globals


@dataclass
class WeightsEntry:
    name: str
    url: Optional[str]
    filename: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    sha256: Optional[str] = None  # hex digest, lowercase

    def _default_cache_dir(self) -> str:
        # Prefer explicit env var; otherwise cache in user dir
        return os.environ.get(
            "LEANYOLO_CACHE_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "leanyolo"),
        )

    def _target_filename(self) -> str:
        if self.filename:
            return self.filename
        if self.url:
            return os.path.basename(urlparse(self.url).path) or f"{self.name}.pt"
        return f"{self.name}.pt"

    def _sha256_of_file(self, path: str, chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def _download_to(self, url: str, dst: str, *, progress: bool = True) -> None:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(dst)) as tmp:
            tmp_path = tmp.name
            with urlopen(url) as r:  # nosec - URL is controlled by caller/tests
                # Stream download to avoid high memory usage
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    tmp.write(chunk)
        os.replace(tmp_path, dst)

    def _load_from_file(self, path: str, map_location: str | torch.device) -> Dict[str, torch.Tensor]:
        # Prefer weights_only=True to avoid importing classes from pickled checkpoints (e.g., ultralytics)
        try:
            return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            # Older torch without weights_only kwarg
            try:
                return torch.load(path, map_location=map_location)
            except Exception:
                # Last resort: explicit weights_only=False for odd cases
                return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
        except Exception as e:
            # Attempt a safe load by dynamically stubbing missing globals (e.g., ultralytics classes)
            try:
                return self._safe_load_with_dynamic_stubs(path, map_location)
            except Exception:
                # Fallback path (may require external modules; only if trusted)
                try:
                    return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[call-arg]
                except Exception:
                    return torch.load(path, map_location=map_location)

    def _safe_load_with_dynamic_stubs(self, path: str, map_location: str | torch.device):
        """Load a checkpoint with weights_only=True by allowlisting missing globals.

        This avoids importing external libraries (e.g., ultralytics) by creating
        minimal stub modules and classes on-the-fly and registering them via
        torch.serialization.add_safe_globals. Only class names required by the
        checkpoint are stubbed, and no conversion is performed here.
        """
        attempted: set[str] = set()
        for _ in range(64):  # guard against infinite loops
            try:
                return torch.load(path, map_location=map_location, weights_only=True)  # type: ignore[call-arg]
            except Exception as ex:  # UnpicklingError with details
                msg = str(ex)
                # Look for pattern like: "Unsupported global: GLOBAL ultralytics.nn.tasks.YOLOv10DetectionModel"
                m = re.search(r"Unsupported global: (?:GLOBAL\s+)?([\w\.]+)\.(\w+)", msg)
                if not m:
                    raise
                mod_path, cls_name = m.group(1), m.group(2)
                fqcn = f"{mod_path}.{cls_name}"
                if fqcn in attempted:
                    raise
                attempted.add(fqcn)
                # Create stub module hierarchy
                cur_mod = None
                parent = None
                for i, part in enumerate(mod_path.split(".")):
                    mod_full = ".".join(mod_path.split(".")[: i + 1])
                    mod_obj = sys.modules.get(mod_full)
                    if mod_obj is None:
                        mod_obj = types.ModuleType(mod_full)
                        sys.modules[mod_full] = mod_obj
                        if parent is not None:
                            setattr(parent, part, mod_obj)
                    parent = mod_obj
                # Define a minimal stub class with a permissive state_dict() accessor
                mod_obj = sys.modules[mod_path]
                if not hasattr(mod_obj, cls_name):
                    # Create a new type under the module
                    Stub = type(
                        cls_name,
                        (object,),
                        {
                            "__module__": mod_path,
                            # Provide a benign state_dict() to satisfy callers; actual tensors are
                            # extracted from nested dicts by downstream utilities.
                            "state_dict": lambda self: {},
                        },
                    )
                    setattr(mod_obj, cls_name, Stub)
                    add_safe_globals([Stub])
                else:
                    add_safe_globals([getattr(mod_obj, cls_name)])
        # If we somehow exit the loop
        raise RuntimeError("Failed to safely load checkpoint with dynamic stubs")

    def get_state_dict(
        self,
        *,
        progress: bool = True,
        map_location: str | torch.device = "cpu",
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verify_hash: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Load state dict via local path, env dir, or download.

        Order of resolution:
        1) Explicit local_path
        2) LEANYOLO_WEIGHTS_DIR/<filename>
        3) cache_dir (or default cache) with optional download from URL
        """

        # 1) Explicit local override
        if local_path is not None:
            return self._load_from_file(local_path, map_location)

        # 2) Repository-level or user-provided weights dir
        env_dir = os.environ.get("LEANYOLO_WEIGHTS_DIR")
        filename = self._target_filename()
        if env_dir:
            candidate = os.path.join(env_dir, filename)
            if os.path.exists(candidate):
                return self._load_from_file(candidate, map_location)

        # 3) Check cache, download if necessary
        cache_dir = cache_dir or self._default_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, filename)

        def valid_hash(path: str) -> bool:
            if not (verify_hash and self.sha256):
                return True
            try:
                return self._sha256_of_file(path) == self.sha256
            except FileNotFoundError:
                return False

        # If cached and hash ok, load
        if os.path.exists(cache_path) and valid_hash(cache_path):
            return self._load_from_file(cache_path, map_location)

        # Otherwise download if URL provided
        if not self.url:
            raise FileNotFoundError(
                f"Weights not found locally ('{cache_path}') and no URL provided. "
                "Place the file in LEANYOLO_WEIGHTS_DIR or pass local_path."
            )

        self._download_to(self.url, cache_path, progress=progress)

        # Verify hash after download
        if not valid_hash(cache_path):
            # Avoid using a corrupted file
            try:
                os.remove(cache_path)
            finally:
                raise RuntimeError(
                    "Downloaded file hash mismatch for weights '"
                    + filename
                    + "'."
                )

        return self._load_from_file(cache_path, map_location)


class WeightsResolver:
    def list(self, model_name: str) -> Iterable[str]:  # pragma: no cover - interface
        raise NotImplementedError

    def get(self, model_name: str, key: str) -> WeightsEntry:  # pragma: no cover - interface
        raise NotImplementedError
