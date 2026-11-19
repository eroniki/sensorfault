"""JSON manifest writer/reader. Manifest includes a content hash for provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)


def write_manifest(manifest: dict, path) -> str:
    """Write manifest JSON (pretty, sorted keys) with ``content_sha256``. Returns the hash."""
    body = {k: v for k, v in manifest.items() if k != "content_sha256"}
    digest = hashlib.sha256(_canon(body).encode()).hexdigest()
    out = {**body, "content_sha256": digest}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, sort_keys=True, indent=2, default=float))
    return digest


def read_manifest(path) -> dict:
    m = json.loads(Path(path).read_text())
    body = {k: v for k, v in m.items() if k != "content_sha256"}
    if hashlib.sha256(_canon(body).encode()).hexdigest() != m.get("content_sha256"):
        raise ValueError("manifest hash mismatch (file edited after writing)")
    return m
