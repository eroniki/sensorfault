"""Dataset manifests: relative path, size and SHA-256 only. Data stay outside git; manifests are committed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def build_manifest(root, paths, meta: dict | None = None) -> dict:
    root = Path(root)
    files = {}
    for rel in sorted(str(p) for p in paths):
        fp = root / rel
        files[rel] = {"size": fp.stat().st_size, "sha256": sha256_file(fp)}
    return {"meta": meta or {}, "files": files}


def verify_manifest(root, manifest: dict) -> list[str]:
    """Return a list of problems (empty if every file matches)."""
    root = Path(root)
    bad = []
    for rel, info in manifest["files"].items():
        fp = root / rel
        if not fp.exists():
            bad.append(f"missing: {rel}")
        elif fp.stat().st_size != info["size"] or sha256_file(fp) != info["sha256"]:
            bad.append(f"changed: {rel}")
    return bad


def write_manifest(manifest: dict, out) -> None:
    Path(out).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
