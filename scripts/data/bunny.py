#!/usr/bin/env python3
"""Regenerate ``src/sensorfault/sim/assets/stanford_bunny_8mm.obj`` from the upstream bunny.

The repository ships the decimated mesh (~70 KB) so tests run offline; this script documents and
reproduces how it was made. Nothing is downloaded unless ``--yes`` is given.

Source and terms (checked 2026-11-18 on http://graphics.stanford.edu/data/3Dscanrep/):
  Stanford Bunny, Stanford University Computer Graphics Laboratory (Turk & Levoy, SIGGRAPH 1994),
  zippered reconstruction, 35947 vertices / 69451 triangles. Use for research is permitted, as is
  free mirroring/redistribution; acknowledge the source; no commercial use without permission.
  The page adds "You can do anything you want to the Stanford bunny".

Downloads, tried in order:
  1. http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz -> bunny/reconstruction/bun_zipper.ply
  2. GitHub mirror alecjacobson/common-3d-test-models data/stanford-bunny.obj (same mesh, MeshLab
     OBJ export; SHA-256 pinned below)
Then vertex-clustering decimation with 8 mm cells (``sim.mesh.decimate_vertex_cluster``).

  python3 scripts/data/bunny.py --yes            # rewrite the shipped asset
  python3 scripts/data/bunny.py --yes --out x.obj
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sensorfault.sim.bunny import ASSET  # noqa: E402
from sensorfault.sim.mesh import (  # noqa: E402
    decimate_vertex_cluster,
    load_obj,
    load_ply_ascii,
    save_obj,
)

STANFORD_TGZ = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
MIRROR_OBJ = (
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/"
    "stanford-bunny.obj"
)
MIRROR_SHA256 = "1eb35d1e21ce99e5ce911353b6be278990713448dd9e8f5c9387f9de39b32205"
CELL = 0.008  # metres (upstream units)
HEADER = """Stanford Bunny, decimated (vertex clustering, 8 mm cells) for sensorfault tests.
Source: Stanford 3D Scanning Repository, Stanford University Computer Graphics Laboratory
http://graphics.stanford.edu/data/3Dscanrep/ (Turk & Levoy, SIGGRAPH 1994).
Research use and free redistribution permitted with acknowledgement; no commercial use.
NOT covered by this repository's MIT licence. Regenerate: python3 scripts/data/bunny.py --yes
Units: metres, y up (upstream frame)."""


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def fetch(tmp: Path):
    try:
        data = _get(STANFORD_TGZ)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("bun_zipper.ply"))
            p = tmp / "bun_zipper.ply"
            p.write_bytes(tf.extractfile(member).read())
        print(f"fetched {STANFORD_TGZ}")
        return load_ply_ascii(p)
    except Exception as e:  # network policy, moved file, ...
        print(f"Stanford download failed ({e}); trying the GitHub mirror")
    data = _get(MIRROR_OBJ)
    digest = hashlib.sha256(data).hexdigest()
    if digest != MIRROR_SHA256:
        raise SystemExit(f"mirror SHA-256 mismatch: {digest}")
    p = tmp / "stanford-bunny.obj"
    p.write_bytes(data)
    print(f"fetched {MIRROR_OBJ} (sha256 ok)")
    return load_obj(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="actually download")
    ap.add_argument("--out", type=Path, default=ASSET)
    ap.add_argument("--cell", type=float, default=CELL)
    a = ap.parse_args()
    if not a.yes:
        print(__doc__)
        print(f"dry run: would write {a.out}; pass --yes to download")
        return 0
    with tempfile.TemporaryDirectory() as td:
        mesh = fetch(Path(td))
    if len(mesh.vertices) != 35947 or len(mesh.faces) != 69451:
        raise SystemExit(f"unexpected mesh size {len(mesh.vertices)} V / {len(mesh.faces)} F")
    low = decimate_vertex_cluster(mesh, a.cell)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_obj(low, a.out, HEADER)
    print(f"wrote {a.out}: {len(low.vertices)} V / {len(low.faces)} F")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
