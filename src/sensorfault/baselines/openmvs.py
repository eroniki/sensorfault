"""A: OpenMVS depth-map fusion wrapper (AGPL-3.0 tool; only invoked as a subprocess).

Pipeline: (1) per-view ``depthNNNN.dmap`` (openmvs_dmap.py, verified against the
OpenMVS ``develop`` source by reading), (2) a minimal Interface ``scene.mvs`` -- NOT
implemented (see ``write_mvs_scene``), (3) ``DensifyPointCloud --geometric-iters 0
--fusion-filter F --number-views-fuse N --postprocess-dmaps 0 --view-neighbors-file ...``,
(4) read the output .ply. Nothing here has been run against a real OpenMVS binary.
Whether ``--geometric-iters 0`` really skips PatchMatch and consumes pre-existing dmaps
is an inference from PLAN.md -- TODO(verify) by smoke test (week 3-4).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np

from sensorfault.baselines._common import ParamFuser, depth_to_normals, finish
from sensorfault.baselines.base import register
from sensorfault.baselines.openmvs_dmap import DMap, write_dmap
from sensorfault.types import DepthFrame, FusionResult


def find_binary(name: str = "DensifyPointCloud") -> str | None:
    env = os.environ.get("OPENMVS_BIN")
    if env:
        cand = Path(env) / name if Path(env).is_dir() else Path(env)
        if cand.exists():
            return str(cand)
    return shutil.which(name)


def write_dmaps(frames: Sequence[DepthFrame], out: Path) -> List[Path]:
    """One dmap per frame; R = world->camera = T_wc[:3,:3]^T, C = camera centre."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        d = np.where(np.isfinite(f.depth) & (f.depth > 0), f.depth, 0).astype(np.float32)
        n = depth_to_normals(d, f.K)
        n = np.nan_to_num(n).astype(np.float32)  # invalid -> zero normal (sentinel)
        conf = (
            f.conf.astype(np.float32)
            if f.conf is not None
            else np.where(d > 0, 1.0, 0.0).astype(np.float32)
        )
        vd = d[d > 0]
        dm = DMap(
            d,
            f.K,
            f.T_wc[:3, :3].T,
            f.T_wc[:3, 3],
            n,
            conf,
            ids=[i],
            image_name=f"image{i:04d}.png",
            dmin=float(vd.min()) if vd.size else 0.0,
            dmax=float(vd.max()) if vd.size else 0.0,
        )
        p = out / f"depth{i:04d}.dmap"
        write_dmap(str(p), dm)
        paths.append(p)
    return paths


def write_view_neighbors(frames: Sequence[DepthFrame], path: Path, n_views: int) -> None:
    """All-vs-all neighbour list, one line per view: ``ref nb1 nb2 ...``.
    TODO(verify): exact format of --view-neighbors-file (not read from source)."""
    n = len(frames)
    with open(path, "w") as fh:
        for i in range(n):
            nb = [j for j in range(n) if j != i][: max(n_views, 1)]
            fh.write(" ".join(map(str, [i] + nb)) + "\n")


def write_mvs_scene(frames: Sequence[DepthFrame], path: Path) -> None:
    """Minimal Interface .mvs (header "MVSI", uint32 version, uint32 reserved, then the
    ARCHIVE-serialised ``Interface`` struct: platforms/cameras/poses, images, vertices).
    The archive is a versioned custom binary format (std::string/vector/Matx encodings,
    per-version fields) that was not read to completion, so writing it blind risks silent
    corruption. TODO: implement from Interface.h (ArchiveSave + Interface::serialize) and
    verify with ``InterfaceMVS`` round trip. Workaround: build the scene with OpenMVS's own
    ``InterfaceCOLMAP`` (see _colmap.py writer) or a C++ shim."""
    raise NotImplementedError("write_mvs_scene: see docstring TODO")


def read_ply_points(path: str | Path):
    """Minimal PLY reader (ascii / binary_little_endian, vertex x y z [nx ny nz])."""
    with open(path, "rb") as f:
        raw = f.read()
    hdr_end = raw.index(b"end_header") + len(b"end_header")
    hdr_end = raw.index(b"\n", hdr_end) + 1
    lines = raw[:hdr_end].decode().splitlines()
    fmt = next(l.split()[1] for l in lines if l.startswith("format"))
    nv = 0
    props = []
    in_vertex = False
    tmap = {
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "uchar": "u1",
        "uint8": "u1",
        "char": "i1",
        "int": "i4",
        "uint": "u4",
        "short": "i2",
        "ushort": "u2",
        "int32": "i4",
        "uint32": "u4",
    }
    for l in lines:
        t = l.split()
        if t[:1] == ["element"]:
            in_vertex = t[1] == "vertex"
            if in_vertex:
                nv = int(t[2])
        elif t[:1] == ["property"] and in_vertex:
            if t[1] == "list":
                raise ValueError("list property in vertex element")
            props.append((t[2], tmap[t[1]]))
    if fmt == "ascii":
        rows = np.loadtxt(raw[hdr_end:].decode().splitlines()[:nv], ndmin=2)
        data = {n: rows[:, i] for i, (n, _) in enumerate(props)}
    elif fmt == "binary_little_endian":
        arr = np.frombuffer(raw, np.dtype([(n, "<" + t) for n, t in props]), nv, hdr_end)
        data = {n: arr[n] for n, _ in props}
    else:
        raise ValueError(f"unsupported PLY format {fmt}")
    pts = np.stack([data["x"], data["y"], data["z"]], -1).astype(np.float32)
    nrm = (
        np.stack([data["nx"], data["ny"], data["nz"]], -1).astype(np.float32)
        if "nx" in data
        else None
    )
    return pts, nrm


@register("openmvs")
class OpenMVS(ParamFuser):
    """Params: fusion_filter (1; 0 merge, 1 voting, 2 newer rule), views_to_fuse (2),
    n_neighbors (32), workdir (None -> tempdir), keep_workdir (False), binary (None),
    extra_args ([]) -- depth/normal thresholds have no CLI flag verified, TODO(verify)."""

    DEFAULTS = dict(
        fusion_filter=1,
        views_to_fuse=2,
        n_neighbors=32,
        workdir=None,
        keep_workdir=False,
        binary=None,
        extra_args=[],
        timeout_s=3600,
    )

    @classmethod
    def available(cls) -> tuple[bool, str]:
        b = find_binary()
        if b is None:
            return False, "DensifyPointCloud not on PATH and $OPENMVS_BIN unset/invalid"
        return True, b

    @staticmethod
    def build_command(
        binary: str, scene: Path, nb_file: Path, out_ply: Path, p: Mapping
    ) -> List[str]:
        return [
            binary,
            str(scene),
            "--geometric-iters",
            "0",
            "--fusion-filter",
            str(int(p["fusion_filter"])),
            "--number-views-fuse",
            str(int(p["views_to_fuse"])),
            "--postprocess-dmaps",
            "0",
            "--view-neighbors-file",
            str(nb_file),
            "-o",
            str(out_ply),
            *map(str, p["extra_args"]),
        ]

    def prepare(self, frames: Sequence[DepthFrame], wd: Path, p: Mapping) -> List[str]:
        write_dmaps(frames, wd)
        nb = wd / "view_neighbors.txt"
        write_view_neighbors(frames, nb, int(p["n_neighbors"]))
        scene = wd / "scene.mvs"
        write_mvs_scene(frames, scene)  # raises NotImplementedError until implemented
        return self.build_command(
            p["binary"] or find_binary() or "DensifyPointCloud", scene, nb, wd / "fused.ply", p
        )

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        t0 = time.perf_counter()
        p = self.params(cfg)
        wd = Path(p["workdir"] or tempfile.mkdtemp(prefix="sf_openmvs_"))
        try:
            cmd = self.prepare(frames, wd, p)
            subprocess.run(cmd, cwd=wd, check=True, timeout=p["timeout_s"])
            pts, nrm = read_ply_points(wd / "fused.ply")
        finally:
            if not p["keep_workdir"] and not p["workdir"]:
                shutil.rmtree(wd, ignore_errors=True)
        sids = list(dict.fromkeys(f.sensor_id for f in frames))
        return finish(
            FusionResult(
                points=pts,
                normals=nrm,
                excluded_sensors=dict.fromkeys(sids, False),
                meta=dict(cmd=cmd, offline=True),
            ),
            t0,
        )
