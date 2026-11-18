"""Triangle-mesh scenes: OBJ/PLY loading, vertex-clustering decimation and a numpy ray caster
that renders posed RGB-D views (z-depth + Lambertian grey). Dependency-free (numpy only).

Camera convention as in ``sim.scene``: x right, y down, z forward (OpenCV), ``T_wc`` maps camera
to world, pixel ``(u, v)`` has ray ``K^-1 [u, v, 1]`` (integer pixel coordinates, the same as the
reference backend). Cost of ``render_rgbd`` is O(H * W * F): fine for meshes of a few thousand
triangles at test resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TriMesh:
    vertices: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(0), self.vertices.max(0)

    def transformed(self, scale: float = 1.0, R: np.ndarray | None = None, t=(0.0, 0.0, 0.0)):
        """``x -> scale * R x + t``."""
        R = np.eye(3) if R is None else np.asarray(R, float)
        v = scale * self.vertices @ R.T + np.asarray(t, float)
        return TriMesh(v, self.faces.copy())


# --------------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------------


def load_obj(path: str | Path) -> TriMesh:
    """Vertices and (fan-triangulated) faces of a Wavefront OBJ; normals/uvs ignored."""
    vs, fs = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
                fs.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return TriMesh(np.asarray(vs, float), np.asarray(fs, np.int64).reshape(-1, 3))


def load_ply_ascii(path: str | Path) -> TriMesh:
    """ASCII PLY with ``element vertex`` (x y z first) and ``element face`` (vertex_indices)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    if not lines or lines[0].strip() != "ply" or "ascii" not in lines[1]:
        raise ValueError(f"{path}: only ASCII PLY is supported")
    end = next(i for i, ln in enumerate(lines) if ln.strip() == "end_header")
    nv = nf = 0
    for line in lines[:end]:
        tok = line.split()
        if tok[:2] == ["element", "vertex"]:
            nv = int(tok[2])
        elif tok[:2] == ["element", "face"]:
            nf = int(tok[2])
    body = lines[end + 1 :]
    v = np.array([[float(x) for x in ln.split()[:3]] for ln in body[:nv]])
    fs = []
    for ln in body[nv : nv + nf]:
        tok = [int(x) for x in ln.split()]
        idx = tok[1 : 1 + tok[0]]
        fs.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
    return TriMesh(v, np.asarray(fs, np.int64).reshape(-1, 3))


def save_obj(mesh: TriMesh, path: str | Path, header: str = "") -> None:
    with open(path, "w") as fh:
        for line in header.splitlines():
            fh.write(f"# {line}\n".replace("# \n", "#\n"))
        for x in mesh.vertices:
            fh.write(f"v {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n")
        for f in mesh.faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def decimate_vertex_cluster(mesh: TriMesh, cell: float) -> TriMesh:
    """Vertex-clustering decimation: vertices in the same ``cell``-sized voxel merge to their mean;
    degenerate and duplicate faces are dropped and unused vertices removed."""
    key = np.floor((mesh.vertices - mesh.vertices.min(0)) / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n).astype(float)
    v = np.stack([np.bincount(inv, mesh.vertices[:, k], n) for k in range(3)], 1) / cnt[:, None]
    f = inv[mesh.faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, first = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(first)]
    used, f = np.unique(f, return_inverse=True)
    return TriMesh(v[used], f.reshape(-1, 3).astype(np.int64))


# --------------------------------------------------------------------------------------------
# Ray casting
# --------------------------------------------------------------------------------------------


def raycast(mesh: TriMesh, o: np.ndarray, d: np.ndarray, chunk: int = 256):
    """Nearest hit of rays ``o + t d`` (Moeller-Trumbore, both face sides). ``d`` (N, 3) need not
    be unit length, so with z-depth rays ``t`` is z-depth. Returns ``t`` (N,) (inf on a miss) and
    the hit face index (N,) (-1 on a miss)."""
    tri = mesh.vertices[mesh.faces]  # (F, 3, 3)
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    s = np.asarray(o, float) - v0  # (F, 3)
    q = np.cross(s, e1)  # (F, 3)
    qe2 = np.einsum("fi,fi->f", q, e2)
    t_out = np.full(len(d), np.inf)
    f_out = np.full(len(d), -1, np.int64)
    for i in range(0, len(d), chunk):
        dd = d[i : i + chunk]  # (m, 3)
        p = np.cross(dd[:, None, :], e2[None])  # (m, F, 3)
        det = np.einsum("mfi,fi->mf", p, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = np.einsum("mfi,fi->mf", p, s) * inv
        v = (dd @ q.T) * inv
        t = qe2[None, :] * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        t = np.where(hit, t, np.inf)
        j = np.argmin(t, axis=1)
        tj = t[np.arange(len(dd)), j]
        t_out[i : i + chunk] = tj
        f_out[i : i + chunk] = np.where(np.isfinite(tj), j, -1)
    return t_out, f_out


def face_normals(mesh: TriMesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-18)


def render_rgbd(
    mesh: TriMesh,
    K: np.ndarray,
    T_wc: np.ndarray,
    h: int,
    w: int,
    light=(0.3, -0.5, -0.8),
    albedo: float = 0.8,
    ambient: float = 0.15,
):
    """Clean z-depth (H, W) float32 (0 = background), grey Lambertian RGB (H, W, 3) uint8 and
    cos(incidence) (H, W) of ``mesh`` seen from camera ``(K, T_wc)``."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    dw = dc.reshape(-1, 3) @ T_wc[:3, :3].T
    t, fid = raycast(mesh, T_wc[:3, 3], dw)
    hit = fid >= 0
    n = np.zeros_like(dw)
    n[hit] = face_normals(mesh)[fid[hit]]
    dn = dw / np.linalg.norm(dw, axis=1, keepdims=True)
    n = np.where((np.sum(n * dn, 1) > 0)[:, None], -n, n)  # face the camera
    cos = np.where(hit, -np.sum(n * dn, 1), 0.0)
    L = -np.asarray(light, float) / np.linalg.norm(light)
    shade = np.where(hit, ambient + albedo * np.clip(n @ L, 0.0, None), 0.0)
    rgb = np.repeat((255 * np.clip(shade, 0, 1)).astype(np.uint8)[:, None], 3, 1)
    depth = np.where(hit, t, 0.0).astype(np.float32)
    return depth.reshape(h, w), rgb.reshape(h, w, 3), cos.reshape(h, w)
