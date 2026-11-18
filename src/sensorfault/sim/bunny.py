"""Stanford bunny scene: posed RGB-D views of a non-planar mesh with an optional injected depth
fault, for gate regression tests (the flat-plane test scene hides model error from relief).

The mesh is a vertex-clustered (8 mm cells, ~2.4k triangles) copy of the zippered Stanford bunny
reconstruction, shipped as ``assets/stanford_bunny_8mm.obj`` (~60 KB) so tests run offline. It is
regenerated from upstream by ``scripts/data/bunny.py``. Source: Stanford 3D Scanning Repository,
Stanford University Computer Graphics Laboratory (http://graphics.stanford.edu/data/3Dscanrep/);
research use and free redistribution with acknowledgement, no commercial use (see NOTICE). The
mesh is NOT covered by the repository's MIT licence.

World frame: z up, the bunny scaled by ``scale`` (default 10, ~1.5 m long) and centred on the
origin with its feet on z = 0. Cameras sit on an arc at ``radius`` around the bunny and look at
its centre (``sim.scene.look_at``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sensorfault.sim.mesh import TriMesh, load_obj, render_rgbd
from sensorfault.sim.scene import intrinsics, look_at
from sensorfault.types import DepthFrame

ASSET = Path(__file__).with_name("assets") / "stanford_bunny_8mm.obj"
# upstream bunny is y-up; rotate +90 deg about x so it stands on the world xy plane (z up)
_Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def bunny_mesh(scale: float = 10.0, path: str | Path | None = None) -> TriMesh:
    """The shipped low-res bunny in world units (z up, centred in xy, feet on z = 0)."""
    m = load_obj(path or ASSET)
    v = m.vertices @ _Y_UP_TO_Z_UP.T
    lo, hi = v.min(0), v.max(0)
    centre = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return TriMesh(scale * (v - centre), m.faces)


@dataclass
class BunnyConfig:
    n_views: int = 6
    hw: int = 32  # square image side (pixels)
    fov_deg: float = 36.0
    radius: float = 3.0  # camera distance from the look-at point (m)
    azimuth_span_deg: float = 60.0  # cameras evenly spread over this arc, centred on azimuth0
    azimuth0_deg: float = -90.0  # -90: cameras on the -y side
    elevation_deg: float = 20.0
    elevation_jitter_deg: float = 5.0  # alternate views +/- this, so the rig is not coplanar
    scale: float = 10.0
    noise: float = 0.005  # additive Gaussian depth noise (m, 1 sigma)
    seed: int = 0
    # injected fault: additive depth bias on a pixel window of one view
    fault_view: int | None = None
    fault_bias: float = 0.05
    fault_region: tuple = (8, 24, 8, 24)  # (r0, r1, c0, c1)


@dataclass
class BunnyViews:
    frames: list  # DepthFrame, sensor ids "v0".."v{n-1}"
    rgb: list  # (H, W, 3) uint8 per view
    depth_clean: list  # (H, W) float32 noiseless depth per view (0 = background)
    fault_mask: list  # (H, W) bool: pixels that received the bias (all False for clean views)
    mesh: TriMesh
    cfg: BunnyConfig

    @property
    def faulty(self) -> set[str]:
        return {f.sensor_id for f, m in zip(self.frames, self.fault_mask) if m.any()}


def camera_poses(cfg: BunnyConfig, target: np.ndarray) -> list[np.ndarray]:
    n = cfg.n_views
    az = np.deg2rad(cfg.azimuth0_deg + cfg.azimuth_span_deg * (np.arange(n) / max(n - 1, 1) - 0.5))
    el = np.deg2rad(
        cfg.elevation_deg + cfg.elevation_jitter_deg * np.where(np.arange(n) % 2, 1, -1)
    )
    pos = target + cfg.radius * np.stack(
        [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1
    )
    return [look_at(p, target) for p in pos]


def make_bunny_views(cfg: BunnyConfig | None = None, mesh: TriMesh | None = None) -> BunnyViews:
    """Render ``cfg.n_views`` posed RGB-D views; background pixels have depth 0 (invalid). Noise is
    added to valid pixels only; the fault adds ``fault_bias`` to the valid pixels of
    ``fault_region`` in view ``fault_view``."""
    cfg = cfg or BunnyConfig()
    mesh = mesh if mesh is not None else bunny_mesh(cfg.scale)
    lo, hi = mesh.bounds
    target = 0.5 * (lo + hi)
    K = intrinsics(cfg.hw, cfg.hw, cfg.fov_deg)
    rng = np.random.default_rng(cfg.seed)
    frames, rgbs, cleans, masks = [], [], [], []
    for i, T in enumerate(camera_poses(cfg, target)):
        d0, rgb, _ = render_rgbd(mesh, K, T, cfg.hw, cfg.hw)
        valid = d0 > 0
        d = d0 + cfg.noise * rng.standard_normal(d0.shape).astype(np.float32) * valid
        m = np.zeros_like(valid)
        if i == cfg.fault_view:
            r0, r1, c0, c1 = cfg.fault_region
            m[r0:r1, c0:c1] = True
            m &= valid
            d = d + np.float32(cfg.fault_bias) * m
        frames.append(DepthFrame(f"v{i}", i, d.astype(np.float32), K, T))
        rgbs.append(rgb)
        cleans.append(d0)
        masks.append(m)
    return BunnyViews(frames, rgbs, cleans, masks, mesh, cfg)
