"""C1: weighted-average TSDF (Curless & Levoy 1996 / KinectFusion) via Open3D, known
poses, no ICP, no sensor elimination (the 'no-elimination floor'). Open3D is imported
lazily. Unverified against a real Open3D install (TODO(verify): API names for the
installed version; ScalableTSDFVolume integrates weight 1 per observation, so per-pixel
confidence is ignored)."""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np

from sensorfault.baselines._common import ParamFuser, finish
from sensorfault.baselines.base import register
from sensorfault.types import DepthFrame, FusionResult


@register("tsdf_open3d")
class TSDFOpen3D(ParamFuser):
    """Params: voxel_length (0.01), sdf_trunc (4*voxel), depth_trunc (10 m)."""

    DEFAULTS = dict(voxel_length=0.01, sdf_trunc=None, depth_trunc=10.0)

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            import open3d  # noqa: F401
        except Exception as e:  # ImportError or binary mismatch
            return False, f"open3d not importable: {e}"
        return True, ""

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        ok, why = self.available()
        if not ok:
            raise RuntimeError(why)
        import open3d as o3d

        t0 = time.perf_counter()
        p = self.params(cfg)
        vl = float(p["voxel_length"])
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=vl,
            sdf_trunc=float(p["sdf_trunc"] or 4 * vl),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )
        for f in frames:
            h, w = f.depth.shape
            d = np.where(np.isfinite(f.depth) & (f.depth > 0), f.depth, 0.0).astype(np.float32)
            color = np.full((h, w, 3), 127, np.uint8)  # dummy colour required by the RGBD API
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color),
                o3d.geometry.Image(np.ascontiguousarray(d)),
                depth_scale=1.0,
                depth_trunc=float(p["depth_trunc"]),
                convert_rgb_to_intensity=False,
            )
            K = f.K
            intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            vol.integrate(rgbd, intr, np.linalg.inv(f.T_wc))  # extrinsic = world-to-camera
        pcd = vol.extract_point_cloud()
        pts = np.asarray(pcd.points, np.float32)
        nrm = np.asarray(pcd.normals, np.float32) if pcd.has_normals() else None
        sids = list(dict.fromkeys(f.sensor_id for f in frames))
        return finish(
            FusionResult(points=pts, normals=nrm, excluded_sensors=dict.fromkeys(sids, False)), t0
        )
