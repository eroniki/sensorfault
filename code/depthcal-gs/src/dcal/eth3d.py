"""ETH3D multi-view (DSLR) interface: COLMAP-text calibration, depth dumps, undistorted ground truth.

Facts used (ETH3D documentation, https://www.eth3d.net/documentation, and checked on the `pipes` scene):
- cameras.txt / images.txt are COLMAP text; pose = world -> camera, Hamilton quaternion (QW QX QY QZ), x right, y down, z forward.
- original images use THIN_PRISM_FISHEYE (12 params: fx fy cx cy k1 k2 p1 p2 k3 k4 sx1 sy1); pre-undistorted images use PINHOLE.
- `ground_truth_depth/dslr_images/*.JPG` are 4-byte float dumps, row-major, at the ORIGINAL (distorted) resolution; missing = +inf;
  they are NOT aligned with the undistorted images, so they are re-sampled here.
- Camera model equations follow COLMAP (src/colmap/sensor/models.h, ThinPrismFisheyeCameraModel); pixel centres are at +0.5.
Whether the dump stores z-depth or ray range is checked on real data by `scripts/eth3d_check_scene.py` (`depth_kind`).
Data are never stored in git; see `manifest.py`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_IMG_EXT = re.compile(r"\.(jpe?g|png)$", re.I)


@dataclass(frozen=True)
class Camera:
    cam_id: int
    model: str
    width: int
    height: int
    params: tuple

    @property
    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    q: tuple      # (w, x, y, z)
    t: tuple
    cam_id: int
    name: str

    @property
    def T_cw(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = qvec_to_R(self.q)
        T[:3, 3] = self.t
        return T


def qvec_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        raise ValueError("zero quaternion")
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def parse_cameras(path) -> dict[int, Camera]:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        out[int(tok[0])] = Camera(int(tok[0]), tok[1], int(tok[2]), int(tok[3]), tuple(float(v) for v in tok[4:]))
    return out


def parse_images(path) -> list[ImagePose]:
    """Extrinsics lines only: the keypoint line that follows each one is numeric triples and has no image extension."""
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()
        if len(tok) >= 10 and _IMG_EXT.search(tok[-1]):
            out.append(ImagePose(int(tok[0]), tuple(float(v) for v in tok[1:5]), tuple(float(v) for v in tok[5:8]),
                                 int(tok[8]), tok[9]))
    return out


def read_depth(path, width: int, height: int) -> np.ndarray:
    """Float32 (H, W), NaN where the file stores +inf. Checks the file size against the camera resolution."""
    p = Path(path)
    if p.stat().st_size != 4 * width * height:
        raise ValueError(f"{p.name}: {p.stat().st_size} bytes, expected {4 * width * height} for {width}x{height}")
    d = np.fromfile(p, dtype="<f4").reshape(height, width)
    return np.where(np.isfinite(d), d, np.nan).astype(np.float32)


# ---- camera models -----------------------------------------------------------------------------------------------

def _thin_prism_distortion(extra, u, v):
    k1, k2, p1, p2, k3, k4, sx1, sy1 = extra
    u2, uv, v2 = u * u, u * v, v * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) + sx1 * r2
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) + sy1 * r2
    return du, dv


def img_from_cam(cam: Camera, xyz: np.ndarray):
    """Project camera-frame points (N, 3) to continuous pixel coordinates (N, 2); returns (xy, valid)."""
    xyz = np.asarray(xyz, float)
    w = xyz[:, 2]
    valid = w > 1e-9
    wz = np.where(valid, w, 1.0)
    u, v = xyz[:, 0] / wz, xyz[:, 1] / wz
    fx, fy, cx, cy = cam.params[:4]
    if cam.model == "PINHOLE":
        uu, vv = u, v
    elif cam.model == "THIN_PRISM_FISHEYE":
        r = np.sqrt(u * u + v * v)
        scale = np.where(r > 1e-12, np.arctan(r) / np.maximum(r, 1e-12), 1.0)
        uu, vv = u * scale, v * scale
        du, dv = _thin_prism_distortion(cam.params[4:12], uu, vv)
        uu, vv = uu + du, vv + dv
    else:
        raise NotImplementedError(cam.model)
    return np.stack([fx * uu + cx, fy * vv + cy], 1), valid


def cam_from_img(cam: Camera, xy: np.ndarray, iters: int = 20) -> np.ndarray:
    """Normal coordinates (u, v) = (x/z, y/z) of the rays through pixels (N, 2); Newton iteration for the distortion."""
    xy = np.asarray(xy, float)
    fx, fy, cx, cy = cam.params[:4]
    ud, vd = (xy[:, 0] - cx) / fx, (xy[:, 1] - cy) / fy
    if cam.model == "PINHOLE":
        return np.stack([ud, vd], 1)
    if cam.model != "THIN_PRISM_FISHEYE":
        raise NotImplementedError(cam.model)
    extra = cam.params[4:12]
    u, v = ud.copy(), vd.copy()
    eps = 1e-7
    for _ in range(iters):
        du, dv = _thin_prism_distortion(extra, u, v)
        fu, fv = u + du - ud, v + dv - vd
        dux, dvx = _thin_prism_distortion(extra, u + eps, v)
        duy, dvy = _thin_prism_distortion(extra, u, v + eps)
        a, b = 1 + (dux - du) / eps, (duy - du) / eps
        c, d = (dvx - dv) / eps, 1 + (dvy - dv) / eps
        det = a * d - b * c
        u, v = u - (d * fu - b * fv) / det, v - (-c * fu + a * fv) / det
    theta = np.sqrt(u * u + v * v)
    scale = np.where(theta > 1e-12, np.tan(theta) / np.maximum(theta, 1e-12), 1.0)
    return np.stack([u * scale, v * scale], 1)


def undistort_depth(depth_src: np.ndarray, cam_src: Camera, cam_dst: Camera, scale: float = 1.0,
                    depth_kind: str = "z", dst_shape: tuple | None = None):
    """Gather ground-truth depth into the undistorted pinhole view (nearest neighbour; invalid stays NaN).

    ``scale`` > 1 shrinks the destination image by that factor (continuous coordinates scale linearly).
    Returns (depth_z, K_dst). Both cameras share centre and orientation, so z-depth is unchanged by the re-projection;
    with ``depth_kind="range"`` the stored value is first converted to z-depth along the ray.
    """
    if cam_dst.model != "PINHOLE":
        raise ValueError("destination camera must be PINHOLE")
    h = int(round(cam_dst.height / scale)) if dst_shape is None else dst_shape[0]
    w = int(round(cam_dst.width / scale)) if dst_shape is None else dst_shape[1]
    sx, sy = cam_dst.width / w, cam_dst.height / h
    fx, fy, cx, cy = cam_dst.params[:4]
    jj, ii = np.mgrid[0:h, 0:w]
    x_full = (ii + 0.5) * sx
    y_full = (jj + 0.5) * sy
    u, v = (x_full - cx) / fx, (y_full - cy) / fy
    rays = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 1)
    xy, ok = img_from_cam(cam_src, rays)
    xi = np.floor(xy[:, 0]).astype(int)
    yi = np.floor(xy[:, 1]).astype(int)
    inb = ok & (xi >= 0) & (xi < cam_src.width) & (yi >= 0) & (yi < cam_src.height)
    out = np.full(h * w, np.nan, np.float32)
    d = depth_src[yi[inb], xi[inb]]
    if depth_kind == "range":
        d = d / np.sqrt(1 + u.ravel()[inb] ** 2 + v.ravel()[inb] ** 2)
    elif depth_kind != "z":
        raise ValueError(depth_kind)
    out[inb] = d
    K = np.array([[fx / sx, 0, cx / sx], [0, fy / sy, cy / sy], [0, 0, 1.0]])
    return out.reshape(h, w), K


@dataclass(frozen=True)
class Eth3DView:
    view_id: int
    name: str
    rgb: np.ndarray        # (H, W, 3) float in [0, 1]
    K: np.ndarray
    T_cw: np.ndarray
    gt_depth: np.ndarray   # (H, W) z-depth, NaN = no ground truth


def load_scene(root, scene: str, scale: float = 8.0, depth_kind: str = "z", names: list[str] | None = None):
    """Load undistorted views of one ETH3D scene, with ground-truth depth re-sampled into them (needs Pillow)."""
    from PIL import Image

    base = Path(root) / scene / scene if (Path(root) / scene / scene).exists() else Path(root) / scene
    cams_u = parse_cameras(base / "dslr_calibration_undistorted" / "cameras.txt")
    poses_u = parse_images(base / "dslr_calibration_undistorted" / "images.txt")
    cams_o = parse_cameras(base / "dslr_calibration_jpg" / "cameras.txt")
    poses_o = {Path(p.name).name: p for p in parse_images(base / "dslr_calibration_jpg" / "images.txt")}
    views = []
    for p in sorted(poses_u, key=lambda q: q.name):
        fname = Path(p.name).name
        if names is not None and fname not in names:
            continue
        cam_u = cams_u[p.cam_id]
        po = poses_o[fname]
        cam_o = cams_o[po.cam_id]
        depth = read_depth(base / "ground_truth_depth" / "dslr_images" / fname, cam_o.width, cam_o.height)
        img = Image.open(base / "images" / "dslr_images_undistorted" / fname).convert("RGB")
        h, w = int(round(cam_u.height / scale)), int(round(cam_u.width / scale))
        rgb = np.asarray(img.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
        gt, K = undistort_depth(depth, cam_o, cam_u, scale=scale, depth_kind=depth_kind, dst_shape=(h, w))
        views.append(Eth3DView(len(views), fname, rgb, K, p.T_cw, gt))
    return views
