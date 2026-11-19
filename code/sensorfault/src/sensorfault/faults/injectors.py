"""Seeded, deterministic fault injectors operating on lists of ``DepthFrame``.

Frames are never mutated; new ``DepthFrame`` objects are returned for faulted frames.
"Time" for a sensor = order of its frames by ``frame_id`` (PLAN.md: Skoltech3D uses
viewpoint order). Onset index = round(onset_frac * n_frames_of_that_sensor).

Fault kinds
-----------
extrinsic_drift   believed pose T_wc -> T_wc @ dT(s); rotation (deg) and/or translation (cm)
depth_bias_const  depth += sign * frac * median_valid_range_of_sensor  (or abs ``bias_m``)
depth_bias_range  depth *= (1 + sign * beta)
multipath         positive bias A * w(concavity); HEURISTIC surrogate, not physical
transient         frame-level bursts: dropout | offset | noise | pixel_outliers
common-mode      params.common_fault_id -> identical realisation across sensors
Profiles: step | ramp | random_walk (scale s_i in [0,1] for step/ramp; random_walk is a
signed cumulative walk normalised to unit end-std).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
from scipy.spatial.transform import Rotation

from sensorfault.types import DepthFrame

from .seeds import derive_rng, seed_entropy, stable_hash

KINDS = ("extrinsic_drift", "depth_bias_const", "depth_bias_range", "multipath", "transient")
PROFILES = ("step", "ramp", "random_walk")


@dataclass
class FaultSpec:
    kind: str
    sensor_id: str
    fault_id: int = 0
    params: dict = field(default_factory=dict)
    onset_frac: float = 0.0
    profile: str = "step"

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}")
        if self.profile not in PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}")
        if not 0.0 <= self.onset_frac <= 1.0:
            raise ValueError("onset_frac must be in [0,1]")


def profile_scales(n: int, onset_frac: float, profile: str, rng: np.random.Generator) -> np.ndarray:
    """Per-frame fault scale s_i (float64, length n)."""
    i0 = int(round(onset_frac * n))
    s = np.zeros(n)
    if i0 >= n:
        return s
    m = n - i0
    if profile == "step":
        s[i0:] = 1.0
    elif profile == "ramp":
        s[i0:] = np.linspace(1.0 / m, 1.0, m)
    elif profile == "random_walk":
        steps = rng.standard_normal(m) / np.sqrt(m)
        s[i0:] = np.cumsum(steps)
    else:
        raise ValueError(profile)
    return s


def _rand_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def _valid(d: np.ndarray) -> np.ndarray:
    return np.isfinite(d) & (d > 0)


def _replace(fr: DepthFrame, **kw) -> DepthFrame:
    d = dict(
        sensor_id=fr.sensor_id,
        frame_id=fr.frame_id,
        depth=fr.depth,
        K=fr.K,
        T_wc=fr.T_wc,
        conf=fr.conf,
    )
    d.update(kw)
    return DepthFrame(**d)


# ---------------------------------------------------------------- multipath surrogate
def multipath_weight(
    depth: np.ndarray,
    K: np.ndarray,
    *,
    win: int = 5,
    c_ref_frac: float = 0.01,
    jump_frac: float = 0.1,
) -> np.ndarray:
    """Concavity weight w in [0,1] per pixel (HEURISTIC).

    1. Back-project to camera-frame points P; normals n from finite differences, oriented
       toward the camera.
    2. Concavity c = (mean_window(P) - P) . n : positive when the neighbourhood lies in
       front of the local tangent plane (concave, as seen from the camera side).
    3. w = clip(c / (c_ref_frac * z), 0, 1); pixels whose 3x3 depth range exceeds
       ``jump_frac * z`` (occlusion boundaries) get w = 0 (depth-gradient gate).
    Invalid pixels get 0. This is a stand-in for interreflection, NOT a physical model.
    """
    h, w_ = depth.shape
    z = np.where(_valid(depth), depth, 0.0).astype(np.float64)
    u, v = np.meshgrid(np.arange(w_), np.arange(h))
    P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
    dx = np.gradient(P, axis=1)
    dy = np.gradient(P, axis=0)
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    flip = np.sum(n * (-P), -1, keepdims=True) < 0  # point toward camera
    n = np.where(flip, -n, n)
    Pm = np.stack([uniform_filter(P[..., c], size=win, mode="nearest") for c in range(3)], -1)
    c = np.sum((Pm - P) * n, -1)
    wgt = np.clip(c / np.maximum(c_ref_frac * z, 1e-9), 0.0, 1.0)
    vmax = maximum_filter(z, size=3, mode="nearest")
    big = np.where(z > 0, z, np.inf)
    vmin = minimum_filter(big, size=3, mode="nearest")
    jump = (vmax - np.where(np.isfinite(vmin), vmin, 0.0)) > jump_frac * np.maximum(z, 1e-9)
    wgt = np.where(jump | (z <= 0) | (nn[..., 0] < 1e-12), 0.0, wgt)
    return wgt


# --------------------------------------------------------------------- injectors
def _inject_extrinsic(frames, rng, p, scales):
    rot_deg = float(p.get("rotation_deg", 0.0))
    t_cm = float(p.get("translation_cm", 0.0))
    ar, at = _rand_unit(rng), _rand_unit(rng)
    out = []
    for fr, s in zip(frames, scales):
        dT = np.eye(4)
        dT[:3, :3] = Rotation.from_rotvec(np.deg2rad(rot_deg) * s * ar).as_matrix()
        dT[:3, 3] = 0.01 * t_cm * s * at
        out.append(_replace(fr, T_wc=fr.T_wc @ dT))
    return out, {"rot_axis": ar.tolist(), "trans_axis": at.tolist()}


def _sign(p, rng) -> float:
    s = p.get("sign", None)
    if s is None:
        return float(rng.choice([-1.0, 1.0]))
    return 1.0 if float(s) >= 0 else -1.0


def _inject_bias_const(frames, rng, p, scales):
    sign = _sign(p, rng)
    if "bias_m" in p:
        mag = float(p["bias_m"])
    else:
        vals = (
            np.concatenate([f.depth[_valid(f.depth)].ravel() for f in frames])
            if frames
            else np.zeros(1)
        )
        med = float(np.median(vals)) if vals.size else 0.0
        mag = float(p.get("frac_of_median_range", 0.02)) * med
    out = []
    for fr, s in zip(frames, scales):
        d = fr.depth.astype(np.float32)
        d = np.where(_valid(d), d + np.float32(sign * mag * s), d)
        d = np.where(_valid(fr.depth) & (d <= 0), 0.0, d).astype(np.float32)
        out.append(_replace(fr, depth=d))
    return out, {"sign": sign, "bias_m": mag}


def _inject_bias_range(frames, rng, p, scales):
    sign = _sign(p, rng)
    beta = float(p.get("beta", 0.02))
    out = []
    for fr, s in zip(frames, scales):
        d = np.where(
            _valid(fr.depth), fr.depth * np.float32(1.0 + sign * beta * s), fr.depth
        ).astype(np.float32)
        out.append(_replace(fr, depth=d))
    return out, {"sign": sign, "beta": beta}


def _inject_multipath(frames, rng, p, scales):
    A = float(p.get("amplitude_m", 0.05))
    out = []
    for fr, s in zip(frames, scales):
        w = multipath_weight(
            fr.depth, fr.K, win=int(p.get("win", 5)), c_ref_frac=float(p.get("c_ref_frac", 0.01))
        )
        bias = np.maximum(A * max(float(s), 0.0) * w, 0.0)  # clamped >= 0
        d = np.where(_valid(fr.depth), fr.depth + bias, fr.depth).astype(np.float32)
        out.append(_replace(fr, depth=d))
    return out, {"amplitude_m": A, "heuristic": True}


def transient_mask(
    n: int, fraction: float, burst_min: int, burst_max: int, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask of ~fraction*n frames covered by bursts of length U{burst_min..burst_max}."""
    m = np.zeros(n, bool)
    target = int(round(fraction * n))
    tries = 0
    while m.sum() < target and tries < 1000:
        L = int(rng.integers(burst_min, burst_max + 1))
        a = int(rng.integers(0, max(n - 1, 1)))
        m[a : a + L] = True
        tries += 1
    if m.sum() > target and target > 0:  # trim last burst overshoot deterministically from the end
        idx = np.flatnonzero(m)
        m[idx[target:]] = False
    return m


def _inject_transient(frames, rng, p, scales):
    n = len(frames)
    mask = transient_mask(
        n,
        float(p.get("fraction", 0.1)),
        int(p.get("burst_min", 3)),
        int(p.get("burst_max", 10)),
        rng,
    )
    mode = p.get("mode", "offset")
    mag = float(p.get("magnitude_m", 0.1))
    out = []
    for fr, on in zip(frames, mask):
        if not on:
            out.append(fr)
            continue
        d = fr.depth
        if mode == "dropout":
            nd = np.zeros_like(d)
        elif mode == "offset":
            nd = np.where(_valid(d), d + np.float32(mag), d)
        elif mode == "pixel_outliers":  # sparse independent outliers (voting-friendly control)
            hit = _valid(d) & (rng.random(d.shape) < float(p.get("pixel_fraction", 0.05)))
            nd = np.where(hit, np.maximum(d + rng.uniform(-mag, mag, d.shape), 1e-3), d)
        elif mode == "noise":
            nd = np.where(_valid(d), np.maximum(d + rng.normal(0, mag, d.shape), 1e-3), d)
        else:
            raise ValueError(mode)
        out.append(_replace(fr, depth=nd.astype(np.float32)))
    return out, {"mode": mode, "active_frames": np.flatnonzero(mask).tolist()}


_INJ = {
    "extrinsic_drift": _inject_extrinsic,
    "depth_bias_const": _inject_bias_const,
    "depth_bias_range": _inject_bias_range,
    "multipath": _inject_multipath,
    "transient": _inject_transient,
}


def inject_faults(
    frames: Sequence[DepthFrame], specs: Sequence[FaultSpec], master_seed: int
) -> Tuple[List[DepthFrame], dict]:
    """Apply ``specs`` in order. Returns (new frame list in the original order, manifest dict).

    Deterministic: result depends only on (frames, specs, master_seed). Each spec draws from
    ``SeedSequence([master, stable_hash(sensor_id), fault_id])``; reordering specs does not
    change any fault's realisation.
    """
    frames = list(frames)
    index: Dict[str, List[int]] = {}
    for i, f in enumerate(frames):
        index.setdefault(f.sensor_id, []).append(i)
    for k in index:
        index[k].sort(key=lambda i: frames[i].frame_id)
    records = []
    for spec in specs:
        spec.validate()
        idxs = index.get(spec.sensor_id)
        if not idxs:
            raise KeyError(f"sensor {spec.sensor_id!r} not in frames")
        # common_fault_id: all specs sharing it draw ONE realisation (axis/sign), for common-mode faults
        rng = (
            derive_rng(master_seed, "__common__", int(spec.params["common_fault_id"]))
            if "common_fault_id" in spec.params
            else derive_rng(master_seed, spec.sensor_id, spec.fault_id)
        )
        sub = [frames[i] for i in idxs]
        scales = profile_scales(len(sub), spec.onset_frac, spec.profile, rng)
        new, resolved = _INJ[spec.kind](sub, rng, spec.params, scales)
        for i, nf in zip(idxs, new):
            frames[i] = nf
        records.append(
            {
                **asdict(spec),
                "seed_entropy": seed_entropy(master_seed, spec.sensor_id, spec.fault_id),
                "sensor_hash": stable_hash(spec.sensor_id),
                "onset_index": int(round(spec.onset_frac * len(sub))),
                "n_frames": len(sub),
                "frame_ids": [int(f.frame_id) for f in sub],
                "scales": [float(x) for x in scales],
                "resolved": resolved,
            }
        )
    manifest = {
        "schema": "sensorfault.fault_manifest/1",
        "master_seed": int(master_seed),
        "faults": records,
        "faulty_sensors": sorted({r["sensor_id"] for r in records}),
    }
    return frames, manifest
