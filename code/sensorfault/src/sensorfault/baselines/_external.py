"""Thin wrappers around external research repos, run via subprocess from
``$SENSORFAULT_THIRD_PARTY/<name>`` (see scripts/third_party/setup.sh). Nothing in this
module was verified against the real repos: command lines and data layouts are marked
TODO(verify). Third-party code is never vendored."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from sensorfault.baselines._colmap import write_colmap_model
from sensorfault.baselines._common import ParamFuser
from sensorfault.baselines.base import register
from sensorfault.types import DepthFrame, FusionResult


def third_party_root() -> Path:
    return Path(os.environ.get("SENSORFAULT_THIRD_PARTY", "third_party"))


class ExternalRepoFuser(ParamFuser):
    repo_dir: str = ""
    marker_files: tuple = ()  # relative paths that must exist in the clone
    needs_weights: bool = False
    weights_env: str = ""
    todo: tuple = ()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        root = third_party_root() / cls.repo_dir
        if not root.is_dir():
            return False, f"repo missing: {root} (run scripts/third_party/setup.sh --no-dry-run)"
        miss = [m for m in cls.marker_files if not (root / m).exists()]
        if miss:
            return False, f"repo {root} lacks expected files {miss} (unverified layout)"
        if cls.needs_weights:
            w = os.environ.get(cls.weights_env, "")
            if not w or not Path(w).exists():
                return False, f"weights missing: set ${cls.weights_env} to an existing path"
        return True, ""

    def _todo_error(self) -> NotImplementedError:
        return NotImplementedError(
            f"{self.name}: not implemented. TODO:\n  - " + "\n  - ".join(self.todo)
        )

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        raise self._todo_error()


@register("senfunet")
class SenFuNet(ExternalRepoFuser):
    """L1 (ECCV 2022). Stub: dataset layout/config not verified."""

    repo_dir = "SenFuNet"
    marker_files = ("README.md",)
    needs_weights = True
    weights_env = "SENFUNET_WEIGHTS"
    todo = (
        "verify repo layout/config yaml (python 3.8, open3d>=0.13, submodule, hard-coded paths)",
        "write dataset adapter (Replica/CoRBS-like folder layout) from DepthFrame lists, 2 sensors per scene",
        "pretrained-weight URL unverified; add fine-tune-on-clean and fine-tune-with-bias rows",
        "run train/test script via subprocess, read output mesh/TSDF -> points",
    )


@register("routedfusion")
class RoutedFusion(ExternalRepoFuser):
    """L2 (CVPR 2020). Single-sensor; sensors would run sequentially. Stub."""

    repo_dir = "RoutedFusion"
    marker_files = ("README.md",)
    needs_weights = True
    weights_env = "ROUTEDFUSION_WEIGHTS"
    todo = (
        "verify repo layout/config; ShapeNet-trained pretrained weights (URL unverified)",
        "write per-sensor sequential dataset adapter (no cross-sensor mechanism)",
        "run fusion via subprocess; extract surface -> points",
    )


class _SplatFuser(ExternalRepoFuser):
    """Secondary same-representation baselines. ``prepare`` writes a COLMAP text model +
    grey placeholder images (RGB is required by both repos); depth supervision files and
    the training command are not verified."""

    def prepare(self, frames: Sequence[DepthFrame], workdir: Path) -> Path:
        write_colmap_model(frames, workdir)
        return workdir

    def fuse(self, frames: Sequence[DepthFrame], cfg: Mapping) -> FusionResult:
        raise self._todo_error()


@register("gs_mcmc_depth")
class GSMCMCDepth(_SplatFuser):
    repo_dir = "3dgs-mcmc"
    marker_files = ("train.py",)
    todo = (
        "add depth loss + accumulated-alpha mask loss to train.py (needs a patch/fork; do not commit upstream code)",
        "feed depth maps as files (format unverified) alongside the COLMAP model from prepare()",
        "fix N (cap_max = init count), cap anisotropy, scale SGLD noise below sigma_n",
        "export Gaussians -> points (centres) or rendered depth fusion",
    )


@register("twodgs_depth")
class TwoDGSDepth(_SplatFuser):
    repo_dir = "2d-gaussian-splatting"
    marker_files = ("train.py",)
    todo = (
        "add external-depth loss; rescale depth-distortion / normal-consistency terms to metric units",
        "placeholder-RGB dataset from prepare() (+ an RGB-available row)",
        "extract mesh/points via render.py (flags unverified)",
    )
