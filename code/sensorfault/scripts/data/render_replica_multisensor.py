#!/usr/bin/env python3
"""STUB: render per-sensor depth along a trajectory in a Replica scene.

Status: NOT IMPLEMENTED (raises NotImplementedError). Spec only.

Purpose (PLAN.md section 6, "Replica"): build a synthetic multi-sensor depth rig
with per-sensor intrinsics, FOV, noise and extrinsics, for room-scale scenes with
concave corners (multipath). Fault injection is NOT done here; faults are applied
downstream by the sim/faults modules so that clean renderings are the single
source of truth.

Backends (pick one; TODO(verify) which is installed/maintained):
  * habitat-sim with the Replica habitat configs (replica_stage.stage_config.json,
    https://github.com/facebookresearch/habitat-sim), or
  * ReplicaRenderer from facebookresearch/Replica-Dataset (headless, Pangolin/EGL).

Inputs
  --scene-dir   $SENSORFAULT_DATA/replica/<scene>
  --trajectory  camera-rig poses file (N x 4x4 world_T_rig); generated from a seed
  --rig         YAML/JSON: list of sensors, each with
                  name, width, height, fx, fy, cx, cy (or hfov), near, far,
                  T_rig_sensor (4x4 extrinsics), noise model id + params
  --seed        integer; every random draw derives from (seed, sensor, frame)
  --out         $SENSORFAULT_DATA/replica_rendered/<scene>/<rig_id>/
Outputs (per sensor, per frame i)
  <out>/<sensor>/depth_clean/{i:05d}.png   uint16 mm, 0 = invalid
  <out>/<sensor>/depth/{i:05d}.png         after the sensor noise model
  <out>/poses.txt, <out>/rig.json, <out>/manifest.json (code git SHA, seed, args,
  SHA-256 of every output) -- the manifest + generator are what get released.
Noise models to support (sim must stay deterministic given the seed): zero-mean
Gaussian with depth-dependent sigma (stereo: sigma ~ z^2), ToF multipath bias near
concave corners (may need a path-tracing pass or a geometric approximation --
document the choice), dropout at grazing angles / far range, quantisation.
Constraints
  * Licence: Replica Research Terms -> rendered images are derivative data and must
    NOT be redistributed. Write only under the gitignored data directory.
  * Hot path deterministic and allocation-light; no hardcoded paths.
"""
from __future__ import annotations

import argparse
import sys


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--rig", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["habitat-sim", "replica-renderer"], default="habitat-sim")
    ap.add_argument("--clean-only", action="store_true", help="skip the sensor noise model")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    raise NotImplementedError(f"render_replica_multisensor is a stub (backend={args.backend})")


if __name__ == "__main__":
    sys.exit(main())
