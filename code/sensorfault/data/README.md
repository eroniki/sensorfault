# data/

> **Design v2 note (2026-11-13):** the current design uses ETH3D (primary), ScanNet++ (optional) and Replica
> (controlled synthetic); a download script for ETH3D is **not written yet** (TODO). The Skoltech3D, Replica-rig
> and DFC2019/MVS3DM material below belongs to design v1 (physical depth sensors) and is kept for reference.

Everything in this directory except this README is gitignored. Point the code at another
location with `SENSORFAULT_DATA=/path` (default `./data`). **We release manifests and
scripts only, never the data or derived renderings.**

| Dataset | Role | Licence | Size | Obtain with |
|---|---|---|---|---|
| Skoltech3D | Primary real data (Kinect v2 ToF, 2 phone ToF, RealSense D435 stereo, SL reference) | CC BY-NC 4.0 (code repo: MIT) | per-scene RealSense depth ~2.6 GiB; fixed part ~11 GiB (Kinect 5, phones 1+1, SL scan 3, calib 0.04); 2 scenes ~ 15 GiB | `scripts/data/skoltech3d.py` |
| Replica | Synthetic multi-sensor rig, room-scale multipath | Replica Research Terms: non-commercial research only, do not redistribute renderings | 33.9 GB compressed (~2x extracted, unverified) | `scripts/data/replica.sh`, `scripts/data/render_replica_multisensor.py` (stub) |
| US3D / DFC2019 | OPTIONAL satellite section | DataPort says CC BY 4.0; contest terms forbid redistribution, publication needs approval | ~289 GB (plan figure, unverified) | instructions only: `scripts/data/us3d_dfc2019.md` (BLOCKED) |
| IARPA MVS3DM (San Fernando, AR) | OPTIONAL satellite | unverified | unverified | instructions only: `scripts/data/mvs3dm_san_fernando.md` |

All download scripts are dry-run by default, need `--yes --accept-license` to transfer,
resume with `curl -C -` / `wget --continue`, and check sizes (upstream publishes no checksums
for Skoltech3D or Replica). Check what you have: `python3 -I scripts/data/verify.py`.

## Release policy
Publish: code, configs, seeds, fault schedules, per-file SHA-256 manifests, scene lists.
Do not publish: any dataset file, any rendering from Replica, any DFC2019 tile, or
numeric dumps from which they can be reconstructed. Results tables/figures are fine
under each licence's attribution rules (Skoltech3D and DFC2019: cite as required).

## Layout contract (`$SENSORFAULT_DATA/`)
```
skoltech3d/
  _zips/  _conf.tsv  _tools/sk3d_data/            # download cache, official tooling
  dataset/calibration/{kinect_v2,phone_right,phone_left,real_sense}/{ir,rgb,...}/{cameras.txt,intrinsics.yaml,undistortion.pt}
  dataset/{scene}/stl/reconstruction/cleaned.ply  # reference scan
  dataset/{scene}/{cam}/depth/undistorted/{pos:04}.png      # real_sense: .../{light}/{pos:04}.png
  dataset/{scene}/{cam}/{mode}/images.txt         # poses (COLMAP text)
replica/
  _parts/                                          # downloaded tar parts
  {scene}/{mesh.ply,textures/,glass.sur,habitat/}  # as upstream download.sh extracts (TODO(verify))
replica_rendered/{scene}/{rig_id}/{sensor}/{depth_clean,depth}/{frame:05d}.png, poses.txt, rig.json, manifest.json
us3d_dfc2019/        # optional, blocked
mvs3dm_san_fernando/ # optional, unverified
```
Zip-internal roots for Skoltech3D (`dataset/`, `raw/`, `addons/`) follow the upstream README path
table; the extraction assumption is marked TODO(verify) in `skoltech3d.py`.
