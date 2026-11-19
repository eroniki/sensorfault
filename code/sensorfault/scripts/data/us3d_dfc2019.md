# US3D / IEEE GRSS DFC2019 (OPTIONAL, BLOCKED until publication approval)

Status: **OPTIONAL satellite section (first to be cut per PLAN.md). Do not download
until the licence question below is settled.** No script is provided on purpose:
access requires an IEEE account login and acceptance of contest terms, which must
be done by a human.

## What it is
- Urban Semantic 3D (US3D): WorldView-3 satellite imagery plus airborne-lidar
  ground truth over Jacksonville FL (JAX) and Omaha NE (OMA), ~100 km^2.
  The DFC2019 contest release is ~20% of US3D in 500 m x 500 m tiles
  (111 train / 10 val / 10 test) [GRSS DFC2019 page, verified 2026-11-13].
- Data owners: DigitalGlobe, IARPA, Johns Hopkins University APL.

## How to get it (human steps)
1. DFC2019 package: IEEE DataPort, DOI 10.21227/c6tm-vw12 (https://ieee-dataport.org/open-access/2019-ieee-grss-data-fusion-contest-data).
   Requires a free IEEE account. TODO(verify): node/1797 page and exact file list.
2. Extended US3D (separate, larger): DOI 10.21227/9frn-7208, listed total ~858 GB
   (JAX, OMA, ATL, point clouds, geopose); page text is inconsistent about whether a
   paid IEEE DataPort subscription is needed. Not what the plan needs; avoid.
3. Size: PLAN.md budgets ~289 GB for the DFC2019 release. TODO(verify) against the
   DataPort file list before starting a transfer.
4. Use a resumable client (`curl -C -` / `wget -c`) on DataPort's signed S3 URLs;
   they expire, so re-fetch the link on resume. Verify sizes against the file list;
   DataPort publishes no checksums that I could verify.

## Licence conflict (important)
- DataPort label: **CC BY 4.0**.
- Contest Terms and Conditions (GRSS page, verified 2026-11-13): "Any dissemination
  or distribution of the data packages by any registered user is strictly forbidden."
  Publications using the data need case-by-case approval by the IADF TC and the data
  owners: send the manuscript to **iadf_chairs@grss-ieee.org**; identify data as
  `grss_dfc_2019`; acknowledge JHU/APL, IARPA and the IADF TC; cite Bosch et al. 2019
  and Le Saux et al. 2019.
- Consequences for this repo: never commit or mirror the data or derived tiles;
  release manifests + scripts only; email the IADF TC (once, only if the satellite
  section stays in the paper) before submission.

## Expected layout once approved (contract used by verify.py)
`$SENSORFAULT_DATA/us3d_dfc2019/` containing `Track1*/` (or the DataPort TAR
contents extracted as-is; TODO(verify) exact folder names after extraction), and
the folder must carry a `README*` file from the package. verify.py only checks that
the directory exists and is non-empty.
