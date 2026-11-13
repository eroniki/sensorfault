# IARPA MVS3DM benchmark (San Fernando, Argentina) -- instructions only

Per PLAN.md, the data used by Gong & Fritsch 2018 is the IARPA Multi-View Stereo 3D
Mapping (MVS3DM) benchmark over **San Fernando, Argentina** (NOT Jacksonville/Omaha).
Optional and secondary; first to cut together with the satellite section.

## What is known (verified from search results 2026-11-13; abstract-level)
- 2016 IARPA MVS3DM challenge; ~50 DigitalGlobe WorldView-2/3 images (pan + MS) over
  ~100 km^2 near San Fernando, Argentina, collected across ~2 years; benchmark
  organised/scored by JHU/APL (Bosch et al., AIPR 2016).
- A derived public set exists: Purdue RVL "SatStereo" (arXiv:1907.04404, GNU GPL v3,
  https://engineering.purdue.edu/RVL/Database/SatStereo/). It was *not* confirmed to be
  San Fernando: the page names Jacksonville (72 stereo pairs); the paper states it
  reuses 8 AOIs from the IARPA MVS challenge. TODO(verify) which AOIs/cities.

## Unverified items (TODO(verify) all)
- Where the original MVS3DM package can be obtained today (the 2016 challenge page,
  https://www.iarpa.gov/challenges/3dchallenge.html, may be gone; JHU/APL benchmark
  repo/pages are the likely successor -- check github.com/pubgeo).
- Licence and redistribution terms of the original imagery (DigitalGlobe-owned;
  possibly released public-domain by the sponsor -- the SatStereo paper claims its
  images are public domain, unconfirmed for MVS3DM itself).
- Size, file format (NITF/TIFF + RPC), ground-truth lidar availability/format.

## Procedure
1. Check pubgeo (JHU/APL) and IARPA pages; record the URL, licence text and size here.
2. Only then write a download script following scripts/data conventions
   (`--dry-run` default, `--accept-license`, resumable, size check).
3. Layout contract (provisional): `$SENSORFAULT_DATA/mvs3dm_san_fernando/`
   non-empty; verify.py checks only existence.
4. Treat image pairs sharing an image as non-independent (shared RPC bias) -- see PLAN.md.
