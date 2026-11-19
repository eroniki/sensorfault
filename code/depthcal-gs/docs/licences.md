# Licences and access terms (checked 2026-11-14; not legal advice)

"Read" = primary text read by a reader agent or by me. Anything else is marked unverified.

## Backbones and baselines
| Item | Licence as read | Use limits / notes |
|---|---|---|
| **Stable Diffusion 2.x** | **STILL UNAVAILABLE (retried 2026-11-14).** The HF API returns HTTP 401 for `stabilityai/stable-diffusion-2`, `-2-1`, `-2-1-base` and `-2-base`, while `stable-diffusion-xl-base-1.0` returns 200 from the same host; no HF token is configured on this machine; the Stability-AI/stablediffusion GitHub LICENSE-MODEL and README raw URLs return 404. Marigold's v1-1 card links the SD2 LICENSE-MODEL and labels it "CreativeML Open RAIL++-M" (label only). | Cannot tell whether gated, moved or removed. **Decision (author): SDXL is the first backbone**; SD 2.x only as a later comparison arm if an authenticated check clears it. |
| **SDXL base 1.0 (first backbone)** | Read in full (LICENSE.md, "CreativeML Open RAIL++-M License dated July 26, 2023"); obligations for derivatives in the next section. | Fine-tuning expressly allowed. No research-only clause (the card's "intended for research" is a statement of intent, not a licence term). |
| SD 3.5 Medium | Read: Stability AI Community License (July 5, 2024), gated click-through | Free for research/non-commercial; registration only for commercial use; revenue ($1M) clause is in the commercial grant. If distributing it or derivatives: licence copy, Notice file, "Powered by Stability AI". Must not be used to create or improve a *foundational generative AI model*; AUP incorporated (AUP not read). |
| FLUX.1 schnell | Card tag Apache-2.0, card text allows scientific use (LICENSE file not read) | Few-step distilled model. |
| FLUX.1 dev | Read: FLUX.1 [dev] Non-Commercial License v1.1.1 | Non-commercial only; restrictions pass to derivatives; excludes surveillance-related R&D. Avoid. |
| Marigold code | Apache-2.0 (README) | |
| Marigold weights | **Conflicting:** README says RAIL++-M; v1-0 card says Apache-2.0; v1-1 card says OpenRAIL++. LICENSE-MODEL not read. | Baseline only. |
| Depth Anything V2 | Cards: Small Apache-2.0; Base, Large CC-BY-NC-4.0; Giant not reachable (401). Licence texts not read. | Not a dependency; at most a baseline. |
| Apple Depth Pro | Read: Apple custom licence (LICENSE), weights "released under the LICENSE terms"; no research-only limit found; ACKNOWLEDGEMENTS.md not read | Not a dependency in v3. |
| AffineGS (2611.07709) | Paper CC BY-NC-SA 4.0 (arXiv badge). **No code link on the abstract page**; project page not read. | Baseline must be re-implemented unless code appears. |
| MVGD, DepthSync | No code or weights found | Cannot be run as baselines for now. |

## Datasets
| Dataset | Licence / terms read | Scenes with GT | Independent-researcher fit |
|---|---|---|---|
| **ETH3D** | No licence or terms text on any page read (datasets, documentation, overview, about, imprint). Only: "Users utilize the contents of this website at their own risk." Paper not read. | **High-res DSLR: 13 training scenes with public GT** (14-76 images each; laser scans + rendered depth maps, 4-byte float binaries), **12 test scenes without GT**. Low-res rig: 5 train / 5 test. | Open download, no login. **Decision (author): use with citation, no redistribution.** Absence of a licence is not a grant: keep data and any derived GT-based files out of git and out of any release; ask the maintainers before releasing derived data. |
| Matterport3D | Read (EULA for Academic Use): non-commercial academic only; trained models are "Derived Information"; recipients must give academic institution | 90 buildings | Ruled out (academic institution). |
| Hypersim | CC BY-SA 3.0 (README). Code licence separate. | 461 scenes, synthetic | Usable; ShareAlike may apply to a released derived dataset (weights unclear): legal check before release. |
| TartanAir | CC BY 4.0 | 30 sim environments | Usable, synthetic. |
| BlendedMVS / BlendedMVG | CC BY 4.0 (README); GL3D (source of the 389 extra scenes) README says MIT. Neither text addresses the Altizure source meshes or photos. | 113 + 389 scenes; depth rendered from reconstructed textured meshes (.pfm); released training images are **blended images** (high frequencies from the rendered mesh, low frequencies from the photo), not the original photos; depth units, scale and z-vs-range not stated; camera format per MVSNet (not read) | Usable (paper, README and GL3D README read 2026-11-15); photo realism and metric scale cannot be assumed |
| Replica | Replica Dataset Research Terms (non-commercial research, no institution requirement) | 18 scenes | Usable; small, synthetic-like. |
| DTU MVS | No licence text found ("freely available") | 124 scenes, lab, structured-light GT | Usable with caution; object-scale. |
| Tanks and Temples | Page contradictory: CC BY 4.0 headline but grant "sole purpose of non-commercial scientific research", no redistribution | laser scans; counts not read | Evaluation only, no redistribution. |
| ARKitScenes | Apple licence, non-commercial | 1661 scenes | Non-commercial; terms DATA.md not read. |
| 7-Scenes | Microsoft non-commercial | 7 | Small; GT from KinectFusion (not laser). |
| KITTI | CC BY-NC-SA 3.0, academic use | outdoor | Outdoor only. |
| Waymo | Non-commercial; institution-representation clause | outdoor | Ruled out for now. |
| Co3D | Code CC BY-NC 4.0; dataset licence not found; no metric GT claim found | | Not suitable (no metric GT). |
| MVS-Synth, MuSHRoom | Not fetched | | Unchecked. |

### Dataset picks (author decision 2026-11-14: no ScanNet++)
Evaluation, primary: **ETH3D** (13 scenes with public GT; use with citation, no redistribution). Labelled and label-fraction pool: **BlendedMVG/BlendedMVS (CC BY 4.0)**, **Hypersim (CC BY-SA 3.0)**, **TartanAir (CC BY 4.0)**; reasons and roles in `PLAN.md` section 8. Replica and DTU stay optional extras. Unlabelled pool: any posed RGB (ETH3D's 12 test scenes as images only; BlendedMVG images with their depth withheld). Real indoor laser ground truth is limited to ETH3D; there is no permissive replacement at larger scale.
