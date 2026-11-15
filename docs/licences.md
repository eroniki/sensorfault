# Licences and access terms (checked 2026-11-14; not legal advice)

"Read" = primary text read by a reader agent or by me. Anything else is marked unverified.

## Backbones and baselines
| Item | Licence as read | Use limits / notes |
|---|---|---|
| **Stable Diffusion 2.x** | **NOT VERIFIED.** `stabilityai/stable-diffusion-2` and `-2-1` (and `-2-1-base`) return HTTP 401 from the HF page and API; the Stability-AI/stablediffusion GitHub raw LICENSE-MODEL returns 404. Marigold's v1-1 card links its LICENSE-MODEL and labels it "CreativeML Open RAIL++-M". | Cannot tell whether gated, moved or removed. Author must check from an authenticated HF session. Until then SD 2.x is **not cleared**. |
| SDXL base 1.0 | Read: CreativeML Open RAIL++-M (July 26, 2023) | Fine-tuning expressly allowed. Derivatives must carry at least the same use-based restrictions, the licence copy, change notices; restrictions must be an enforceable term in any agreement. No research-only clause (the card's "intended for research" is not a licence term). |
| SD 3.5 Medium | Read: Stability AI Community License (July 5, 2024), gated click-through | Free for research/non-commercial; registration only for commercial use; revenue ($1M) clause is in the commercial grant. If distributing it or derivatives: licence copy, Notice file, "Powered by Stability AI". Must not be used to create or improve a *foundational generative AI model*; AUP incorporated (AUP not read). |
| FLUX.1 schnell | Card tag Apache-2.0, card text allows scientific use (LICENSE file not read) | Few-step distilled model. |
| FLUX.1 dev | Read: FLUX.1 [dev] Non-Commercial License v1.1.1 | Non-commercial only; restrictions pass to derivatives; excludes surveillance-related R&D. Avoid. |
| Marigold code | Apache-2.0 (README) | |
| Marigold weights | **Conflicting:** README says RAIL++-M; v1-0 card says Apache-2.0; v1-1 card says OpenRAIL++. LICENSE-MODEL not read. | Baseline only. |
| Depth Anything V2 | Cards: Small Apache-2.0; Base, Large CC-BY-NC-4.0; Giant not reachable (401). Licence texts not read. | Not a dependency; at most a baseline. |
| Apple Depth Pro | Read: Apple custom licence (LICENSE), weights "released under the LICENSE terms"; no research-only limit found; ACKNOWLEDGEMENTS.md not read | Not a dependency in v3. |
| AffineGS (2611.07709) | Paper CC BY-NC-SA 4.0 (arXiv badge). **No code link on the abstract page**; project page not read. | Baseline must be re-implemented unless code appears. |
| MVGD, DepthSync | No code or weights found | Cannot be run as baselines for now. |

## Datasets (see also docs section in PLAN.md)
| Dataset | Licence / terms read | Scenes with GT | Independent-researcher fit |
|---|---|---|---|
| **ETH3D** | No licence or terms text on any page read (datasets, documentation, overview, about, imprint). Only: "Users utilize the contents of this website at their own risk." Paper not read. | **High-res DSLR: 13 training scenes with public GT** (14-76 images each; laser scans + rendered depth maps, 4-byte float binaries), **12 test scenes without GT**. Low-res rig: 5 train / 5 test. | Open download, no login. Absence of a licence is not a grant: use for research with citation, ask the maintainers before releasing derived data. |
| **ScanNet++ v2** | **Read** (TUM Terms of Use PDF, decoded): non-commercial research/education only; data may be shared only with colleagues who sign the same terms; GDPR deletion duty; TUM may terminate access; **clause 7: if the researcher is employed by a for-profit entity, the employer is also bound and the researcher represents authority to bind them**; requires requester and a PI/supervisor signature (handwritten). Footnote: independent applicants must be "established researchers with a publicly verifiable research profile". | 1006 scenes; splits 856 train, 50 val; test scenes have no GT. Laser scan + iPhone LiDAR depth. | **Flagged. Not applied for.** Needs: verifiable research profile for independent status, and clause 7 would bind any for-profit employer, which conflicts with the "no employer" rule. Treat as ruled out unless the author decides otherwise. Terms are silent on derived model weights; no-redistribution means pseudo-labels derived from it cannot be shared. |
| Matterport3D | Read (EULA for Academic Use): non-commercial academic only; trained models are "Derived Information"; recipients must give academic institution | 90 buildings | Ruled out (academic institution). |
| Hypersim | CC BY-SA 3.0 (README). Code licence separate. | 461 scenes, synthetic | Usable; ShareAlike may apply to a released derived dataset (weights unclear): legal check before release. |
| TartanAir | CC BY 4.0 | 30 sim environments | Usable, synthetic. |
| BlendedMVS / BlendedMVG | CC BY 4.0 | 113 + 389 scenes; depth rendered from textured meshes; metric scale not stated | Usable; real-image/mesh-rendered depth, good mixed pool. |
| Replica | Replica Dataset Research Terms (non-commercial research, no institution requirement) | 18 scenes | Usable; small, synthetic-like. |
| DTU MVS | No licence text found ("freely available") | 124 scenes, lab, structured-light GT | Usable with caution; object-scale. |
| Tanks and Temples | Page contradictory: CC BY 4.0 headline but grant "sole purpose of non-commercial scientific research", no redistribution | laser scans; counts not read | Evaluation only, no redistribution. |
| ARKitScenes | Apple licence, non-commercial | 1661 scenes | Non-commercial; terms DATA.md not read. |
| 7-Scenes | Microsoft non-commercial | 7 | Small; GT from KinectFusion (not laser). |
| KITTI | CC BY-NC-SA 3.0, academic use | outdoor | Outdoor only. |
| Waymo | Non-commercial; institution-representation clause | outdoor | Ruled out for now. |
| ScanNet v2 | Unverified (host TLS error) | | Unknown. |
| Co3D | Code CC BY-NC 4.0; dataset licence not found; no metric GT claim found | | Not suitable (no metric GT). |
| MVS-Synth, MuSHRoom | Not fetched | | Unchecked. |

### Shortlist if ScanNet++ is out
Labelled training pool: **BlendedMVG/BlendedMVS (CC BY 4.0, 500 scenes)** as the main real-ish pool, **Hypersim** and **TartanAir** (synthetic, as Marigold uses synthetic data), Replica for small indoor tests. Evaluation: **ETH3D 13 scenes** (primary, with leave-one-scene-out), DTU as a second object-scale test. Unlabelled pool for the semi-supervised term: any posed RGB, including ETH3D's 12 test scenes (images only) and BlendedMVS images used without their depth. Real indoor laser-GT at ScanNet++ scale has no permissive replacement in this list.
