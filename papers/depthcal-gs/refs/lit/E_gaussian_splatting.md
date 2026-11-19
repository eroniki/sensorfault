# E: Gaussian splatting / 2DGS / depth priors / uncertainty

All 13 verified by fetching https://arxiv.org/abs/ID (title, authors, year, venue field checked). Venues stated only where the arXiv page states them.

## kerbl2023gaussians
- What: 3DGS: explicit anisotropic 3D Gaussians with tile-based differentiable rasterisation for real-time radiance fields.
- Relation: Foundation of the splatting machinery behind our reference.
- URL: https://arxiv.org/abs/2308.04079
- Caveats: arXiv page says ACM TOG 42(4), July 2023.

## huang2024twodgs
- What: 2DGS: flat 2D Gaussian surfels with ray-splat intersection, depth distortion and normal consistency regularisers for accurate surfaces.
- Relation: The surface model we fit from the other views and render into the held-out view.
- URL: https://arxiv.org/abs/2403.17888
- Caveats: arXiv page lists no venue (related DOI 10.1145/3641519.3657428, SIGGRAPH 2024 likely, unconfirmed); cited as arXiv.

## ye2024gsplat
- What: gsplat: open-source CUDA library for Gaussian splatting (incl. 2DGS, depth rendering).
- Relation: Our implementation backend.
- URL: https://arxiv.org/abs/2409.06765
- Caveats: Comment says JMLR MLOSS; volume/pages not verified, cited as arXiv.

## guedon2023sugar
- What: SuGaR: regularises Gaussians to align with the surface, then extracts mesh.
- Relation: Earlier splat-to-surface baseline for geometry from splats.
- URL: https://arxiv.org/abs/2311.12775
- Caveats: arXiv-only on page (CVPR 2024 not confirmed).

## yu2024gof
- What: GOF: opacity-field level-set mesh extraction from Gaussians for unbounded scenes.
- Relation: Alternative splat surface; we use rendered 2DGS depth instead.
- URL: https://arxiv.org/abs/2404.10772
- Caveats: arXiv-only on page (SIGGRAPH Asia 2024 not confirmed).

## chen2024pgsr
- What: PGSR: planar Gaussians, unbiased depth, multi-view geometric consistency for surfaces.
- Relation: Alternative reference; its multi-view consistency idea resembles our check.
- URL: https://arxiv.org/abs/2406.06521
- Caveats: Page lists DOI 10.1109/TVCG.2024.3494046 (TVCG) but no venue text; cited as arXiv.

## turkulainen2025dnsplatter
- What: DN-Splatter: adds monocular/sensor depth and normal priors to Gaussian splatting for indoor scenes and meshing.
- Relation: Closest depth-prior splatting work; same group/data as MuSHRoom; priors enter training, whereas we test depth samples against held-out splat.
- URL: https://arxiv.org/abs/2403.17822
- Caveats: Venue per arXiv comment: WACV 2025.

## zhu2023fsgs
- What: FSGS: few-shot Gaussian splatting with monocular-depth-guided densification.
- Relation: Shows depth priors in sparse-view splats; relevant to splat reliability when few views remain.
- URL: https://arxiv.org/abs/2312.00451
- Caveats: arXiv-only on page (ECCV 2024 not confirmed).

## li2024dngaussian
- What: DNGaussian: sparse-view Gaussians with global-local depth normalisation and hard/soft depth regularisation.
- Relation: Same: monocular depth regularises splats in sparse views.
- URL: https://arxiv.org/abs/2403.06912
- Caveats: CVPR 2024 per arXiv comment.

## chung2023depthreg
- What: Depth-regularised 3DGS for few-shot images using monocular depth with smoothness constraints.
- Relation: Same: depth-prior splatting under sparse views.
- URL: https://arxiv.org/abs/2311.13398
- Caveats: arXiv-only on page.

## zhang2024corgs
- What: CoR-GS: two Gaussian fields co-regularised via point and rendering disagreement for sparse views.
- Relation: Disagreement between fields as a reliability signal, akin to our per-pixel variance.
- URL: https://arxiv.org/abs/2405.12110
- Caveats: ECCV 2024 per arXiv comment.

## jiang2023fisherrf
- What: FisherRF: Fisher-information uncertainty for radiance fields and active view selection.
- Relation: Principled per-Gaussian uncertainty; our variance is rendered-reference variance, a different construction.
- URL: https://arxiv.org/abs/2311.17874
- Caveats: arXiv-only on page (ECCV 2024 not confirmed).

## ren2023mushroom
- What: MuSHRoom: indoor room dataset with multi-sensor captures for joint reconstruction and novel view synthesis.
- Relation: Benchmark data for evaluation.
- URL: https://arxiv.org/abs/2311.02778
- Caveats: arXiv-only on page.

## UNVERIFIED
(none; no splat-specific per-pixel-variance leave-view-out prior work found or checked)
