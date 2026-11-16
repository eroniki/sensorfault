# C: guided / multi-view diffusion (all entries below verified via arXiv metadata, export.arxiv.org API, matching https://arxiv.org/abs/ID)
Ours: explicit leave-view-out 2DGS render with variance, calibrated spread, a fault test; closed-form scale/shift alignment; guidance only at last denoising steps.

- depthsync2025 | DepthSync (Dong, Zhao, Xu, Shan, Zhang; 2025; ICCV 2025 per arXiv comment). Diffusion-guidance to synchronize scale/geometry across video depth windows. Differs: guides toward cross-window/warped consistency among predictions, not an independent leave-view-out 2DGS render with variance; no calibration or fault test. URL https://arxiv.org/abs/2507.01603. Caveat: author list VERIFIED (5 authors); ICCV only from arXiv comment, bib uses arXiv entry.
- guizilini2025mvgd | MVGD (Guizilini, Irshad, Chen, Shakhnarovich, Ambrus; 2025). Multi-view diffusion producing novel views and depth directly from posed images. Differs: conditions the generator on views (trained), ours is a single-image prior guided at inference by a 2DGS render. https://arxiv.org/abs/2501.18804. Caveat: no venue in arXiv metadata; kept arXiv-only.
- chung2023dps | DPS (ICLR 2023). Likelihood-gradient guidance through Tweedie estimate. Differs: ours is a DPS-style gradient with a rendered, variance-weighted target and post-alignment. https://arxiv.org/abs/2209.14687. Caveat: Mccann capitalisation as in arXiv.
- chung2022mcg | Manifold-constrained gradients (NeurIPS 2022). Projects measurement gradient onto the data manifold. Differs: precursor of the guidance step; no multi-view geometry. https://arxiv.org/abs/2206.00941.
- bansal2023universal | Universal Guidance. Training-free guidance with arbitrary loss/forward/backward guidance. Differs: generic; ours uses geometry-specific target with uncertainty. https://arxiv.org/abs/2302.07121. Caveat: arXiv-only per metadata.
- he2023mpgd | MPGD. Training-free guidance keeping samples on manifold via autoencoder/shortcut. Differs: generic guidance, not geometry. https://arxiv.org/abs/2311.16424. Caveat: arXiv metadata lists no venue.
- gao2024cat3d | CAT3D. Multi-view diffusion to create consistent views for 3D reconstruction. Differs: generates views for reconstruction; ours uses reconstruction to constrain depth sampling. https://arxiv.org/abs/2405.10314.
- wu2023reconfusion | ReconFusion. Diffusion prior regularizes sparse-view NeRF. Differs: diffusion prior supports reconstruction (opposite direction), no leave-view-out evaluation. https://arxiv.org/abs/2312.02981.
- shi2023mvdream | MVDream. Multi-view diffusion for 3D generation. Differs: object generation, no real posed-view guidance. https://arxiv.org/abs/2308.16512.
- tang2023mvdiffusion | MVDiffusion (NeurIPS 2023). Correspondence-aware multi-view image generation. Differs: image (not depth) consistency via learned attention. https://arxiv.org/abs/2307.01097. Caveat: NeurIPS 2023 per arXiv comment.
- viola2025marigolddc | Marigold-DC (ICCV 2025 per arXiv). Zero-shot depth completion by guiding Marigold with sparse depth. Differs: guidance from sparse measured depth; ours from a leave-view-out 2DGS render with variance, plus calibration. https://arxiv.org/abs/2412.13389.
- gregorek2024steeredmarigold | SteeredMarigold. Steers diffusion for completion of largely incomplete depth. Differs as above. https://arxiv.org/abs/2409.10202. Caveat: arXiv-only per metadata.

## UNVERIFIED (not in .bib)
- FreeDoM (Yu et al., ICCV 2023): not found by title query in arXiv API; not verified.
- DiffusionDepth (arXiv 2303.05021; Duan, Guo, Zhu) was verified by API but is not geometric-guidance; omitted as marginal.
