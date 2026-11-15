# Prior art, read in full (2026-11-14)

Sources read: Marigold arXiv 2312.02145v2 (HTML) + README + HF cards; MVGD arXiv 2501.18804v1 (HTML) + project page;
DepthSync arXiv 2507.01603v2 (HTML). Read by delegated reader agents from verbatim text, not memory.
Not read: CVPR camera-ready versions (MVGD v1 only), code (none found for MVGD or DepthSync), AffineGS body (abstract only;
its description comes from sensorfault design note sec. 8.5).

## Marigold (Ke et al., CVPR 2024)
- Fine-tunes Stable Diffusion v2 (v-objective, text off). Depth replicated to 3 channels through the frozen SD VAE; RGB latent
  concatenated; first U-Net layer widened (weights duplicated, halved). Only the U-Net is trained.
- **Affine-invariant** target: per-view 2%/98% percentile normalisation to [-1, 1] (Eq. 3). Scale/shift fitted only at evaluation.
- Trained on synthetic Hypersim (~54K samples/365 scenes) + Virtual KITTI 2 (~20K): 18K iterations, batch 32 (16x gradient
  accumulation), "approximately 2.5 days on a single Nvidia RTX 4090".
- Inference: DDIM, 10 runs with different noise, each member jointly aligned (scale+shift, Eq. 4) then **pixel-wise median**.
  Model cards: an "Uncertainty map" is produced only when ensemble size > 2. Paper reports std/max-min over 10 runs as a
  "consistency" number (App. B.3) and **never evaluates spread against error: no calibration result**.
- Single image only; no multi-view input, no guidance, no fault detection.

## MVGD (Guizilini et al., CVPR 2025; arXiv 2501.18804)
- Pixel-space RIN diffusion, **trained from scratch** (~6k A100-GPU-hours, 300k steps, 61M samples, 19 datasets, 256 px).
  Conditions on 2-5 posed views through learned image tokens + Plucker-style raymaps. No 3D representation, no render.
- Depth is scale-aware, **not metric and not affine-invariant**: scene-scale normalisation from conditioning camera translations
  (Sec. 3.2); "No test-time alignment was used".
- **No uncertainty output.** Only test-time ensembling over E = 5 samples (aggregation rule not stated), no spread reported.
- **Not leave-view-out for depth:** stereo/video depth "the target image is also used as conditioning" (Sec. 4.4).
- No sampling-time guidance. ETH3D not used. "Code and pre-trained models will be made available upon publication": release unconfirmed.
- Stated limits: no dynamics; target camera fixed at origin.
- **Difference for us:** (1) pretrained image backbone, fine-tune cost (~hundreds of GPU-h) instead of ~6k; (2) the tested view's data
  is excluded by construction (unguided mode uses only its own RGB; the render uses only other views); (3) explicit surface render
  with per-pixel variance as guidance; (4) calibration of the sample spread and a fault-injection protocol, none of which MVGD reports.
  MVGD is also a possible *baseline* if weights are released (its multi-view conditioning is a learned alternative to render guidance).

## DepthSync (arXiv 2507.01603)
- Training-free guidance on DepthCrafter (Stable Video Diffusion), video windows (90-110 frames), 5 Euler steps, guidance only in the
  last two steps ("noise in the predicted clean latent during early denoising stages can mislead the guidance process", App. D).
- **Geometry guidance (backward):** decode x0-hat to depth, back-project to point clouds, poses from CoTracker3 + PnP-RANSAC,
  losses: depth reprojection L1 against pairs within distance 3 in the window, 3D tracking L2, normal consistency, edge-aware
  smoothness; AdamW, 80 iterations per step. **Scale guidance (forward):** least-squares scale/shift of the current window to the
  previous window's latent as pseudo-label.
- **Not leave-one-out:** the target frame's own decoded depth is compared with neighbours' warped depth, and the scale pseudo-label
  is built from the window's own prediction.
- No uncertainty, no calibration, no fault detection. Cost: 961 s per 90 frames on one A100 (vs 9 s for DepthCrafter). No code link found.
- **Difference for us:** guide target is a fused 2DGS surface from other views with per-pixel render variance (a likelihood, not a
  pairwise warp loss); leave-j-out by construction; affine alignment fitted to the *render*; the sample spread is the product.
  Adopt from them: late-step-only guidance and the 5-step budget as a starting schedule; use their 961 s/90 frames as a cost anchor.

## Others (abstract level only, not read in full)
ChronoDepth, DepthCrafter, RollingDepth (video diffusion depth, temporal consistency); OracleGS (2509.23258), UAR-Scenes
(2503.15742) (diffusion priors improving 3DGS, opposite direction); Pixel-Perfect Depth (2601.05246) (pixel-space diffusion depth).
Novelty claims are limited to what the three read-in-full papers support; a final prior-art sweep is needed before submission.
