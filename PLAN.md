# PLAN v3: calibrated, multi-view-consistent depth sampling with an affine-invariant diffusion model

Working title; repo name is a placeholder. Local private repo, no remote, nothing public.
Affiliation: Independent Researcher. Elsevier template, preprint first, no pay-to-publish venue.
Separate paper from `eroniki/sensorfault`; its design note (main @ cf745d1) is background only.
No employer IP, data or names. Supersedes v1 (GMRF bias field) and v2 (metric log-depth).
Evidence files: `docs/prior-art.md`, `docs/licences.md`. Marks: **[EST]** estimate; **[UNVERIFIED]** not read from a primary source.

## 1. Contribution

A diffusion model that samples an **affine-invariant** per-view depth posterior from RGB, can be guided at sampling time by a 2DGS surface built from the other views (never view j), and whose sample spread is a **calibrated** covariance of view j's *local shape*, used in the sensorfault-style leave-view-out fault test.

## 2. Decisions recorded (author, 2026-11-14)

1. Test statistic uses **unguided** samples; **guided** sampling is the product.
2. Target is **affine-invariant depth**, not metric. Reason: a single RGB image leaves global scale nearly free, so metric samples spread along one near-rank-one scale direction and the test loses power on local geometry, where faults live. Camera embedding dropped (see 3.1).
3. If ScanNet++ is out, use a shortlist (section 8). Status: **effectively out**, see 8 and the flag in section 11.
4. SD 2.x first, **provided its licence checks out**. Status: **not verified** (HF returns 401); see 7 and section 11.

## 3. Model

### 3.1 Representation
- Latent diffusion, Marigold recipe: frozen VAE, depth replicated to 3 channels, RGB latent concatenated, widened first layer, only the U-Net trained, v-objective. Initialise from the **image** generator, not a depth model.
- Target: per-view percentile-normalised depth (2%/98% to [-1, 1], Marigold Eq. 3). Depth, not disparity, because the 2DGS render is depth and the test fits scale/shift in depth space.
- **No camera embedding.** Affine-invariance removes the need for metric scale, and Marigold works without intrinsics. Known cost: unknown focal length leaves a residual shape ambiguity that partly leaks into the fitted shift; we log it as a measured limitation on ETH3D, and keep intrinsics only for projection/render geometry outside the network.
- Honest novelty note: affine-invariant fine-tuned latent diffusion depth with ensembling **is Marigold**. Our contribution is everything after it (guidance, calibration, fault test, semi-supervision).

### 3.2 Unguided mode U (measurement)
- M samples (start M = 10) from RGB of view j only. Align the samples to each other (Marigold's joint scale/shift, Eq. 4), then take the pixel-wise mean m and the **post-alignment sample covariance** Sigma_U (low-rank + diagonal, shrinkage). Removing the per-sample affine wobble is the point: what remains is local-shape variability.
- Rank budget: the aligned samples have at most M-3 informative directions; a diagonal/stationary-kernel part is needed for the rest (decision after step 5.3).

### 3.3 Guided mode G (product)
- DPS-style guidance at late steps only (start: last 2 of ~5-10 steps, following DepthSync's finding that early x0-hat is too noisy to guide).
- Guide target: depth render D_r^{-j} and per-pixel variance Sigma_r^{-j} from a structure-only 2DGS fitted on the other views' depth. Loss at step t: `min_{s,t} || s*dec(x0hat) + t - D_r^{-j} ||^2_{Sigma_r^{-1}}` over pixels with >= 2 supporting views. (s,t) solved in closed form per step (variable projection), so the guided sample inherits the render's scale/shift.
- Hard rule: nothing derived from view j's RGB enters D_r^{-j}; enforced by view-id masks and a unit test.
- Chicken-and-egg: round 0 = U samples -> 2DGS; round r>0 = G samples -> refit. Leave-j-out field by influence-function step (sensorfault note sec. 2) or K-fold view blocks. 2 rounds first.

## 4. The test statistic (unguided samples, two fitted parameters)

For view j with unguided mean m (affine-aligned samples) and leave-j-out render D_r^{-j} with variance Sigma_r^{-j}:

1. **Fit only a per-view scale and shift** of the U samples to the render: `(s, t) = argmin || s*m + t*1 - D_r^{-j} ||^2_{W}`, robust (Huber/IRLS) so that a fault covering a minority of pixels does not drag the fit, with W = (s^2 Sigma_U + Sigma_r^{-j})^{-1} (one fixed-point iteration).
2. **Residual** `r = s*m + t*1 - D_r^{-j}`. Test the local shape: `T = r^T (s^2 Sigma_U + Sigma_r^{-j})^{-1} r`.
3. **Account for the two parameters:**
   - Degrees of freedom: reference `chi2` with `n_eff - 2` (the same projection as the (a_i, b_i) step in the sensorfault note), where n_eff comes from the correlation of the whitened residual.
   - **Threshold calibrated empirically with the identical pipeline**: for clean held-out views run the same fit, residual and whitening, and set the threshold from the empirical null (conformal / permutation). The fit's cost is then inside the null by construction, rather than assumed from `chi2`. Report both thresholds.
   - Known blind spot, stated up front: any fault in span{m, 1} (global scale/shift of the whole view) is absorbed by (s, t) and is **undetectable by design**; only local-shape faults are tested. The robust fit also requires the minority assumption (fault < half the supported pixels); beyond it the fit absorbs the fault, and we report the fraction at which power collapses.
- "Barely circular": (s, t) use view j's samples and the render, but the render never uses view j, and two numbers cannot encode local shape. Check empirically: power of (a) (s,t) fitted on the whole view vs (b) (s,t) fitted on a held-out half of pixels, to bound the leakage.

## 5. Training: semi-self-supervised

- **Labelled views:** standard v-loss on percentile-normalised GT depth.
- **Unlabelled views:** scale-and-shift-invariant render-and-compare: `min_{s,t} || s*dec(x0hat) + t - D_r^{-j} ||^2` against the stop-gradient leave-j-out render, variance-weighted, >= 2 supporting views; plus photometric warping to other views' RGB (independent of the 2DGS). The invariance means the render never imposes a scale.
- Safeguards: EMA teacher builds the 2DGS; labelled data always in each batch; render supervision only where supported.
- **Label fraction:** rho_scene = fraction of training scenes with GT. Stage 1 supervised (rho = 100% of the labelled pool); stage 2 ramps the unlabelled share 0 -> 50% over the first third, then constant. Grid {100, 30, 10, 3, 0}%. At 0% there is no metric anchor but none is needed now (affine-invariant), so the 0% arm is a meaningful test of render-and-compare alone.
- Protocols never mixed: **inductive** (test scenes unseen) and **transductive** (test-scene RGB and poses used by the unlabelled term, never its GT, and never view j in its own render).

## 6. Calibration and the fault-injection test

- Calibration metrics, in aligned (affine-fixed) space against GT after a per-view GT-fit of (s,t): NLL, PIT, interval coverage, regression ECE, CRPS, AUSE, and clean-view T/(n_eff - 2) -> 1. Marigold reports no spread-vs-error evaluation, so there is no prior number to beat; the baseline is Marigold's own M-sample spread.
- Optional per-scene temperature on Sigma_U (leave-view-out, as sensorfault sec. 9), ablated.
- **Fault-injection families** (mirror sensorfault: region, smooth warp, edge-aligned region; confirm against its configs):
  1. Depth-space on the U sample (control for the statistic and the empirical null; includes the (s,t) cost).
  2. Image-space on I_j (tests the prior): paste/removal, glass-like or mirror regions, blur, exposure, inpainted plausible patch; sweep implied depth shift 0.25-4 sigma.
  3. Pose/focal error on view j (pose-hold rule from the sensorfault note).
  4. Adversarial-plausible: normal-direction shift with natural image statistics.
- Metrics: minimum detectable fault (sigma) at fixed alpha, power curves, clean false-alarm rate, **absorption ratio** (fraction of injected displacement reproduced by the U mean), and power vs fault area fraction (the robust-fit collapse point).
- Mitigations if absorption is high: lower guidance (n/a to the test), tempered sampling, train with injected faults as negatives, fewer capacity-heavy conditioning, or report as a measured limitation.

## 7. Backbones (details in `docs/licences.md`)

| Candidate | Status |
|---|---|
| **SD 2.x** | **Licence NOT verified**: HF pages and API return 401, the Stability GitHub LICENSE-MODEL is 404. Needs an authenticated check. The only evidence is Marigold's v1-1 card labelling the SD2 licence "CreativeML Open RAIL++-M". Not cleared. |
| **SDXL base 1.0** | Licence read: CreativeML Open RAIL++-M. Fine-tuning allowed; restrictions and the licence must pass to derivatives. 2.6B U-Net: needs LoRA/8-bit optimiser on the 24 GB TITAN. Candidate fallback. |
| SD 3.5 Medium | Stability Community License read: free for research; "Powered by Stability AI" and a Notice file if distributing; may not be used to improve a foundational generative model. MMDiT, different fine-tune recipe from Marigold. |
| FLUX.1 schnell | Apache-2.0 per card (LICENSE not read); few-step distilled. FLUX.1 dev: non-commercial, avoid. |

Marigold weights are SD2-derived depth models: baseline only. Recommendation: confirm SD 2.x access this week; if unavailable or its terms unacceptable, take **SDXL** as the first backbone (licence verified) and keep SD 2.x for a comparison arm only if cleared.

## 8. Datasets (details in `docs/licences.md`)

- **ETH3D:** **13 training scenes with public ground truth** (DSLR, 14-76 images each, laser scans and rendered depth maps), 12 test scenes without GT (use as unlabelled/transductive RGB). No licence text on the site: cite, do not redistribute, ask the maintainers before releasing derived data. 13 scenes means leave-one-scene-out evaluation.
- **ScanNet++: effectively ruled out** (flag in 11). Terms read: non-commercial; handwritten requester + PI signatures; independent applicants must be established researchers with a verifiable research profile; clause 7 binds a for-profit employer; no sharing of data.
- **Shortlist:** BlendedMVG/BlendedMVS (CC BY 4.0, 113 + 389 scenes, depth rendered from meshes, metric scale unconfirmed), Hypersim (CC BY-SA 3.0, 461 scenes, synthetic), TartanAir (CC BY 4.0, 30 environments), Replica (research terms, 18 scenes), DTU (no licence text, 124 scenes, object-scale). Matterport3D, Waymo, KITTI, ARKitScenes, 7-Scenes are non-commercial/institutional and not planned. Real indoor laser-GT at ScanNet++ scale has no permissive replacement here.
- Labelled pool for stage 1: BlendedMVG + Hypersim (+ TartanAir); evaluation: ETH3D (primary), DTU (secondary). Unlabelled pool: any posed RGB.

## 9. Prior art and where we differ (details in `docs/prior-art.md`)

- **Marigold:** single image, same affine-invariant recipe and median-of-aligned ensemble; spread never evaluated against error. We add multi-view guidance, calibration, and the test.
- **MVGD (CVPR 2025):** pixel-space diffusion trained from scratch (~6k A100-h), learned multi-view conditioning, scale-aware (not metric, not affine-invariant), **no uncertainty output, no guidance, no 3D render, and for depth the target image is also used as conditioning (not leave-view-out)**; code/weights unconfirmed. We fine-tune a pretrained backbone, exclude view j by construction, guide with an explicit surface carrying variance, calibrate, and inject faults.
- **DepthSync (2507.01603):** training-free guidance on DepthCrafter, pairwise warp/tracking losses between neighbouring frames of a video window, **not leave-one-out** (target's own decoded depth enters), no uncertainty or fault test, 961 s/90 frames on an A100, no code found. We use a fused leave-j-out surface with a likelihood weighted by render variance, and adopt their late-step-only guidance as the starting schedule.
- **AffineGS (2611.07709):** dense edge-aware affine disparity rectification of a depth network's output, no uncertainty/fault test; not read beyond the abstract. No code link found; re-implement if needed.
- Not read in full (abstract-level): ChronoDepth, DepthCrafter, OracleGS, UAR-Scenes, Pixel-Perfect Depth. A final sweep is required before any novelty claim.

## 10. Compute [EST] (nothing rented; no large jobs)

Anchors, now read from the sources: Marigold fine-tune ~2.5 days on one RTX 4090 (18K iterations, batch 32); MVGD ~6k A100-h from scratch (we are far below); DepthSync guidance ~961 s per 90 frames on an A100.

| Stage | A100-h low-high | Where |
|---|---|---|
| 0. Debug (256 px, LoRA/8-bit Adam, toy guidance, unit tests) | 20-40 | TITAN RTX (24 GB, fp16 only) |
| 1. Supervised fine-tune (Marigold-scale, 18K it, batch 32) | 40-80 | 1 rented GPU |
| 2a. 2DGS fits (labelled+unlabelled pool, ~300 scenes x 15-30 min x 2-3 rounds); lower if the pool is BlendedMVG subset | 100-400 | small GPU |
| 2b. Semi-self-supervised training with render-and-compare | 100-250 | rented |
| 3. Label-fraction grid (5 values, shortened) | 150-400 | rented |
| 4. Sampling and calibration eval (M=10, guided ~5-15 s/sample [EST from DepthSync cost]) | 10-60 per config, ~8 configs | rented/TITAN |
| 5. Fault-injection study | 20-60 | rented |
| 6. Baselines (Marigold re-run, AffineGS re-implementation) | 20-60 | either |
| **Total** | **~500-1300** | |

Gate before renting: 1 ETH3D scene on the TITAN with the unguided model plus the statistic (section 4) on synthetic faults; rent only if it beats a trivial baseline.

## 11. Open items needing the author

1. **SD 2.x access:** unverifiable from here (401). Check from a logged-in HF session; else SDXL.
2. **ScanNet++ clause 7 and eligibility:** as read, the terms bind a for-profit employer and require a verifiable research profile and a PI/supervisor signature. Do not apply unless the author decides this fits the "no employer" rule.
3. ETH3D has no stated licence: acceptable to use for research with citation and no redistribution?
4. MVGD/DepthSync/AffineGS code: none found; the AffineGS project page was not read.

## 12. Next steps (no compute)

1. Author resolves items 11.1-11.3.
2. Read the AffineGS body and project page; update section 9.
3. Write the unit test "render for view j uses no data from view j" and a CPU toy of the statistic with (s,t) fitting and the empirical null (synthetic Gaussian samples + injected faults) to check the two-parameter cost and the robust-fit collapse point, before any model exists.
4. ETH3D data interface (GT reprojection, view masks), manifests and hashes only; no data in git.

Python/PyTorch, diffusers, gsplat, 2DGS. No latency targets.
