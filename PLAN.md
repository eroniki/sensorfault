# PLAN v4: calibrated, multi-view-consistent depth sampling with an affine-invariant diffusion model

Working title; repo name is a placeholder. Local private repo, no remote, nothing public.
Affiliation: Independent Researcher. Elsevier template, preprint first, no pay-to-publish venue.
Separate paper from `eroniki/sensorfault`; its design note (main @ cf745d1) is background only.
No employer IP, data or names. Supersedes v1 (GMRF bias field), v2 (metric log-depth) and v3.
Evidence files: `docs/prior-art.md`, `docs/licences.md`. Marks: **[EST]** estimate; **[UNVERIFIED]** not read from a primary source.

## 1. Contribution

A diffusion model that samples an **affine-invariant** per-view depth posterior from RGB, can be guided at sampling time by a 2DGS surface built from the other views (never view j), and whose sample spread is a **calibrated** covariance of view j's *local shape*, used in the sensorfault-style leave-view-out fault test.

## 2. Decisions recorded (author, 2026-11-14)

1. Test statistic uses **unguided** samples; **guided** sampling is the product.
2. Target is **affine-invariant depth**; no camera embedding (reason in 3.1).
3. **ScanNet++ stays in.** The author applies personally as an Independent Researcher (non-commercial, own time and resources). Application text is drafted in `docs/scannetpp-application-draft.md`; **nothing submitted by me**. Clause 7 and the eligibility rule are quoted verbatim in `docs/licences.md`; the open points are listed in section 11.
4. **Backbone: SD 2.x was retried and is still unreachable (HTTP 401), so SDXL with LoRA (or an 8-bit optimiser) on the TITAN RTX is the first backbone.** Its OpenRAIL++-M obligations for derivatives are in section 7 and `docs/licences.md`.
5. **ETH3D: use with citation, no redistribution.**

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
   - Blind spot, measured in the toy (4.1): a **constant offset** is absorbed exactly by t and is invisible. A pure scale error is *not* invisible, because the fitted scale also rescales the sample covariance; so the statistic also checks that the implied scale agrees with the sample spread. The robust fit assumes a minority of faulty pixels; the collapse point is to be measured (not stressed by the toy so far).
- "Barely circular": (s, t) use view j's samples and the render, but the render never uses view j, and two numbers cannot encode local shape. The toy measures the cost directly: power with the fit against power with the true (a, b) (4.1).

### 4.1 CPU toy findings (`scripts/toy_power.py`, `docs/toy-results.md`, tests in `tests/test_statistic.py`)

Synthetic Gaussian world, 16x16 grid, M = 10 samples, 200 clean and 80 trials per fault cell, one seed: **power estimates carry about +-0.05; this checks the statistic, it is not evidence about a real model.** The toy assumes a calibrated posterior (truth is one more draw), which a real diffusion model will not satisfy.
- **The two-parameter fit is nearly free for local faults.** With the oracle covariance the minimum detectable fault (80% power, units of the per-pixel total sd) is 0.88 with the fitted (s, t) against 0.89 with the true (a, b) for a 12% square, and 0.84 vs 0.89 for an edge-aligned region. With the covariance held fixed the fit lowers T by 2.4 on average (the two-parameter cost; asserted in tests).
- **Blind spot confirmed:** a constant offset has power 0.05 with the fit and 1.00 with the true (a, b). A fault proportional to the relief (a scale error) is **not** blind (power 1.00): the fitted scale also rescales the sample covariance, so a mismatch between the implied scale and the sample spread is flagged. I had called global scale/shift faults undetectable by design; only the constant shift is.
- **Smooth faults are hard:** the smooth warp needs about 2.1 against 0.9 for sharp-edged regions (oracle), qualitatively consistent with the sensorfault note, sec. 8 (not re-derived here).
- **The covariance estimate dominates power.** A stationary-kernel fit to the pooled sample autocovariance gave null T/(n-2) = 1.31 (the alignment shrinks the aligned deviations' variance and short-lag correlation) and MDF 2.2-2.4, with the warp undetectable up to 4. The chi2 threshold fails for it (27% false alarms at a nominal 5%), while the empirical-null threshold restores the level by construction. A rank-(M-1) sample covariance shrunk to its diagonal did better here (MDF 1.5, warp 1.2) but its behaviour rests on the toy's exact calibration and on faults lying outside the span of the sample deviations; it is not a recommendation.
- **Consequences:** (1) always threshold from the empirical clean-view null of the identical pipeline; (2) the covariance estimator (stationary kernel family, nugget, alignment-bias correction, temperature) is a main design question, to be settled on real samples; (3) the robust Huber fit and the plain fit were indistinguishable here, so the robust fit is justified only by the large-area case, which this toy does not stress (power stayed high to 60% area at kappa = 2.5): the minority-fault collapse point is **not** yet measured.

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

## 7. Backbone: SDXL (first), with its OpenRAIL++-M obligations

- **Why SDXL:** SD 2.x (Marigold's base) returns HTTP 401 on every repo tried and the Stability GitHub licence file is 404, so it cannot be verified or obtained from here; SDXL's licence was read in full and its repo is reachable (HTTP 200).
- **Training recipe (Marigold-style on SDXL):** frozen VAE, depth replicated to 3 channels, RGB latent concatenated, widened first conv (duplicated weights, halved), v-objective. **LoRA on the 2.6B U-Net (rank 32-64 [EST]) plus a fully trained widened input conv**, because full fine-tuning does not fit 24 GB: fp32 weights 10.4 GB + grads 10.4 GB + 8-bit Adam states about 5 GB [EST] already exceed it before activations. Frozen fp16 base weights (about 5 GB) + LoRA + gradient checkpointing at 512-768 px should fit [EST, to be measured in stage 0]. The TITAN RTX (Turing) has no bf16: fp16 autocast with a grad scaler, VAE kept in fp32 (SDXL's VAE is known to overflow in fp16, from memory, unverified).
- **Risk:** LoRA may fit a Marigold-style depth head less well than the full fine-tune Marigold used; stage 0 compares LoRA ranks and a "widened conv + LoRA" ablation on one scene before committing. Note the box this session ran on reports an RTX 2070 (8 GB), not the TITAN RTX; stage 0 sizing assumes the 24 GB card.
- **Obligations if SDXL-derived weights are ever shared** (read from LICENSE.md; details and quotes in `docs/licences.md`): pass Attachment A use-based restrictions on as an enforceable term in any licence governing the weights, with notice to recipients; give recipients a copy of the licence; mark modified files with a prominent change notice; keep attribution notices. Anything trained on SDXL-generated output or initialised from SDXL is a Derivative. Research use, a preprint and releasing code have no pass-through duty ("This provision does not apply to the use of Complementary Material").
- **Other candidates** (unchanged): SD 3.5 Medium (Community License read: free for research; "Powered by Stability AI" notice and a Notice file on distribution; may not be used to improve a foundational generative model); FLUX.1 schnell (Apache-2.0 per card; LICENSE not read; few-step distilled); FLUX.1 dev excluded (non-commercial). SD 2.x remains a possible comparison arm only if an authenticated check clears it.
- Marigold weights stay baseline-only (licence labelling conflicts across its README and cards).

## 8. Datasets (details in `docs/licences.md`)

- **ETH3D (decided: use with citation, no redistribution):** 13 training scenes with public ground truth (DSLR, 14-76 images each, laser scans and rendered depth maps), 12 test scenes without GT (used as unlabelled/transductive RGB). No licence text on the site. 13 scenes means leave-one-scene-out evaluation. Keep data and derived GT files out of git and any release.
- **ScanNet++ (in, pending the author's application):** 1006 scenes (856 train / 50 val; test scenes have no GT); laser-scan and iPhone-LiDAR depth. Terms: non-commercial only, no sharing of the data (clause 4), so derived per-scene pseudo-labels and depth renders must not be released either; GDPR deletion duty; access can be terminated. Until access is granted the plan must work without it (fallbacks below). **Not used before the token arrives.**
- **Fallback / complementary pool:** BlendedMVG/BlendedMVS (CC BY 4.0, 113 + 389 scenes, depth rendered from meshes, metric scale unconfirmed), Hypersim (CC BY-SA 3.0, 461 scenes, synthetic), TartanAir (CC BY 4.0, 30 environments), Replica (research terms, 18 scenes), DTU (no licence text, 124 scenes, object-scale). Matterport3D, Waymo, KITTI, ARKitScenes, 7-Scenes are non-commercial or institutional and not planned.
- Labelled pool for stage 1: BlendedMVG + Hypersim (+ TartanAir) now; add ScanNet++ if approved. Evaluation: ETH3D (primary), DTU, and ScanNet++ validation scenes if approved. Unlabelled pool: any posed RGB.

## 9. Prior art and where we differ (details in `docs/prior-art.md`)

- **Marigold:** single image, same affine-invariant recipe and median-of-aligned ensemble; spread never evaluated against error. We add multi-view guidance, calibration, and the test.
- **MVGD (CVPR 2025):** pixel-space diffusion trained from scratch (~6k A100-h), learned multi-view conditioning, scale-aware (not metric, not affine-invariant), **no uncertainty output, no guidance, no 3D render, and for depth the target image is also used as conditioning (not leave-view-out)**; code/weights unconfirmed. We fine-tune a pretrained backbone, exclude view j by construction, guide with an explicit surface carrying variance, calibrate, and inject faults.
- **DepthSync (2507.01603):** training-free guidance on DepthCrafter, pairwise warp/tracking losses between neighbouring frames of a video window, **not leave-one-out** (target's own decoded depth enters), no uncertainty or fault test, 961 s/90 frames on an A100, no code found. We use a fused leave-j-out surface with a likelihood weighted by render variance, and adopt their late-step-only guidance as the starting schedule.
- **AffineGS (2611.07709), read in full:** per-pixel affine map between monocular disparity (Depth-Anything-V2-Large) and the 3DGS render, re-estimated every iteration from view j's **own** disparity and **own** render (no leave-view-out), guided-filtered; no uncertainty or fault test; initialised from sensor depth; one RTX 3090, about 1.5 h per MuSHRoom scene. **Public code exists** (github.com/eldentse/AffineGS, Apache-2.0 per its sidebar; built on DN-Splatter/Nerfstudio), so it can be run as a baseline; its Table 1 is an in-sample oracle analysis. Details in `docs/prior-art.md`.
- Not read in full (abstract-level): ChronoDepth, DepthCrafter, OracleGS, UAR-Scenes, Pixel-Perfect Depth. A final sweep is required before any novelty claim.

## 10. Compute [EST] (nothing rented; no large jobs)

Anchors read from sources: Marigold fine-tune about 2.5 days on one RTX 4090 for SD 2 (865M U-Net, 18K iterations, batch 32); MVGD about 6k A100-h from scratch; DepthSync guidance about 961 s per 90 frames on an A100; AffineGS about 1.5 h per scene on an RTX 3090. SDXL's U-Net is about 3x larger than SD 2's and works best at higher resolution, so stage costs below are 2-3x the earlier SD 2 figures; LoRA lowers memory more than compute.

| Stage | A100-h low-high | Where |
|---|---|---|
| 0. Debug (256-512 px, LoRA vs LoRA+conv ablation, toy guidance, memory sizing) | 20-50 | TITAN RTX (24 GB, fp16 only) |
| 1. Supervised fine-tune (Marigold-scale 18K iterations, batch 32, SDXL LoRA, 512-768 px) | 100-250 | 1 rented GPU |
| 2a. 2DGS fits (pool of ~300 scenes x 15-30 min x 2-3 refit rounds; fewer if the pool is BlendedMVG-only) | 100-400 | small GPU |
| 2b. Semi-self-supervised training with render-and-compare | 200-500 | rented |
| 3. Label-fraction grid (5 values, shortened) | 300-800 | rented |
| 4. Sampling and calibration eval (M = 10, guided about 10-30 s per sample on SDXL [EST from DepthSync]) | 20-120 per config, ~8 configs | rented/TITAN |
| 5. Fault-injection study | 40-120 | rented |
| 6. Baselines (Marigold re-run; AffineGS, about 1.5 h/scene on a 3090) | 30-80 | either |
| **Total** | **~900-2500** | |

Cheapest cuts: SD 3.5 or a smaller backbone, fewer rho values, M = 5, 10-step sampler. Gate before renting: one ETH3D scene on the TITAN with the unguided model, the statistic on injected faults and the covariance estimator of 4.1; rent only if it beats a trivial baseline.

## 11. Open items needing the author

1. **ScanNet++ application** (drafted, not submitted by me): eligibility ("established researchers with a publicly verifiable research profile"; an unaffiliated applicant is not listed), the institutional-email request on the registration form, and **clause 7** (applies to a researcher "employed by a for-profit, commercial entity" regardless of whose time is used; signing represents authority to bind the employer). I recommend the written question to TUM in the draft before signing. Until the token arrives, no ScanNet++ results are planned.
2. SD 2.x authenticated check (optional comparison arm only).
3. Choice of covariance estimator and the alignment-bias correction (4.1), on real samples.
4. Whether a released model would be shared at all (SDXL derivative terms apply only then).

## 12. Status and next steps

Done (no compute beyond the CPU toy): prior art read (`docs/prior-art.md`, incl. AffineGS); licences and ScanNet++ terms quoted (`docs/licences.md`, `docs/scannetpp-application-draft.md`); leave-view-out interface and leak test (`src/dcal/surface.py`, `tests/test_no_leak.py`: a poisoned-view check, a perturb-invariance check, a sensitivity check and a mutation test that catches a leaky fit); statistic and toy (`src/dcal/stat.py`, `toy.py`, `scripts/toy_power.py`). 11 tests pass.
Next: (1) author decisions in section 11; (2) swap the toy surface for the real 2DGS surface behind the same `fit_surface/render` interface and re-run `check_leave_out`; (3) ETH3D data interface (GT reprojection, view masks, manifests/hashes only); (4) stage 0 on the TITAN RTX; (5) a toy with a miscalibrated posterior, to stress the empirical-null approach and the minority-fault collapse point.

Python/PyTorch, diffusers, gsplat, 2DGS (the CPU toy needs only numpy/scipy). No latency targets.
