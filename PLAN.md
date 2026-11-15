# PLAN v2: calibrated, multi-view-consistent depth sampling with a diffusion model

Working title; repo name is a placeholder. Local private repo, no remote, nothing public.
Affiliation: Independent Researcher. Elsevier template, preprint first, no pay-to-publish venue.
Separate paper from `eroniki/sensorfault`: nothing here is merged into that method. Its design note
(`docs/design-3dgs-bayes.md`, main @ cf745d1) is background only; section 8 was read, not reused as method.
No employer IP, data or names.

Supersedes PLAN v1 (GMRF bias field, now dropped). No bias field, no correction term, no dependency on another depth network.

Marks: **[EST]** = back-of-envelope, not measured. **[UNVERIFIED]** = from a web search snippet or memory; check the primary source before it goes in the paper.

## 1. Contribution (one sentence)

A diffusion model that samples a per-view depth posterior from RGB, can be guided at sampling time by a 2DGS surface built from the *other* views, and whose sample spread is a **calibrated** covariance usable in the sensorfault leave-view-out fault test.

## 2. Model

- **Backbone:** latent image diffusion model, fine-tuned (not trained from scratch). Initialise from the *image* generator, not from Marigold or any depth model, to keep "independent of other networks". See section 7.
- **Target:** metric log-depth of view j, encoded by the backbone VAE (replicated to 3 channels, Marigold-style trick). Metric because ETH3D and ScanNet++ poses are metric. Condition on RGB latent (channel concat) plus a camera embedding (focal length / FoV), since metric depth is not recoverable from pixels alone. Decision pending: affine-invariant target plus separate scale head, if metric training proves unstable.
- **Two sampling modes**, kept strictly separate (this matters for the fault test, section 5):
  - **U (unguided):** RGB-only posterior p(D_j | I_j). This is the "measurement" with its own covariance Sigma_eps,j.
  - **G (guided):** same sampler plus guidance toward the leave-j-out 2DGS render (below). This is the consistent-depth product.
- **Guidance (diffusion posterior sampling style):** at step t, form x0-hat, decode to depth, compare with the render D_render^{-j} under that render's variance Sigma_surf^{-j}; add the gradient of the Gaussian log-likelihood to the score with a strength schedule. The render comes from a structure-only 2DGS fitted on the other views' depth and **never uses view j's RGB-derived depth** (hard rule; asserted in code by view-id masks).
- **Chicken-and-egg:** the 2DGS is built from other views' depth, which themselves need guidance. Plan: round 0 = U samples per view, fit 2DGS (gsplat/2DGS, depth + normal), round r>0 = G samples from round r-1 fields, refit. Leave-j-out field via the influence-function step (sensorfault design note sec. 2) or K-fold view blocks. 2 rounds first.
- **Output:** M samples per view -> mean, per-pixel variance, and low-rank-plus-diagonal covariance (rank <= M-1, shrinkage), so Sigma_r for the fault test is a stationary-kernel-free, sample-based matrix. Metric: samples are in log-depth; convert with delta method before whitening.

## 3. Training: semi-self-supervised

- **Supervised term (labelled views):** standard v/eps-prediction loss on GT depth latents. Truth sources: ETH3D laser scans (reprojected, masked where invalid) and ScanNet++ (LiDAR/laser depth) **after licence check** [UNVERIFIED, do not download until confirmed].
- **Self-supervised term (unlabelled views):** render-and-compare. Take x0-hat for view j at a random t, decode, compare with the stop-gradient leave-j-out 2DGS render of the same scene (depth, plus normal consistency), weighted by render variance. Photometric cross-view warping against the *RGB* of other views gives an independent signal not routed through the 2DGS.
- **Anti-collapse safeguards:** EMA teacher produces the 2DGS inputs (not the student); render supervision only where >= 2 other views support the pixel; supervised batches always present.
- **Label fraction:** control rho_scene = fraction of *training scenes* with GT, plus pixel density within labelled scenes (as delivered). Schedule:
  - Stage 1: rho = 100% on the labelled training pool, supervised only (the usable upper bound).
  - Stage 2: add the unlabelled term, ramping the unlabelled share of each batch 0 -> 50% over the first third, then fixed.
  - Study grid: rho_scene in {100, 30, 10, 3, 0}%. At 0% there is no metric anchor except posed metric cameras and photometric consistency; whether it bootstraps is a research question, reported honestly either way.
- **Leakage rule:** two protocols, never mixed. (a) *Inductive*: test scenes are never seen in training, labelled or not. (b) *Transductive / test-time*: the unlabelled term may use the test scene's RGB and poses but never its GT, and the tested view j is excluded from the render it is compared with. Report both.
- ETH3D has only ~13 scenes with public GT [UNVERIFIED count]: use scene-wise cross-validation for ETH3D evaluation and ScanNet++ (if licensed) as the main training pool.

## 4. Uncertainty and calibration

- Spread over M samples (start M = 10) gives the per-view covariance (section 2).
- Report calibration, not only accuracy, against GT on held-out scenes: NLL, PIT histogram and interval coverage (50/80/95%), regression ECE, CRPS, AUSE (sparsification), and the whitened-residual statistic T/(n-2) -> 1 on clean views, as in sensorfault.
- Ablate a per-scene temperature on Sigma (fit leave-view-out, as sensorfault sec. 9) vs no post-hoc scaling. A diffusion model's raw spread is not guaranteed calibrated; expect to need it, and say so.
- Expected failure: sample spread captures *model* (aleatoric-like) variability but under-represents bias common to all samples; report on dev GT.

## 5. Fault test and the key risk

Risk: a learned prior has no Gaussian guarantee and may **explain a real fault as plausible depth** (a "hallucinated consistency"). Under sensorfault's flat-prior analysis this is the absorption problem; here it is non-linear and has no closed-form detectability.

- **Test statistic uses mode U only.** Residual r_j = (unguided sample mean for view j) - (leave-j-out render), whitened by Sigma_eps,j (sample covariance) + Sigma_surf^{-j}. Mode G is the product and is *not* tested against its own guidance (that would be circular). Open decision: confirm with author that "sample spread" in the brief means U-mode spread.
- **Fault-injection design** (families mirror sensorfault: region, smooth warp, edge-aligned region; confirm against its configs):
  1. *Depth-space injection:* add the fault to view j's U-sample (tests the statistic only, a control). Compare power with the Gaussian baseline; this isolates the non-Gaussian null.
  2. *Image-space injection (tests the prior):* alter I_j so that the true geometry is unchanged but the image implies different depth: object paste/removal, mirror/glass-like regions, blur, exposure, a plausible inpainted patch; magnitude swept so implied depth shift is 0.25-4 sigma.
  3. *Geometry-space injection:* wrong pose or focal length for view j. Pose faults need the pose-hold rule from the sensorfault note.
  4. *Adversarial-plausible:* shift a region by delta along the surface normal while keeping image statistics natural (inpainting model, or a patch from another scene at matching scale), the case where a prior is most likely to absorb.
- **Metrics:** minimum detectable fault (MDF) in sigma at fixed false-alarm alpha, power curves, false-alarm rate on clean views, and the **absorption ratio** = fraction of the injected displacement reproduced by the U-sample mean. Absorption ratio ~ 1 and low power = the prior is hiding faults.
- **Null:** do not assume chi-squared. Calibrate the threshold empirically (conformal / permutation on clean held-out views), then check the Gaussian threshold as a comparison.
- **Mitigations to test if absorption is high:** (i) lower-capacity conditioning (drop camera embedding), (ii) lower guidance strength, (iii) tempered sampling, (iv) train with injected faults as negatives, (v) report as limitation with measured MDF.

## 6. Baselines and prior art

- **Baselines:** Marigold-style depth diffusion (single view, ensemble spread as uncertainty), same samples-per-view; AffineGS (arXiv 2611.07709: dense edge-aware affine disparity field, no uncertainty, no fault test; characterised from the sensorfault note sec. 8.5, **I have not read the paper itself**). Cheap controls: per-view aligned RMS and ensemble/MDE-confidence tests, as in sensorfault.
- **Prior art (shallow search 2026-11-14, titles/abstract-level only; must read before any novelty claim):**
  - Marigold (Ke et al.): single-view latent diffusion depth, ensembling; no multi-view guidance.
  - MVGD, Guizilini et al., CVPR 2025: multi-view geometric diffusion generating novel-view images and depth from posed images, scale-aware, no explicit 3D representation. **Closest work.** Likely difference: we guide with an explicit 2DGS render of the other views, never the tested view, and evaluate calibration and fault detection. Must verify what MVGD reports about uncertainty.
  - DepthSync (arXiv 2507.01603): diffusion guidance for scale- and geometry-consistent *video* depth; closest on the guidance mechanism. Check how its guidance is built (render vs depth sync) and whether it is leave-view-out.
  - ChronoDepth, DepthCrafter, RollingDepth: video-diffusion depth, temporal not multi-view-geometric consistency.
  - OracleGS (arXiv 2509.23258), UAR-Scenes (2503.15742): diffusion priors used to *improve 3DGS*, the opposite direction.
  - Pixel-Perfect Depth (arXiv 2601.05246): pixel-space diffusion depth; relevant to flying pixels at edges.
- **Where we differ (to be confirmed by reading):** (1) leave-view-out guidance from an explicit surface; (2) sample spread evaluated as calibrated covariance; (3) a fault-injection protocol for the learned prior; (4) semi-self-supervision with a controlled label fraction.

## 7. Backbone candidates and licences (all [UNVERIFIED]: web snippets or memory; read each model card/LICENSE)

| Candidate | Why | Licence as found | Use limits to flag |
|---|---|---|---|
| Stable Diffusion 2.x (865M UNet, VAE) | Marigold's base; small; fits fine-tuning budget | CreativeML OpenRAIL++-M (memory) | Use-based restrictions; derivatives must carry them |
| SDXL | Higher resolution | OpenRAIL++-M (snippet, "use-case restrictions") | Same; heavier (2.6B) |
| SD 3.5 Medium | DiT, strong base | Stability Community License (memory: free under a revenue threshold, registration) | Revenue cap; read terms for derivatives and preprints |
| FLUX.1 schnell | DiT, distilled | Apache-2.0 (snippets agree) | Distilled, few-step: guidance and calibration may behave differently |
| FLUX.1 dev | Strong | Non-commercial (snippets) | Non-commercial; avoid |
| PixArt / Qwen-Image family | DiT alternatives | not checked | check |

Recommended first backbone: **SD 2.x**, because it is the Marigold base (clean baseline comparison), fits a 24 GB card, and the licence is workable for a non-commercial preprint. Dependent models, as baselines only: Marigold v1-0 (Apache-2.0 on its card; v1-1 reported as OpenRAIL++-M by a third-party card, conflicting), Depth Anything V2 (README: Small Apache-2.0, Base/Large/Giant CC-BY-NC-4.0), Depth Pro (licence not found). GeoWizard, Lotus licences not confirmed.

## 8. Datasets

ETH3D (high-res multi-view, laser GT; evaluation-first). ScanNet++ (training pool and second benchmark; **licence/terms check before download**). No employer data. No data in git; only manifests and hashes.

## 9. Compute estimate [EST] (nothing rented, no large jobs run)

Assumptions: A100-80GB-equivalent hours; an H100 is roughly 2x faster. SD-2 UNet at ~512x768; DDIM/few-step sampling for evaluation. Marigold reports ~2.5 days on one RTX 4090 for its fine-tune (memory, [UNVERIFIED]), used only as an anchor.

| Stage | What | GPU-h low-high | Where |
|---|---|---|---|
| 0. Debug | Tiny subset, 256 px, LoRA or 8-bit Adam, toy guidance loop, unit tests | 20-40 | TITAN RTX (24 GB, Turing: fp16 only, no bf16) |
| 1. Supervised fine-tune | Labelled pool, metric log-depth + camera conditioning | 40-80 | rented, 1 GPU |
| 2a. 2DGS fits | ~300 scenes x 15-30 min x 2-3 refit rounds | 150-450 (mostly small-GPU) | rented or TITAN RTX |
| 2b. Semi-self-supervised training | Render-and-compare, EMA teacher, 2 rounds | 100-250 | rented |
| 3. Label-fraction grid | 5 values of rho_scene, shortened schedule | 150-400 | rented |
| 4. Sampling and calibration eval | ~1000 views x M=10 x 10-50 steps, U and G, per config | 5-40 per config; ~8 configs | rented / TITAN RTX |
| 5. Fault-injection study | 4 families x ~5 magnitudes x ~100 views x M samples | 20-60 | rented |
| 6. Baselines (Marigold, AffineGS) | Same views/samples | 20-50 | either |
| **Total** | | **~550-1350 A100-h** | |

Biggest uncertainty: the number of ScanNet++ scenes and 2DGS fit cost. Cheapest cut: drop to 100 scenes (halves 2a) and fewer rho values. Decision rule before renting: run stage 0 and a 1-scene stage 1 on the TITAN RTX; only rent if the unguided model beats a trivial baseline on that scene.

## 10. First step (no large jobs)

1. Read MVGD, DepthSync and the Marigold paper in full; update section 6 with verified differences. (No compute.)
2. Verify licences in section 7 and ScanNet++ terms; record in `docs/licences.md`.
3. Stage 0 on TITAN RTX: build the data interface for one ETH3D scene (GT reprojection, view-id masks) and a unit test that the guidance render for view j provably uses no data from view j.
4. Mode-U fault-test control (injection family 1) on synthetic samples: confirms the statistic and the empirical-null code before any model exists.

## 11. Open decisions for the author

- Is the test statistic built from U-mode samples (recommended) and G-mode the product?
- Metric log-depth with camera embedding, or affine-invariant plus scale head?
- ScanNet++ if licence allows; otherwise what second labelled source?
- Backbone: SD 2.x first (recommended)?

## 12. Housekeeping

Python/PyTorch, gsplat, 2DGS, diffusers. No latency targets. Elsevier template, preprint first.
