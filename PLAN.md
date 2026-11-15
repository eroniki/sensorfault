# PLAN: semi-self-supervised diffusion of a bias field over learned depth maps

Working title; repo name is a placeholder. Private repo, no remote yet.
Affiliation: Independent Researcher. Elsevier template, preprint first, no pay-to-publish venue.
Separate from `eroniki/sensorfault`: nothing here is imported into or merged with that method.

Status legend: **[CHECK]** = carried over from sensorfault design doc sec. 8, to be verified here, not settled.
Note: `docs/design-3dgs-bayes.md` is not on sensorfault `main` (only `main` exists); the numbers below come
from the author's summary and must be re-derived by the test in step 1.

## 1. Formulation

- Representation: 2DGS surfels (gsplat), posed RGB views, monocular depth (Depth Pro) per view.
- Per-view observation model, in disparity: `d~_i(u) = (1 + a_i(u)) d_i*(u) + c_i(u) + e`, `e ~ N(0, sigma^2)`,
  with `b_i = (a_i, c_i)` the bias field (AffineGS-style affine form, but per-pixel and with uncertainty).
- "Diffusion" = spatial propagation: prior on `b_i` is an edge-aware Gaussian Markov random field,
  precision `Q_i = lam * L_edge(I_i) + eps * I` (graph Laplacian with RGB-edge weights). The posterior mean is the
  solution of a sparse linear system, i.e. a steady-state diffusion of sparse evidence. Not a generative model.
- Self-supervision: evidence for `b_i` comes from multi-view consistency. Depth of view i is warped via the surfels into
  view j; disagreement is the residual that `b_i` must explain.
- Output per view: posterior mean `b_i_hat(u)` and variance/covariance (sparse-Cholesky or Hutchinson diagonal).
- **[CHECK]** Bias is view-dependent *per surfel*: fit on the other views, evaluate at view j's distance and angle.
- **[CHECK]** With a Gaussian prior, estimating the per-view field and marginalising it give the same test.
  Estimator: 5-fold cross-fitted mean, calibrated prior covariance, edge-awareness in the mean only.
- Test: same leave-view-out fault test as sensorfault. Metric is minimum detectable fault in units of sigma.

## 2. What "semi" means (open; options, recommendation first)

1. **Sparse metric anchors (recommended).** Fraction `rho` of views/pixels carry metric depth (ETH3D laser scans,
   ScanNet++ iPhone LiDAR); the rest are self-supervised only. Report accuracy and fault-test power vs `rho`
   in {0, 1, 5, 10, 100}%. Clean label-efficiency story. Rule: anchors never include the held-out view.
2. **SfM-track anchors.** Use sparse COLMAP points as the "labelled" pixels. No external sensor, but these are not
   independent of the pose estimate, so risk of circularity.
3. **Teacher-student pseudo-labels.** Fit fields on view subset A, use the posterior mean (weighted by its variance)
   as targets on B, iterate. Cheapest on labels, highest risk of confirmation bias; needs the uncertainty to be calibrated.
4. **Learned prior.** Meta-learn `lam` and edge weights from labelled scenes, apply label-free elsewhere.

Decision needed from author before step 3.

## 3. Datasets

- **ETH3D** (high-res multi-view, laser GT): primary, since it gives ground truth for the bias itself.
  Open question: size of Depth Pro's image-specific distortion here.
- **ScanNet++**: second dataset, iPhone + laser. Check licence/access terms before downloading; keep data out of git.
- No employer data, no named third parties.

## 4. Baseline

AffineGS (arXiv 2611.07709): dense edge-aware affine disparity field, no uncertainty, no fault test.
Compare on depth error after correction, calibration of uncertainty (NLL, coverage), and detectable fault size.

## 5. First experiment (small, no large jobs)

1. **Toy power test (CPU, minutes).** Recreate `bias_power.py` as `tests/test_bias_power.py`: 32x32 toy, square and warp
   faults. Expect to reproduce: no field 0.41 sigma (square) / 0.49 sigma (warp); per-view edge-aware field
   matched to fault 1.63 sigma. Must pass or the sec. 8 claims are revised. Also test the cross-fit vs marginal equivalence numerically.
2. **Depth Pro distortion on ETH3D (one GPU, a few scenes).** Per image, fit affine disparity (global, then per-pixel
   smooth) against laser-GT depth; report residual in sigma units. This sets whether a per-pixel field is needed
   at all versus a global affine. Run on the TITAN RTX only after approval.
3. Only then: GMRF field on one ETH3D scene with 2DGS, option 1 at `rho` = 0 and 100%.

## 6. Risks

- Per-view field absorbs real faults (the sensorfault concern): the test must be run with the field cross-fitted, never fit on the held-out view.
- If Depth Pro distortion on ETH3D is small relative to sigma, the paper's contribution shrinks to uncertainty/calibration; step 5.2 decides.
- Posterior covariance over dense fields is large; use sparse GMRF and diagonal approximations first.

## 7. Housekeeping

Private repo only, no push until approved. Python/PyTorch, gsplat, 2DGS. No latency targets yet.
