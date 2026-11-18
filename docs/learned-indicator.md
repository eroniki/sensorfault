# Learned fault indicator for the view gate

A research plan for replacing the statistical decision of the view gate with a learned indicator
function, trained by self-supervised and unsupervised learning only.

## Problem

Depth maps from a monocular depth model fail regionally: on glass, water and texture-free surfaces a
connected region of a view can be wrong by many standard deviations. The view gate finds such regions by
leaving view `j` (and a block of `B` neighbouring frames of the same strip) out of the fused surfel map,
rendering the remaining views' surface into view `j` (the *cavity*), and comparing.

The current gate turns the cavity residual into a decision with a statistical test: whitening by the
residual covariance `Sigma_r`, a scene temperature `gamma`, a per-view heterogeneity allowance `omega`, an
energy statistic and a region scan with a Bonferroni threshold, and a Holm step-down across views and
rounds. That decision assumes Gaussian noise with one scale per scene and an exact surface model. It
breaks when either assumption fails: a noise level 30% above the calibrated one without a temperature
gives false alarms on every view, and thin structures that 2D Gaussian surfels cannot represent give
false alarms on clean views.

## Constraint

No fault labels are used for training or for model selection. Every training signal comes from the data
itself. Ground truth (exact synthetic depth, injected fault masks, laser depth) is used for evaluation
only, and models are selected by held-out likelihood, never by detection rate.

## Idea

Keep the cavity as the evidence and learn the decision. The cavity is itself the self-supervision: it
predicts view `j` from the other views, so a view cannot vouch for its own fault.

- **Output:** a per-pixel fault probability `zeta_j(u)` for the tested view `j`, pooled to a view score.
- **Inputs (feature stack):** cavity residual `r_j` (aligned by a per-view scale and offset, divided by the
  calibrated noise shape `sigma(range, texture, incidence)`), whitened residual, rendered depth sd
  `sqrt(V_{-j})`, accumulated opacity, untestable-pixel mask, range and incidence, RGB of view `j`, and the
  depth model's confidence where one exists.
- **Kept:** fusion of information blocks, leave-block-out subtraction and Newton step, support rules,
  per-view alignment. Whole-view affine error is absorbed by the alignment and is never a fault.
- **Replaced:** scene temperature, heterogeneity allowance, degrees-of-freedom model, energy statistic,
  region scan with Bonferroni, Holm step-down.
- **Threshold:** split-conformal on development scenes assumed clean at scene level, or a threshold drawn
  from the fitted inlier model's own null by sampling. Choosing between the two is part of the study.

## Formulations

| Option | What is learned | Role |
| --- | --- | --- |
| U1 Amortised contamination model | A net gives an outlier prior `pi(u)` and an inlier scale `s(u)`, fitted by maximising, over all views, `sum log[(1 - pi) N(r; 0, s^2 (sigma^2 + V)) + pi p_out(r)]`; `zeta` is the outlier posterior | Main method. With constant `pi` and `s = gamma` it reduces to a robust form of the statistical gate |
| U2 Geometry-to-appearance distillation | An image-only student predicts U1's `zeta` from RGB | Fault prior on untestable pixels, which no other view sees |
| U3 Density model | A normalising flow or autoencoder on residual patches, trained with a trimmed loss; `zeta` from negative log density | Alternative without a mixture assumption |
| U4 Self-training | A student trained on the statistical gate's own flags | Baseline |
| U5 Cross-model disagreement | Disagreement between several depth models as an input or pretext | Ablation; blind to errors all models share |

Design rules for U1:

- The scale head `s(u)` sees geometric inputs only (range, incidence, rendered sd), never RGB. Otherwise the
  net can learn that glass means high noise and explain a fault away. The prior head `pi(u)` may see RGB.
- `p_out` is fixed and normalised over a bounded residual range, so flagging a pixel costs likelihood unless
  its residual is large. A Beta prior keeps the mean of `pi` small, and a cap on the flagged fraction per
  view keeps a view from being written off whole.
- Architecture: a small U-Net on about eight channels, or frozen DINOv2 features of the RGB for the prior
  head.

## Data

| Data | Training input | Evaluation only |
| --- | --- | --- |
| MatrixCity aerial blocks with off-the-shelf depth models (Depth Pro, Metric3D v2, UniDepthV2, Depth Anything V2) | Cavity residual stacks of all views | Exact depth, giving true error maps |
| MatrixCity with injected faults (`sensorfault.faults.injectors`) | The same stacks | Injected masks, for a sweep over area and magnitude |
| Stanford bunny and relief scenes (`sensorfault.sim.bunny`) | Unit tests, thin-structure regime | Exact masks |
| UseGeo, ETH3D | Per-scene fitting is allowed, since no labels are needed | Laser depth |

Evaluation label rule: a pixel is faulty when its aligned error exceeds `3 sigma` inside a connected region
covering at least 2% of the view. Evaluation masks are stored apart from the training data. Splits go by
block or scene, never by view, because neighbouring frames of a strip repeat the same depth error.

## Use inside the reconstruction

1. **Hard:** the outer loop (train, freeze, fuse, test) is unchanged; each round drops the view with the
   highest pooled score above the threshold.
2. **Soft:** each round, the depth loss of pixel `u` in view `j` is weighted by `w_j(u) = 1 - zeta_j(u)`, so
   the good parts of a faulty view still contribute (soft EM). The indicator is refitted between rounds and
   frozen while the surface is fitted, so the surface fit cannot lower its loss by flagging everything.

## Experiments

| Step | Question | Depends on |
| --- | --- | --- |
| L0 | Feature stacks per view, evaluation masks in a separate store | cavity renders |
| L1 | U1 on injected faults: does the outlier posterior recover the withheld masks, match the statistical gate on clean Gaussian faults, and stay calm when the noise level is 30% off? | L0 |
| L2 | U1 on real depth-model errors against the statistical gate and depth-model confidence (go / no-go) | depth maps of all views from the four models |
| L3 | U2 on untestable pixels in a low-overlap scene | L2 |
| L4 | Hard versus soft use inside the 2DGS fit: held-out AbsRel and F-score | GPU surfel fit |
| L5 | UseGeo and ETH3D: trained on MatrixCity versus fitted on the test scene | real data |
| L6 | Ablations: U3, U4, U5, no cavity (residual against the fused surface), RGB in the scale head, no rendered sd, no conformal step | L4 |

If U1 does not beat the statistical gate and depth-model confidence on real errors at a matched false-alarm
rate (L2), the approach is reported as a negative result.

Stress regimes, where the statistical gate is known to fail: noise level off by 30% without a temperature,
the same error repeated on three neighbouring frames, thin structures and views at the edge of the
coverage, low overlap.

Baselines: the statistical gate, aligned RMS per view, depth-model confidence, robust depth loss
(Huber, Geman-McClure), RobustNeRF, SpotLessSplats, no gate, and an oracle gate with true masks as an upper
bound.

Metrics: pixel average precision and F1 against the evaluation masks; view detection rate at 5% per-view
false alarm; realised false-alarm rate on clean views against the target; held-out AbsRel and F-score of
the fused surface; transfer across depth models and from synthetic to real scenes.

## Risks

- **Identifiability.** Without labels, nothing forces a large residual to be called a fault rather than
  noise. The geometric-only scale head, the fixed bounded outlier density and the cavity are the defences;
  L1 measures them against withheld masks.
- **Shared errors.** An error repeated by every view that sees a region is invisible to any cross-view
  signal. Leave-block-out removes this only within the block length.
- **Label leakage through model selection.** Selecting by detection rate would make the method supervised
  in practice; the selection rule is fixed before L2.
- **The guarantee.** Conformal control is marginal and assumes exchangeable scenes.
- **Related work.** SpotLessSplats, NeRF On-the-go and WildGaussians learn outlier masks without labels
  from image features and training loss. The distinction is leave-block-out evidence, a geometry-only
  scale head and a calibrated false-alarm rate.
