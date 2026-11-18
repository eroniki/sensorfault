# Experiment Design: Depth and Image Messages on the Same Gaussian Primitives

This note designs the experiments for a fault indicator in which every 2D Gaussian surfel
receives two kinds of message: a depth message from each frame's monocular depth map, and an
image message from each frame's pixels through the differentiable renderer. The fault test is
the tension between the two families, per primitive or per surface entity. A light head on
frozen features carries the verdict to surface where the image message is silent. One monocular
depth model is used per frame; all backbones are frozen. The note first states what has to be
true for this to work, as quantities that can be measured, corrects the parts of the conception
that the code and the literature contradict, and gives a staged cheap test. Notation follows the
main document (`docs/learned-indicator.md`, Section 7); new symbols are in Section 0.

Status of every number: *(repo)* is quoted from the repository; *(checked)* was read on a
paper's or project's own page while writing; *(recalled)* was not; everything else is a design
choice or an estimate. Nothing has been measured; every statement about what an experiment will
show is a prediction.

## 0. Setting, symbols, and what exists

| Symbol | Meaning |
| --- | --- |
| $\mathbf q_k = (\nu, \tau_1, \tau_2)$ | placement of surfel $k$: offset along its normal and two tilts |
| $\mathbf a_k$ | appearance of surfel $k$: colour or feature coefficients, opacity |
| $\mathbf g^D, \Lambda^D$; $\mathbf g^I, \Lambda^I$ | gradient and Gauss-Newton information of the depth loss and of the image loss with respect to placement |
| $\mathbf G_m$ | entity direction: the map from three entity parameters (offset, two tilts of the group) to the placements of its surfels |
| $\mathbf r_m$, $T_m$ | tension gap of entity $m$ and its test statistic |
| $\varsigma$ | image share of the information, $\Lambda^I (\Lambda^I + \Lambda^D)^{-1}$ |
| $\rho_{\mathrm{cap}}$ | capture range: largest placement error, in pixels of parallax, at which the image gradient has the right sign |
| $\chi_{\mathrm{abs}}$ | absorbed fraction: share of an injected tension that disappears when appearance is re-optimised |
| $\psi$ | honesty ratio: nominal image information divided by the information realised under injection |

**What exists** *(repo)*.

- `src/sensorfault/gs/fit.py`: a depth-only 2DGS fit with placeholder colour. Its own compositor
  renders ray-surfel intersection depth; the stock gsplat depth blends per-surfel centre depth
  and does not respond to tilt. Scales are frozen, opacity is frozen after a warm-up.
  Information blocks ($10 \times 10$ per surfel and view) come from a closed-form Jacobian in a
  dense numpy implementation with memory proportional to the number of surfels times a chunk.
- `src/sensorfault/gs/gate.py`: the leave-block-out consensus test on those blocks. It has run on
  96 by 54 pixel frames and 81 152 surfels, with synthetic depth noise, never with a real depth
  model.
- `experiments/rgb_train_2dgs.py`: a separate photometric 2DGS trainer (spherical harmonics, free
  opacity, default densification) on block 3 at 960 by 540: 1.5 million Gaussians, 31 minutes,
  3.2 GB, test PSNR 23.1 dB. Its surface is within 1 m of truth on 81% of land cells and 5% of
  water cells; on water it lies a median 5.7 m *above* the true plane, with a 95th percentile of
  48 m.
- A depth cache with four views; MatrixCity block 3 with exact depth (185 frames in strips, 46%
  water); no segmenter and no feature encoder in the repository. gsplat is not installed in the
  environment this note was written in; its interface was read from the documentation.

**What the rasteriser offers** *(checked, gsplat documentation)*. `rasterization_2dgs` accepts
colours with an arbitrary number of channels (rendered in chunks of 32), so features can be
splatted; returns rendered normals, normals from depth, a distortion map and median depth;
exposes absolute gradients of projected means. It does not document per-pixel compositing
weights as an output. It is differentiable in reverse mode only, as far as the documentation
shows; forward-mode products are not mentioned.

**Three facts that shape the design.**

1. *The two halves live at different resolutions.* The depth gate works with tens of thousands
   of surfels; the image message needs surfels near pixel scale, of the order of a million. The
   dense information code cannot be used at that size. Information has to come from the
   renderer's backward pass.
2. *A single surfel has one colour.* Moving one surfel along its normal changes the image only
   where its footprint edge meets a different colour. Texture is carried by many surfels
   together. The image message about placement is therefore a property of a *group moving
   coherently*, and per-surfel diagonal blocks understate it and misattribute it. The entity is
   the natural unit of the test for this reason as well as for power.
3. *On the one surface where the existing photometric model fails, it does not abstain.* On
   water it builds geometry tens of metres off. "Curvature vanishes on blank surface, so
   abstention is automatic" is contradicted by the repository's own photometric reconstruction
   unless appearance is restricted; Section 1.3 makes that a measurement.

## 1. The conception as measurable quantities

### 1.1 The image message of an entity

For rendered colour or feature $C(u) = \sum_k w_k(u)\, \mathbf c_k$ and residual
$\mathbf e(u) = C(u) - I(u)$ with pixel noise $\sigma_c$, the message of frame $i$ about the
three parameters of entity $m$ is

```math
\mathbf g^I_{mi} = \sum_{u} \frac{\mathbf J_{mi}(u)^\top \mathbf e_i(u)}{\sigma_c^2}, \qquad
\Lambda^I_{mi} = \sum_{u} \frac{\mathbf J_{mi}(u)^\top \mathbf J_{mi}(u)}{\sigma_c^2}, \qquad
\mathbf J_{mi}(u) = \sum_{k \in m} \frac{\partial C_i(u)}{\partial \mathbf q_k}\, \mathbf G_{m,k} .
```

$\mathbf g^I_{mi}$ costs one backward pass per frame for all entities at once (the per-surfel
gradient contracted with $\mathbf G_m$). $\Lambda^I_{mi}$ does not come from one backward pass.
Three estimators are compared in the experiments:

- *finite difference of the gradient* under a group displacement: one extra render and backward
  pass per direction, with non-adjacent entities displaced together so that a few passes cover
  all entities;
- *random probing*: back-propagating random sign patterns on the residual image gives
  $\mathbf J^\top \mathbf v$ with expectation $\mathbf J^\top \mathbf J$ of its outer product;
- *per-frame scatter*: $\sum_i \mathbf g^I_{mi} \mathbf g^{I\top}_{mi}$, which needs no extra pass
  and estimates the variance of the gradient directly, whatever the residual distribution.

The third matters because $\sigma_c$ is not known and photometric residuals are neither Gaussian
nor independent between pixels. A nominal $\Lambda^I$ is a claim; the experiments check it by
injection (Section 1.4).

### 1.2 The tension statistic, and how circularity is avoided

At a joint fit the gradients cancel, $\mathbf g^D_m + \mathbf g^I_m = 0$; with
$\mathbf g = \mathbf g^I_m$ the gap between what the two families would each choose, and its test,
are

```math
\mathbf r_m = \bigl[(\Lambda^D_m)^{-1} + (\Lambda^I_m)^{-1}\bigr]\, \mathbf g, \qquad
T_m = \mathbf g^\top \bigl[(\Lambda^D_m)^{-1} + (\Lambda^I_m)^{-1}\bigr]\, \mathbf g \ \sim\ \chi^2_3 \ \text{ under the linear Gaussian null} .
```

The joint fit is the wrong place to evaluate it, for two reasons that are both testable. The
tested messages helped place the entity, so the statistic is deflated by their leverage. And at
a compromise placement the appearance has been fitted to a slightly wrong surface, which blurs
the texture and weakens the image gradient.

The design therefore uses a **score test on held-out frames at the depth-placed surface**. The
entity's placement is fitted from depth messages only; appearance is assigned under a restricted
regime (Section 1.3) from a set of fitting frames; the image gradient is evaluated on judging
frames that took no part in either. Under the null,

```math
\operatorname{Var}\bigl(\mathbf g^I_m(\hat{\mathbf q}^D)\bigr) = \Lambda^I_m + \Lambda^I_m (\Lambda^D_m)^{-1} \Lambda^I_m, \qquad
T^{\mathrm S}_m = \mathbf g^{I\top}_m \bigl[\Lambda^I_m + \Lambda^I_m (\Lambda^D_m)^{-1} \Lambda^I_m\bigr]^{-1} \mathbf g^I_m .
```

No joint optimisation is needed to detect; the joint fit is needed only to repair. The
experiments compare three versions: in-sample at the joint fit, held-out frames at the joint fit,
and the score test, and report the size of the optimism of the first two (B2).

A fault that every frame repeats shows here as it should: all depth messages agree, so
$\Lambda^D$ is large and the depth-placed surface is confidently wrong; the image gradient on any
frame is then large relative to its variance. Depth consensus sees nothing because it compares
depth messages with each other.

### 1.3 Appearance can absorb geometry

A misplaced surface can still reproduce the images if appearance is free: view-dependent colour
shifts texture with viewpoint, opacity fades a wrong surface, new primitives fill in. The
existing photometric trainer has all three freedoms. Four regimes are compared:

| Regime | Appearance | What it can absorb |
| --- | --- | --- |
| R0 | spherical harmonics of degree 3, free opacity, densification (the existing trainer) | almost any placement error |
| R1 | one colour per surfel, opacity and scales frozen, no densification; colours fitted on the fitting frames | blur only |
| R2 | colours taken from a single reference frame per entity | nothing; the render of another frame is that frame's content displaced by the placement error, with occlusion by the rasteriser. The hand-warped stabilised clip is this regime for one depth map |
| R3 | frozen dense features splatted in place of colour (R1 or R2 sourcing), loss in feature space | as its sourcing; tolerant of exposure and of fine misregistration |

The **absorbed fraction** makes this a number. Displace an entity by a known amount, hold the
geometry, re-optimise appearance for a fixed budget on the fitting frames, and measure the
tension on the judging frames before and after:
$\chi_{\mathrm{abs}} = 1 - T_{\text{after}} / T_{\text{before}}$. A regime is usable for the test
if $\chi_{\mathrm{abs}}$ is small while healthy surface is still rendered acceptably.

### 1.4 Calibration by injection, capture range, honesty

"Move this group of Gaussians by a known offset and tilt and re-render every view" is the
calibration experiment. For a ladder of displacements $\boldsymbol\delta$ of entity $m$:

- the **exchange identity**: the change of the image gradient should equal
  $\Lambda^I_m \boldsymbol\delta$ and the change of the loss
  $\tfrac12 \boldsymbol\delta^\top \Lambda^I_m \boldsymbol\delta - \mathbf g^\top \boldsymbol\delta$;
- the **honesty ratio** $\psi$: nominal over realised information, per surface class. On blank
  or glittering surface a fitted model can claim information that an actual displacement does
  not confirm;
- the **capture range** $\rho_{\mathrm{cap}}$: a photometric loss is not convex in placement.
  Beyond a displacement of about the texture scale the gradient saturates or changes sign. A
  monocular fault of 3-5% of depth at 216 px of parallax between adjacent survey frames is a
  shift of 6 to 11 px at full resolution, several surfel footprints. Whether a gradient at the
  depth-placed surface points the right way at that distance is the first thing to learn. The
  ladder is repeated on blurred and downsampled frames and on feature maps (R3), which widen the
  basin at the price of precision; where the gradient fails, a sweep of renders over the offset
  replaces it, and the test is then a plane sweep computed by a rasteriser.

Injection also supplies unlimited labels at no cost for one learned component (Section 1.6).

### 1.5 Entities and features on primitives

Per-frame masks from a segmenter in image mode are lifted by voting: rendering mask channels as
"colours" and back-propagating their inner product with the masks gives, for each surfel, its
compositing-weighted vote for each mask in one backward pass, without per-pixel weights from the
rasteriser. Surfels with similar vote vectors across frames form an entity. Frozen features are
lifted the same way and rendered to any view with the multi-channel rasteriser. What has to be
measured is how wrong geometry corrupts the lifting: purity and fragmentation of lifted entities
on the depth-placed surface against entities lifted on the exact surface, as a function of the
local placement error.

Outline evidence comes for free in this form: at an owned outline an entity's surfels border
different colours, so $\mathbf J_m$ is large there even when the interior is blank. Ownership is
handled by the rasteriser's occlusion instead of by a rule. The layer test of the main document
becomes a comparison of two image messages on the same entity, one restricted to pixels near the
outline and one to the interior.

### 1.6 What is learned

Nothing in the test itself is learned. Three light heads on frozen features lifted to entities
remain, each with a stated source of supervision:

1. the **prior head** $\pi(\boldsymbol\phi_m)$, trained on entities where the tension is
   informative and applied where $\Lambda^I$ is negligible, with the propensity of being
   informative as the map of support and an *unsupported* state outside it;
2. an **honesty head** that predicts $\log \psi$ from features, trained on injection responses.
   It learns on which appearances the image message overstates its information. It can correct
   scale; it cannot learn bias, because injection moves the model and not the truth;
3. the per-stratum noise scales of both families, estimated from held-out scatter.

Admissible inputs to the prior head are features of single frames and of raw clips. Rendered
residuals and gradients are the witness itself and are excluded.

## 2. The cheap test, in two stages

Exact depth is used for scoring only, and to construct known displacements.

### Stage A: is the image message a measurement? (existing photometric model, no new fit)

Inputs: the trained photometric 2DGS model of block 3 and its frames; a second copy retrained
under regime R1 from the same initial points (about half an hour on the 8 GB card, by the first
run's time); groups of surfels defined by ground tiles of 20 m and by lifted image masks.
Displacement ladder: 0.25 to 32 px of parallax along the normal, and tilts of matching size.

- **A1, exchange identity and capture range.** Response of gradient and loss to displacement on
  judging frames; $\rho_{\mathrm{cap}}$ at 90% correct sign, per surface class, at three image
  scales and for R1, R2, R3.
- **A2, honesty and abstention.** $\psi$ per class for the three estimators of Section 1.1. On
  water and shadow in particular: is nominal information small where the response is nil?
- **A3, absorbed fraction** $\chi_{\mathrm{abs}}$ for R0 to R3 at 500 appearance steps, with the
  PSNR each regime reaches on healthy surface.
- **A4, what the image family builds where it should abstain.** For the R0 and R1 models, the
  signed distance of the photometric surface from the exact one on water, and whether it is
  behind (a virtual image) or in front (floaters). The repository's numbers say in front for R0.

**Stage A stops the line if any of:**

- **KA1.** On textured land, at every image scale and regime, $\rho_{\mathrm{cap}}$ is below the
  parallax shift of a fault of three monocular error scales. A gradient at the depth-placed
  surface then cannot see the faults that matter, and what remains is a sweep.
- **KA2.** On blank and water classes $\psi > 3$ for every estimator, the per-frame scatter
  included. Abstention is then not automatic and the test would accuse correct depth wherever
  the image model hallucinates.
- **KA3.** No regime has $\chi_{\mathrm{abs}} < 0.5$ while staying within 1 dB of R1 on healthy
  surface.

Cost: two model copies, a few thousand renders; an estimate of two GPU-hours and 500 lines.

### Stage B: the tension against real monocular faults (joint model)

Inputs: Depth Pro on the 185 frames; a depth-placed surfel surface at image resolution; masks
and features lifted; frames split by ground tile into fitting and judging sets.

- **B1, detection.** $T^{\mathrm S}_m$ per entity against the true placement error of the
  depth-placed surface: AUC, average precision, true-positive rate at 5% false positives,
  overall, per class, and on entities with high repetition coefficient. Comparators: the
  hand-warped stabilised-clip gap (the special case); the consensus gap; disagreement between
  the depth map and a photometric-only surface; and three published devices re-implemented on
  the same data (Section 3): a virtual-stereo inconsistency mask, an instance-level density
  uncertainty, a reprojection trusted region; single-image baselines last.
- **B2, calibration and circularity.** Distribution of the statistic on truly healthy entities
  against $\chi^2_3$, for in-sample, held-out and score-test versions, before and after
  correction by $\psi$.
- **B3, when to listen.** Four fits: depth only; image only; both fused by information with no
  gate; both with the placement gate. Harm-weighted surface error, on entities by image share
  $\varsigma$. Against an oracle that chooses per entity with exact depth.
- **B4, transfer.** The prior head on entities with negligible $\Lambda^I$, against the
  per-cluster fault rate and a supervised probe trained on disjoint tiles.
- **B5, lifting under wrong geometry** (Section 1.5).

**Stage B stops the line if any of:**

- **KB1, rendering adds nothing.** The tension matches the hand-warped gap within 0.02 AUC with
  no larger coverage, overall and on entities bordered by occluders, where the rasteriser's
  visibility should help most.
- **KB2, no usable null.** Healthy entities have a statistic whose robust scale is outside
  $[0.5, 2]$ of $\chi^2_3$ in the score-test version after correction.
- **KB3, fusion already answers the question.** The gated fit does not beat ungated
  information-weighted fusion on the harm-weighted metric. This is plausible: where the image
  share is near one the pixels overrule the prior without any test, and where it is zero the
  test is silent. The gate can matter only at intermediate shares and through the head.
- **KB4, not better than the crowd.** On repeated faults the tension does not exceed the best
  published comparator by 0.05 average precision.
- **KB5, no transfer, or the prior is not worth keeping** on unwitnessed entities, as in the main
  document.

**It is supported if all of:** capture range covers three monocular error scales in some regime;
$\psi$ within a factor of two on all classes after the honesty head; a regime with
$\chi_{\mathrm{abs}} < 0.2$; the score test calibrated; AUC of 0.9 on repeated faults where
$\varsigma > 0.5$ with consensus near chance; the gate beating ungated fusion on entities with
$0.1 < \varsigma < 0.9$; the head at AUC 0.75 on unwitnessed entities. All thresholds are
conventions fixed before measurement.

**Uncertainty.** Cluster bootstrap over ground tiles; the count of distinct faulty entities is
shown with every interval. Water is half the block and few entities.

## 3. Prior work that the experiments must beat or contain

| Work | What it does | Status | Relation |
| --- | --- | --- | --- |
| "In Depth We Trust: Reliable Monocular Depth Supervision for Gaussian Splatting" | renders depth from a view and a virtual stereo partner, flags pixels whose reprojected depths disagree, applies scale-invariant monocular supervision only there, plus a depth-gradient alignment loss; 3DGS and 2DGS | *(checked)* | decides where to use the prior from the *rendered model's* self-consistency; no test of the prior, no information, no gradients of the two losses |
| MonoInstance | segments each view, associates instances by distance of back-projected point clouds, defines uncertainty of monocular depth per instance as low point density in the fused cloud, weights the depth loss by it | *(checked)* | entity-level and pose-based, but the evidence is depth consensus: an error every view repeats gives a dense cloud and low uncertainty |
| GVGS | block-wise affine calibration of monocular depth to rendered depth on a quadtree, in a trusted region from reprojection and Gaussian visibility | *(checked)* | repairs by local recalibration against the model's own depth; does not ask whether that depth is right |
| PUP 3D-GS | per-Gaussian Fisher information of the reconstruction loss on mean and scale, from outer products of image gradients on coarse patches; used to prune | *(checked)* | the same object as $\Lambda^I$, for another purpose; no depth prior, no test |
| FisherRF, Bayes' Rays, Gauss-Newton or Levenberg-Marquardt optimisers for splatting | Fisher information for view selection, post-hoc spatial uncertainty, second-order fitting | *(recalled)* | information of one loss; no comparison of two families |
| Depth-regularised and normal-regularised splatting (several) | add a monocular depth or normal loss, sometimes with a confidence weight | *(recalled; one confidence-weighted SLAM variant seen by title only)* | the prior as a loss term with a heuristic weight |
| Feature splatting, lifting of segment-anything masks onto Gaussians (several) | render features; assign masks to Gaussians by accumulated weights or contrastive features | *(titles seen; contents recalled)* | components this design reuses |
| Residual-supervised distractor masks on frozen features | learn a mask from the photometric residual of the model being fitted | *(recalled)* | learns from an in-sample residual; no held-out judge |

One search was made for work on conflict between depth and photometric gradients per Gaussian;
it returned none, which is weak evidence. What appears to remain new, from the side of the
experiments: a test with a null and an error bar between the two message families on the same
primitives, evaluated as a score test on held-out frames; repeated faults as the stated target,
which the consensus-based devices above cannot see by construction; calibration of the image
message by injection, with capture range, honesty ratio and absorbed fraction as reported
quantities; and the four-fit answer to when a primitive should listen to which family.
Depth-regularised splatting with a confidence weight is the baseline to which all of this
collapses if KB3 fires.

## 4. Follow-on experiments, in order

| | Question | Metric | Baseline | Falsifiable hypothesis |
| --- | --- | --- | --- | --- |
| F1 | Lumped per-surfel blocks or exact entity information? | error of predicted response under injection | exact finite difference | per-surfel diagonal blocks understate entity information by more than a factor of two on textured surface; random probing with 8 probes is within 20% |
| F2 | Colour or frozen features in the loss? | capture range, honesty, detection under exposure change and degraded frames | R1 colour | feature-metric rendering doubles the capture range and loses less than half the precision |
| F3 | Frame spacing | all of stage A and B1 on dense synthetic video at strides 1 to 16 and on survey stills | hand-warped gap | capture range in pixels is constant, so the detectable fault shrinks in proportion to parallax; at small parallax a regime exists where the image share is near one half and the gate matters |
| F4 | Reflections and transparency | outline message against interior message on the same entity; sign of the photometric surface relative to the depth-placed one | none | bounded reflectors show a significant interior-behind gap with a consistent outline; on rendered water with floaters the honesty head removes the false accusation |
| F5 | Repeated faults (headline) | detection on entities by repetition coefficient, witnessed and unwitnessed separately | consensus gate, robust loss, frame removal, the three published devices | consensus-based methods within 0.05 of chance at high repetition; the tension at AUC 0.9 where witnessed; the head at a true-positive rate of 0.30 at 5% false positives where not |
| F6 | Repair | held-out depth error and surface F-score, precision and recall separately | the four fits; depth-regularised splatting with a confidence weight | the placement gate keeps monocular shape and closes half the gap between ungated fusion and the oracle on entities of intermediate image share |
| F7 | Label budget | calibration and ranking on unwitnessed entities at 0, 1, 5, 20 labelled frames | linear probe at equal budget | labels improve calibration, not ranking |
| F8 | Real aerial data with laser depth | all of the above on laser-covered surface; pose covariance in $\Lambda^I$ | as F6 | the score test stays calibrated under real pose error once poses are marginalised or refined on the fitting frames only |

## 5. What a rendered survey block can and cannot show

- The photometric model's failure on water is floaters above the plane, not a mirror image
  below; this block tests false information, not the reflection case.
- Glass is below useful entity size at this range.
- Exact poses, no exposure change and no noise favour the image message; decisions are read with
  degraded frames and perturbed poses (`experiments/h7_pose_check.py`).
- Survey spacing gives large parallax, so most textured surface has an image share near one,
  where the answer does not need a test. Intermediate shares need dense video with exact depth,
  which is not on disk.

## 6. Protocol

1. Exact depth enters scoring and the construction of injected displacements only.
2. Frames are split by ground tile into fitting and judging sets before anything is fitted; the
   judging frames never touch placement, appearance, lifting or noise scales of the entities
   they judge.
3. The appearance regime used for the test is fixed by stage A, before stage B is scored.
4. Heads are trained on tension labels and injection responses of the scene itself; labelled
   data, when used, come from scenes never evaluated.
5. Backbones, segmenter and depth model are frozen; only surfel parameters and the three heads
   are fitted.
6. Evaluation data live in `results/k0/eval_only/`; a unit test fails if a fitting script opens
   it. Thresholds and hypotheses are committed and tagged as `prereg/TAGGING.md` prescribes
   before the scoring script is first run. Block 3 is a development block afterwards.

## 7. Implementation plan: what exists and what must be built

| # | Stage | Script | Does | Reuses | Must be built |
| --- | --- | --- | --- | --- | --- |
| 1 | A | `k0_rgb_r1.py` | retrain the photometric model under R1 | `rgb_train_2dgs.py`, `rgb_prepare.py` | switches for harmonics degree, opacity freeze, no densification; frame split by tile |
| 2 | A | `k0_groups.py` | ground-tile groups; mask voting onto surfels | multi-channel render | voting by backward pass; a segmenter in image mode |
| 3 | A | `k0_inject.py` | displace groups, render judging frames, record loss and gradient | renderer call of `rgb_train_2dgs.py` | entity directions $\mathbf G_m$; the three information estimators; R2 colouring; R3 feature targets |
| 4 | A | `k0_absorb.py` | re-optimise appearance at held geometry | trainer loop | geometry freeze |
| 5 | B | `k0_cache.py` | Depth Pro on 185 frames, float32 | `depthcache.run.run` | none |
| 6 | B | `k0_depth_surface.py` | depth-placed surfels at image resolution with depth information from autograd | ray-plane compositor of `gs/fit.py` | information without the dense numpy path; alignment per frame |
| 7 | B | `k0_tension.py` | score test per entity on judging frames | steps 2, 3, 6 | the statistic and its variance; comparators |
| 8 | B | `k0_fourfits.py` | depth only, image only, fused, gated | `gs/fit.py`, trainer | a joint loss; the placement gate |
| 9 | B | `k0_heads.py` | prior and honesty heads on lifted features | none | feature lifting; propensity |
| 10 | both | `k0_truth.py`, `k0_score.py` | truth, tables, one line per criterion | `matrixcity_io`, `align_depth`, `eval/metrics.detection_pr`, a tile-resampling wrapper for `eval/stats.bootstrap_ci` | entity-level truth; true placement error of the depth-placed surface |

Order: 1 to 4 first; they need no depth model and no joint fit, and they decide KA1 to KA3.
Roughly 500 lines for stage A and 1500 for stage B.

## 8. What this design cannot settle

- **Whether a gradient at the depth-placed surface reaches faults of realistic size.** If not,
  the method is a rendered plane sweep and the differentiable part does no work in detection.
- **Bias of the image family.** Floaters and virtual images are confident and wrong; injection
  calibrates scale, not bias. Only the outline-against-interior comparison and labelled
  evaluation address it.
- **The role of the primitives.** If KB1 and KB3 both fire, the surfels are bookkeeping and the
  note says so: the same detection is available from warped frames, and the same repair from
  information-weighted fusion.
- **Intermediate image share**, where the gate could matter, is rare at survey spacing and needs
  dense video with exact depth.
- **Unverified:** gsplat behaviour beyond its documentation (forward-mode products, per-pixel
  weights, depth definition in 2DGS mode beyond what the repository found); memory and time of a
  joint fit on 8 GB; all prior work marked recalled; the completeness of the literature search.
- **Thresholds are conventions** fixed in advance, not derived from a cost model.
