# Experiment Design: Monocular Depth Judged by Motion Parallax

This note designs the experiments for a fault indicator in which a monocular depth map is judged
by motion parallax from the same posed video. A frozen tracker gives correspondences; with known
poses each track is a triangulated depth with an uncertainty set by its parallax; where
triangulation is well conditioned, the gap between monocular and triangulated depth is a fault
label that costs nothing; a light head on frozen features is trained on those labels and carries
them to pixels where the tracker abstains. One depth model is used per image. The note starts
with a cheap test that can stop the idea before any method code is written. Notation follows the
main document (`docs/learned-indicator.md`, Section 7); new symbols are in Section 0.

Status of every number: values quoted from the repository are marked *(repo)*; facts about
external models and datasets are marked *(checked)* when read on the project's own page or paper
while writing this note and *(recalled)* when not; everything else is a design choice or an
estimate. Nothing in this note has been measured, and every statement about what an experiment
will show is a prediction.

## 0. Setting, symbols and what exists

**Unit.** A pair (view $j$, pixel $u$) on an analysis grid of $480 \times 270$. The method's
native unit is a (view, surfel) message; Section 4, F6 moves there.

| Symbol | Meaning |
| --- | --- |
| $\tau$ | a track: image positions $\mathbf u_t$ of one scene point in frames $t \in T_\tau$ |
| $p$ | parallax of a track in pixels: image displacement caused by camera translation, $p = f\, b_\perp / d$ for baseline $b_\perp$ across the ray |
| $\sigma_{\mathrm{px}}$ | standard deviation of the tracker's position error, per axis |
| $d^{\triangle}_j(u)$, $\sigma^{\triangle}_j(u)$ | triangulated depth of the track through pixel $u$ of view $j$, and its standard deviation |
| $\epsilon^{\mathrm{rp}}$ | root-mean-square reprojection residual of a track after triangulation |
| $x^{\mathrm{mo}}_j(u)$ | motion witness: standardised gap between aligned monocular depth and triangulated depth |
| $\mathcal W_j$, $\mathcal A_j$ | pixels of view $j$ where the witness is well conditioned; pixels where it abstains |
| $\ell_j(u) \in \{0, 1\}$ | witness label on $\mathcal W_j$ |
| $e(\boldsymbol\phi)$ | propensity: probability that a pixel with frozen features $\boldsymbol\phi$ lies in $\mathcal W$ |
| $n$ | frame stride of a subsampled video |
| $\varepsilon_{\mathrm{mo}}$ | relative error scale of the aligned monocular depth |

**What exists** *(repo)*. MatrixCity `small_city` aerial block 3: 185 training views in strips
and 19 test views, $1920 \times 1080$, exact depth, median depth 338 m; water is 46% of the
ground area (flat at $-3.4$ m). A depth cache with four model wrappers; four views cached; after
alignment to exact depth the median relative error is 3-5% for Depth Pro (the default model), 1-2%
for Depth Anything V2, below 1% for UniDepthV2; Depth Pro takes 0.54 s per view and 6.3 GB.
`depthcache.align.align_depth` fits a per-view scale or affine map to a *sparse* reference with
outliers. A photometric 2DGS model of the block, whose surface is within 1 m of truth on 81% of
land cells and 5% of water cells. Wrappers for OpenMVS depth maps (`baselines/openmvs_dmap.py`).
Loaders with strip keys (`experiments/matrixcity_io.py`). No tracker, matcher or flow model is in
the repository. Compute: one 8 GB GPU; a 24 GB card has been used for depth models.

**A fact about the data that the design must face.** The block is a survey: still frames in
strips, not a dense video. The frame spacing and overlap of block 3 are not recorded in the
repository and are the first thing to compute from `transforms.json`. The real set planned so far
(UseGeo) is also stills at 80% forward overlap *(repo)*. Section 1.2 shows that this places both
sets in the regime where the witness is far more precise than the monocular depth, which is the
regime in which the question "why keep the monocular prior" is hardest.

## 1. What the witness can know: three regimes

### 1.1 Precision of a triangulated depth

For a camera translating by $b_\perp$ across the ray, a point at depth $d$ moves by
$p = f\, b_\perp / d$ pixels, so $d = f\, b_\perp / p$ and

```math
\frac{\sigma^{\triangle}}{d} = \frac{\sigma_{\mathrm{px}}}{p} \quad \text{(two frames)}, \qquad
\frac{\sigma^{\triangle}}{d} = \frac{\sigma_{\mathrm{px}}}{\sqrt{\sum_{t \in T_\tau} (p_t - \bar p)^2}} \approx \frac{\sigma_{\mathrm{px}}}{P}\sqrt{\frac{12}{N}} \quad \text{($N$ equally spaced frames, total parallax $P$)} .
```

The second form assumes position errors that are independent between frames. Tracker errors are
not: drift is correlated along a track. The $\sqrt{N}$ gain of dense frames is therefore an
assumption to be measured (M1), and until it is, the two-frame form with the end points of the
track is the honest one. Pose error adds to $\sigma_{\mathrm{px}}$ through the reprojection of the
pose covariance $\Sigma_q$, as in the main document's residual variance.

With known poses, two frames constrain a correspondence to a line, which leaves one of the two
position components unchecked: an error along the epipolar line is read as depth. In a straight
nadir strip all epipolar lines run along the flight direction, so the two-frame check sees only
cross-track error. Three or more frames overdetermine the depth, and the reprojection residual
$\epsilon^{\mathrm{rp}}$ then checks both components. The witness therefore requires
$\lvert T_\tau\rvert \ge 3$, and a forward-backward tracking cycle as a second check.

### 1.2 Regimes

Let $\varepsilon_{\mathrm{mo}}$ be the relative error scale of the aligned monocular depth and
$n_{\mathrm{eff}}$ the effective number of independent tracks in a fault region. Define the
crossover parallax $p^\ast = \sigma_{\mathrm{px}} / \varepsilon_{\mathrm{mo}}$.

| Regime | Parallax | Per pixel | Per region |
| --- | --- | --- | --- |
| R1, blind | $p < p^\ast / (3\sqrt{n_{\mathrm{eff}}})$ | triangulation worse than monocular | a regional shift of $\varepsilon_{\mathrm{mo}}$ is not detectable |
| R2, judge | between | triangulation worse than monocular | a regional shift of $\varepsilon_{\mathrm{mo}}$ is detectable at 3 standard errors |
| R3, replace | $p > p^\ast$ | triangulation better than monocular | detectable; triangulation could stand in for the prior on $\mathcal W$ |

R2 is the only regime in which the phrase "the witness judges the dense prior without replacing
it" is literally true on tracked pixels: no single triangulated depth is as good as the
monocular one, yet their regional mean exposes a fault. It exists for short windows of dense
video, for forward motion, and near the focus of expansion.

In a survey block it barely exists. A point seen across a whole pass moves by about the image
extent along the track, $P \approx 1000$ px; even adjacent stills at 80% overlap give
$p \approx 0.2 \times 1080 = 216$ px. With $\sigma_{\mathrm{px}} = 1$ px this is a relative depth
error of 0.1% to 0.5%, against 3-5% for Depth Pro on the four cached views. Whatever
$\sigma_{\mathrm{px}}$ turns out to be within reason, tracked pixels of a survey block are in R3
by an order of magnitude. Two consequences shape everything below.

1. On $\mathcal W$ the monocular depth is the weakest sensor present, and a reconstruction that
   adds the tracks as depth messages with their own $\sigma^{\triangle}$ will overrule it there
   without any indicator. The indicator's value on survey data lies on $\mathcal A$ only.
2. The claim about faults that every view repeats splits in two. On $\mathcal W$ the witness sees
   such faults directly, because triangulation does not use any depth map; this part is nearly
   guaranteed and nearly worthless, since the tracks would have corrected the surface anyway. On
   $\mathcal A$ nothing sees them except the head, which extrapolates from appearance. The paper's
   claim lives or dies on $\mathcal A$.

### 1.3 A caution about reflections

A reflection in a *planar* mirror of a static scene is the image of a static virtual point behind
the mirror. It satisfies the epipolar constraint in every frame and triangulates, with small
residual, to the virtual depth. So calm water and flat glass do not necessarily produce tracks
that leave their epipolar lines: they can produce confident, self-consistent tracks at the wrong
depth. Only curved or moving reflectors (ripples, waves) break the constraint. Likewise an object
moving along the flight direction stays on its epipolar line. The self-check is an honest
abstention flag for tracker failure and for most motion, and is not one for planar specular
surfaces. Whether the witness then mislabels correct monocular depth as faulty there is measured
in M2; it is the most likely source of systematically wrong labels.

## 2. The cheap test (K0)

**Question.** Is the witness a sound label where it claims to be; do faults exist where it
abstains; does a head trained on witness labels rank faults on abstaining pixels; and does the
monocular prior beat the alternatives there once it is gated?

Exact depth is used for scoring only. No alignment, threshold, label or head parameter touches it.

### 2.1 Inputs

- **Depth.** One identity pass of Depth Pro over the 185 training views (about 100 s of GPU), and
  one of Depth Anything V2 (about 65 s) as a second decision model, stored as float32.
- **Geometry, before any model is run.** From `transforms.json`: frames per strip, forward and
  side overlap, and for every pixel (using exact depth, evaluation side) the number of frames that
  see it and the total parallax $P$. This costs nothing and decides which regime of Section 1.2
  the block is in.
- **Correspondences from frozen models.** Three witnesses are run so that the claim "learned
  tracking holds where patch correlation abstains" is itself tested:

  | Witness | Model | Verified | Remarks |
  | --- | --- | --- | --- |
  | Point tracker | CoTracker3, online (windowed) and offline checkpoints | exists, tracks grids of queries, licence CC-BY-NC *(checked)* | trained on video with small inter-frame motion; a 20% image shift between survey stills is outside that; internal resolution near $384 \times 512$ *(recalled)*, which sets $\sigma_{\mathrm{px}}$ at several full-resolution pixels |
  | Dense matcher | RoMa, chained along the strip | *(recalled: published 2024, built on a frozen DINOv2 encoder; licence not checked)* | made for wide baselines; pairwise, so tracks come from chaining |
  | Optical flow | SEA-RAFT or RAFT, chained | *(recalled)* | all-pairs correlation at 1/8 of $1920 \times 1080$ needs about 5.6 GB in single precision (computed from the volume size); run at half resolution |
  | Classical control | plane-sweep patch correlation along the ray | no model | the witness that tracking replaces |

  None of these has been run here; memory and time on 8 GB are unmeasured.
- **Frozen features** for the head, cached once (Section 4, F3 lists the candidates). The cheap
  test uses one image encoder and the tracker's own outputs.

### 2.2 The witness

1. **Tracks.** Queries on the analysis grid of every frame; tracked through the frames of the
   same strip in which the point stays visible, and into the neighbouring strips where the model
   allows.
2. **Triangulation with known poses.** Linear triangulation then two Gauss-Newton steps on the
   reprojection error; $\sigma^{\triangle}$ from the inverse normal matrix with
   $\sigma_{\mathrm{px}}$ estimated label-free as the robust scale of the reprojection residuals of
   all tracks with at least four frames.
3. **Conditioning and abstention.** $u \in \mathcal W_j$ if the track has at least three frames,
   $\epsilon^{\mathrm{rp}}$ is below three times its nominal value, the forward-backward cycle
   closes within $2\sigma_{\mathrm{px}}$, and $\sigma^{\triangle}/d^{\triangle}$ is below one third
   of the monocular error scale. Otherwise $u \in \mathcal A_j$, with the reason recorded (lost,
   off-epipolar, cycle failure, low parallax, near a depth discontinuity of $d^{\triangle}$).
4. **Alignment.** The per-view scale and offset $(a_j, b_j)$ are fitted to the triangulated
   depths on $\mathcal W_j$ with `align_depth`, which was written for a sparse reference with
   outliers. This is the label-free reference the cache documentation asks for. The monocular
   error scale $\varepsilon_{\mathrm{mo}}$ is the robust scale of the aligned gap on $\mathcal W_j$.
5. **Score and label.**

```math
x^{\mathrm{mo}}_j(u) = \frac{\log d^{\mathrm{al}}_j(u) - \log d^{\triangle}_j(u)}{\sqrt{\varepsilon_{\mathrm{mo}}^2 + \bigl(\sigma^{\triangle}_j(u) / d^{\triangle}_j(u)\bigr)^2}}, \qquad
\ell_j(u) = 1 \ \text{if } \lvert x^{\mathrm{mo}}_j(u)\rvert > 3 \text{ in a connected region of at least 2\% of the view}, \quad
\ell_j(u) = 0 \ \text{if } \lvert x^{\mathrm{mo}}_j(u)\rvert < 2 ,
```

   and no label in between. The region rule matches the evaluation mask so that label and target
   describe the same object.

### 2.3 Evaluation targets

The true-fault mask $f$, the ignore band, the surface classes (water, ground, roof, facade, edge
and thin, texture-free) and the repetition strata by $\varrho(X)$ are those of the main
document's Section 5.1. They are written to `results/k0/eval_only/` and read by the scoring script
only. Exact depth and poses also give exact correspondences for any static pixel, so the
tracker's position error is measured directly, without a flow benchmark.

### 2.4 Measurements

- **M1, witness accuracy and calibration.** End-point error of tracks against exact
  correspondences, per class and per frame gap; error of $d^{\triangle}$; the ratio of that error
  to $\sigma^{\triangle}$ (robust standard deviation of the standardised error, target one); the
  same ratio as a function of the number of frames used, which tests the $\sqrt{N}$ assumption;
  coverage $\lvert\mathcal W\rvert$ per class for each witness. This is where the learned
  trackers are compared with plane-sweep correlation on weakly textured ground.
- **M2, the self-check.** How well $\epsilon^{\mathrm{rp}}$ and the cycle error detect tracks
  with end-point error above 3 px (ROC), per class. On water: the share of tracks that pass all
  checks, and the error of their triangulated depth against the water surface. A cluster of
  passing tracks at a consistent wrong depth is the virtual-image case of Section 1.3.
- **M3, label quality on $\mathcal W$.** Precision and recall of $\ell$ against $f$, overall, per
  class and on the common-mode stratum; every disagreement attributed to a cause (alignment,
  tracker, discontinuity, reflection, threshold band).
- **M4, where the faults are.** The table of fault by coverage: $\Pr(f = 1 \mid \mathcal W)$ and
  $\Pr(f = 1 \mid \mathcal A)$; the share of faulty units, and of common-mode faulty units, in
  each set, per class; the number of distinct fault regions in ground space that lie mostly in
  $\mathcal W$ (training positives) and mostly in $\mathcal A$ (the test).
- **M5, transfer.** A logistic head and a two-layer head on cached frozen features, trained on
  witness labels of the block itself (transductive, no ground truth), scored on $\mathcal A$
  against $f$: AUC, average precision, true-positive rate at 5% false positives, overall and
  within each surface class, and as a function of propensity decile (Section 3). Comparators that
  use no labels: a constant; local texture energy; the cavity residual of the main document where
  it is defined; the per-cluster fault rate of a 16-cluster k-means on the same features (a head
  that only learned "which kind of surface"). Comparator with labels: a linear probe on the same
  features trained on true masks of a disjoint set of ground tiles.
- **M6, replacement.** On $\mathcal A$, per class, the relative depth error of: (i) monocular
  depth with the global alignment of step 4; (ii) the same restricted to pixels with $f = 0$,
  which is what a perfect indicator would keep; (iii) monocular depth warped locally to the
  triangulated anchors (a smooth scale field interpolated from $\mathcal W$); (iv) triangulated
  depth filled into $\mathcal A$ by harmonic and by piecewise-planar interpolation, no monocular
  depth at all; (v) OpenMVS depth maps; (vi) a feed-forward multi-view model on windows of frames.
  For (vi), VGGT exists and has a track head; its original checkpoint is non-commercial and no way
  to supply known poses is documented *(checked)*; its memory per number of frames on 8 GB is not
  verified, so (vi) is optional in the cheap test.

**Uncertainty.** Cluster bootstrap over $100 \times 100$ m ground tiles, 1000 resamples; the
number of distinct fault regions is reported beside every interval.

### 2.5 Decision thresholds, fixed in advance

Read on the decision model (Depth Pro; Depth Anything V2 if Depth Pro's prevalence is outside
1-30%), the best of the three learned witnesses chosen by coverage at calibrated
$\sigma^{\triangle}$ (a label-free criterion), with images degraded and poses perturbed as in
Section 5. The numbers are conventions fixed before measurement.

**Precondition.** Prevalence between 1% and 30%; at least 20 fault regions mostly in
$\mathcal W$ and at least 20 mostly in $\mathcal A$; $\mathcal W$ covers at least 20% of land
pixels. If it fails, the block says nothing about the idea and the test moves to dense synthetic
video (Section 4, F2).

**The approach is abandoned if any of:**

- **K1, the label is unsound.** On $\mathcal W$ the precision or the recall of $\ell$ against $f$
  has an upper interval bound below 0.90; or the standardised triangulation error has robust
  standard deviation outside $[0.5, 2]$ after the label-free estimate of $\sigma_{\mathrm{px}}$.
- **K2, nothing left to judge.** Fewer than 10% of faulty units lie in $\mathcal A$. All faults
  then sit where triangulation overrules the prior anyway, and on survey data the indicator has
  no job. (The idea could survive only in regime R2, tested in F2.)
- **K3, no transfer.** On $\mathcal A$ the head's AUC has lower bound at or below 0.65, or its
  average precision does not exceed the best label-free comparator by 0.05.
- **K4, a surface-type detector and nothing more.** On $\mathcal A$ the per-cluster fault rate
  matches the head within 0.02 average precision, *and* within every surface class the head's
  AUC has lower bound at or below 0.55. The method then reduces to "distrust this kind of
  surface", which a handful of labels or a rule would give.
- **K5, the prior is not worth keeping.** On $\mathcal A$, in every surface class with at least
  five fault regions, some alternative without monocular depth, (iv), (v) or (vi), has median
  relative error no larger than (ii), the monocular depth under a perfect indicator. No indicator
  can then make the prior useful.

**It is supported if all of:** precision and recall of $\ell$ at least 0.95 on $\mathcal W$; at
least 30% of faulty units and of common-mode faulty units in $\mathcal A$; the head on
$\mathcal A$ has AUC at least 0.75 overall and at least 0.65 inside two or more surface classes,
and a true-positive rate of at least 0.30 at 5% false positives on common-mode units; and in at
least one surface class (ii) beats every alternative of M6 by an interval that excludes zero.

K5 is the criterion most likely to fire on this block, for a simple reason: its largest
abstaining region is a flat water plane, which planar interpolation from the shoreline reproduces
exactly. A verdict from K5 on block 3 alone is therefore read per class, and the classes that
decide are the non-planar ones: facades, thin structures, shadowed and occluded ground.

### 2.6 Cost

| Item | Cost (estimate; nothing measured) |
| --- | --- |
| Overlap and parallax from poses | seconds |
| Depth, 370 passes | about 3 min GPU |
| Tracker, matcher, flow over about 185 frames in strips | tens of minutes each; memory on 8 GB to be measured |
| Triangulation, alignment, labels | minutes, CPU |
| Features and heads | minutes; head memory negligible |
| M6 fills and OpenMVS | under an hour |
| New code | about 1000 lines |

## 3. Measuring transfer without ground truth

The head is trained where the witness speaks and used where it does not. That is selection on
the covariates: $\mathcal W$ is textured and diffuse, $\mathcal A$ is water, glass, texture-free
and occluded. Three quantities describe the shift, and each is an experiment.

**Propensity.** $e(\boldsymbol\phi) = \Pr(u \in \mathcal W \mid \boldsymbol\phi)$ is estimated
without any label by a logistic regression of the coverage flag on the frozen features, cross-fitted
over ground tiles. Where $e$ is near zero the head has seen nothing like the pixel; importance
weighting cannot correct that, and the output there is an extrapolation by construction. The share
of $\mathcal A$ with $e < 0.05$ is reported for every scene; it is the share of the image on which
the method has, in the strict sense, no evidence.

**Prevalence shift.** $\Pr(f \mid \mathcal A) \ne \Pr(f \mid \mathcal W)$ in general, and the
difference is not identified from unlabelled data: nothing on $\mathcal A$ observes $f$. The head
can rank on $\mathcal A$; its calibration there is unknown. This is the one thing a few labels
from a synthetic scene buy that nothing else does, and the label-budget experiment (F4) is designed
around it.

**Counterfactual abstention.** Pixels of $\mathcal W$ can be made to play the part of
$\mathcal A$ while their witness label is kept:

- *Withheld strata.* Remove whole feature clusters, or the lowest-propensity decile of
  $\mathcal W$, from the head's training set and score the head on them against $\ell$.
- *Short windows.* Compute labels from full tracks; give the pipeline a window so short that the
  conditioning rule of Section 2.2 fails on low-parallax pixels; score the head on the pixels
  that became abstentions.
- *Degraded texture.* Blur or downsample the frames until the tracker loses a region; train and
  apply the head on the degraded frames; score against labels from the clean frames.

Each gives the head's accuracy as a function of distance from its training support, on any video,
with no ground truth. The proposal is to report this curve on real data, where laser references
are sparse or missing on exactly the surfaces of interest. Its validity is itself a hypothesis
(F3): on synthetic data with exact depth, does the curve at low propensity predict the true
accuracy on $\mathcal A$? If it does, the method carries its own evaluation to scenes without
reference data. If it does not (because water is not blurred land), the note says so and real-data
claims about $\mathcal A$ need reference depth after all.

## 4. Follow-on experiments, in order

Each is run only if the one before does not stop the line. Intervals are over ground tiles within
a scene and over scenes across scenes.

### F1. Which witness, at which frame density

- **Question.** Which frozen correspondence model gives the largest $\mathcal W$ at calibrated
  uncertainty, and how does that depend on frame spacing?
- **Data.** Block 3 as stills; a dense synthetic video with exact depth and poses, subsampled at
  strides $n \in \{1, 2, 4, 8, 16\}$. TartanAir provides long sequences with exact depth, pose and
  optical flow, an ocean environment and reflections from glass and puddles *(checked)*; its motion
  is not aerial survey motion and its total size is 4 TB, so a few sequences are taken. Mid-Air is
  a synthetic low-altitude drone set *(checked that it exists; its frame rate and depth format are
  recalled)*. MatrixCity's authors describe a plugin that renders camera trajectories in Unreal
  Engine scenes *(checked on the project page; whether it is released and usable for a dense
  flight over block 3 is not verified)*.
- **Metric.** M1 and M2 per stride and per class.
- **Baseline.** Plane-sweep patch correlation; SIFT matching with geometric verification.
- **Hypothesis.** At strides typical of video the point tracker has the largest calibrated
  coverage on texture-free ground, exceeding plane-sweep correlation by at least 10 points of
  coverage in that class; at survey spacing the dense matcher overtakes the tracker. Falsified if
  plane-sweep correlation matches the learned witnesses within 5 points in every class, in which
  case tracking adds cost and no coverage.

### F2. The judge regime

- **Question.** Is there a range of parallax in which triangulated depth is worse than monocular
  depth per pixel and still detects regional faults?
- **Data.** Dense synthetic video; window length swept from 3 frames up, so that total parallax
  runs from a few pixels to $p^\ast$ and beyond. Causal windows only, as an online system would
  have.
- **Metric.** Median relative error of $d^{\triangle}$ against that of the monocular depth;
  region-level AUC of the pooled witness $\sum_{u \in \text{region}} x^{\mathrm{mo}} / \sqrt{n_{\mathrm{eff}}}$
  against $f$; detection delay in frames.
- **Baseline.** The single-pixel witness; the cavity residual over the same window.
- **Hypothesis.** For windows with $p^\ast / (3\sqrt{n_{\mathrm{eff}}}) < P < p^\ast$ the pooled
  witness reaches region-level AUC of 0.9 while its per-pixel depth error exceeds the monocular
  one. Falsified if the two crossovers coincide within a factor of two in $P$, which would mean
  the witness is useful only once it could replace the prior. This experiment is the direct
  answer to "why keep the monocular prior" for tracked pixels, and the natural core of an
  online, robotics-facing version of the work.

### F3. Transfer from textured to untextured surfaces

- **Question.** Does the head rank faults on $\mathcal A$, which frozen features carry that, and
  does counterfactual abstention predict it?
- **Data.** MatrixCity blocks and the dense synthetic sequences with water and glass.
- **Frozen feature sets compared** (cached once; the head trains from the cache):

  | Features | Why | Status |
  | --- | --- | --- |
  | Per-image self-supervised encoder (DINOv2 at three sizes; DINOv3 including a variant pretrained on overhead imagery) | appearance; the reference | DINOv2 widely used; DINOv3 and its overhead variant *(recalled; weights and licence to verify)* |
  | Encoder of the depth model itself | encodes what the depth model was unsure about; free, the pass is run anyway | available for the cached models; Depth Pro peaks at 6.3 GB *(repo)* |
  | The tracker's own outputs and features: visibility, confidence, correlation features at the query | free; says *why* the witness abstained | CoTracker3 returns visibility *(checked)*; access to internal features is a code change, not verified |
  | Multi-view geometry model, aggregated features over a window (VGGT) | sees parallax inside its features, which a per-image encoder cannot | exists *(checked)*; memory on 8 GB per window length not verified |
  | Video encoder (V-JEPA 2 or similar) | temporal features without a geometry head | *(recalled; not checked)* |

- **Metric.** As M5, on $\mathcal A$, per class and per propensity decile; error of the
  counterfactual-abstention estimate against the true value.
- **Baseline.** Per-cluster fault rate; texture energy; a linear probe with true labels from a
  disjoint scene.
- **Hypotheses.** (i) The head exceeds the per-cluster rate by 0.05 average precision on
  $\mathcal A$. (ii) Multi-view features exceed per-image features by 0.05 on $\mathcal A$.
  (iii) The counterfactual estimate at the lowest propensity decile is within 0.05 AUC of the
  true value on $\mathcal A$. Each falsified separately.
- **Caution.** Features of a multi-view geometry model may encode that model's own depth. A head
  on them can then act as a comparison between two depth estimators. That is permitted by the
  frozen-backbone scope, but it is a different mechanism from appearance transfer and is reported
  as such, with an ablation that removes the model's depth-head inputs.

### F4. Label budget

- **Question.** With the witness supplying labels for free, what do true labels still buy?
- **Budgets.** 0, 1, 5, 20 labelled views and one whole labelled synthetic scene, ten draws each,
  from scenes never evaluated.
- **Methods.** Witness-trained head; the same with labelled views used only to recalibrate
  prevalence and threshold on $\mathcal A$; the same with a supervised term; and the reference
  line, a linear probe on the same frozen features trained on the labelled views alone.
- **Metric.** Average precision and expected calibration error on $\mathcal A$, as curves over
  the budget; the budget at which the reference line catches the method.
- **Hypothesis.** Labels improve calibration on $\mathcal A$ (calibration error halves at 5
  views) and not ranking (average precision changes by less than 0.02); at zero labels the method
  reaches 80% of the probe's average precision at one full scene; and with labels from synthetic
  scenes only, the probe loses at least 0.05 on real data where the witness-trained head, which is
  fitted on the target scene, does not. Falsified separately. If the probe with five synthetic
  labelled views matches the method on real $\mathcal A$, the witness is unnecessary as a source of
  supervision.

### F5. Faults that every view repeats

- **Question.** Does the indicator detect them, and where?
- **Data.** The common-mode stratum of natural errors; and, to obtain such errors on
  $\mathcal A$ in controlled number, depth maps in which one view's natural fault region is made
  consistent across all views that observe it.
- **Metric.** True-positive rate at 5% false positives and AUC, separately on
  $\mathcal W \cap$ common-mode and $\mathcal A \cap$ common-mode.
- **Baseline.** The cavity residual, a robust loss and view removal (all predicted at chance);
  the depth model's confidence; disagreement between several depth models and instability under
  input transformations, as single-image baselines that cost extra depth passes.
- **Hypothesis.** On $\mathcal W \cap$ common-mode the witness alone reaches AUC 0.9 while the
  consensus baselines are within 0.05 of 0.5. On $\mathcal A \cap$ common-mode the head reaches a
  true-positive rate of 0.30 at 5% false positives and exceeds every single-image baseline.
  Falsified separately; the second part is the claim that matters.

### F6. Reconstruction

- **Question.** Does gating the monocular messages by $1 - \zeta$ improve the surfel surface?
- **Data.** MatrixCity blocks and dense synthetic sequences; held-out frames by block.
- **Metric.** Held-out depth error and surface F-score, overall, on $\mathcal W$ and on
  $\mathcal A$, and on fault regions; precision and recall of the surface reported separately,
  because removing a wrong surface on $\mathcal A$ leaves a hole.
- **Baselines.** No gate; robust depth loss; view removal; the cavity residual gate; the depth
  model's confidence; and the three that decide whether the indicator is needed at all:
  (a) *tracks as depth messages, monocular depth ungated*, where information-form fusion already
  lets the precise tracks overrule the prior on $\mathcal W$; (b) the same with the monocular
  noise inflated uniformly on $\mathcal A$; (c) no monocular depth: tracks plus interpolation, or
  OpenMVS, or a feed-forward multi-view model. Bound: an oracle gate.
- **Hypothesis.** On fault regions inside $\mathcal A$ the indicator closes at least half the gap
  between baseline (a) and the oracle, and beats (b) and (c) on non-planar classes; on clean
  regions it changes error by less than 2%. Falsified if (a) is within noise of the oracle (the
  tracks already do the work) or if (c) wins on $\mathcal A$ (the prior is not needed).

### F7. The same judge for another prior

- **Question.** Is the method specific to monocular depth?
- **Design.** Replace the monocular depth map by the dense depth of a feed-forward multi-view
  model and rerun K0 and F6. Such a model uses parallax itself, so its errors on $\mathcal W$
  should shrink; its errors on water and glass are, like the monocular ones, repeated by every
  view.
- **Hypothesis.** The witness-trained head detects faults of the multi-view prior on
  $\mathcal A$ with AUC within 0.05 of its value for the monocular prior. If it holds, the
  contribution is a judge for any dense learned depth prior, which survives the day multi-view
  models displace monocular ones. If the multi-view prior has no faults left to find, that is the
  strongest form of the objection and is reported.

### F8. Real aerial data

- **Question.** Does it hold on real imagery with real pose error?
- **Data.** UseGeo, stills with laser depth maps (reader is a stub; download pending *(repo)*).
  A real aerial *video* with poses and a laser reference has not been identified; a multi-modal
  UAV dataset with LiDAR appeared in a search and is not verified. Without one, the judge regime
  (F2) is shown on synthetic video only, and this is stated as a limit.
- **Metric.** M1 to M4 on laser-covered pixels; the head on $\mathcal A$ where laser exists; the
  counterfactual-abstention curve everywhere; reconstruction as F6.
- **Hypothesis.** Witness labels keep precision and recall of 0.90 under real pose error once
  $\Sigma_q$ enters $\sigma^{\triangle}$; the counterfactual curve on real data lies within 0.05
  AUC of the synthetic one at equal propensity. Missing laser returns are reported as unscored.

## 5. What a rendered survey block can and cannot show

- **Water** is 46% of block 3 and the photometric reconstruction fails on it *(repo)*, so it is a
  large abstaining region. Whether the monocular models are wrong there is unknown. Whether the
  renderer produces view-consistent reflections on it is unknown; if it produces none, the
  virtual-image failure of Section 1.3 cannot appear and M2 on this block is uninformative about
  it. TartanAir's reflective scenes are the fallback.
- **Glass** at 340 m range in nadir views is below the 2% region size; not tested here.
- **Flattery.** Rendered frames have no noise, blur, exposure change or rolling shutter, and the
  poses are exact. A tracker benefits from all of it. Controls: frames degraded (gain and gamma
  jitter of 20%, noise of 3/255, blur of 0.7 px, JPEG quality 75) before tracking and before the
  depth model; poses perturbed at the level a bundle adjustment leaves, with the machinery of
  `experiments/h7_pose_check.py`; decisions read in the degraded setting.
- **Training-set overlap.** Trackers and depth models are trained largely on synthetic video.
  Whether MatrixCity or TartanAir is in the training data of any model used must be checked per
  model; CoTracker3's baseline checkpoints were trained on Kubric only *(checked)*, which makes
  them a clean control for that question.
- **Moving objects.** The released block's traffic state is not verified; the dataset supports
  controllable vehicle and pedestrian density *(checked)*.

## 6. Protocol

1. **No ground truth in the loop.** Tracks, triangulation, alignment, labels, propensity and the
   head use images, poses and one depth map per image. Evaluation masks live in `eval_only/`; a
   unit test fails if a fitting script imports the exact-depth reader.
2. **Thresholds of the witness** (conditioning, residual, cycle, the 2 and 3 standard-deviation
   label bounds) are set from nominal values and from track statistics, never from a detection
   metric.
3. **Transductive fitting is declared.** The head is fitted on the scene it is applied to, from
   witness labels. Labelled data, when used (F4), come from scenes that are never evaluated, in
   two tiers: synthetic only, or a few real views from a designated set separated on the ground
   from all evaluation views by one image footprint.
4. **Model selection** for the head (capacity, stopping, feature set) uses held-out witness
   labels on $\mathcal W$ and the counterfactual-abstention score, both label-free. The feature-set
   comparison of F3 is reported in full, not as a selected winner.
5. **Splits by block or scene.** Block 3 is used by the cheap test and is thereafter a
   development block; headline numbers come from other blocks, other sequences and real data.
6. **Frozen means frozen.** Released weights; no fine-tuning of tracker, depth model or encoder.
7. **Registration.** The thresholds of Section 2.5 and the hypotheses of Section 4 are committed
   and tagged as `prereg/TAGGING.md` prescribes before the scoring script is first run.
8. **Intervals** are over ground tiles and scenes, with the count of fault regions shown.

## 7. Implementation plan for the cheap test

Scripts under `experiments/` with prefix `k0_`, config `configs/experiments/k0_kill_test.yaml`,
outputs under `results/k0/`. Nothing in `src/` changes.

| # | Script | Inputs | Outputs | Reuses |
| --- | --- | --- | --- | --- |
| 1 | `k0_geometry.py` | `transforms.json` | frames per strip, overlaps, neighbour lists; no GPU | `matrixcity_io.load_block` (poses, strip keys) |
| 2 | `k0_cache.py` | training images | identity depth cache, float32, scene `block3train` | `depthcache.run.run`, `datasets.matrixcity_views` |
| 3 | `k0_tracks.py` | images (clean and degraded), neighbour lists, witness name | tracks per strip: positions, visibility, confidence, cycle error | one frozen tracker, matcher or flow model; plane-sweep control written here |
| 4 | `k0_triangulate.py` | tracks, poses | $d^{\triangle}$, $\sigma^{\triangle}$, $\epsilon^{\mathrm{rp}}$, parallax, abstention reason, on the analysis grid | `matrixcity_io.project`, `pixel_rays_world` |
| 5 | `k0_witness.py` | depth cache, step 4 | $(a_j, b_j)$, $\varepsilon_{\mathrm{mo}}$, $x^{\mathrm{mo}}$, $\ell$, $\mathcal W$, $\mathcal A$ | `depthcache.align.align_depth` with $d^{\triangle}$ as sparse reference |
| 6 | `k0_features.py` | images, tracker outputs | cached features, k-means strata, propensity $e$ | none |
| 7 | `k0_head.py` | features, labels | $\zeta$ on all pixels; counterfactual-abstention scores | none; takes no path to `eval_only/` |
| 8 | `k0_fill.py` | step 4 and 5 outputs | depth on $\mathcal A$ for alternatives (i), (iii), (iv), (v) of M6 | `baselines/openmvs_dmap.py` |
| 9 | `k0_truth.py` | exact depth, depth cache | `eval_only/`: $f$, ignore band, classes, $\varrho$, tiles, exact correspondences and parallax | `matrixcity_io.load_block(step=4)`, `align_depth`, `WATER_Z` of `surface_dsm.py`, `baselines/_common.depth_to_normals` |
| 10 | `k0_score.py` | all of the above | tables M1 to M6 and one line per criterion with value, interval and verdict | `eval/metrics.detection_pr`; `eval/stats.bootstrap_ci` resamples one value per scene, so a tile-resampling wrapper is added here |
| 11 | `tests/test_k0_isolation.py` | - | fails if scripts 3 to 8 touch exact depth | - |

Order: 1, 2 and 9 first. They give the regime of the block, the prevalence and the number of
fault regions before any tracker is installed; if the precondition cannot hold, the remaining
steps move to dense synthetic video.

## 8. What is sharp here, and what is not

**Not new.** Checking a monocular depth map against multi-view geometry is old, and so is
training an image head on labels produced by geometric consistency. A paper that says only
"triangulate, compare, distil" would be read as a multi-view-stereo check on monocular depth, and
rightly.

**Two ideas in the experimental design that are sharper than that.**

1. *The regime map with the abstention set as the object of study.* Stating, and measuring along
   a ladder of frame strides and window lengths, where the witness is blind, where it judges
   without being able to replace, and where it replaces; and reporting every result separately on
   $\mathcal W$ and $\mathcal A$, with baselines (tracks as messages, interpolation, feed-forward
   multi-view depth) that make the monocular prior justify itself. This turns the standing
   objection into the paper's main figure instead of a paragraph in the limitations.
2. *Counterfactual abstention as a label-free estimate of transfer.* Making tracked pixels play
   untracked ones gives an accuracy-against-propensity curve on any posed video. If F3 shows that
   the curve predicts true accuracy on abstaining pixels, the method can be evaluated on real
   water and glass, where reference depth does not exist. It is not known whether anyone has used
   this device for depth-fault detection; the literature has not been searched for it, and
   importance-weighted validation under covariate shift is the obvious relative.

Whether these suffice for a top venue depends on results that do not exist yet. If transfer to
$\mathcal A$ is weak or reduces to a surface-type prior, there is no method contribution and at
most a measurement paper.

## 9. What this design cannot settle

- **Survey blocks sit in the replace regime.** On the available data the judge regime is absent,
  the witness's success on tracked pixels is uninteresting, and everything rests on transfer.
  The judge regime needs dense video with exact depth, which has to be obtained.
- **Calibration on abstaining pixels is not identified without labels.** Ranking may transfer;
  the fault rate does not.
- **Planar reflectors can yield confident wrong labels**, and block 3 may be unable to show it.
- **The flat water plane favours interpolation** over any prior; conclusions about the value of
  the prior must come from non-planar abstaining regions, which may be few in nadir views.
- **Tracker precision at full resolution is unmeasured**, as are memory and time of every
  external model on 8 GB.
- **Thresholds are conventions**, fixed in advance to prevent reinterpretation, not derived from
  a cost model.
