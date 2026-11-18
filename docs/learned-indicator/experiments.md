# Experiment Design: a Fault Indicator Combined from Several Cues on Frozen Features

This document designs the experiments for an indicator that treats several cheap signals as noisy
fault detectors, combines them with a label model whose accuracies are estimated from the agreement
pattern of the detectors, and distils the result into a light head on a frozen backbone. It starts
with a cheap test that can kill the idea before any method code is written. Notation follows the
main document (`docs/learned-indicator.md`, Section 7); new symbols are introduced in Section 0.

Status of every number below: values quoted from the repository are marked *(repo)*; everything
else is a design choice or an estimate and is marked as such. No experiment in this document has
been run.

## 0. Setting, symbols and what exists

**Unit.** A unit is a pair (view $j$, pixel $u$) on an analysis grid of $480 \times 270$ (every
fourth pixel of the $1920 \times 1080$ frame, the grid `experiments/surface_dsm.py` already uses).
The method's native unit is a (view, surfel) pair; the pixel grid is used for the kill test because
it needs no fitted surface. Section 3 moves to surfels.

**Symbols added to the main document's notation.**

| Symbol | Meaning |
| --- | --- |
| $m_0$; $m$ | the depth model whose maps are reconstructed from (primary); any other depth model |
| $q \in \{1,\dots,Q\}$ | cue index |
| $x^{(q)}_j(u) \ge 0$ | score of cue $q$; larger means more evidence for a fault |
| $\lambda^{(q)}_j(u) \in \{1, 0, \varnothing\}$ | vote of cue $q$: fault, no fault, abstain |
| $\mathrm{se}_q$, $\mathrm{sp}_q$ | $P(\lambda^{(q)}=1 \mid f=1)$ and $P(\lambda^{(q)}=0 \mid f=0)$, among non-abstaining units |
| $\pi$ | fault prevalence $P(f=1)$ |
| $\zeta^{\mathrm{LM}}_j(u)$ | posterior fault probability of the label model |
| $\varrho(X)$ | repeat fraction of a ground point $X$: share of the views observing $X$ whose depth is faulty there |
| $L$ | label budget (number of labelled views) |

**What exists in the repository and constrains the design.**

- MatrixCity `small_city` aerial block 3: 185 training views and a 19-view test split, exact
  float32 depth, loaded by `experiments/matrixcity_io.py` (`load_block`, `project`,
  `pixel_rays_world`). Median true depth is 338 m *(repo)*.
- Water is 46% of the block's ground area, a flat plane at $-3.4$ m; land starts at $+0.5$ m
  *(repo, `results/surface/summary.json`, constant `WATER_Z = -2.0` in `surface_dsm.py`)*.
- A photometric 2DGS model of the block exists (`experiments/rgb_train_2dgs.py`,
  `surface_render_rgb.py`): test PSNR 23.1 dB; its surface is within 1 m of truth on 81% of land
  cells and on 5% of water cells, median error $+5.7$ m on water *(repo)*.
- Depth cache (`sensorfault.depthcache`): four wrappers, `run.run(...)` accepts a custom list of
  `datasets.View` objects, `align.align_depth` fits a per-view scale (metric models) or an affine
  map in inverse depth (Depth Anything V2) robustly. Only four test views are cached so far. On
  those, after alignment to exact depth, the median relative error is 3-5% (Depth Pro), 9.5-14%
  (Metric3D v2), below 1% (UniDepthV2) and 1-2% (Depth Anything V2); peak memory 6.3, 4.4, 4.6 and
  2.1 GB; 0.54, 1.35, 0.22 and 0.35 s per view on a 24 GB card *(repo, `docs/depth-cache.md`)*.
  UniDepth's confidence reads as predicted error; Metric3D's changes sign between views *(repo)*.
- Experiment E1 (`experiments/e1_matrixcity_injection.py`) injected regional faults into
  *synthetic* depth noise on surfels placed on the exact geometry. It has not seen a real depth
  model. Its result that a per-view aligned RMS detects 98.1% against the gate's 100% *(repo)* shows
  that injected faults do not discriminate between detectors; the experiments below therefore rest
  on natural errors of real depth models.
- `sensorfault.faults.injectors` holds the legacy sensor-fault kinds (extrinsic, constant and
  range bias, multipath surrogate, transient); the regional injection used in E1 lives in the E1
  script. `sensorfault.sim.bunny` gives a small scene with exact masks.
- Compute: one 8 GB GPU for the method; a 24 GB card has been used for the depth models.

**Learning scope.** Every learned component is a light head on frozen features; no backbone is
fine-tuned. Training is unsupervised (label model), self-supervised (distillation from the label
model's posterior) or semi-supervised (a small labelled set next to a large unlabelled one). The
amount of labels is an experimental variable (Section 3, F2). Fully supervised training of a large
model is out of scope and appears only as an evaluation-side upper bound on cue scores.

---

## 1. The kill test (K0)

**Question.** Do at least three cues carry information about the true faults of a real depth
model, are their errors close enough to conditionally independent for a label model to estimate
their accuracies, and is there headroom above the best single cue, in particular on faults that
depth consensus cannot see?

Exact depth is used for scoring only. Nothing computed from it enters a cue, an alignment used by a
cue, a threshold of a vote, or a parameter of the label model.

### 1.1 Data and depth-model runs

Scene: block 3, the 185 training views (cache scene name `block3train`; the 19 test views share
frame indices with the training views and need their own scene name).

Runs, at full resolution, stored as float32 (float16 quantises a Depth Pro map to about 0.27 m
steps *(repo)*, which is of the order of the equivariance differences to be measured):

| Group | Models | Input transformation | Runs |
| --- | --- | --- | --- |
| Identity | Depth Pro, Metric3D v2, UniDepthV2, Depth Anything V2 | none | $4 \times 185 = 740$ |
| Flip | Depth Pro, UniDepthV2, Depth Anything V2 | horizontal flip | $3 \times 185$ |
| Rotation | same three | $90^\circ$ and $270^\circ$ (a nadir image has no canonical up) | $3 \times 370$ |
| Rescale | same three | downsample to $960 \times 540$ | $3 \times 185$ |
| Context | same three | four overlapping crops of $1152 \times 648$ (60% of each side) | $3 \times 740$ |

Total $740 + 3 \times 1480 = 5180$ forward passes. With the measured times this is
$185 \times 2.46\,\mathrm{s} + 1480 \times 1.11\,\mathrm{s} \approx 35$ minutes of GPU time plus model loading; the
times were measured on a 24 GB card and UniDepth ran a slow fallback, so one hour is the planning
figure. Depth Pro peaks at 6.3 GB including 0.5 GB of other processes *(repo)*; it should fit 8 GB
but this is not verified. Metric3D is run on the identity only: its 300 m clamp must be raised
(`max_depth_m`) because 72% of true depth is beyond it *(repo)*, and it is the slowest model.
Storage: transformed outputs are mapped back to the identity frame and kept at $480 \times 270$
float32 (0.5 MB each, about 2.3 GB); identity outputs are kept at full resolution.

Focal length passed to the wrappers: unchanged for flip, rotation and crop; halved for the
rescale. Depth Pro reads the scene as a 3-5 m miniature *(repo)*; that is a whole-view scale error,
removed by alignment and not a fault.

### 1.2 Alignment used by the cues (label-free)

Each cue needs depth in a common gauge. Two alignments, neither using exact depth:

- **Within a view, between maps of the same image** (cues c2 and c3): a robust affine fit in
  inverse depth of map A to map B over all pixels (RANSAC then least squares, the algorithm of
  `align.fit_affine_inverse`). No reference outside the image is needed.
- **Across views** (cues c1 and c4): with known metric poses the scale of a depth map is
  determined by multi-view consistency. Initialise the per-view scale from the camera height above
  a ground plane (median depth equals flying height over ground for a nadir view), then alternate:
  reproject all views outside the block $\mathcal B_j$ into view $j$, take the per-pixel median, and
  refit $(a_j, b_j)$ robustly (Huber weights, as in the main document's Section 3). Five rounds.

A control alignment uses 1% of the exact depth pixels as a stand-in for triangulated tie points
(the "run b" protocol of `docs/depth-cache.md`). It leaks two parameters per view from the
evaluation data and is reported only to bound what the label-free alignment costs.

Known weak point: with water covering 46% of the area, some views are mostly water. If a model is
wrong on water, the robust alignment of such a view can lock onto the water. Views whose two
alignments (from the largest and from the second-largest consistent pixel set) are both
self-consistent are reported as view-level ambiguous and analysed separately.

### 1.3 True-fault mask (evaluation only)

For the primary model $m_0$ and view $j$:

1. Align $\hat d_j$ to the exact depth $d_j$ with `align_depth` (median scale, or RANSAC affine for
   Depth Anything V2). This is the best-case alignment; it is an evaluation device.
2. Relative error $e_j(u) = (\hat d^{\mathrm{al}}_j(u) - d_j(u)) / d_j(u)$. Robust scale
   $\hat\sigma_j = 1.4826 \cdot \operatorname{MAD}_u\, e_j(u)$.
3. $f_j(u) = 1$ if $\lvert e_j(u)\rvert > 3\hat\sigma_j$ and $u$ lies in a connected region of
   such pixels covering at least 2% of the view (the main document's definition). $f_j(u) = 0$ if
   $\lvert e_j(u)\rvert < 2\hat\sigma_j$. Everything else (the band between 2 and 3 $\hat\sigma_j$,
   small exceedance regions, and pixels within 3 grid cells of a true depth discontinuity) is
   *ignore* and enters neither class.
4. Sensitivity variants, reported but not used for the decision: thresholds 2 and 4
   $\hat\sigma_j$; region sizes 0.5% and 5%; an absolute threshold $\lvert e\rvert > 0.03$; and
   the alignment of Section 1.2 in place of step 1. A threshold-free companion is the Spearman
   correlation between each score and $\lvert e\rvert$.

The mask is a property of ($m_0$, view, pixel). The test is run with each of Depth Pro,
UniDepthV2 and Depth Anything V2 as $m_0$; the pre-declared decision model is Depth Pro (the
plan's default). If Depth Pro's prevalence $\pi$ is outside $[0.01, 0.30]$, the decision model is
the one whose prevalence is closest to 0.05.

**Fault classes derived from the scene** (no semantic labels are needed; MatrixCity's material
or reflectance layers, if released for the aerial blocks, would refine this, which is not verified):

| Class | Definition from exact data |
| --- | --- |
| Water | true height below $-2$ m |
| Ground | horizontal normal, height within 2 m of the local ground level |
| Roof | horizontal or sloped normal, elevated |
| Facade | normal within $30^\circ$ of horizontal |
| Edge and thin | within 3 grid cells of a depth discontinuity (scored separately, outside the main mask) |
| Texture-free | local gradient energy of the image below its 10th percentile (from the image, not from depth) |

**Repetition strata.** Each faulty unit is back-projected with the exact depth to its ground point
$X$; $\varrho(X)$ is the share of views observing $X$ that are faulty there with the same sign of
error. Strata: *isolated* ($\varrho < 1/3$), *mixed*, *common-mode* ($\varrho \ge 2/3$). The
common-mode stratum is where depth consensus is blind by construction and is the stratum on which
the idea must earn its keep.

Masks, classes and strata are written to `results/k0/eval_only/` and read by the scoring script
only.

### 1.4 Cues

All scores are computed for $m_0$ on the analysis grid. Votes use a threshold that needs no
labels: each score is turned into a robust z-value with the scene-wide median and MAD of that
score, and $\lambda^{(q)} = 1$ where it exceeds 3, $0$ where it is below 2, and abstains in
between or where the cue is undefined.

- **c1, geometric consensus.** Cavity depth $\hat D_{-j}$ is the per-pixel median of the aligned
  depth maps of all views outside $\mathcal B_j$ reprojected into view $j$ (z-buffered); its spread
  gives $V_{-j}$; $x^{(1)} = \lvert z_j(u)\rvert$ with $z_j$ as in the main document. $B \in \{1, 3\}$.
  Abstains outside $\Omega_j$ (fewer than three contributing views). This is a stand-in for the
  surfel cavity of `gs/gate.py` and needs no fit; the stand-in is replaced in F0.
- **c2, cross-model disagreement.** For each other model $m$, align its map to that of $m_0$
  within the view and take $x^{(2,m)} = \lvert \log \hat d^{m_0} - \log \hat d^{m \to m_0}\rvert$.
  Three pairwise scores are kept as separate cues and also pooled as their median.
- **c3, equivariance failure.** Each transformed output of $m_0$ is mapped back, aligned to the
  identity output within the view, and $x^{(3)}$ is the standard deviation of log depth over the
  available transformations. Sub-scores per transformation family (flip and rotation, rescale,
  context) are kept. Abstains where fewer than three transformed outputs cover the pixel.
- **c4a, multi-view photo-consistency.** Using $\hat d_j$ in the cross-view gauge and the known
  poses, the four neighbours with the largest baseline (two along the strip, two across) are
  warped into view $j$; a $7 \times 7$ zero-mean normalised cross-correlation cost is evaluated at
  $\hat d_j$ and at 32 depth hypotheses around it. $x^{(4a)}$ is the cost at $\hat d_j$ minus the
  minimum over the sweep, taking the best two neighbours per pixel. The cue abstains where the
  cost profile is flat (range of the profile below a fixed fraction of its noise level), which
  covers texture-free ground and water.
- **c4b, disagreement with the photometric surface.** $x^{(4b)} = \lvert \log \hat d_j - \log D^{\mathrm{rgb}}_j\rvert$
  with $D^{\mathrm{rgb}}_j$ the median depth rendered from the existing RGB 2DGS model. Cheap
  because the model exists; a proxy for the influence of a depth message on held-out colour loss.

Reference scores that are not part of the method: UniDepthV2's and Metric3D's own confidence;
the per-view aligned RMS (view level); a constant.

Sensitivity of the photometric cues is bounded by geometry: a depth error $\delta$ at depth $d$
moves the correspondence by $f\, b\, \delta / d^2$ pixels for baseline $b$ and focal length $f$.
This is computed from `transforms.json` for every neighbour pair before c4a is interpreted; if
a $3\hat\sigma$ error moves the correspondence by less than one pixel, c4a cannot work on this
block regardless of image quality.

### 1.5 Measurement (a): how much each cue knows

For each cue, against $f$, over non-ignored units: area under the ROC curve (AUC), average
precision (AP) together with $\pi$ and the lift $\mathrm{AP}/\pi$, true-positive rate at 5%
false-positive rate, coverage (share of units where the cue does not abstain), and
$\mathrm{se}_q$, $\mathrm{sp}_q$ of the vote. Each is reported overall, per class and per
repetition stratum. Metrics for an abstaining cue are computed on its covered units and,
separately, with abstention counted as "no fault".

**Uncertainty.** Units are strongly dependent: neighbouring pixels share an error and neighbouring
views repeat it. Confidence intervals come from a cluster bootstrap over ground tiles of
$100 \times 100$ m (all units whose ground point falls in a tile are resampled together; 1000
resamples). Block 3 spans about $800 \times 700$ m, so there are about 56 tiles, half of them
water. The number of distinct fault regions in ground space is reported next to every interval;
with fewer than 20 distinct regions the intervals describe one scene and nothing more.

### 1.6 Measurement (b): dependence between cue errors given the class

The label model assumes that votes are independent given $f$. Three statistics, each with a
cluster-bootstrap interval:

1. **Conditional log odds ratio** for each pair of cues and each class,

```math
\mathrm{LOR}_{qq' \mid y} = \log \frac{P(\lambda^{(q)}{=}1, \lambda^{(q')}{=}1 \mid f{=}y)\; P(\lambda^{(q)}{=}0, \lambda^{(q')}{=}0 \mid f{=}y)}{P(\lambda^{(q)}{=}1, \lambda^{(q')}{=}0 \mid f{=}y)\; P(\lambda^{(q)}{=}0, \lambda^{(q')}{=}1 \mid f{=}y)} ,
```

   estimated from the $2 \times 2$ table of jointly non-abstaining units with 0.5 added to each
   cell. Conditional independence means $\mathrm{LOR} = 0$ in both classes. This is the primary
   statistic because it does not depend on the marginals and has a direct reading: at
   $\mathrm{LOR} = 1$ one cue erring multiplies the odds of the other erring by 2.7.
2. **Conditional Spearman correlation** of the continuous scores within each class; it catches
   dependence that the vote threshold hides.
3. **Conditional mutual information** $I(\lambda^{(q)}; \lambda^{(q')} \mid f)$ in bits, with the
   plug-in estimator on the vote table, reported as a fraction of $\min\{H(\lambda^{(q)} \mid f), H(\lambda^{(q')} \mid f)\}$.

Two further analyses decide what the dependence means:

- **Does a scene class explain it?** The same three statistics are computed within strata of a
  label-free clustering (k-means with 16 clusters on frozen patch features, Section 3 F2 names the
  backbones). If dependence is large given $f$ and small given ($f$, cluster), cues share a
  cause that the image identifies, and a label model stratified by cluster is the remedy. If it
  stays large, the remedy is a model with explicit dependence, whose structure is not identifiable
  from three cues.
- **Consequence for the label model.** The error of the label-free accuracy estimates,
  $\lvert \widehat{\mathrm{se}}_q - \mathrm{se}_q\rvert$ and $\lvert \widehat{\mathrm{sp}}_q - \mathrm{sp}_q\rvert$,
  with the right-hand values from the masks. This is what dependence costs in the end and is the
  quantity the decision uses.

What is expected before measuring, stated so that the result can contradict it: c2 and c3 share
the network $m_0$ and should be positively dependent given no fault (both fire on depth edges and
fine structure); c1 and c4b both rest on multi-view geometry and should be dependent given a fault
on water; the pair (c1, c3) should be the closest to independent.

### 1.7 Measurement (c): gain of the label-free combination

Combinations that use no labels:

| Name | Rule |
| --- | --- |
| Majority | share of non-abstaining votes that say fault |
| Mean rank | mean of the scores' scene-wide ranks |
| Max rank | maximum of the ranks |
| DS | Dawid-Skene two-class model on the votes with abstention, fitted by expectation-maximisation from 10 starts, initialised at majority vote |
| DS-binned | the same with each score cut into five quantile bins (a non-parametric conditional density per cue) |
| Triplet | method-of-moments accuracies from every cue triple, then the naive Bayes posterior |
| DS-stratified | DS-binned with accuracies and prevalence per feature cluster |

The class called "fault" in the fitted model is fixed by the convention that it is the smaller
class and that c1's sensitivity plus specificity exceeds one. Whether that convention picks the
right class is itself scored (Section 1.9, K5).

Comparators: the best single cue chosen with the masks (the strict comparator); c1 alone (the
comparator a practitioner would use without labels); the depth models' confidences.

Reported for each: AP, AUC, true-positive rate at 5% false-positive rate, the expected calibration
error of $\zeta^{\mathrm{LM}}$, overall and per stratum, and paired bootstrap differences against
the comparators.

### 1.8 Measurement (d): upper bounds (evaluation only)

Fitted with the masks, cross-fitted over ground tiles (train on four fifths of the tiles, score
on the rest, tiles with a 50 m buffer):

- **Supervised naive Bayes** on the binned scores: the label model's family with parameters set
  from labels.
- **Logistic regression** and **gradient-boosted trees** (depth 3) on the raw scores: the best
  that these cues allow.

These split the distance to the best single cue into three parts:

```math
\underbrace{\mathrm{AP}_{\mathrm{GBT}} - \mathrm{AP}_{\mathrm{best\ single}}}_{\text{headroom } H}
= \underbrace{\mathrm{AP}_{\mathrm{GBT}} - \mathrm{AP}_{\mathrm{sup.\ NB}}}_{\text{cost of assuming independence}}
+ \underbrace{\mathrm{AP}_{\mathrm{sup.\ NB}} - \mathrm{AP}_{\mathrm{DS}}}_{\text{cost of having no labels}}
+ \underbrace{\mathrm{AP}_{\mathrm{DS}} - \mathrm{AP}_{\mathrm{best\ single}}}_{\text{realised gain } G} .
```

The bounds are bounds for these cues, not for the problem: an image-only supervised probe
(Section 3, F2) can exceed them.

### 1.9 Decision thresholds, fixed in advance

All thresholds refer to the decision model, the 3-$\hat\sigma$, 2% mask, the label-free
alignment, and images degraded as in Section 2 (the "realistic" setting). An interval means the
95% cluster-bootstrap interval. The numerical values are conventions chosen before seeing data;
they have no derivation behind them and are stated so that the outcome cannot be reinterpreted.

**Precondition P (is the test informative at all).** $0.01 \le \pi \le 0.30$ for at least one
model, at least 20 distinct fault regions in ground space, and the common-mode stratum holds at
least 10% of the faulty units. If P fails, the block cannot test the idea: the result is "no
information", not "killed", and the test moves to a second block and to real data (K0-R below).

**Kill** if any of:

- **K1, no headroom.** $H < 0.05$ in AP overall *and* on the common-mode stratum (upper interval
  bound below 0.05). No combination of these cues can help.
- **K2, nothing new where consensus is blind.** On the common-mode stratum no cue other than c1
  has AUC with lower bound above 0.60.
- **K3, fewer than three usable detectors.** Fewer than three cues, counting c2's pairs as one,
  have overall AUC with lower bound above 0.60 at coverage of at least 50%.
- **K4, the label model cannot use them.** Every label-free combination, including the
  stratified one, has $G < 0$ with the interval excluding zero, while $H \ge 0.05$. This kills the
  zero-label version; the semi-supervised version is then tested in F2 before the idea is dropped.
- **K5, the model finds something else.** The label model's fault class predicts a scene class
  better than it predicts faults: AP of $\zeta^{\mathrm{LM}}$ against the water mask exceeds its AP
  against $f$ by more than 0.10.

**Support** if all of:

- **S1.** At least three cues have overall AUC $\ge 0.70$, and at least one cue other than c1 has
  AUC $\ge 0.70$ on the common-mode stratum.
- **S2.** Some triple of cues has $\lvert \mathrm{LOR}\rvert \le 1$ for all three pairs in both
  classes, unstratified or within feature clusters, and the label-free accuracy estimates of that
  triple are within 0.10 of the true values.
- **S3.** The best label-free combination has $G \ge 0.05$ in AP with the interval above zero and
  $G \ge H/2$.
- **S4.** On the common-mode stratum the combination's true-positive rate at 5% false-positive
  rate is at least 0.30 and at least twice that of c1.

Anything between is **inconclusive**: the idea continues only as the semi-supervised variant, and
only if S1 and $H \ge 0.05$ hold.

**K0-R, the same test on a real scene.** Before any method code is written, Sections 1.3 to 1.8
are repeated on two ETH3D scenes with laser depth (valid on 10-12% of pixels *(repo)*; the truth
must be warped from the distorted to the undistorted geometry, see `docs/depth-cache.md`). These
are terrestrial, not aerial, and serve one purpose: to check that the synthetic verdict on
measurements (a) and (b) is not an artefact of rendering. If the sign of the verdict differs
between MatrixCity and ETH3D, the real scene decides.

### 1.10 Cost of the kill test

| Item | Cost (estimate) |
| --- | --- |
| Depth-model runs, 5180 passes | about 1 h GPU, 2.5 GB kept |
| Frozen features for the clustering, 185 views, one backbone | minutes; under 1 GB |
| c1 (reprojection of 185 views at $480 \times 270$) | minutes, CPU |
| c4a (plane sweep, 4 neighbours, 32 hypotheses, $960 \times 540$) | under 30 min GPU, under 2 GB |
| Label models, bounds, bootstrap | minutes, CPU |
| New code | 800 to 1000 lines; three to four evenings |
| K0-R on two ETH3D scenes | about 80 views; under 30 min GPU for the same transformation set |

The 185 training images (about 1 GB for the block *(repo)*) must be on disk under
`data/matrixcity/block3train/`.

---

## 2. What a rendered scene can and cannot show

**Which real failure classes occur.**

- *Water* occurs and is large: 46% of the area. The existing photometric reconstruction fails on
  it (5% of cells within 1 m, median $+5.7$ m, normals off by $30^\circ$ median *(repo)*). So in
  this render the photometric judge is not unrealistically good on water: it is bad. Whether the
  monocular models are wrong on this water is unknown; a flat plane may be easy for them. Two
  outcomes are possible and both matter. If the models are right and the photometric cue fires,
  water is a large block of correlated false alarms of c4a, c4b and possibly c1, and K5 tests
  whether the label model then learns "water" instead of "fault". If the models are wrong on water
  in the same way in every view, water is the common-mode stratum.
  The exact depth on water is the water surface (flat at $-3.4$ m), which is the surface a survey
  wants; real laser references are sparse or absent on water, so this is the one place where water
  can be scored at all.
- *Glass and specular facades* are in doubt. The depth buffer returns the glass pane, so truth is
  well defined, but at a range of about 340 m in near-nadir views a window is far below the 2%
  region size of the fault definition, and facades are barely visible. Rendered reflections are
  also not guaranteed to move across views as real reflections do. Glass is therefore treated as
  untested on this data. The 0.5% region-size variant and the facade class report whatever is
  there; no conclusion about glass is drawn from MatrixCity.
- *Texture-free surfaces* (roads, flat roofs, water) occur and are the class the rendering
  represents best.
- *Not represented:* moving objects between frames, illumination change along a flight, haze,
  sensor noise, motion blur, rolling shutter, lens distortion residuals, pose error.

**Which cues a render flatters.**

- *c4a and c4b* benefit from noise-free images, identical exposure in all frames, static
  lighting and exact poses. On textured land they are expected to be better than they would be on
  real imagery.
- *c1* benefits from exact poses.
- *c2 and c3* depend on how the depth models treat rendered images. Several of the models were
  trained partly on synthetic data; whether MatrixCity or closely related renders were among it is
  not known and must be checked in the models' data lists. If so, errors on this block are smaller
  and less varied than on real aerial images (UniDepthV2 below 1% after alignment on four views
  *(repo)* is a warning), which makes precondition P the likely point of failure.

**Controls.**

1. *Image degradation for the photometric cues only.* Per-view gain and gamma jitter of
   $\pm 20\%$, Gaussian noise of 3/255, a Gaussian blur of 0.7 px and JPEG quality 75, applied to
   the images c4a reads and to a retrained copy of the RGB model for c4b. The depth models see
   the degraded images too in a second pass (740 identity runs more), so that c2 is not compared
   across different inputs. The decision thresholds of Section 1.9 are read in this setting; the
   clean setting is reported next to it and the difference is the size of the flattery.
2. *Pose perturbation.* Poses perturbed at the level left by a bundle adjustment (0.5 px
   reprojection), with the machinery of `experiments/h7_pose_check.py`; c1 and c4 recomputed.
3. *Sensitivity floor.* The disparity $f\, b\, \delta / d^2$ of a $3\hat\sigma$ error per
   neighbour pair, as in Section 1.4; a photometric cue that looks good below one pixel of
   disparity is reading something other than depth (most likely the image content that also
   predicts the fault) and is flagged.
4. *Image-only shortcut.* A cue can score well because it detects a *class* (water) that happens
   to be faulty, not the fault. The within-class tables of Section 1.5 show this: a cue whose AUC
   is high overall and near 0.5 inside every class is a class detector. A class detector is useful
   but transfers only if the class is faulty elsewhere too.
5. *Real replica.* K0-R of Section 1.9.
6. *A second block.* A block with less water changes prevalence and the class mix; verdicts that
   flip between blocks are reported as scene-dependent. One block tar is about 15.8 GB *(repo)*.

One property of the method limits the damage of all this: a label model re-estimates the
accuracies of its cues on every scene, so accuracies measured on a render are never carried to
real data. What can be wrong is the decision to continue, in either direction, which is why the
real replica precedes the method.

---

## 3. Follow-on experiments, in order

Each is run only if the one before it does not stop the line. "Statistical gate" is the existing
test of `gs/gate.py` with its Holm loop. All intervals are cluster-bootstrap intervals over ground
tiles or, across scenes, over scenes.

### F0. Cues on surfels

- **Question.** Do the kill-test verdicts hold when c1 comes from the leave-block-out surfel
  cavity and the unit is a (view, surfel) message with information $(\Lambda_{ik}, \eta_{ik})$?
- **Data.** Block 3, Depth Pro; a depth-supervised surfel fit.
- **Metric.** AUC and AP of each cue on messages (truth: share of the message's pixels that are
  faulty, thresholded at one half), and separately the share of faulty messages that land on
  surfels supported only by faulty views.
- **Baseline.** The pixel-level stand-in of K0.
- **Hypothesis.** Message-level AUC of every cue is within 0.03 of its pixel-level value.
  Falsified otherwise. A separate count decides whether a ray-level (see-through) cue is needed:
  if more than 20% of faulty messages sit on fault-only surfels, c1 at the surfel is blind to them
  and a free-space violation score is added as a cue.

### F1. Label model without labels versus with

- **Question.** How close is the label-free estimate of the cue accuracies to the supervised one,
  and what does the difference cost?
- **Data.** Block 3 and one more block; three depth models as $m_0$.
- **Metric.** $\lvert \widehat{\mathrm{se}}_q - \mathrm{se}_q\rvert$,
  $\lvert \widehat{\mathrm{sp}}_q - \mathrm{sp}_q\rvert$, $\lvert \hat\pi - \pi\rvert$; AP of the
  posterior; frequency of class swaps over 10 starts and over scenes.
- **Baseline.** Supervised naive Bayes; majority vote.
- **Hypothesis.** Accuracy errors at most 0.10; AP within 0.03 of supervised naive Bayes; no
  class swap in any scene under the convention of Section 1.7. Falsified by a larger gap or a
  single swap that the convention does not catch.

### F2. Label budget and frozen backbones

- **Question.** At an equal number of labelled views, does the method beat a supervised linear
  probe on frozen features, and what do a few labels buy?
- **Budgets.** $L \in \{0, 1, 5, 20\}$ labelled views and one whole labelled scene (185 views);
  10 random draws of the labelled views per budget. Labelled views come from a block or scene
  that is never evaluated (Section 4).
- **Methods at each budget.**
  1. *Reference line:* logistic regression on frozen patch features, trained on the $L$ labelled
     views only. Undefined at $L = 0$.
  2. The same probe with the cue scores appended to the features.
  3. Label model with labels as anchors: labelled units enter the likelihood with $f$ observed;
     this fixes which class is the fault and pins the accuracies.
  4. Head distilled from the label model's posterior on all unlabelled views plus a supervised
     loss on the $L$ labelled views.
  5. Head trained by pseudo-labels and consistency under the transformations of Section 1.1, no
     cues: the generic semi-supervised baseline.
  6. Logistic regression on the cue scores alone, trained on the labelled views.
  The conformal threshold of the main document is calibrated on the labelled views when $L > 0$.
- **Frozen backbones** (features computed once and cached; the head trains from the cache, so the
  method's own memory is that of a linear layer or a two-layer network):

  | Backbone | Parameters | Why | 8 GB at inference (estimate, fp16, not measured) |
  | --- | --- | --- | --- |
  | DINOv2 ViT-S/14, ViT-B/14, ViT-L/14 with registers | 21 M, 86 M, 0.3 B | general image features; size sweep | all fit; ViT-L on a $518 \times 924$ input about 2 GB, on tiles at native resolution about 4 GB |
  | DINOv2 ViT-g/14 | 1.1 B | upper end of what fits | fits on $518$-px tiles, about 5 GB |
  | DINOv3 ViT-L, including the variant pretrained on satellite imagery | 0.3 B | closest pretraining domain to nadir aerial views; existence and licence of the released weights recalled from memory, to verify | as ViT-L |
  | Encoder of the depth model $m_0$ (Depth Anything V2 ViT-L; Depth Pro; UniDepthV2) | 0.3 B and up | features that already encode what the depth model is unsure about; free, because the forward pass is run anyway | measured peak of the full models: 2.1, 6.3, 4.6 GB *(repo)* |
  | Cue scores only (no image features) | none | separates what the image adds | none |

  Larger backbones (for example a 7 B model) are excluded: their weights alone exceed 8 GB in
  half precision. Feature storage for 185 views at $37 \times 66$ patches and 1024 dimensions in
  half precision is about 0.9 GB per backbone.
- **Metric.** AP and true-positive rate at 5% false-positive rate on evaluation scenes, as curves
  over $L$; the crossover budget $L^\ast$ at which the reference line reaches the method.
- **Hypotheses.** (i) At $L \le 5$ the best of methods 3 and 4 exceeds the reference line by at
  least 0.05 AP. (ii) At $L = 0$ the method reaches at least 80% of the AP of the reference line
  at $L = 185$. (iii) Five labelled views remove every class swap and halve the accuracy errors
  of F1. (iv) Method 2 exceeds method 1 at every budget, i.e. the cues add information the
  features do not have. Falsified individually. If (i) and (iv) both fail, a linear probe with a
  handful of labels is the better method and the cue machinery is not justified; this is a second
  kill criterion and is reported as such.
- **Expected weak point.** Where faults are tied to a visible class (water), a probe on frozen
  features with five labelled views is likely to be very strong within a dataset. The probe's
  weakness is transfer from synthetic labels to real images, which F7 tests.

### F3. Violation of independence and its effect

- **Question.** How fast does the label model degrade as cue errors become dependent, and does
  stratification repair it?
- **Data.** (a) Controlled: votes simulated from the measured $\mathrm{se}_q$, $\mathrm{sp}_q$
  with a pairwise dependence set by a shared latent cause, $\mathrm{LOR}$ swept from 0 to 3, in
  one class or both. (b) Natural: cue subsets ordered by their measured dependence, including the
  deliberately redundant set of the three c2 pairs.
- **Metric.** Accuracy-estimate error and AP loss against supervised naive Bayes as functions of
  $\mathrm{LOR}$; the same for the stratified model and for a model with one extra pairwise term.
- **Baseline.** Majority vote, which assumes nothing.
- **Hypothesis.** AP loss stays below 0.03 for $\lvert\mathrm{LOR}\rvert \le 1$ and the
  stratified model keeps it below 0.03 up to the dependence measured on real cues. Falsified if
  the plain model falls below majority vote at the measured dependence and stratification does not
  recover it.
- **Not identifiable, by construction.** With three cues the dependence structure cannot be
  estimated from votes alone; with more cues it can only under assumptions on which pairs are
  dependent. The structure used is therefore fixed from how the cues are built (c2 and c3 share a
  network; c1 and c4 share geometry) and never fitted to masks.

### F4. Distilled head versus label model

- **Question.** Does a head on frozen features, trained on $\zeta^{\mathrm{LM}}$, match the label
  model where cues exist and add anything beyond it?
- **Data.** Blocks as in F1; head trained on some blocks, evaluated on others and transductively
  on the same block.
- **Metric.** AP on testable units; AP with one cue withheld from the head's teacher and the head
  asked to predict that cue's region; calibration.
- **Baseline.** $\zeta^{\mathrm{LM}}$ itself; a head trained on c1's votes alone (the main document's
  student); the reference line of F2.
- **Hypothesis.** The head is within 0.02 AP of the label model on testable units and above the
  c1-only student by at least 0.05 AP. Falsified otherwise. A head that exceeds its teacher is
  reported with the within-class analysis of Section 2, control 4, because the likely reason is a
  class shortcut.

### F5. Transfer to untestable regions

- **Question.** Where no other view observes the scene ($O_{-j}$ below threshold; c1 and c4
  abstain), does the indicator still work?
- **Data.** Strip ends and block borders of MatrixCity blocks; a low-overlap subsample (every
  second strip, every second frame) that makes more pixels untestable while their truth is known.
- **Metric.** AP on untestable units only, for the label model restricted to c2 and c3, for the
  distilled head, and for both combined.
- **Baseline.** Depth-model confidence; the reference line of F2 at $L = 5$.
- **Hypothesis.** The head's AP on untestable units is at least 80% of its AP on testable units
  of the same views and exceeds depth-model confidence by at least 0.05. Falsified otherwise.
  The low-overlap subsample controls for the fact that naturally untestable pixels sit at image
  borders, where features differ.

### F6. Errors repeated in all views of a region (headline)

- **Question.** Does the indicator detect faults that every view of a region repeats, which no
  method built on depth consensus can see?
- **Data.** The common-mode stratum of natural errors on MatrixCity blocks (water if the models
  fail there); and, as a controlled complement, natural single-view faults transplanted: the
  depth map of a faulty region is replaced, in *all* views observing it, by a surface consistent
  across those views (the faulty depth of one view back-projected and re-rendered into the
  others). This keeps the image content of a real failure while making the error common-mode. An
  injection into depth alone cannot serve here, because c2, c3 and c4 read the image.
- **Metric.** True-positive rate at 5% false-positive rate and AP on common-mode units; the same
  per cue.
- **Baseline.** The statistical gate, c1 alone, the main document's mixture indicator on c1's residual
  (all three predicted to be at chance), depth-model confidence, and the reference line of F2.
- **Hypothesis.** The three consensus-based baselines have AUC within 0.05 of 0.5 on this
  stratum; the combined indicator reaches a true-positive rate of at least 0.30 at 5% false
  alarms. Falsified if the indicator is at chance too, or if the consensus baselines are not
  (which would mean the stratum is mis-defined).
- **Caveat on the transplant.** A transplanted common-mode error is geometrically consistent by
  construction and so is as hard as possible for c1 and c4a; a natural common-mode error can be
  slightly inconsistent between views. Both are reported; the natural one is the claim.

### F7. Use in reconstruction

- **Question.** Do the message weights $w_{ik} = 1 - \zeta_{ik}$ improve the fused surface?
- **Data.** MatrixCity blocks with Depth Pro and Depth Anything V2 depth; held-out views by
  block of frames.
- **Metric.** Held-out depth error (AbsRel, and the share of pixels within 1 m) and surface
  F-score at 1 m and 2 m against the exact surface, overall, on faulty regions, and on the
  common-mode stratum; change on fault-free regions.
- **Baselines.** No gating; the statistical gate (view removal); a robust depth loss (Huber and
  Geman-McClure); weighting by the depth model's own confidence; the main document's mixture indicator;
  an oracle with true masks. Same poses, depth maps, alignment, initialisation and schedule for
  all.
- **Hypothesis.** On faulty regions the indicator closes at least half of the F-score gap
  between no gating and the oracle and exceeds the best non-oracle baseline by an interval that
  excludes zero; on fault-free regions it changes held-out AbsRel by less than 2% relative.
  On the common-mode stratum the robust loss and the statistical gate are predicted to equal no
  gating. Falsified by failure of either part.
- **Known gap.** Removing a common-mode fault leaves a hole, since no view has correct depth
  there. The metric must separate "wrong surface removed" (precision rises) from "surface
  missing" (recall falls); both are reported, and F-score alone is not used to claim a gain there.

### F8. Real aerial data with laser depth

- **Question.** Does any of it hold on real nadir imagery?
- **Data.** UseGeo (three sets of eight strips, per-image laser depth maps; reader is a stub and
  the download is pending); H3D if images and poses turn out to be released. Zero-label fitting
  per set; the label-budget variants with labels from MatrixCity only, and with a few labelled
  real views from a designated set (Section 4).
- **Metric.** As F2 and F7, on laser-covered pixels; the false-alarm rate realised by the
  conformal threshold.
- **Baseline.** As F7, plus the reference line of F2 trained on synthetic labels and on five real
  labelled views.
- **Hypothesis.** The zero-label method on real data exceeds the probe trained on synthetic
  labels by at least 0.05 AP (the probe does not transfer; the label model does not need to), and
  is within 0.05 AP of the probe trained on five real labelled views. Falsified otherwise.
- **Limits.** Laser returns are sparse or missing on water and glass, so the classes of most
  interest are the ones the reference scores worst; missing reference is reported as "unscored",
  never as "no fault".

---

## 4. Protocol hygiene

**Three kinds of data, kept apart.**

| Kind | May contain | May be used for |
| --- | --- | --- |
| Unlabelled | images, poses, depth-model outputs, cues, frozen features, of any scene including evaluation scenes | fitting the label model and the head (transductive fitting on the evaluated scene is the intended use and is declared as such) |
| Labelled, tier S | fault masks from exact depth on designated synthetic *label blocks* | semi-supervised terms, anchors, threshold calibration, selection among a pre-declared short list |
| Labelled, tier R | fault masks from laser depth on a few views of a designated real *label set* | the same, in runs marked as using real labels |
| Evaluation | masks, classes and strata of evaluation scenes | scoring only |

**Rules.**

1. *Splits by block or scene.* A scene is either a label scene or an evaluation scene for the
   whole study. For MatrixCity: block 3 is an evaluation block for the zero-label experiments and
   the kill test; label blocks are other blocks. For UseGeo: one of the three sets is the label
   set, the other two are evaluation sets; within the label set the labelled views come from one
   strip. Where label and evaluation areas could overlap on the ground, a buffer of one image
   footprint separates them, so no ground point is seen by both a labelled and an evaluated view.
2. *Evaluation masks live elsewhere.* They are written by one script to an `eval_only/`
   directory; fitting scripts take no path into it; the scoring script is the only reader and
   writes only aggregate tables. A unit test asserts that the fitting modules do not import the
   truth loader.
3. *What may be tuned on what.* Without labels: vote thresholds (robust z of the cue's own
   distribution), number of bins and clusters, head capacity and stopping are selected by held-out
   likelihood of the votes on views not used for fitting, or by agreement between label models
   fitted on disjoint cue subsets. With labels: only on the label tier in force. Never on an
   evaluation metric. The list of cues, the dependence structure of F3 and the class convention
   are fixed before any mask is read.
4. *The kill test is the one exception and is contained.* It reads masks of block 3 to decide
   whether to build the method. After it, block 3 is spent as an untouched test block for
   everything the kill test could have influenced: the choice of cues and thresholds. Later
   headline numbers therefore come from other blocks and from real data; block 3 numbers are
   reported as development results.
5. *Alignment.* No cue uses an alignment fitted to exact or laser depth. The evaluation
   alignment of Section 1.3 is used to define truth only.
6. *Frozen means frozen.* Backbones and depth models are used with released weights; nothing is
   fine-tuned; the head sees cached features. Whether an evaluation dataset was in a backbone's or
   a depth model's training data is checked and stated; where it cannot be determined it is
   stated as unknown.
7. *Label budgets.* Labelled views are drawn by a seeded rule before any result is seen; every
   budget is run with 10 draws and reported as median and range. The reference probe and the
   method receive the same labelled views.
8. *Registration.* Thresholds of Section 1.9, the hypotheses of Section 3 and the scene roles are
   committed and tagged (as `prereg/TAGGING.md` prescribes) before the scoring script is first run.
9. *Seeds and intervals.* Every stochastic step is seeded; intervals are over ground tiles within
   a scene and over scenes across scenes, never over pixels.

---

## 5. Implementation plan for the kill test

New scripts go under `experiments/` with the prefix `k0_`, a config
`configs/experiments/k0_kill_test.yaml`, outputs under `results/k0/`. Existing modules are used
as they are; nothing in `src/` needs to change for the kill test.

| # | Script | Inputs | Outputs | Reuses |
| --- | --- | --- | --- | --- |
| 1 | `k0_cache.py` | block 3 training images and `transforms.json`; list of models and transformations | depth cache under one scene name per transformation (`block3train`, `block3train.hflip`, `.rot90`, `.rot270`, `.half`, `.crop0` to `.crop3`), float32; a JSON with the inverse mapping of each transformation | `depthcache.run.run(..., views=...)` with `depthcache.datasets.View` objects whose `load_rgb` applies the transformation and whose `focal_px` is adjusted; `depthcache.datasets.matrixcity_views`; `ModelCache` for resume. Scene names must match `[A-Za-z0-9._-]+`. `model_options: {metric3d_v2: {max_depth_m: 1000}}` |
| 2 | `k0_align.py` | identity cache; poses | per-view $(a_j, b_j)$ for each model in the cross-view gauge; within-view maps between models and between transformations; list of ambiguous views | `depthcache.load_depth`, `align.fit_scale`, `align.fit_affine_inverse` (called with another map as reference); `matrixcity_io.load_block` for poses only (`project`, `pixel_rays_world`) |
| 3 | `k0_cue_consensus.py` | aligned identity maps of $m_0$; poses; block size $B$ | `c1.npz` per view: $\hat D_{-j}$, $V_{-j}$, support count, $z_j$, abstain mask | `matrixcity_io.project`; strip keys from `load_block` to form $\mathcal B_j$; reprojection logic of `baselines/_common.reproject_to_ref` and `baselines/median_fusion.py` |
| 4 | `k0_cue_models.py` | identity maps of all models; within-view maps | `c2.npz` per view: one score per model pair and the pooled score | step 2 outputs |
| 5 | `k0_cue_equiv.py` | transformed caches of $m_0$; inverse mappings | `c3.npz` per view: pooled score, per-family scores, coverage count | step 1 mapping file; step 2 |
| 6 | `k0_cue_photo.py` | images (clean and degraded); aligned $\hat d_j$; poses; rendered depth of the RGB model | `c4a.npz` (cost difference, flatness, abstain mask), `c4b.npz`; a table of disparity per $\hat\sigma$ for each neighbour pair | half-resolution images of `rgb_prepare.py`; `surface_render_rgb.py` for $D^{\mathrm{rgb}}_j$; torch `grid_sample` for the sweep |
| 7 | `k0_features.py` | images; backbone name | cached patch features (half precision) and a k-means assignment per pixel of the analysis grid | none in the repository; one frozen backbone |
| 8 | `k0_label_model.py` | `c1` to `c4` files; cluster assignment | `zeta_<combination>.npz` per view; fitted $\widehat{\mathrm{se}}_q$, $\widehat{\mathrm{sp}}_q$, $\hat\pi$ per start; held-out vote likelihood | none; numpy only. Takes no path to `eval_only/` |
| 9 | `k0_truth.py` | exact depth; identity cache; images (for the texture class) | `results/k0/eval_only/`: $f_j$, ignore mask, classes, $\varrho$, ground-tile index, mask variants | `matrixcity_io.load_block(step=4)`; `depthcache.align.align_depth`; `WATER_Z` and the height grid of `surface_dsm.py`; `baselines/_common.depth_to_normals` |
| 10 | `k0_score.py` | outputs of steps 3 to 9 | `results/k0/summary.json` and `summary.md`: tables of Sections 1.5 to 1.8 and one line per criterion K1 to K5, S1 to S4 and P with the measured value, interval and verdict | `eval/metrics.detection_pr`; `eval/stats.bootstrap_ci` and `paired_wilcoxon`, with a cluster-resampling wrapper if the existing function resamples units; scikit-learn for the supervised bounds |
| 11 | `tests/test_k0_isolation.py` | - | fails if any of scripts 2 to 8 imports the exact-depth reader or opens `eval_only/` | - |

Order of work: 1 and 9 first, and from them the prevalence, the number of fault regions and the
size of the common-mode stratum (precondition P). If P fails on block 3, steps 2 to 8 are not
worth writing for this block and the effort goes to K0-R. Then 2, 4, 5 (cheapest cues), 3, 6, 7, 8,
10.

---

## 6. What this design cannot settle

- **The kill test can fail to be informative.** If the depth models make few regional errors on a
  rendered block, precondition P fails and nothing is learned about the idea. This is a likely
  outcome for at least one model.
- **Dependence is measured, not removed.** The test says whether the cues are close enough to
  independent on the scenes examined. It cannot say that they will be on another scene, and the
  dependence structure is not identifiable without labels. The stratified label model is a remedy
  only for dependence that a visible class explains.
- **A class detector can pass for a fault detector.** On a block where water is both half the
  scene and possibly the main fault, a high overall score may reflect "this is water". The
  within-class tables expose this but cannot turn it into evidence about other classes.
- **Glass and specular facades are not tested** by nadir views at this range, rendered or real,
  and laser references are poor on them. Claims about these classes need oblique or terrestrial
  data that the current plan does not include beyond the two ETH3D scenes of K0-R.
- **Thresholds are conventions.** The values in Section 1.9 and the hypothesis margins in
  Section 3 are fixed in advance to prevent reinterpretation; they are not derived from a cost
  model of reconstruction error.
- **Memory figures for the backbones are estimates** until measured on the 8 GB card; the Depth
  Pro encoder at 6.3 GB is the tightest.
