# Judging a Monocular Depth Prior by Motion Parallax: a Self-Supervised Fault Indicator on Frozen Features

Supporting derivations are in `docs/learned-indicator/`: the statistical theory (`theory.md`), the
definition of the evidence (`cues.md`), the surfel-space formulation (`surfel.md`), objections and
prior work (`objections-and-prior-work.md`), and the experiment design (`experiments.md`).

## Abstract

Depth-supervised Gaussian surfel reconstruction from aerial video trusts one monocular depth map per
frame. These maps fail regionally, and because neighbouring frames see a surface in the same way, the
failure is often repeated by most frames that observe it. A gate built on agreement between depth maps
is blind to the repeated part of a fault. The same video carries a second, independent measurement
that no depth map shares: with known poses, the motion of a tracked point is a depth measurement
whose error bar follows from its parallax. We use it as a witness against the prior. A track supplies
one number that depends on depth and many that do not; the latter calibrate the witness on the scene
itself and tell when it must abstain. Where the witness is informative, its gap to the monocular depth
is a fault label that costs nothing and does not care how many frames repeat the error. A light head
on frozen features is fitted to those labels by a likelihood that knows the noise of each label, and
carries the judgement to pixels the tracker cannot follow. We show that the witness is most precise
exactly where the prior is least needed, so the value of the indicator lies entirely in that transfer,
and we treat it as a selective-labels problem: a label-free map of where the head interpolates and
where it extrapolates, an output state for "unsupported", parallax evidence carried into untextured
planar regions from their trackable boundaries, and depth consensus as an examiner of the transferred
head. One quantity measures the premise without ground truth: the share of a monocular error that the
other frames repeat. A cheap test with thresholds fixed in advance decides whether the approach has
anything to add. Nothing in this document has been measured.

## 1. Introduction

A static scene is observed by a posed aerial video flown in strips. An off-the-shelf monocular model
gives one depth map per frame, and a surface of 2D Gaussian surfels is fitted to the depth maps. Most
depth pixels are accurate up to a per-frame scale and offset. A minority, grouped in regions that
follow appearance (water, glass, texture-free ground), are wrong by many standard deviations.

The established defence is consistency: a frame is compared with the surface that the other frames
support, and a large residual marks a fault. We call every method of this kind a *consensus gate*. It
has a blind spot that no refinement removes. If the frames that observe a region share the error,
their residuals against each other are small, and the error is fused into the surface. In reliability
theory this is the statement that a bias common to all redundant observations has no minimal
detectable size; in satellite navigation a fault common to a constellation is covered only by a
second, independent one.

The second constellation is already in the data. The input is a video with poses, so every scene point
that can be followed through the frames has a depth fixed by its image motion. This measurement shares
nothing with the monocular model: not the network, not the per-frame alignment, not the errors that
recur from frame to frame. It can therefore see a fault that every depth map repeats.

Using it raises an objection that shapes the whole design: *if motion parallax measures depth well
enough to judge the monocular prior, why keep the prior?* The answer is not that triangulation is too
noisy. For a nadir strip with normal overlap, a tracked point is one to two orders of magnitude more
precise than monocular depth (Section 4.4). The prior is kept for coverage: it is the only dense depth
on surfaces the tracker cannot follow, and those surfaces are also where monocular models fail most.
So the witness is strongest where the prior matters least, and the work an indicator can do is to
learn, where the witness speaks, when to distrust the prior, and to apply that where it is silent.

**Claim.** Motion parallax from the same video supplies fault labels for a monocular depth prior that
are free, carry a known error, and are not blinded by repetition across frames. A head on frozen
features trained on them detects prior faults on surfaces without parallax evidence better than
surface type alone predicts, and gating on it improves the surface there, where a consensus gate, a
robust loss and reconstruction without the prior do not.

**Scope of the learning.** Self-supervised, on frozen models only: a frozen point tracker or matcher,
a frozen image (or video) encoder, and the frozen monocular depth model. The learned parts are a light
head on frozen features and a few scale parameters per appearance stratum. No fault labels are needed
for training. A small labelled set from a scene with exact depth is optional and has one role: to test
the assumptions and to calibrate the indicator where the witness never speaks.

**Contributions.**

1. A parallax witness for a depth prior that validates itself: of the $2L$ numbers of a track of
   length $L$, one carries depth and $2L - 1$ are free of both the depth and the prior's error, and
   these calibrate the witness per scene and decide abstention (Sections 3.2 and 3.3).
2. The repetition coefficient: the share of a monocular error that the surface fused from the other
   frames repeats, estimable on real data without ground truth as a function of the left-out block
   (Section 4.3). It turns the blind spot of consensus gates from an argument into a measurement.
3. The harm-detectability relation: for a fault of given effect on the surface, detectability by
   parallax grows with the share of information that parallax already supplies. Direct testing pays
   only where the two sources are comparably informative, and evaluation must be weighted by harm
   (Section 4.4).
4. Transfer from tracked to untracked surface as a selective-labels problem, with a label-free map of
   interpolation and extrapolation, a third output state, anchored planes that carry parallax into
   untextured regions, and depth consensus as the examiner (Section 4.6).
5. A decision protocol with thresholds fixed in advance, including the outcome that no indicator can
   make the prior worth keeping (Section 5).

## 2. Related Work

**Fault detection by redundancy.** Baarda's data snooping and solution separation in receiver
autonomous integrity monitoring test an observation against the solution from the others, and both
literatures state the limit we start from: common-mode faults need independent redundancy.

**Monocular depth with multi-view geometry.** Scale-aligned monocular depth regularises neural
surfaces and splatting; structure-from-motion systems have begun to use monocular priors with
uncertainties; recent work weights or masks depth supervision in splatting by photometric or
cross-view checks. Multi-view stereo has long filtered depth maps by geometric consistency. Checking a
monocular depth against triangulated points is therefore not new in itself. What we add is the
treatment of the check as a witness with its own identified error law, the measurement of what it
cannot reach, and the transfer to that region.

**Self-supervised confidence.** Confidence for stereo matching has been trained from labels generated
by consistency checks, without ground truth. The head here is of that family; the differences are the
label model with a per-instance known noise and the explicit treatment of selection.

**Masks and uncertainty in radiance fields and splatting.** Heads on frozen features supervised by
the rendering residual predict distractor masks. Their supervision is a residual against a model the
masked view helped to fit; ours is an independent measurement.

**Frozen trackers, matchers and multi-view models.** Learned point trackers, dense matchers and
feed-forward multi-view geometry models provide correspondences or depth without training on the
scene. They are used here frozen, as instruments, and the last group also as baselines that need no
monocular prior.

**Learning under selection and from noisy labels.** Selective labels, propensity weighting and
forward-corrected losses for known label noise are standard; we use them, we do not extend them.

## 3. Background

### 3.1 Surface, messages and depth consensus

With $d_i(u)$ the true depth of pixel $u$ in frame $i$, the monocular depth is modelled as

```math
\hat d_i(u) = a_i\bigl(d_i(u) + \beta\, d_i(u)^2\bigr) + b_i + \varepsilon_i(u),
```

with per-frame scale and offset $(a_i, b_i)$, a shared distance bias $\beta$, and noise of nominal
standard deviation $\sigma_i(u)$. A *fault* is a connected region of a frame whose depth deviates from
this model by many $\sigma$.

The rendered depth of a pixel is $\hat D_i(u) = \sum_k w_{uk} d_k$ with normalised compositing weights
$w_{uk}$ and depth Jacobians $\mathbf j_{uk}$. With a pixel weight map $\omega_i(u) \in [0, 1]$, frame
$i$ sends surfel $k$ the message

```math
\Lambda_{ik}[\omega_i] = \sum_u \omega_i(u)\, w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta_{ik}[\omega_i] = \sum_u \omega_i(u)\, w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e_i(u)}{\sigma_i^2(u)} .
```

The information uses $w_{uk}$ and not $w_{uk}^2$: by Jensen's inequality the lumped block-diagonal
matrix majorises the Hessian of the frame's loss, so a step $\Lambda^{-1}\boldsymbol\eta$ cannot
overshoot. For a tested frame $j$ and a block $\mathcal B_j$ of $B$ neighbouring frames, subtracting
the block's messages and rendering the result into frame $j$ gives the *cavity* depth $\hat D_{-j}$.
The **consensus gap** is the aligned monocular depth minus the cavity depth. It is the evidence of a
consensus gate.

### 3.2 The parallax gap

All quantities are in inverse depth $\xi = 1/d$ along the ray of the reference pixel, in which image
motion is linear. Let $\hat\xi_M$ be the aligned monocular value and $\epsilon_M = \hat\xi_M - \xi$ its
error. A point at inverse depth $\xi$ on the ray of pixel $u$ of frame $j$ projects into frame $i$ at
$\mathbf p_i(\xi)$, which traces the epipolar line of $u$. A frozen tracker reports
$\mathbf q_i = \mathbf p_i(\xi) + \boldsymbol\epsilon_i$ in the $L$ frames where it follows the point.
With the parallax gain $\mathbf b_i = \partial \mathbf p_i / \partial \xi$ at $\hat\xi_M$, its length
$b_i$ (focal length times the baseline across the ray, for lateral motion), and unit vectors
$\hat{\mathbf l}_i$ along and $\hat{\mathbf n}_i$ across the epipolar line, each frame gives one scalar
that depends on depth and one that does not:

```math
e_i = \hat{\mathbf l}_i^\top \bigl( \mathbf q_i - \mathbf p_i(\hat\xi_M) \bigr) = - b_i\, \epsilon_M + \epsilon_i^{\parallel}, \qquad
o_i = \hat{\mathbf n}_i^\top \bigl( \mathbf q_i - \mathbf p_i(\hat\xi_M) \bigr) = \epsilon_i^{\perp} .
```

Weighted least squares on the along-line residuals gives the **parallax gap** and its information,

```math
r_T = - \frac{\sum_i b_i\, e_i / \sigma_i^2}{\Lambda_T} = \epsilon_M - \epsilon_T, \qquad
\Lambda_T = \sum_{i} \frac{b_i^2}{\sigma_i^2}, \qquad \operatorname{Var}(\epsilon_T) = \Lambda_T^{-1} ,
```

with $\sigma_i$ the tracker noise in pixels. The gap is the mean displacement of the track from where
the monocular depth says it should be, in units of parallax; no triangulated point is formed and no
track is stored (five running sums per pixel). The error bar is per pixel and comes from geometry: in
depth units it is tracker noise in ground units times the ratio of flying height to effective
baseline. For independent tracker errors the information grows with the cube of track length; if the
tracker drifts, with the square. Which holds is a property of the tracker to be measured.

### 3.3 What a track knows about itself

A track of $L$ frames supplies $2L$ scalars. One combination carries the depth. The others are, under
a static rigid scene, free of the depth and of the monocular error:

```math
A_\perp = \sum_{i} \frac{o_i^2}{\sigma_i^2} \sim \chi^2_{L}, \qquad
A_\parallel = \sum_{i} \frac{(e_i + b_i\, r_T)^2}{\sigma_i^2} \sim \chi^2_{L-1} .
```

Depth consensus has no such statistics: each of its residuals mixes the error of the judged map with
the errors of the maps it is compared with. They serve three purposes without using the monocular
depth. *Abstention:* a track whose statistics exceed their quantile has left the static rigid model
(a mover, an occlusion, a rippling reflection, a tracker failure). *Noise scale:* pooled per appearance
stratum they estimate the tracker noise, so $\Lambda_T$ is calibrated on the scene. *Anisotropy:* in a
strip all epipolar lines are nearly parallel to the flight direction, and the ratio of the two
statistics tests whether the tracker is less certain along that direction, which is the component the
depth depends on. A further diagnostic separates measured from interpolated motion: on weak texture a
learned tracker fills in motion from its surroundings, and such tracks are consistent across frames;
the share of a track's information that comes from image evidence at the point itself is reported and
low-evidence tracks abstain.

**What the self-check cannot see.** A track error proportional to the parallax gain shifts $r_T$ and
leaves both statistics untouched. It is to the witness what a repeated fault is to consensus, and its
sources are specific.

- *Planar reflections.* The mirror image of a static scene in still water or flat glass is a static
  rigid scene. Its tracks satisfy the epipolar constraint and triangulate consistently, behind the
  reflecting surface. If the monocular depth places the surface correctly, the witness accuses a
  correct prior.
- *Motion along the strip at constant speed.* Traffic on a road parallel to the flight line produces
  a displacement proportional to the baseline: a wrong depth that fits perfectly.
- *Borrowed parallax.* At an occlusion edge, or inside a uniform patch, the tracker carries the
  parallax of a neighbouring surface.

Remedies that need no labels are partial: a cross strip changes the epipolar direction and the time
gap, which exposes movers and borrowed parallax but not reflections; a virtual image lies behind the
surface, so on reflective strata the fault model discounts that sign; and the anchored planes of
Section 4.6 identify tracks that pierce a surface defined by its boundary. Pixels where a conflict
cannot be attributed are labelled unknown, not faulty.

**Frame spacing.** A survey flight with 80% forward overlap moves the image by about a fifth of its
width between frames. That is the regime of wide-baseline matchers, not of point trackers built for
small motion. Which instrument applies is set by the frame rate of the input, and both are covered by
the formulation, since it uses only positions and their noise.

## 4. Method

### 4.1 Overview

For a pixel $u$ of frame $j$ let $f \in \{0, 1\}$ be the latent fault state. The indicator is its
posterior given frozen features $\boldsymbol\phi$, the parallax gap and the consensus gap:

```math
\operatorname{logit} \zeta_j(u) = \underbrace{\operatorname{logit} \pi(\boldsymbol\phi)}_{\text{head on frozen features}}
+ \underbrace{\log \frac{g^{1}(r_T \mid s, \Lambda_T)}{g^{0}(r_T \mid s, \Lambda_T)}}_{\text{parallax, where a track passes its checks}}
+ \underbrace{\log \frac{h^{1}(r_C \mid s, \varrho)}{h^{0}(r_C \mid s, \varrho)}}_{\text{consensus, where the cavity is defined}} .
```

Here $s = s(\boldsymbol\phi)$ is an appearance stratum (a cluster of frozen features), $g^0$ is the
inlier law of the monocular error convolved with the known witness noise, $g^1$ the corresponding
fault law, and $\varrho$ the repetition coefficient of Section 4.3. As $\Lambda_T \to 0$ the parallax
term tends to zero and $\zeta \to \pi(\boldsymbol\phi)$: an abstaining pixel is the limit of an
uninformative one, and there is no hard boundary between tested and untested pixels.

The three terms have different jobs. Parallax decides where it speaks. The head carries the decision
to where it does not. Consensus needs no texture, so it remains as evidence on untracked pixels for
the part of a fault that is not repeated, and as the examiner of the head (Section 4.6).

### 4.2 Self-supervised fit

The head $\pi_\theta$ and the inlier spread $\varsigma_s$ per stratum are fitted by maximising the
marginal likelihood of the gaps,

```math
\mathcal J(\theta, \varsigma) = - \sum_{n} \omega\, \log \Bigl[ \bigl(1 - \pi_\theta(\boldsymbol\phi_n)\bigr)\, g^{0}(r_{T,n} \mid s_n, \Lambda_{T,n}) + \pi_\theta(\boldsymbol\phi_n)\, g^{1}(r_{T,n} \mid s_n, \Lambda_{T,n}) \Bigr] ,
```

with $\omega$ a deflation for spatial correlation. The gradient with respect to the head's logit is
$\zeta_n - \pi_n$: the head is regressed onto the posterior.

- *Labels with known noise.* A thresholded gap is a label whose miss and false-alarm rates depend on
  $\Lambda_T$ through $g^0$ and $g^1$. Since those rates are known per pixel, the marginal likelihood
  is the consistent objective and no clean subset is selected by hand.
- *Soft selection.* A pixel contributes in proportion to its label information

  ```math
  \iota_n = \frac{\mathbb E\,(\zeta_n - \pi_n)^2}{\pi_n (1 - \pi_n)} \in [0, 1] ,
  ```

  zero for an uninformative witness and one for a perfect one. No threshold on parallax is needed.
- *The head cannot copy its target.* It receives frozen features and never the gap. Copying would
  require the features to encode the noise of the witness, which is excluded when the encoder does
  not share the tracker's computation. For video or multi-view encoders this must be re-examined
  (Section 4.7). The head for a ground tile is fitted on other tiles, since all frames of a tile
  share its faults.

### 4.3 What is identified without labels

Splitting a track into its forward and backward halves gives two witnesses with independent errors;
with the monocular value they are three measurements of one quantity with mutually independent
errors, which identifies the error law of each (the practical form is a deconvolution of the gap with
the known witness kernel). The assumptions are a static rigid scene on accepted tracks, tracker errors
independent between the halves, and independence of tracker error and monocular error given the
stratum.

| Quantity | Identified from |
| --- | --- |
| tracker noise and its anisotropy, per stratum | $A_\perp$, $A_\parallel$ |
| witness error variance | difference of the half-track gaps |
| monocular error law and fault prevalence, per stratum | the gaps, with the known kernel |
| repetition coefficient $\varrho(B)$ | covariance of consensus gap and parallax gap |
| fault posterior on tracked pixels | Bayes' rule with the above |
| witness errors proportional to parallax (Section 3.3) | not identified within a strip |
| anything on appearances that are never tracked | not identified |

**The repetition coefficient.** Let $\epsilon_C$ be the error of the cavity depth and define, by
projection, $\epsilon_C = \varrho\, \epsilon_M + \eta_C$ with $\eta_C \perp \epsilon_M$. Then
$\varrho = 0$ means that the other frames do not share the error of frame $j$, and $\varrho = 1$ that
the surface fused from them carries the same error, so that consensus sees nothing: the consensus gap
is $r_C = (1 - \varrho)\, \epsilon_M - \eta_C$. With the witness error independent of both,

```math
1 - \hat\varrho = \frac{\widehat{\operatorname{Cov}}(r_C, r_T)}{\widehat{\operatorname{Var}}(r_T) - \overline{\Lambda_T^{-1}}} ,
```

an errors-in-variables slope with known error variance; for one region with a significant mean
parallax gap it is the ratio of the mean consensus gap to the mean parallax gap. A curve
$\hat\varrho(B)$ that falls to zero once the left-out block exceeds a few frames says that repetition
is local and a longer block cures it. A plateau above zero is the fault that every frame repeats, and
its height is the fraction of the error that no consensus method can see. This is measurable on real
video without reference depth, and on a scene with exact depth the same curve from true errors checks
the estimator.

### 4.4 Why the prior is kept: harm and detectability

Let $\vartheta = \Lambda_T / \Lambda_M$ be the ratio of parallax information to the information of the
monocular depth at a pixel, and $\varsigma = \vartheta / (1 + \vartheta)$ the parallax share. If both
are fused, a fault of size $\Delta$ in the prior moves the surface by its *harm*
$h = \Delta\,(1 - \varsigma)$. The noncentrality of the comparison between the two is
$\Delta^2 \Lambda_M\, \varsigma$. Eliminating $\Delta$,

```math
\delta^2 = h^2\, \Lambda_M\, \frac{\varsigma}{(1 - \varsigma)^2} :
```

for a fault of given harm, detectability by parallax rises steeply with the parallax share and
vanishes with it. For small faults the expected gain of direct testing is proportional to
$\vartheta / (1 + \vartheta)^3$, largest at $\vartheta = \tfrac12$; for a fault of ten inlier standard
deviations the useful band is roughly $0.2 \le \vartheta \le 3$.

| Regime | Where | Witness | Prior |
| --- | --- | --- | --- |
| replace, $\vartheta \gg 1$ | tracked, diffuse, good baseline | supplies labels at almost no noise | nearly irrelevant once tracks are fused |
| judge, $\vartheta \approx 1$ | short tracks, far range, strip ends, low overlap, dense video at small baseline | tests the prior directly | matters and can be tested |
| blind, $\vartheta \ll 1$ or no track | texture-free, reflective, occluded, single coverage | none | the only depth evidence |

With assumed survey numbers (5 cm ground sampling, one pixel of tracker noise, five frames, monocular
inlier error of one percent of the flying height) $\vartheta$ is of the order of 100. The typical
tracked pixel is in the first row. Three consequences follow.

1. The contribution of an indicator is in the third row, by transfer, and in the second row, by direct
   test. Detection quality on the first row measures the regime in which the answer does not matter.
2. Evaluation is weighted by harm: each faulty pixel counts with $1 - \varsigma$ of its surfel, or the
   three regimes are reported separately.
3. Whether tracks also enter the surface fit is a separate decision. As data they are optimal in the
   first two rows; as a judge only, a false witness costs a gated prior and not a wrong surface. The
   position taken here: fuse the witness where its checks pass and the stratum is not reflective,
   judge everywhere, and split tracks by ground tile into a fitting set and a judging set so that the
   judge is not part of the judged surface.

### 4.5 Frozen features and the head

Let $\boldsymbol\phi_j(u)$ be frozen patch features upsampled to working resolution, from a
self-supervised image encoder and from the encoder of the depth model, whose forward pass is run
anyway. They define the strata $s$, and they are the input of the head
$\pi_\theta(\boldsymbol\phi) = \operatorname{sigm}(\mathbf w^\top \boldsymbol\phi + w_0)$ or a two-layer
variant, with the distance to the nearest edge of the monocular depth map as a covariate, because
short tracks cluster at depth edges where monocular depth is also worst.

Features can be lifted onto surfels with the compositing weights, which gives a per-surfel feature
fused over frames and a per-(frame, surfel) feature; disagreement between the two is appearance that
changes with viewpoint, a cue for reflective and transparent material that costs no labels.

### 4.6 Transfer to surface without parallax

The head is trained where $\iota$ is large and used where it is small. This is learning from
selectively observed labels, and selection is on trackability, which the features also encode.

**Conditions.** Let $o$ mark informative pixels and $e(\boldsymbol\phi) = \Pr(o = 1 \mid \boldsymbol\phi)$
be the propensity. What is learned on $o = 1$ holds on $o = 0$ if (S1) among pixels that look alike to
the encoder, those the tracker follows are faulty as often as those it does not, and (S2)
$e(\boldsymbol\phi) > 0$ where the head is applied. Under both, the risk on untracked pixels is an
importance-weighted risk on tracked ones.

**A label-free map of support.** The propensity is estimated by a second head on the same features,
trained to predict $o$, without any fault label. It splits untracked pixels into an *interpolation
zone* (appearances that are sometimes tracked) and an *extrapolation zone* (appearances essentially
never tracked). The effective sample size of the importance weights states how many tracked pixels
stand behind a prediction. Weakly textured roofs and asphalt are expected in the first zone; open
water and large glass in the second.

**Three output states.** The indicator reports *trusted*, *faulty* or *unsupported*. In the
extrapolation zone the head's output is fixed by the geometry of the feature space and by nothing in
the training signal, and it is reported as unsupported rather than as a probability.

**Anchored planes.** Water, glazing and flat roofs are untextured inside and bounded by structure
that tracks well: shoreline, frames and mullions, parapets. If the interior is planar, the boundary
tracks determine it. In information form on the plane parameters, with anchors $t$ of precision
$\lambda_t$ along the normal at plane coordinates $\mathbf y_t$ and $\mathbf h_t = [1, -\mathbf y_t^\top]$,

```math
\Lambda_S = \sum_{t} \lambda_t\, \mathbf h_t^\top \mathbf h_t, \qquad
v_S(\mathbf y) = [\,1,\ -\mathbf y^\top] \Lambda_S^{-1} [\,1,\ -\mathbf y^\top]^\top + \varsigma^2_{\mathrm{pl}} ,
```

where $\varsigma_{\mathrm{pl}}$ allows for departure from planarity. For anchors spread evenly around
the region the predictive variance inside their hull is at most three times that at the centroid; for
water the plane is horizontal and one parameter, the level, remains. The monocular depth of the
interior is then judged against the plane with variance $v_S$. This moves planar regions with
trackable boundaries from the blind regime to the judge regime, gives labels on exactly the
appearances that otherwise fall in the extrapolation zone, and identifies virtual images: an interior
track that triangulates far behind the anchored plane is a reflection or a view through glass, and is
removed as a judge and recorded as a material attribute. The assumption is planarity of the interior,
tested only at the boundary and by the shape of the monocular surface itself; a non-planar region
abstains. Planar priors are standard in multi-view stereo; the use here is as a tested judge.

**Counterfactual abstention.** Tracked pixels can be made to play untracked ones while their label is
kept: withhold whole feature clusters or the lowest-propensity decile from training; shorten the
window until low-parallax pixels lose their conditioning; blur the frames until the tracker loses a
region. Each gives the accuracy of the head as a function of distance from its training support, on
any posed video and without reference depth. That this curve predicts true accuracy on untracked
surface is a hypothesis to be tested on a scene with exact depth.

**Consensus as examiner.** Parallax abstains without texture; consensus needs none and is blind only
to the repeated share of a fault. On untracked pixels the consensus gap still exists, so the rate of
consensus outliers must increase with the head's score there, and with $\varrho$ transported from
neighbouring strata the expected rate is predicted. The head is trained by the witness that sees
repeated faults and examined by the witness that works without texture. Beyond every label-free check
lies the intersection: a fault that all frames repeat on a surface that is never tracked and has no
anchored plane. There the claim rests on transfer and on labelled evaluation.

### 4.7 Frozen models

*Correspondence.* A point tracker for dense video, a dense matcher for survey spacing, and plane-sweep
patch correlation as the non-learned reference. The witness uses positions, a visibility flag and, if
available, a confidence; its noise is estimated from the ancillaries and not taken from the model.

*Features.* A per-image encoder cannot see parallax, which keeps the head independent of the witness.
A video or multi-view encoder sees parallax inside its features. That may help on weak texture, and it
may let the head reproduce the witness including its errors on reflections. It is an ablation, with
the copy check of Section 4.2 applied.

*Multi-view geometry models.* Feed-forward models that output depth for a window of frames are
baselines that need no monocular prior, and candidates for a second witness.

Existence, licences, memory needs and behaviour on aerial imagery of specific models are to be
verified before any is fixed.

### 4.8 Use in reconstruction

The indicator is a pixel field; surfels do the bookkeeping.

1. **Track messages.** A track is a depth message along its reference ray with precision $\Lambda_T$.
   It enters the lumped information blocks as a second, sparse depth map without scale, offset or
   distance bias.
2. **Anchored alignment.** The per-frame $(a_i, b_i)$ and $\beta$ are fitted against parallax instead
   of against the surface, with the indicator's weights, so no nuisance parameter is fitted to flagged
   pixels and the alignment no longer depends on the surface it helps to test.
3. **Gated fusion.** With $\omega_i(u) = 1 - \zeta_i(u)$ the monocular messages of Section 3.1 give
   the gated surface to first order in closed form, and the refit uses the same weight map in the
   depth loss.
4. **Four fits of one experiment.** Tracks only; monocular only; both fused without a gate; both with
   the gate. "The prior adds nothing" is the outcome that the first matches the last on the
   harm-weighted metric.

Unsupported regions are reported as unconstrained rather than filled from neighbours. The indicator is
held fixed while the surface is fitted.

### 4.9 Calibration

On tracked pixels the posterior is calibrated by the identified error laws and checked on held-out
tracks. On untracked pixels the head ranks; its calibration there is not identified, because nothing
observes the fault state. A threshold with a stated false-alarm rate comes from split-conformal
calibration on regions (not pixels) of a small labelled set, with covariate weighting. This is the
main use of labels.

## 5. Experiments

### 5.1 Evaluation targets

On data with exact depth: the *true-fault mask* (after the best affine alignment, relative error above
three robust standard deviations inside a connected region of at least 2% of the frame; a band around
depth discontinuities ignored); *repetition strata* by the share of observing frames that are faulty
with the same sign at a ground point (isolated below one third, common-mode from two thirds);
*surface classes*; and the *regime* of each pixel by $\vartheta$. Pixels are split into $\mathcal W$,
where the witness is informative, and $\mathcal A$, where it abstains.

### 5.2 The deciding measurement

On a synthetic aerial block with exact depth, used for scoring only, with images degraded and poses
perturbed so that matching is not flattered:

- M1. Accuracy and calibration of the witness: track error against exact correspondences, error of the
  gap against its stated error bar, coverage per surface class.
- M2. The self-check: how well the ancillaries detect bad tracks; on water and glass, the share of
  tracks that pass every check at a consistent wrong depth.
- M3. Label quality on $\mathcal W$ against the true mask, with every disagreement attributed.
- M4. Where the faults are: the share of faulty pixels, and of common-mode ones, in $\mathcal A$.
- M5. Transfer: a head trained on witness labels of the block, scored on $\mathcal A$ against the true
  mask, per surface class and per propensity decile, against a constant, local texture energy, the
  consensus gap, the per-cluster fault rate (a head that only learned the kind of surface), and a
  linear probe on the same features trained on true masks of disjoint ground tiles.
- M6. Replacement: on $\mathcal A$, depth error of the monocular prior under a perfect indicator,
  against triangulated depth interpolated into $\mathcal A$ (harmonic and piecewise planar), a
  multi-view stereo pipeline, and a feed-forward multi-view model.
- M7. The repetition coefficient $\varrho(B)$ from true errors and from the label-free estimator.

**Precondition.** Fault prevalence between 1% and 30%; at least 20 distinct fault regions mostly in
$\mathcal W$ and 20 mostly in $\mathcal A$; $\mathcal W$ covers at least 20% of land pixels.

**The approach is abandoned if any of:**

- K1, the label is unsound: precision or recall of the witness label on $\mathcal W$ below 0.90, or
  the standardised witness error far from unit scale after the label-free noise estimate.
- K2, nothing left to judge: fewer than 10% of faulty pixels lie in $\mathcal A$.
- K3, no transfer: on $\mathcal A$ the head's AUC has lower bound at or below 0.65, or it does not
  exceed the best label-free comparator by 0.05 in average precision.
- K4, a surface-type detector and nothing more: on $\mathcal A$ the per-cluster fault rate matches the
  head, and within every surface class the head is near chance.
- K5, the prior is not worth keeping: on $\mathcal A$, in every surface class with enough fault
  regions, some alternative without monocular depth is at least as accurate as the prior under a
  perfect indicator.

**It is supported if all of:** witness labels reach 0.95 precision and recall on $\mathcal W$; at
least 30% of faulty pixels and of common-mode faulty pixels lie in $\mathcal A$; the head on
$\mathcal A$ has AUC of at least 0.75 overall and at least 0.65 inside two or more surface classes,
with a true-positive rate of at least 0.30 at 5% false positives on common-mode pixels; and in at
least one surface class the prior under a perfect indicator beats every alternative of M6. The
numerical values are conventions fixed before the measurement.

A large flat water body is reproduced exactly by planar interpolation from its shoreline, so K5 is
read per class and decided on non-planar classes: facades, thin structures, shadowed and occluded
ground. Parts of the measurement need no tracker: with correspondences taken from exact depth, M4, M6
and M7 run on a CPU and already decide K2 and K5.

### 5.3 Follow-on experiments

| | Question | Hypothesis |
| --- | --- | --- |
| F1 | Which witness at which frame density | Along a ladder of frame strides, point trackers win at small motion and dense matchers at survey spacing; information grows between the square and the cube of track length |
| F2 | The judge regime | On dense video at small baseline, where triangulation is worse per pixel than the prior, the regional parallax gap still detects faults |
| F3 | Transfer | The counterfactual-abstention curve predicts true accuracy on untracked surface; anchored planes move water and glazing out of the extrapolation zone |
| F4 | Label budget | A few labelled regions fix calibration on untracked pixels; ranking does not need them |
| F5 | Repeated faults | $\hat\varrho(B)$ has a plateau above zero; on the common-mode stratum the indicator detects what the consensus gap, a robust loss and frame removal miss |
| F6 | Reconstruction | The gated fit beats tracks only, monocular only and ungated fusion on the harm-weighted metric, and changes clean scenes by less than 2% |
| F7 | Another prior | The same witness judges a different depth source without change |
| F8 | Real aerial data | The label-free quantities ($\hat\varrho$, propensity map, counterfactual curve) are stable on real video, and agree with laser depth where it exists |

### 5.4 Baselines and bounds

No gate; robust depth loss; removal of whole frames; the consensus gate; the depth model's own
confidence; disagreement between several depth models and instability under input transformations;
the rule "where the tracker abstains, distrust the prior"; the per-cluster fault rate; a linear probe
on frozen features at equal label budget; residual-supervised masks on frozen features; zero-shot
depth completion from the triangulated points; a multi-view stereo pipeline; a feed-forward multi-view
model with its own confidence. Bounds: an oracle gate with true masks, and exact correspondences in
place of the tracker.

### 5.5 Protocol

Evaluation masks, classes and strata are written to a store that only the scoring code reads.
Thresholds, strata and exclusion bands are chosen by held-out likelihood of the gaps or on labelled
scenes disjoint from evaluation, never by a detection metric on evaluation scenes. Splits are by
ground tile or scene. Intervals are cluster-bootstrap intervals over ground tiles, with the number of
distinct fault regions reported beside each.

## 6. Limitations

- **Repeated and never tracked.** A fault that all frames repeat, on a surface that is never tracked
  and has no anchored plane, is beyond every label-free check. The head's output there is an
  extrapolation and its error cannot be estimated from the video.
- **The witness has its own repeated faults.** Planar reflections, constant-speed motion along the
  strip and borrowed parallax pass its self-check. On glass and water these give confident wrong
  labels on exactly the class the transfer depends on.
- **Selection on trackability.** Trackable pixels inside an untrackable material are atypical (frames,
  dirt, reflections), so the labels of such a stratum may be drawn from the wrong population.
- **The prior may not be worth keeping.** Where triangulation, interpolation or a multi-view model
  covers the untracked surface as well as a perfectly gated prior, no indicator has a job. The
  deciding measurement includes this outcome.
- **Survey stills are not video.** At normal photogrammetric overlap the judge regime is nearly empty
  and the method reduces to labels plus transfer; the judge regime needs dense video.
- **Anchored planes assume planarity** that is tested only at the boundary.
- **Frozen encoders are trained mostly on ground-level images** and may separate materials poorly in
  nadir views.
- **Gating removes constraint where it is scarce.** Detection and repair are different problems; gated
  and unsupported regions are reported as unconstrained.

## 7. Notation

| Symbol | Meaning |
| --- | --- |
| $\hat d_i(u)$; $a_i, b_i, \beta$; $\sigma_i(u)$ | monocular depth; per-frame scale and offset, shared distance bias; nominal noise |
| $w_{uk}$, $\mathbf j_{uk}$; $\Lambda_{ik}[\omega]$, $\boldsymbol\eta_{ik}[\omega]$ | compositing weight and depth Jacobian; lumped message under pixel weights $\omega$ |
| $\mathcal B_j$, $B$; $\hat D_{-j}$ | block left out with frame $j$, its size; cavity depth |
| $\xi$, $\hat\xi_M$, $\epsilon_M$ | inverse depth, aligned monocular value, its error |
| $\mathbf q_i$, $\mathbf b_i$, $b_i$ | tracked position in frame $i$, parallax gain and its length |
| $e_i$, $o_i$ | track residual along and across the epipolar line |
| $r_T$, $\epsilon_T$, $\Lambda_T$ | parallax gap, witness error, parallax information |
| $A_\perp$, $A_\parallel$ | ancillary statistics of a track |
| $r_C$, $\epsilon_C$, $\varrho$ | consensus gap, cavity error, repetition coefficient |
| $\Lambda_M$, $\vartheta$, $\varsigma$ | information of the monocular depth, information ratio, parallax share |
| $h$, $\delta^2$ | harm of a fault, noncentrality of its test |
| $f$, $\zeta$, $\pi(\boldsymbol\phi)$ | latent fault state, indicator, head output |
| $\boldsymbol\phi$, $s$ | frozen features, appearance stratum |
| $\iota$, $e(\boldsymbol\phi)$ | label information, propensity of informative parallax |
| $\mathcal W$, $\mathcal A$ | pixels where the witness is informative, where it abstains |
| $\Lambda_S$, $v_S$ | information and predictive variance of an anchored plane |
