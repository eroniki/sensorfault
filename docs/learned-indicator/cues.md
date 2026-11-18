# Evidence Definitions: a Depth Prior Judged in a Compensated Video, with the Masklet as Unit

This note defines the evidence on which a fault indicator for monocular depth is built when the input
is a posed aerial image sequence flown in strips, there is one monocular depth map per frame, and the
models available are frozen backbones that are native to video and to dense prediction: a video
segmenter that follows regions through a clip (masklets) and a video encoder with spatio-temporal
tokens. It states, for each kind of evidence, the quantity, its nuisance, how its null is obtained,
when it abstains and how that is detected, what it responds to, and what is identified. It then
records what was checked about the candidate models, where the contribution lies, what would end the
idea, and the cheapest tests.

The note makes one construction central that the other definitions follow from: the **compensated
clip**. Each neighbouring frame is resampled into a reference geometry through the monocular depth and
the known poses. If the depth is right, the compensated clip is a still image. Everything that still
moves in it is evidence, and its motion is the depth error multiplied by a known gain. A tracker, a
video encoder and a video segmenter are three ways of reading that residual motion.

Notation follows the main document: frame $j$, pixel $u$ with ray $\mathbf x = K_j^{-1} \tilde u$,
inverse depth $\xi = 1/d$, aligned monocular value $\hat\xi_M$ with error $\epsilon_M$, projection
$\mathbf p_i(\xi)$ into frame $i$, parallax gain $\mathbf b_i = \partial \mathbf p_i / \partial \xi$ of
length $b_i$ along the unit epipolar direction $\hat{\mathbf l}_i$, parallax gap $r_T$ with information
$\Lambda_T$, ancillary statistics $A_\perp$ and $A_\parallel$, information ratio
$\vartheta = \Lambda_T / \Lambda_M$ and parallax share $\varsigma = \vartheta / (1 + \vartheta)$,
repetition coefficient $\varrho$.

**Status of the statements.** Nothing here has been measured; all statements about behaviour are
predictions. Facts about third-party models are either *checked* (read from the project page or the
paper, listed in Section 8) or marked "(recalled)". Statements about the project's own code were
checked against the source.

## 1. The Unit: a Surface Entity

### 1.1 Why the pixel in a frame is the wrong unit

A fault that every frame repeats is a property of the surface as the depth model sees it, not of a
view. Write the monocular error at surface point $X$ in frame $i$, for $X$ in a region $m$, as

```math
\epsilon_{M,i}(X) = \underbrace{\mu_m + \alpha_m(X)}_{\text{repeated by all frames}} + \underbrace{\beta_{m,i}}_{\text{per-frame shift of the region}} + \underbrace{\gamma_i(X)}_{\text{residual}} ,
```

with $\sum_X \alpha_m(X) = 0$ over the region and $\sum_i \beta_{m,i} = 0$. The first two terms do not
depend on the frame. The repetition coefficient is their share of the error variance at the block
length considered. The claim "the verdict belongs to the masklet, with per-frame deviations as a
second-order term" is the claim that $\mu_m$ dominates: faults are regional, follow objects and
materials, and repeat. That is a hypothesis about the depth model with a direct measurement, the
variance components of this decomposition on a scene with exact depth (Section 10, test 1). If
$\alpha_m$ dominates (faults cut through regions) or $\beta_{m,i}$ dominates (faults change from frame
to frame), the masklet is the wrong unit, for power in the first case and for meaning in the second.

### 1.2 What only motion sees

Let $r_i(X) = \epsilon_{M,i}(X) - \epsilon_T(X)$ be the parallax gap of frame $i$ at $X$, with
$\epsilon_T$ the error of the witness, which belongs to the surface point and is common to all frames
that are judged against the same correspondences. Then

```math
r_i(X) - r_{i'}(X) = \epsilon_{M,i}(X) - \epsilon_{M,i'}(X), \qquad
\bar r(X) = \frac{1}{L} \sum_i r_i(X) = \mu_m + \alpha_m(X) + \bar\gamma(X) - \epsilon_T(X) .
```

Differences between frames are free of the witness and contain no repeated term: they are depth
consensus, expressed on the surface. The frame mean contains the repeated term and the witness error,
and nothing else contains the repeated term. The two sources of evidence of the main document are
therefore two contrasts of one table (surface point by frame): its within-row contrasts and its row
means. Motion is needed for exactly one quantity per region, the mean field, and all correspondences in
the tube of the region can be pooled into it.

### 1.3 Region fault model

The frame-independent part is modelled per region as an affine field in inverse depth over the rays of
a reference frame,

```math
\mu_m + \alpha_m(X) \approx \boldsymbol\rho_m^\top \mathbf x, \qquad \boldsymbol\rho_m \in \mathbb R^3 .
```

This is the natural family: a plane $\mathbf n^\top X = d_\pi$ has inverse depth
$\xi(\mathbf x) = \boldsymbol\pi^\top \mathbf x$ with $\boldsymbol\pi = \mathbf n / d_\pi$, so the
difference of two planes is of this form, and a planar surface placed on a wrong plane (a lake put at
the wrong level or tilted, a glass roof pushed inwards) is a fault with exactly three parameters. A
constant shift is the special case $\boldsymbol\rho \propto$ the optical axis. Departure from the
affine field inside the region is left to a per-pixel term of the second order.

### 1.4 Where masklets come from, and what a frozen segmenter does not give

A masklet is a set of masks $\{M_{m,i}\}$ of one region in the frames where it is visible. Two points
about the available models were checked and change the construction.

- *No model returns an exhaustive partition of a video without prompts.* The video mode of the
  segmenters is driven by prompts: clicks, boxes or masks on a frame, or, in the concept-promptable
  model, a short noun phrase or an exemplar, which returns all matching instances with identities.
  Automatic mask generation exists for single images. An exhaustive set of masklets must therefore be
  built: generate masks on key frames, use each as a prompt, propagate, and merge duplicates. Cost
  grows with the number of regions; the concept-promptable model is reported near real time for about
  five concurrent objects, and an aerial frame has hundreds of regions.
- *With poses and depth, association through time is not the segmenter's job.* A mask in frame $j$ can
  be carried to any frame through the depth and the poses, or lifted onto the surfels it covers. The
  segmenter is needed for something else: it follows the region by appearance, independently of the
  depth. The disagreement between the region carried by geometry and the region followed by
  appearance is a measurement (Section 4).

The unit of the verdict is thus a pair: the appearance masklet, and the set of surfels $\mathcal K_m$
it covers. Concept prompts ("water", "glass roof", "solar panel", "road") can name strata without
labels; nothing checked says how the model behaves on materials or on nadir aerial imagery.

## 2. The Compensated Clip

### 2.1 Definition

For a reference frame $j$ and a clip of frames $i \in \mathcal C_j$, resample each frame into the
geometry of $j$ through the monocular depth of $j$,

```math
\tilde I^{(j)}_i(u) = I_i\bigl( \mathbf p_i(\hat\xi_M(u)) \bigr), \qquad i \in \mathcal C_j ,
```

with a mask where the point is occluded in $i$ or leaves the image (a depth test against the same
depth map). The sequence $(\tilde I^{(j)}_i)_i$ is the compensated clip. For a static, diffuse scene and
a correct depth every frame of it equals $I_j$ up to resampling and exposure: **the null hypothesis is
stillness.** If the depth is in error by $\epsilon_M$, the content at $u$ is displaced in frame $i$ by

```math
\Delta_i(u) = - b_i\, \epsilon_M(u)\ \hat{\mathbf l}_i + O(\epsilon_M^2),
```

along the epipolar direction, proportional to the gain of that frame. This residual motion is the
along-line residual $e_i$ of the main document, now present in the pixels of a video.

A second compensation is defined on the surface. Resampling every frame onto the fused surfel surface
(an orthographic stack per surface patch) gives a clip that is still when the *fused surface* is right.
Its residual motion measures the error of the surface that is actually built, in which a repeated
fault survives in full and a non-repeated one is averaged down. The frame-level clip judges a depth
map; the surface-level clip judges the result. Both are used: the first to gate messages, the second
to verify that gating removed the error. Checking a surface model by the consistency of images
rectified through it is established in photogrammetry (recalled); the construction is not new, its use
as the input of frozen video models is.

### 2.2 Why this is the right input for video-native models

1. **It removes the frame-spacing problem.** Survey frames at 80% forward overlap are displaced by a
   fifth of the image, far outside what video encoders, trackers and video segmenters are trained on
   (checked for the video encoder: clips of 16 to 64 frames of ordinary video, the longer ones spanning
   about 16 seconds). After compensation the motion is $b_i \epsilon_M$: zero where the depth is right,
   and a few pixels to a few tens of pixels on a fault. The compensated clip is a small-motion video
   whatever the spacing of the flight, so one construction covers dense video and survey stills.
2. **Its reference is in the training distribution.** The video encoder was trained with still images
   repeated in time as clips (checked). The null clip, $I_j$ repeated, is an ordinary input, and so is
   a clip with small residual motion.
3. **The expectation is produced by geometry, not by a learned predictor.** No frozen video model with
   a predictor conditioned on camera pose was found. The action-conditioned predictor released with
   the video encoder takes seven-dimensional end-effector displacements of a robot arm, was trained on
   about 62 hours of manipulation video from an uncalibrated camera, and is sensitive to camera
   placement (checked). It cannot predict how tokens move under a camera motion. The unconditional
   predictor "watches as an observer". The only way to obtain "what the depth and the poses predict"
   from frozen parts is the explicit warp.

The nearest classical idea is plane-plus-parallax: stabilise a sequence on a dominant plane, and the
residual image motion is parallax relative to the plane, directed along epipolar lines (recalled).
Here the reference surface is the monocular depth instead of a plane, and the residual parallax is the
error of the prior.

### 2.3 What compensation distorts

Resampling changes local scale and blurs; disocclusions put wrong content where the mask fails;
exposure differences and view-dependent appearance show as flicker without motion. These are nuisance
responses of every reader of the clip. They are handled by the finite-difference design of Section 3.2
(the same distortions are present in the perturbed clips and cancel to first order), by the occlusion
mask, and by per-frame gain normalisation.

## 3. Three Readers of the Compensated Clip

### 3.1 Explicit reader: residual tracks

A frozen point tracker or flow model is run on the compensated clip, where it works in its trained
regime. It returns displacements $\hat\Delta_i(u)$ from the reference frame. Then

```math
e_i = \hat{\mathbf l}_i^\top \hat\Delta_i, \quad o_i = \hat{\mathbf n}_i^\top \hat\Delta_i, \qquad
r_T = - \frac{\sum_i b_i\, e_i / \sigma_i^2}{\Lambda_T}, \quad \Lambda_T = \sum_i \frac{b_i^2}{\sigma_i^2},
```

```math
A_\perp = \sum_i \frac{o_i^2}{\sigma_i^2} \sim \chi^2_L, \qquad A_\parallel = \sum_i \frac{(e_i + b_i r_T)^2}{\sigma_i^2} \sim \chi^2_{L-1} ,
```

exactly the parallax gap and its ancillary statistics, with the tracker noise $\sigma_i$ estimated per
stratum from the ancillaries (across-line and along-line separately, because of the aperture effect).
Displacements are mapped back through the local Jacobian of the warp where the resampling is far from
the identity. This reader is the calibrated reference: it has an error bar from geometry and
$2L - 1$ degrees of freedom of self-check.

### 3.2 Latent reader: response of a frozen video encoder

A distance between the tokens of two clips is a number without a null, which is the defect that a
correlation score without a profile has. The latent reader is defined so that it returns the same
quantity as the explicit one, in the same units, with an error bar.

Let $\Phi(\cdot)$ be the frozen video encoder, returning one token per tubelet (checked for the
candidate: two frames by 16 by 16 pixels; at 384 pixels that is a 24 by 24 grid per pair of frames).
Let $\mathcal V(\xi)$ denote the compensated clip built with the depth field $\xi$, and
$\mathcal V_0$ the still clip. For a token $\tau$ covering a patch of the reference frame, form

```math
\mathbf g_\tau = \Phi_\tau\bigl(\mathcal V(\hat\xi_M)\bigr) - \Phi_\tau(\mathcal V_0), \qquad
\mathbf J_\tau = \frac{\Phi_\tau\bigl(\mathcal V(\hat\xi_M + \delta)\bigr) - \Phi_\tau\bigl(\mathcal V(\hat\xi_M - \delta)\bigr)}{2\delta} ,
```

where $\delta$ is a known perturbation of inverse depth, constant over the token's support (two more
forward passes). $\mathbf J_\tau$ is the measured response of the encoder to a depth change at this
place, in this clip, with this baseline. The latent gap is the one-parameter Gauss-Newton step that
brings the compensated clip to stillness,

```math
r_\Phi(\tau) = \frac{\mathbf J_\tau^\top W \mathbf g_\tau}{\mathbf J_\tau^\top W \mathbf J_\tau}, \qquad
\Lambda_\Phi(\tau) = \frac{\mathbf J_\tau^\top W \mathbf J_\tau}{s_\Phi^2}, \qquad
A_\Phi(\tau) = \frac{\lVert \mathbf g_\tau - \mathbf J_\tau r_\Phi(\tau) \rVert_W^2}{s_\Phi^2} ,
```

with $W$ a fixed metric on token space (identity after whitening over the scene) and $s_\Phi$ the
token noise, estimated from clips that the explicit reader certifies as still. This is feature-metric
alignment restricted to the one direction that geometry allows, read from a frozen video encoder.

- **Abstention is measured, not assumed.** Where the tokens do not respond to a depth perturbation
  (no baseline, no structure the encoder uses, token dominated by occlusion mask), $\mathbf J_\tau$ is
  small and $\Lambda_\Phi$ is small. This is the latent counterpart of the evidence fraction of a
  track, and it does not need a separate diagnostic.
- **Ancillaries.** The temporal token index gives replicate estimates $r_{\Phi}(\tau, t)$ of one
  scalar, because the gain $b_i$ is already inside the warp; their dispersion plays the role of
  $A_\parallel$. A second finite difference, perturbing the warp across the epipolar direction, gives
  an across-line component that must be zero for a static rigid scene: the role of $A_\perp$.
  $A_\Phi$ tests whether the token difference is explained by a depth change at all; flicker and
  resampling artefacts fall into it.
- **Linear range.** The step is valid while the residual motion is within the range in which the
  encoder's response is linear, which is unknown and is measured by a response curve: inject known
  depth perturbations on a scene with exact depth and record $r_\Phi$ against the truth. The same
  curve can be measured on any posed video around the monocular depth, without labels, since $\delta$
  is chosen by the experimenter. For large faults a coarse-to-fine schedule over $\delta$ is needed.
- **Honesty check.** On tokens where both readers are informative, $r_\Phi$ regressed on the pooled
  $r_T$ must have slope one and the residual spread predicted by $\Lambda_\Phi^{-1} + \Lambda_T^{-1}$.
  The latent reader is used beyond the reach of the explicit one only in strata where this holds at
  the boundary of that reach.

What the video encoder might add over the explicit reader is response on structure too weak or too
large-scale for a point tracker (shading gradients, texture statistics, region outlines inside the
token), and over per-frame features compared frame by frame, sensitivity to motion smaller than a
token. Neither is established. A per-image encoder applied to the same compensated frames is the
necessary ablation; if it does as well, the video encoder is not needed for the witness.

Paired comparison is how prediction error in this model family has been validated: surprise at
physically impossible events was shown by comparing two matched clips, not by an absolute threshold
(checked). The finite-difference design is paired by construction.

### 3.3 Region reader: masklet drift

The video segmenter is prompted with the mask of region $m$ in the reference frame and propagates it
through the compensated clip. Under the null the masklet does not move. Its outline in frame $i$ is
compared with the reference outline: at boundary point $u_b$ with unit normal $\mathbf n_b$ to the
contour, the signed distance $c_{b,i}$ from the reference contour to the propagated one is observed.
A contour reveals only the displacement component along its normal, so

```math
c_{b,i} = - b_i \bigl( \hat{\mathbf l}_i^\top \mathbf n_b \bigr)\, \boldsymbol\rho_m^\top \mathbf x_b + \epsilon^{c}_{b,i} ,
```

for a region whose error is the affine field of Section 1.3. Weighted least squares over boundary
points and frames gives the region gap and its information,

```math
\hat{\boldsymbol\rho}_m = - \Lambda_m^{-1} \sum_{b, i} \frac{b_i (\hat{\mathbf l}_i^\top \mathbf n_b)\, c_{b,i}}{\sigma_c^2}\, \mathbf x_b, \qquad
\Lambda_m = \sum_{b, i} \frac{b_i^2 (\hat{\mathbf l}_i^\top \mathbf n_b)^2}{\sigma_c^2}\, \mathbf x_b \mathbf x_b^\top ,
```

with the residual sum of squares $A_m \sim \chi^2_{N_{\mathrm{eff}} - 3}$ as the self-check for
planarity of the error, rigidity and consistent delineation. No interior texture is used. This is the
statement "the motion of a region fixes its plane" in estimable form, and it shows where that
statement needs qualifying.

1. **Contour segments parallel to the epipolar direction carry nothing.** Within a strip all epipolar
   lines are nearly parallel to the flight direction, so only the leading and trailing parts of an
   outline inform. A region elongated along the strip (a canal, a road) is constrained at its two ends
   only, and its tilt across the strip is weakly determined; $\Lambda_m$ shows this in its smallest
   eigenvalue. A cross strip supplies the other direction.
2. **An outline belongs to the nearer surface.** The boundary between a roof and the ground beside it
   is the edge of the roof. For the ground region the same contour is an occluding contour and moves
   with the roof's depth. Each boundary segment must be assigned to the side that owns it; segments
   owned by the neighbour are dropped for this region. Ownership is an ordinal judgement across the
   edge, taken from the sign of the depth step in the monocular map and confirmed by the direction in
   which the neighbouring texture is covered and uncovered in the uncompensated frames. A shoreline
   overhung by trees is owned by the trees; using it would place the water at canopy height.
3. **Contact boundaries are shared.** Where there is no depth step (shore at water level, a painted
   marking) both regions own the contour.
4. **Unbounded regions abstain.** A region that leaves the image or the clip on the informative sides
   has an ill-conditioned $\Lambda_m$.
5. **Boundary noise is large and partly systematic.** $\sigma_c$ is the jitter of the segmenter's
   outline under the null, measured on clips certified still. It is expected to be pixels, not
   fractions of a pixel. With survey baselines this still gives a sharp estimate, because the gain
   $b_i$ is hundreds of pixels per unit relative depth. The systematic part is the danger: an outline
   that follows a glint, a shadow that moves, or a different notion of the region's extent from frame
   to frame shifts $\hat{\boldsymbol\rho}_m$ and is caught only as far as it breaks the affine model
   ($A_m$).

Running the segmenter on the compensated clip rather than on the raw frames has two effects: it is in
its small-motion regime, and the measured drift is the gap itself rather than a homography from which
the prediction has to be subtracted.

### 3.4 Pooling the pixel readers over a masklet

Interior evidence from the explicit and the latent reader is pooled into the same three parameters,

```math
\Lambda^{\mathrm{int}}_m = \omega \sum_{u \in m} \Lambda(u)\, \mathbf x_u \mathbf x_u^\top, \qquad
\hat{\boldsymbol\rho}^{\mathrm{int}}_m = \bigl(\Lambda^{\mathrm{int}}_m\bigr)^{-1} \omega \sum_{u \in m} \Lambda(u)\, r(u)\, \mathbf x_u ,
```

with $\omega$ the deflation for spatial correlation of the reader's noise. The region statistic and
its minimum detectable fault are

```math
T_m = \hat{\boldsymbol\rho}_m^\top \Lambda_m^{\mathrm{tot}} \hat{\boldsymbol\rho}_m \sim \chi^2_3, \qquad
\delta_{\min}(m, \mathbf x) = \bigl( z_{1-\alpha/2} + z_{1-\beta_{\mathrm p}} \bigr) \sqrt{ \mathbf x^\top \bigl(\Lambda_m^{\mathrm{tot}}\bigr)^{-1} \mathbf x } ,
```

where $\Lambda^{\mathrm{tot}}_m$ adds interior and boundary information when they agree. Abstention is
graded by $\delta_{\min}$ as before, now per region. Per-frame deviations $\beta_{m,i}$ follow from the
between-frame contrasts of Section 1.2 and need no witness.

**Interior against boundary.** The two estimates of $\boldsymbol\rho_m$ use disjoint data. Their
difference, standardised, is a test with a meaning of its own: if the boundary says the region is
where the prior puts it and the interior says it is deeper, something is seen *through or in* the
surface.

## 4. The Reflection Problem at Region Level

The mirror image of a static scene in a flat reflector is a rigid static scene. Its residual motion in
the compensated clip is consistent parallax at the virtual depth; all ancillaries of the pixel readers
pass. From interior parallax alone a mirror cannot be told from an opening, and adding the boundary
does not settle it either: a courtyard has its rim at roof level and its floor below, a lake has its
shore at water level and its virtual scene below. The geometry is the same. Three things do
discriminate, in decreasing order of strength.

1. **Mirror consistency.** A virtual point is the reflection of a real point across the reflector's
   plane. With the plane $\boldsymbol\pi_m$ from the region reader, reflect each interior point that
   lies behind the plane and test whether it lands on reconstructed surface above the plane with
   matching appearance. If it does, the interior evidence is a reflection: it is withdrawn as a judge,
   the region is marked reflective, and the boundary estimate stands alone. The test needs the
   reflected object to be in the map (buildings beside the water) and fails silently for reflected sky,
   which shows up instead as residual parallax at infinity.
2. **Appearance that changes with viewpoint.** In the surface-level compensated clip a diffuse region
   is still in content as well as position; a reflective one changes content with the frame.
3. **Concept strata.** A promptable segmenter can propose "water" or "glass" regions, in which the sign
   "deeper than the boundary plane" is discounted.

Interior evidence behind the plane with no mirror counterpart and no reflective signature remains
unattributed (a real depression, or a view through clear glass onto real geometry) and the region is
labelled unknown.

## 5. What Is Identified

| Region type | Reader that identifies $\boldsymbol\rho_m$ | Regime |
| --- | --- | --- |
| Textured, diffuse | explicit and latent interior readers; boundary as a check | replace or judge, by $\vartheta$ |
| Weak texture, bounded | boundary reader; latent reader where its response is measured non-zero | judge |
| Untextured, planar, owned or contact boundary on the informative sides | boundary reader | judge |
| Planar reflector with reconstructed surroundings | boundary reader; interior withdrawn by mirror consistency | judge, with material attribute |
| Untextured and non-planar | none ($A_m$ rejects the affine model) | blind |
| Untextured and unbounded in the clip, or elongated along a single strip | none ($\Lambda_m$ ill-conditioned) | blind |
| Region delineated inconsistently by the segmenter | none | blind, and not detected if the inconsistency is affine |

Not identified by any reader: a whole-frame error in the nuisance group of the depth model; a fault on
a blind region; whether an unattributed "deeper interior" is real. The blind class is smaller than
"untextured", which is the gain of the region reader, and it is exactly characterised by two
statistics that are computed without labels ($A_m$ and the eigenvalues of $\Lambda_m$).

## 6. What the Head May See

The head is a probe on frozen tokens, pooled over the masklet tube, trained on region labels
$(T_m, \delta_{\min})$ and applied to regions where the readers abstain. Two rules follow from the
definitions.

- **Tokens of the raw clip, never of the compensated clip.** The compensated clip contains the
  monocular depth by construction; its tokens are the witness. The head sees the raw frames of the
  region (video tokens, which encode how the region moves and how its appearance changes with the
  view, and per-image tokens).
- **Copying is a label problem before it is a feature problem.** A probe cannot subtract a depth map it
  is not given, so it cannot reproduce the gap. What it can learn is any regularity in the labels,
  and false labels on reflections are a regularity with a clear signature in motion-aware tokens. The
  remedy is to correct the labels (Section 4), and to report head accuracy separately on regions whose
  interior evidence was withdrawn.

Selection remains: labelled regions are those some reader can judge, and that is decided by appearance
and shape. The propensity is now per region, and the transfer axis is $\delta_{\min}(m)$. The region
reader moves planar water and glass from the unlabelled to the labelled side, which is what makes a
test of transfer to those materials possible at all without reference depth.

## 7. Cost and Code

| Item | Exists in the code | To add |
| --- | --- | --- |
| Poses, intrinsics, projection | `types.DepthFrame`; `baselines/_common` (`project`, `project_into`, `backproject_cam`, `depth_to_normals`) | backward warp with occlusion mask; orthographic stack on surfels |
| Images | `depthcache/datasets.View.load_rgb`; image preparation scripts under `experiments/` | an image path into the frames used by `gs/`, which carry depth only |
| Alignment of monocular depth to sparse reference depth | `depthcache/align.align_depth` (robust scale; consensus fit affine in inverse depth) | call with the readers' depths as reference |
| Frozen model wrappers and caches | pattern in `depthcache/` (lazy loading, on-disk cache with manifest) | wrappers for a segmenter, a video encoder, a tracker; none is installed |
| Readers, region statistics, ownership, mirror test | nothing | array-level module |
| Between-frame contrasts | `gs/gate.ViewGateTest` forms the leave-out residual internally | export per pixel; aggregate per region |

Forward passes per reference frame: the latent reader needs three passes per tile and clip (depth and
two perturbations) plus one cached pass for the still clip, and two more for the across-line check;
with 16-pixel tokens at 384 pixels an aerial frame of several thousand pixels on a side is of the
order of a hundred tiles at native resolution. The region reader needs one propagation per region and
clip. Neither cost has been timed, and memory needs of the candidate models were not found on their
project pages.

## 8. What Was Checked about the Models

Checked against the project pages and papers:

| Model | Checked facts | Consequence |
| --- | --- | --- |
| V-JEPA 2 (video encoder) | Tubelets of 2 frames by 16 by 16 pixels; 256 pixels, one variant at 384; clips of 16 frames, 64 in the final training phase (about 16 seconds); checkpoints of about 0.3, 0.6 and 1 billion parameters; trained on internet video and on still images repeated as clips; code under MIT with Apache 2.0 parts; the page states no licence for the weights; the standard loader returns the encoder | Tokens are coarse; still clips are in distribution; no aerial data in training |
| V-JEPA 2-AC (predictor) | About 0.3 billion parameters on the frozen encoder; conditioned on 7-dimensional end-effector displacement and state; about 62 hours of robot video at 4 frames per second and 256 pixels; uncalibrated camera; errors when the camera placement changes | Not a camera-pose-conditioned predictor; unusable for the expectation |
| V-JEPA 2.1 | Adds a loss on visible tokens and supervision at intermediate layers to obtain dense, temporally consistent features; 384 pixels; about 0.08 to 2 billion parameters; dense probes reported for indoor depth, semantic segmentation and video object segmentation; training mix of curated images and ground-level video | The candidate for the latent reader and for head features; sensitivity to small residual motion not reported |
| Intuitive-physics study on V-JEPA | Surprise is the predictor's error on following frames; decisions by comparison of paired clips; short temporal context named as a limitation | Supports paired designs only |
| SAM 2 / 2.1 (video segmenter) | Code and checkpoints under Apache 2.0; about 39 to 224 million parameters; video segmentation is prompt-driven with a streaming memory; automatic mask generation for images only | Usable as region reader; exhaustive masklets must be constructed |
| SAM 3 / 3.1 | About 848 million parameters; a detector and a memory tracker on a shared encoder; prompts are short noun phrases, exemplars and visual prompts; returns masklets with identities for all matching instances; no promptless mode described; video cost grows with the number of objects; own licence; checkpoints behind an access request; the later release shares memory across objects for speed | Source of concept strata; behaviour on materials and on aerial imagery not documented |

Not checked, recalled only: input resolution and memory use of the segmenters; whether the
unconditional predictor weights of the video encoder are distributed in loadable form; every property
of point trackers, flow models and dense matchers; the terms of the non-standard licences; all prior
work named in Section 9. No model has been run.

A model that predicts tokens under a given camera motion, which would supply the expectation directly,
was not found among frozen video encoders. Camera-controlled video generators exist (recalled) but are
generators, not dense encoders, and were not examined.

## 9. Contribution and Nearest Prior Work

Prior work (all recalled): plane-plus-parallax decomposition of image motion; feature-metric
alignment and feature-metric losses for self-supervised depth; planar priors and plane sweeping in
multi-view stereo; lifting of segmenter masks onto Gaussian splats; verification of surface models by
consistency of rectified images; prediction error of a joint-embedding video model as a
violation-of-expectation signal (checked). Multi-view supervision of monocular depth is standard.

What is put forward here, with an honest grading:

1. **The compensated clip with stillness as the null**, as the common input of frozen video models.
   It turns any frame spacing into small motion, keeps the models in their training regime, and makes
   the residual motion equal to the prior's error times a known gain. The decomposition is classical;
   using the monocular prior as the reference surface and frozen video-native models as readers is,
   to the best of recalled knowledge, not.
2. **A calibrated latent reader**: the encoder's response to a known depth perturbation, measured in
   place, converts token differences into a gap with information and ancillaries, and makes abstention
   a measured quantity. This is the sharpest new element. Its worth depends entirely on the response
   curve of Section 10.
3. **The region reader with ownership and the interior-against-boundary test**, and mirror consistency
   as the resolution of the reflection problem where the reflected scene is reconstructed.
4. **Consensus and parallax as two contrasts of one surface-by-frame table**, which states precisely
   what motion is needed for: one mean field per region.

Items 1 and 2 together are a plausible core for a vision venue; item 3 is what a robotics venue would
value, because it yields a material attribute and a testable plane for water and glass. None of them
has empirical support yet.

## 10. What Would End the Idea, and the Cheapest Tests

1. **Faults are not properties of regions.** The variance components of Section 1.1 show that
   $\mu_m$ is a small share. *Test 1, no model, CPU:* on the scene with exact depth and cached
   monocular depth, lift a fixed over-segmentation of each frame through the exact depth to obtain
   oracle regions, and decompose the monocular error into the four terms, for region sizes from
   superpixels to connected components of the true fault mask.
2. **Residual motion of real faults is outside the readers' range.** *Test 2, no model, CPU:* from the
   same data compute $b_i \epsilon_M$ in pixels at the working resolution of each candidate. If typical
   faults move content by much less than a pixel at token resolution, or by more than a token at the
   resolution that fits in memory, the latent reader has no operating point.
3. **The encoder does not respond to residual parallax.** *Test 3, one frozen model:* response curve
   of the latent gap against injected depth perturbations on compensated clips built from exact depth:
   slope, linear range, noise floor, against a per-image encoder on the same clips.
4. **Masklets are not stable under the null.** *Test 4, one frozen model:* propagate masks through
   compensated clips built from exact depth; the outline jitter is $\sigma_c$ and the rate of lost or
   merged masklets is the coverage of the region reader on aerial imagery.
5. **Ownership cannot be decided**, so that boundary estimates are biased towards neighbours.
6. **The latent reader fails the honesty check** against the explicit one on their overlap.
7. **The blind class of Section 5 still contains most harmful faults.**

Tests 1 and 2 need only arrays that are already cached and decide whether tests 3 and 4 are worth a
GPU.

## 11. Not Known

- The variance components of monocular error by region and frame, and so whether the masklet is the
  right unit.
- The response of any frozen video encoder to small residual motion, its linear range, and whether it
  exceeds that of per-image features.
- Outline jitter and identity stability of video segmenters on compensated aerial clips; behaviour of
  concept prompts on water and glass seen from above.
- Whether boundary ownership can be assigned reliably from an erroneous depth map.
- How often the reflected scene is reconstructed, so that mirror consistency applies.
- Cost and memory of all three readers on full-resolution aerial frames.
- How the exact depth of the synthetic scene treats glass and water, and the frame spacing of that
  scene.
