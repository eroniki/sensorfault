# Does the Stabilised Video Stand Still? Judging a Monocular Depth Prior per Surface Entity with Frozen Video Models

Supporting derivations are in `docs/learned-indicator/`: the statistical theory (`theory.md`), the
definition of the evidence (`cues.md`), the surfel-space formulation (`surfel.md`), objections and
prior work (`objections-and-prior-work.md`), and the experiment design (`experiments.md`).

## Abstract

Depth-supervised Gaussian surfel reconstruction from posed aerial video trusts one monocular depth map
per frame. These maps fail on whole surfaces, and they fail the same way in every frame that sees the
surface, so a gate built on agreement between depth maps cannot see the failure. We judge the prior
with the video itself, through one construction. Resample the neighbouring frames into a reference
view through the depth map under test. If the depth is right, the result is a still image repeated.
Where it is wrong, the region slides along the epipolar direction in step with the camera, by the
depth error times a known gain. Every instrument then asks one question, whether a region of the
stabilised clip stands still, and three frozen instruments read the answer: optical flow or tracking
inside the region, the motion of the region's own outline, and the tokens of a video encoder. An
injected depth perturbation is exactly exchangeable with the unknown error, which gives any frozen
instrument, including a black-box encoder, a metric estimate, a self-measured sensitivity and an
abstention rule without labels. The unit of judgement is a surface entity followed through the clip
(a masklet from a frozen segmenter), with a three-parameter error per entity. In this model depth
consensus observes only what differs between frames and parallax observes only what the frames share,
so the two are complementary and the repeated fault is, by construction, the quantity parallax
estimates. An outline moves even when the interior has no texture, and disagreement between outline
and interior identifies reflecting and transparent surfaces. A light head on frozen features carries
the verdict to entities no instrument reaches. Nothing in this document has been measured; a staged
test with thresholds fixed in advance decides whether the approach holds.

## 1. Introduction

A static scene is observed by a posed aerial video flown in strips. An off-the-shelf monocular model
gives one depth map per frame, and a surface of 2D Gaussian surfels is fitted to the depth maps. Most
depth pixels are accurate up to a per-frame scale and offset. A minority, grouped in regions that
follow objects and materials (water, glass, texture-free roofs), are wrong by many standard
deviations.

The established defence is consistency between depth maps: a frame is compared with the surface the
other frames support. Every such *consensus gate* shares a blind spot. If the frames that observe a
surface make the same error, their residuals against each other are small and the error is fused. In
reliability theory a bias common to all redundant observations has no minimal detectable size.

Three observations shape the approach.

*The repeated fault belongs to the surface, not to a view.* If every frame misreads a lake the same
way, the object of the verdict is the lake. The natural unit is a surface entity followed through the
video, and a verdict on it is consistent across frames by construction.

*The video already contains an independent measurement.* With known poses, the image motion of a
surface fixes its depth, and this shares nothing with the monocular network.

*Frozen video models are trained on small motion; survey frames are far apart.* Stabilising the clip
by the depth under test removes the motion that the geometry explains and leaves only the residual,
which is small wherever the depth is roughly right. Flow networks, video encoders and mask
propagation then work in the regime they were trained in, whatever the frame spacing.

The objection to any such scheme is that a measurement good enough to judge the prior could replace
it. Section 4.6 answers quantitatively: where a surface is well witnessed the prior is indeed
redundant in its placement, and it is kept for its shape inside entities and for the entities that no
instrument reaches. The contribution of an indicator is confined to those, and evaluation is weighted
accordingly.

**Claim.** For a monocular depth prior on posed video, (i) the error on a surface entity separates into
a part all frames share and a part that differs between frames, of which depth consensus sees only the
second; (ii) the shared part is estimable per entity from the stabilised clip by frozen instruments,
with a self-measured error bar, including on untextured entities through their outline; and (iii) a
head on frozen features trained on these verdicts detects faults on unwitnessed entities better than
surface type alone predicts, and gating on it improves the surface where a consensus gate, a robust
loss and reconstruction without the prior do not.

**Scope of the learning.** Self-supervised on frozen models: a frozen video segmenter, a frozen flow
network or tracker, a frozen video encoder, a frozen image encoder and the frozen monocular depth
model. The learned parts are a light readout on encoder tokens, a light prior head, and a few variance
parameters per appearance stratum. No fault labels are needed for training. A small labelled set from
a scene with exact depth is optional, for testing assumptions and for calibration on unwitnessed
entities.

**Contributions.**

1. The depth-stabilised clip as a single test object with a sharp null, and the exchange identity
   that calibrates any frozen instrument on it by injected perturbations (Sections 3.2 and 4.3).
2. A surface-entity error model in which depth consensus and parallax observe complementary variance
   components, with the repetition coefficient as an intraclass correlation estimable without ground
   truth (Section 4.4).
3. Outline parallax for untextured entities, with boundary ownership, and the boundary-against-interior
   test that identifies reflection and transparency without labels (Section 4.2).
4. A gate that acts on the three placement parameters of an entity's monocular message and keeps its
   shape, so that detection and repair are one operation (Section 4.8).
5. The harm-detectability relation and a selective-labels treatment of transfer, both in entity units
   (Sections 4.6 and 4.7).
6. A staged decision protocol whose first stage needs no video model (Section 5).

## 2. Related Work

**Fault detection by redundancy.** Data snooping in geodesy and solution separation in navigation
integrity test an observation against the solution from the others, and state the limit we start
from: common-mode faults need independent redundancy.

**Plane plus parallax and layered motion.** Stabilising an image sequence by a reference plane and
reading depth from the residual motion is classical, as is the decomposition of video into motion
layers. The stabilised clip generalises the reference plane to a reference depth map under test, and
the layer test of Section 4.2 is a two-layer decision per entity.

**Feature-metric and photometric reprojection.** Warping by depth and pose and comparing images or
deep features is the loss of self-supervised depth and of feature-metric alignment. The comparison of
warped and real clips inside an encoder is of this family. What is added is the calibration by
injection, which turns a unitless distance into an estimate with a measured sensitivity.

**Monocular priors with multi-view geometry.** Scale-aligned monocular depth regularises neural
surfaces and splatting; structure-from-motion with monocular priors and uncertainties, and recent
reliability weighting of depth supervision in splatting, are the nearest systems. Segmentation-guided
multi-view stereo and planar splatting use regions and planes for reconstruction; the overlap with
entity-level judging must be examined before any novelty claim.

**Masks and uncertainty in radiance fields and splatting.** Heads on frozen features supervised by
the rendering residual predict distractor masks; their supervision is a residual against a model the
masked view helped to fit.

**Video foundation models.** Joint-embedding predictive video encoders provide spatiotemporal tokens
and have been used to measure violation of expectation; promptable segmenters provide regions with
identities through a video. Section 4.9 states what was verified about them and what the approach
cannot take from them.

**Learning under selection and from noisy labels.** Selective labels, propensity weighting and
likelihoods with known label noise are standard and are used without extension.

## 3. Background

### 3.1 Surface, messages and depth consensus

With $d_i(u)$ the true depth of pixel $u$ in frame $i$, the monocular depth is modelled as

```math
\hat d_i(u) = a_i\bigl(d_i(u) + \beta\, d_i(u)^2\bigr) + b_i + \varepsilon_i(u),
```

with per-frame scale and offset, a shared distance bias and nominal noise $\sigma_i(u)$. The rendered
depth of a pixel is $\hat D_i(u) = \sum_k w_{uk} d_k$ with normalised compositing weights $w_{uk}$ and
Jacobians $\mathbf j_{uk}$. With a pixel weight map $\omega_i(u) \in [0, 1]$, frame $i$ sends surfel $k$

```math
\Lambda_{ik}[\omega_i] = \sum_u \omega_i(u)\, w_{uk}\, \frac{a_i^2\, \mathbf j_{uk}^\top \mathbf j_{uk}}{\sigma_i^2(u)}, \qquad
\boldsymbol\eta_{ik}[\omega_i] = \sum_u \omega_i(u)\, w_{uk}\, \frac{a_i\, \mathbf j_{uk}^\top e_i(u)}{\sigma_i^2(u)} .
```

The information uses $w_{uk}$ and not its square: the lumped block-diagonal matrix majorises the
Hessian of the frame's loss, so a step $\Lambda^{-1}\boldsymbol\eta$ cannot overshoot. For a tested
frame $j$ and a block $\mathcal B_j$ of $B$ neighbouring frames, subtracting the block's messages and
rendering into frame $j$ gives the *cavity* depth; the **consensus gap** $r_C$ is the aligned
monocular depth minus the cavity depth.

### 3.2 The depth-stabilised clip

All depth quantities are in inverse depth $\xi = 1/d$ along the reference ray, in which image motion
is linear. Let $\hat\xi_{M}$ be the aligned monocular value of the reference frame $j$,
$\epsilon_M = \hat\xi_M - \xi$ its error, $\mathbf p_i(\xi)$ the projection into frame $i$ of the point
at inverse depth $\xi$ on the ray of $u$, and $\mathbf b_i = \partial \mathbf p_i / \partial \xi$ the
parallax gain (focal length times the baseline across the ray, for lateral motion). For a window of
frames $i$, resample each into the reference view through the depth under test:

```math
\tilde I_{i \to j}(u) = I_i\bigl( \mathbf p_i(\hat\xi_{M}(u)) \bigr) \approx I_j\bigl( u + \mathbf b_i\, \epsilon_{M}(u) \bigr) .
```

**Null.** If the depth of $u$ is correct and the surface is static, diffuse and visible, the
stabilised clip is a still image repeated.

**Under an error** the content at $u$ moves by $-\mathbf b_i\, \epsilon_M(u)$: along the epipolar
direction, proportional to the signed baseline and to the error. A faulty region slides back and
forth in step with the camera while everything correct stands still. The direction and the time
course are known, which separates a depth error from illumination change and from independent motion.

**Exchange identity.** Stabilising with $\hat\xi_M + \delta$ adds exactly $-\mathbf b_i \delta$ to the
residual motion. For any statistic $D$ computed from the stabilised clip by any instrument,

```math
D(\delta) = \mathcal D\bigl( \epsilon_M + \delta \bigr) :
```

the statistic under an injected perturbation is the instrument's response function translated by the
unknown error. The identity is exact in the geometry and needs no model of the instrument.

**What else moves.** Independently moving objects; view-dependent appearance (highlights, reflections,
transparency); occlusion, which duplicates the foreground next to depth edges; pose error, a smooth
field over the frame; exposure change. A band at rendered depth discontinuities is excluded and
reported as abstention; the rest is handled by the statistics of Section 4.

## 4. Method

### 4.1 Surface entities

A frozen segmenter gives regions in each frame. With known poses, regions are associated across
frames geometrically: masks are compared after stabilisation, where a correctly placed region does not
move, so association does not depend on a memory tracker and is not disturbed by frame spacing. The
result is a *masklet* $m$: masks $\mathcal M_{mj}$ of one surface entity in the frames $j$ that see it.
Nested masks give a tree of entities.

The working hypothesis is stated for the error of the prior, not for the surface. Monocular networks
tend to get the shape of an object roughly right and its placement wrong. The inverse depth of a plane
is affine in the pixel, $\xi(u) = \mathbf N^\top \mathbf k_u$, and the error on an entity is modelled
as affine in the same coordinates,

```math
\epsilon_{M,j}(u) = \mathbf k_u^\top \boldsymbol\epsilon_{mj} + \eta_j(u), \qquad u \in \mathcal M_{mj} ,
```

three coefficients per entity and frame (an offset and two tilts) and a remainder. The surface may be
curved. Under this model a few informative locations anywhere on the entity determine the error of
all its pixels, textured or not. Whether monocular error is in fact low-order on entities is the first
thing to measure (Section 5).

### 4.2 Reading the stabilised clip: explicit parallax

**Per pixel.** A frozen flow network or tracker on the stabilised clip reports the residual
displacement. With components $e_i$ along and $o_i$ across the epipolar line and instrument noise
$\sigma_i$, weighted least squares over the $L$ frames gives the **parallax gap** and its information,

```math
r_T = - \frac{\sum_i b_i\, e_i / \sigma_i^2}{\Lambda_T} = \epsilon_M - \epsilon_T, \qquad
\Lambda_T = \sum_{i} \frac{b_i^2}{\sigma_i^2}, \qquad \operatorname{Var}(\epsilon_T) = \Lambda_T^{-1} .
```

Of the $2L$ numbers of a track, one combination carries the depth and $2L - 1$ are free of both the
depth and the monocular error: $A_\perp = \sum_i o_i^2/\sigma_i^2$ and
$A_\parallel = \sum_i (e_i + b_i r_T)^2/\sigma_i^2$. These *ancillary* statistics decide abstention
(a mover, an occlusion, an instrument failure), estimate the instrument noise per stratum on the scene
itself, and test for the aperture problem along the flight direction. Depth consensus has no such
statistics.

**Per entity, interior.** Pooling the pixel gaps with the affine basis,

```math
\hat{\boldsymbol\epsilon}_{mj}^{\mathrm{int}} = \bigl(\Lambda_{mj}^{\mathrm{int}}\bigr)^{-1} \sum_{u \in \mathcal M_{mj}} \Lambda_T(u)\, \mathbf k_u\, r_T(u), \qquad
\Lambda_{mj}^{\mathrm{int}} = \omega \sum_{u \in \mathcal M_{mj}} \Lambda_T(u)\, \mathbf k_u \mathbf k_u^\top ,
```

with $\omega$ a deflation for spatial correlation.

**Per entity, outline.** In the stabilised clip the mask of a misplaced entity moves with its content.
Segmenting each stabilised frame, or prompting the segmenter with the reference mask and reading how
far the returned mask snaps, gives for a boundary point with outward normal $\hat{\boldsymbol\nu}(u)$ a
normal displacement

```math
c_i(u) = - \bigl( \hat{\boldsymbol\nu}(u)^\top \mathbf b_i \bigr)\, \mathbf k_u^\top \boldsymbol\epsilon_{mj} + \text{noise}, \qquad
\Lambda_{mj}^{\partial} = \sum_{i} \sum_{u \in \partial^{\mathrm{own}} \mathcal M_{mj}} \frac{(\hat{\boldsymbol\nu}^\top \mathbf b_i)^2}{\sigma_\partial^2}\, \mathbf k_u \mathbf k_u^\top .
```

This is parallax without interior texture: a lake or a pane has nothing to track inside, but its
outline moves. Four qualifications bound the claim.

- *Direction.* Only the component of the boundary normal along the epipolar direction counts. In a
  strip, leading and trailing edges carry the information and edges parallel to the flight line carry
  none. A closed outline has both.
- *Ownership.* A boundary between two regions belongs to the nearer one. For the farther region it is
  an occlusion edge that moves with the occluder: a lake fringed by trees would read canopy height.
  Segments are classified as attached (a marking or crease, usable by both sides), owned, or foreign
  and excluded. The classification needs only the depth order across the boundary.
- *Extent.* An entity cut by the frame border or without a closed outline has a rank-deficient
  $\Lambda^\partial$ and abstains in the missing directions. A horizontal water surface needs one
  parameter.
- *Systematic error.* With many boundary points the noise is small; the limit is whether the
  segmenter draws the boundary of a water surface or a pane at the same physical place in every view.
  This is testable without ground truth, by the repeatability of boundaries on correctly placed
  entities.

**Entity ancillaries and the layer test.** With more informative locations than unknowns an entity
has redundancy of its own. The scatter of interior gaps about the affine fit, $A_m$, measures the part
of the error that does not follow the entity. The residual of boundary displacements about the fit
flags mixed ownership. And the two estimates $\hat{\boldsymbol\epsilon}^{\partial}$ and
$\hat{\boldsymbol\epsilon}^{\mathrm{int}}$ use different image evidence: their difference is zero for
an opaque diffuse surface, while a static outline with an interior that lies *behind* is the signature
of a reflecting or transparent surface. The verdict about the surface is then taken from the outline,
and the interior is recorded as a material attribute and removed as a judge.

This settles, for bounded reflectors, the main failure of pixel-level parallax: the mirror image of a
static scene in still water or flat glass is a rigid scene, its tracks pass every per-track check and
triangulate consistently at the wrong depth. What remains unresolved is a reflector with no attached
boundary in view, an object moving along the strip at constant speed, and parallax borrowed by the
instrument from a neighbouring surface.

### 4.3 Reading the stabilised clip: a frozen video encoder

A frozen video encoder maps a clip to spatiotemporal tokens. Fed the stabilised clip, its tokens on an
entity are compared with the tokens of the still image repeated; let $D$ be their distance pooled over
the entity. The raw distance is not evidence: it has no units, it responds to warp artefacts and
exposure, and it is as blind as a tracker where the image has no structure. The exchange identity
turns it into an estimator.

**Latent sweep.** Since $D(\boldsymbol\delta) = \mathcal D(\boldsymbol\epsilon_{mj} + \boldsymbol\delta)$
for a perturbation $\boldsymbol\delta$ of the entity's three coefficients, and the response is minimal
at zero residual motion,

```math
\hat{\boldsymbol\epsilon}_{mj}^{\,\mathrm{lat}} = - \arg\min_{\boldsymbol\delta} D(\boldsymbol\delta) ,
```

a plane sweep in latent space with three parameters per entity. The estimate is metric because the
perturbation is.

**Information by injection.** The curvature of $D$ at its minimum, from the same sweep by finite
differences, is the measured sensitivity of the instrument on this entity. An entity on which an
injected fault of the size of interest does not move the statistic has zero information, and the
encoder abstains there by its own measurement. The noise of the minimiser is calibrated by the scatter
between forward and backward half-windows and by regression on the explicit gap where both exist.

**Readout by injection.** A light head $w_\theta$ on the tokens amortises the sweep to one pass. Its
training signal is exact and free: perturb the depth of random entities by known
$\boldsymbol\delta$ and require

```math
w_\theta\bigl( \mathrm{stab}(\hat\xi_M + \boldsymbol\delta) \bigr) - w_\theta\bigl( \mathrm{stab}(\hat\xi_M) \bigr) = \boldsymbol\delta .
```

An injected perturbation produces the same residual motion as a real error of that size, so there is
no domain gap in the signal. The readout is a learned witness and inherits every limit of the witness.

The explicit gap with its error bar is the calibrated reference. The latent witness is admitted only
if it agrees with the explicit one within stated errors where both are informative. Whether a frozen
video encoder responds to small residual motion better than per-image features is open; if it does
not, it adds nothing as a witness and remains a source of features for the prior.

### 4.4 What the two kinds of evidence see

After transport to a common frame, the entity coefficients are decomposed as

```math
\boldsymbol\epsilon_{mj} = \boldsymbol\mu_m + \boldsymbol\alpha_{mj}, \qquad
\boldsymbol\mu_m \mid s \sim (1 - \pi_s)\, \mathcal N(\mathbf 0, \Sigma_{0s}) + \pi_s\, p_{\mathrm{out}}, \qquad
\boldsymbol\alpha_{mj} \sim (\mathbf 0, \Sigma_{\alpha s}) ,
```

with $s$ the appearance stratum. $\boldsymbol\mu_m$ is the error every frame makes on this surface,
$\boldsymbol\alpha_{mj}$ what frame $j$ adds, and $\eta$ what does not follow the entity. The fault
state $F_m$ of an entity is the outlier component of $\boldsymbol\mu_m$.

The cavity depth of frame $j$ is fused from the other frames, so on entity $m$ its error is
$\boldsymbol\mu_m$ plus an average of other frames' $\boldsymbol\alpha$. The consensus gap is therefore
$\boldsymbol\alpha_{mj}$ minus that average, with no $\boldsymbol\mu_m$ in it.

**Depth consensus observes the per-frame term. Only parallax observes the surface term.** The two are
complementary and together complete: consensus tests contrasts between frames, a witness on the
entity tests their mean, and one witness per entity suffices.

The **repetition coefficient**, the share of a frame's error that the cavity repeats, is the
intraclass correlation

```math
\varrho = \frac{\operatorname{tr} \Sigma_\mu}{\operatorname{tr} \Sigma_\mu + \operatorname{tr} \Sigma_\alpha}
```

for per-frame terms independent beyond the block; its dependence on the block size $B$ is the temporal
correlation of $\boldsymbol\alpha$ and its plateau is the surface term. It is estimated without ground
truth from the two variance components, or as the errors-in-variables slope
$1 - \hat\varrho = \widehat{\operatorname{Cov}}(r_C, r_T) / (\widehat{\operatorname{Var}}(r_T) - \overline{\Lambda_T^{-1}})$.
A plateau above zero is the fault no consensus method can see, measured on real video.

| Quantity | Identified from |
| --- | --- |
| instrument noise per stratum | track ancillaries; half-window scatter |
| witness information per entity | geometry and noise scale; injected response for the encoder |
| per-frame component $\Sigma_\alpha$ | scatter of entity coefficients across frames |
| law of the surface term, fault prevalence | deconvolution over the entities of a stratum with the known kernel |
| share of error that follows entities | $A_m$ |
| second layer (reflection, transparency) | outline against interior |
| repetition $\varrho(B)$ | variance components, or slope against the consensus gap |
| anything on an entity with rank-deficient information and no feature overlap | not identified |

The exchangeable units are entities, not pixels. The sample size of every estimate is the number of
entities in a stratum, which is the honest count for regional faults, and the number of tests is the
number of entities rather than of all connected regions.

### 4.5 Verdict and prior

Combining the readings and the frames,

```math
\bar{\boldsymbol\epsilon}_m = \Bigl( \sum_j \Lambda_{mj} \Bigr)^{-1} \sum_j \Lambda_{mj}\, \hat{\boldsymbol\epsilon}_{mj}, \qquad
\operatorname{logit} \zeta_m = \operatorname{logit} \pi(\boldsymbol\phi_m)
+ \log \frac{\bigl( p_{\mathrm{out}} * \mathcal N(\mathbf 0, V_m) \bigr)(\bar{\boldsymbol\epsilon}_m)}{\mathcal N\bigl( \bar{\boldsymbol\epsilon}_m;\ \mathbf 0,\ \Sigma_{0s} + V_m \bigr)} ,
```

with $V_m = \Lambda_m^{-1} + \Sigma_{\alpha s} / J_m$ for $J_m$ frames. The verdict is one number per
surface entity and the same in every frame. As the information goes to zero the likelihood ratio goes
to one and $\zeta_m \to \pi(\boldsymbol\phi_m)$. Offset, tilt and, through $A_m$, shape faults are
reported separately. The pixel indicator is the entity verdict refined by the pixel's own evidence,

```math
\operatorname{logit} \zeta_j(u) = \operatorname{logit} \Pr(f \mid \zeta_m)
+ \log \frac{g^{1}}{g^{0}}\bigl(r_T(u) - \mathbf k_u^\top \hat{\boldsymbol\epsilon}_{mj}\bigr)
+ \log \frac{h^{1}}{h^{0}}\bigl(r_C(u)\bigr) ,
```

with the consensus term carrying the per-frame part. The verdict is placed at the coarsest node of the
entity tree whose children are homogeneous, so a fault on part of an entity is not diluted.

**Prior head.** $\pi(\boldsymbol\phi_m)$ is a light head on frozen features pooled over the entity's
masks and frames. It is fitted by the marginal likelihood of the entity gaps, which is learning from
labels whose noise is known per unit through $V_m$; the label information
$\iota_m = \mathbb E(\zeta_m - \pi_m)^2 / (\pi_m(1 - \pi_m))$ weights each entity automatically. Its
input must not contain the residual motion. Tokens of the stabilised clip do, and are excluded. Tokens
of the raw clip and of single frames contain appearance and motion signatures of surfaces (rippling,
flicker, layered motion) but not the depth under test, and are admissible. The remaining risk is
learning the witness's bias rather than its noise, which is why verdicts on entities with a detected
second layer come from the outline only. The check is that held-out likelihood does not improve when
the head is also given the entity's own witness statistics. Cross-fitting is by ground tile.

### 4.6 Why the prior is kept: harm and detectability

Let $\varsigma$ be the share of an entity's placement information that the witness supplies. If both
are fused, a placement fault $\Delta$ of the prior moves the surface by its *harm*
$h = \Delta\,(1 - \varsigma)$, and the noncentrality of the comparison is $\Delta^2 \Lambda_M \varsigma$.
Hence

```math
\delta^2 = h^2\, \Lambda_M\, \frac{\varsigma}{(1 - \varsigma)^2} :
```

for a fault of given harm, detectability rises steeply with the witness share and vanishes with it.
Direct testing pays only where the two sources are comparably informative.

| Regime | Entities | Witness | Prior |
| --- | --- | --- | --- |
| replace | textured, or with a well-owned outline | fixes placement; supplies labels | redundant in placement; kept for shape within the entity |
| judge | short windows, far range, small entities, partly owned outlines | tests the prior directly | matters and can be tested |
| blind | featureless and unbounded in view, foreign outlines only, shape faults | none | the only depth evidence |

With assumed survey numbers a tracked pixel is of the order of a hundred times more informative than
monocular depth, and an outline of a few thousand pixels fixes an offset to centimetres. Outline
parallax therefore moves water and flat roofs with visible boundaries out of the blind regime, which
is a gain for the witness and also removes them from where the prior matters. The contribution of an
indicator is measured by the harm-weighted share of faulty surface that stays blind, a number that
can be computed before any head exists. Evaluation weights each faulty unit by $1 - \varsigma$ or
reports the regimes separately.

### 4.7 Transfer to unwitnessed entities

The prior head is trained where $\iota_m$ is large and used where it is small, and selection is on
witnessability, which the features also encode. What is learned transfers if (S1) among entities that
look alike to the encoder the witnessed ones are faulty as often as the unwitnessed ones, and (S2)
unwitnessed entities have witnessed look-alikes.

- *A label-free map of support.* A propensity head predicts from the same features whether an entity
  is witnessed. It splits unwitnessed entities into an interpolation zone and an extrapolation zone,
  and the effective sample size of the importance weights states how many witnessed entities stand
  behind a prediction.
- *Three output states.* Trusted, faulty, or unsupported. In the extrapolation zone the output is
  fixed by the geometry of the feature space and nothing in the training signal, and is reported as
  unsupported.
- *Counterfactual abstention.* Witnessed entities are made to play unwitnessed ones with their verdict
  kept: withhold feature clusters, shorten the window, hide the outline, blur the interior. This gives
  accuracy against distance from the training support on any posed video. That the curve predicts true
  accuracy on unwitnessed entities is a hypothesis to test on a scene with exact depth.
- *Consensus as examiner.* The consensus gap exists without texture. Among unwitnessed entities its
  outlier rate must rise with the head's score. What stays beyond every label-free check is a
  surface-level fault on an unwitnessed entity; there the claim rests on transfer and labelled
  evaluation.

Material strata can come from noun-phrase prompts of a promptable segmenter ("water", "glass roof").
As covariates of prevalence and noise, wrong strata cost efficiency only. Where they switch the model
they must be confirmed by the layer test, which is geometric.

### 4.8 Use in reconstruction

Surfels do the bookkeeping. Entities are lifted onto surfels with the compositing weights, so an
entity is a set of surfels and a three-parameter direction in their joint state.

1. **Placement gate.** The monocular message of an entity is split into its component along the three
   placement parameters and the remainder. The gate $1 - \zeta_m$ acts on the placement component
   only; the shape of the monocular depth within the entity is kept. Detection and repair are then one
   operation: a faulty entity keeps its monocular shape and takes its placement from the witness, or,
   if unwitnessed, is left free in placement and reported as such.
2. **Witness messages.** The entity gap is a message on the same three parameters with information
   $\Lambda_m$; pixel gaps are sparse depth messages without scale, offset or distance bias.
3. **Anchored alignment.** Per-frame scale, offset and the distance bias are fitted against the
   witness with the indicator's weights, so no nuisance parameter is fitted to flagged entities.
4. **Sample splitting.** Witness evidence is divided by ground tile into a fitting set and a judging
   set, so the judge is not part of the judged surface.
5. **Four fits of one experiment.** Witness only; monocular only; both fused without a gate; both with
   the gate. "The prior adds nothing" is the outcome that the first matches the last on the
   harm-weighted metric.

The indicator is held fixed while the surface is fitted. Unsupported entities are reported as
unconstrained in placement.

### 4.9 Frozen models: what the approach can and cannot take from them

Checked against the models' own papers and repositories; everything else is to be verified before a
model is fixed.

- *Video encoder.* The V-JEPA 2 family tokenises clips into tubelets of two frames by sixteen by
  sixteen pixels at 256 or 384 pixels and 16 to 64 frames; a later member is trained for dense
  features. Its action-conditioned predictor is conditioned on robot end-effector actions. **No
  released predictor is conditioned on camera pose**, so the expectation cannot be formed inside the
  model and must be formed by the explicit warp of Section 3.2. No aerial training data is named.
- *Video segmenter.* The SAM 2 family propagates prompted masks through video and produces automatic
  masks for single images only. SAM 3 accepts noun-phrase and exemplar prompts and returns masklets
  with identities, has no promptless mode, and its cost grows with the number of objects. **Masklets
  therefore need seeding and association**, which is why association is done geometrically here. The
  family is known to be weak on glass and mirrors.
- *Not verified.* Memory and speed on a small GPU, licence terms of gated weights, behaviour of any of
  these models on nadir aerial imagery, water or glazing, and at survey frame spacing.

## 5. Experiments

### 5.1 Evaluation targets

On data with exact depth: the true inverse-depth error of every monocular depth pixel; the true-fault
mask (relative error above three robust standard deviations inside a connected region of at least 2%
of the frame, a band at depth discontinuities ignored); surface classes; and for each entity its true
coefficients $\boldsymbol\epsilon_{mj}$, their decomposition into $\boldsymbol\mu_m$ and
$\boldsymbol\alpha_{mj}$, and its witness share $\varsigma$.

### 5.2 The deciding measurement, in two stages

**Stage A: is the entity the right unit, and is anything left to judge?** Needs monocular depth
passes, automatic single-image masks and association by exact geometry. No tracker, no video encoder,
no head.

- A1. Share of true fault energy explained by an affine function of inverse depth per mask, against
  ground tiles and superpixels of equal count.
- A2. Variance components: $\Sigma_\mu$ against $\Sigma_\alpha$, and $\varrho(B)$ from true errors.
- A3. With exact correspondences and exact outlines in place of instruments, the witness share of each
  entity, and the harm-weighted share of faulty surface in the blind regime.
- A4. Replacement: on blind entities, depth error of the prior under a perfect placement gate against
  interpolation of witnessed depth, a multi-view stereo pipeline and a feed-forward multi-view model.

**Stage B: do frozen instruments deliver it?**

- B1. Entity formation: do frozen segmenters return water, glazing and roofs as regions in nadir
  frames, with boundaries repeatable to a few pixels across views.
- B2. Witness accuracy and calibration: explicit gap, outline gap and latent sweep against true
  coefficients; stated against actual error; the layer test on water and glass.
- B3. Does the video encoder add to the witness: injected-response curvature of video tokens against
  per-image features on the same stabilised clips.
- B4. Copy check and transfer: the prior head on unwitnessed entities against the true mask, per
  propensity decile, against a constant, the per-stratum fault rate, the consensus gap, and a linear
  probe trained on true masks of disjoint ground tiles.

**The approach is abandoned if any of:**

- K1, wrong unit: masks explain no more fault energy than tiles of equal count (A1).
- K2, nothing repeated: $\varrho(B)$ falls to zero within a few frames, so a longer leave-out block
  already cures the problem (A2).
- K3, nothing left to judge: less than 10% of harm-weighted faulty surface is in the blind or judge
  regime (A3).
- K4, the prior is not worth keeping: on blind entities of every non-planar surface class, some
  alternative without monocular depth matches the prior under a perfect gate (A4).
- K5, no entities: frozen segmenters do not form repeatable regions on the surfaces of interest (B1).
- K6, unsound witness: entity verdicts on witnessed entities have precision or recall below 0.90, or
  stated errors are off by more than a factor of two (B2).
- K7, no transfer: on unwitnessed entities the head does not exceed the per-stratum fault rate, or is
  near chance within every surface class (B4).

**It is supported if all of:** masks explain at least half of fault energy and clearly more than
tiles; $\varrho$ has a plateau above 0.3; at least 30% of harm-weighted faulty surface lies outside the
replace regime; the witness reaches 0.95 precision and recall with calibrated errors; and the head on
unwitnessed entities has AUC of at least 0.75 overall and at least 0.65 inside two or more surface
classes. A failure of B3 alone removes the encoder as a witness and leaves the rest standing. The
numerical values are conventions fixed before the measurement.

### 5.3 Follow-on experiments

| | Question | Hypothesis |
| --- | --- | --- |
| F1 | Frame density | Stabilisation makes the instruments insensitive to frame spacing; information per frame follows the baseline |
| F2 | The judge regime | On dense video at small baseline the entity gap detects faults that the pixel gap cannot |
| F3 | Outline parallax | Owned outlines place untextured entities to within their stated error; foreign outlines are detected by the boundary residual |
| F4 | Layer test | Outline against interior separates reflecting and transparent entities from misplaced ones |
| F5 | Transfer | The counterfactual-abstention curve predicts true accuracy on unwitnessed entities |
| F6 | Repeated faults | The indicator detects surface-level faults that the consensus gap, a robust loss and frame removal miss |
| F7 | Reconstruction | The placement gate beats witness only, monocular only and ungated fusion on the harm-weighted metric, and changes clean scenes by less than 2% |
| F8 | Real aerial video | The label-free quantities are stable on real data and agree with laser depth where it exists |

### 5.4 Baselines and bounds

No gate; robust depth loss; removal of whole frames; the consensus gate; the depth model's own
confidence; disagreement between several depth models and instability under input transformations;
the pixel-level parallax gap without entities; raw feature-metric reprojection error without
calibration; the rule "unwitnessed, so distrust the prior"; the per-stratum fault rate; a linear probe
on frozen features at equal label budget; residual-supervised masks; zero-shot depth completion from
witnessed depth; a multi-view stereo pipeline; a feed-forward multi-view model with its own
confidence. Bounds: an oracle gate with true masks, and exact correspondences and outlines in place of
the instruments.

### 5.5 Protocol

Evaluation quantities are written to a store that only the scoring code reads. Thresholds, strata and
exclusion bands are chosen by held-out likelihood of the gaps or on labelled scenes disjoint from
evaluation. Splits are by ground tile or scene. Intervals are cluster-bootstrap intervals over ground
tiles, with the number of entities and of distinct fault regions reported beside each. A rendered
scene has perfect poses and synthetic materials: images are degraded, poses perturbed, and verdicts
repeated on real scenes, which decide where the two disagree.

## 6. Limitations

- **Surface-level faults on unwitnessed entities** are beyond every label-free check.
- **The entity may be the wrong unit.** If monocular error cuts across segments, entity verdicts
  dilute it and the pixel formulation is what remains.
- **Everything rests on frozen segmenters forming repeatable regions** on water, glazing and roofs in
  nadir views, which is exactly where such models are known to be weak.
- **Reflectors without an attached boundary in view, constant-speed motion along the strip and
  borrowed parallax** pass the witness's own checks.
- **Outline ownership is inferred from a depth map that may be wrong**, and curved occluders slide.
- **The video encoder may add nothing as a witness.** It is as blind as a tracker without image
  structure, and no released predictor accepts a camera pose.
- **Where a surface is witnessed the prior is redundant in placement.** The value of the method is
  bounded by the harm-weighted blind share, which may be small on survey data.
- **Shape faults within an entity** are detected only where interior evidence exists.
- **Frozen models are trained mostly on ground-level video.**

## 7. Notation

| Symbol | Meaning |
| --- | --- |
| $\hat d_i(u)$; $a_i, b_i, \beta$; $\sigma_i(u)$ | monocular depth; per-frame scale and offset, shared distance bias; nominal noise |
| $w_{uk}$, $\mathbf j_{uk}$; $\Lambda_{ik}[\omega]$, $\boldsymbol\eta_{ik}[\omega]$ | compositing weight and depth Jacobian; lumped message under pixel weights |
| $\mathcal B_j$, $B$; $r_C$ | block left out with frame $j$, its size; consensus gap |
| $\xi$, $\hat\xi_M$, $\epsilon_M$ | inverse depth, aligned monocular value, its error |
| $\mathbf p_i(\xi)$, $\mathbf b_i$, $b_i$ | projection into frame $i$, parallax gain and its length |
| $\tilde I_{i \to j}$ | frame $i$ stabilised into the reference view by the depth under test |
| $e_i$, $o_i$; $A_\perp$, $A_\parallel$ | residual displacement along and across the epipolar line; track ancillaries |
| $r_T$, $\Lambda_T$ | pixel parallax gap and its information |
| $m$, $\mathcal M_{mj}$, $\mathbf k_u$ | surface entity (masklet), its mask in frame $j$, affine basis in inverse depth |
| $\boldsymbol\epsilon_{mj}$, $\boldsymbol\mu_m$, $\boldsymbol\alpha_{mj}$, $\eta$ | entity error coefficients, surface term, per-frame term, remainder |
| $\Lambda_{mj}^{\mathrm{int}}$, $\Lambda_{mj}^{\partial}$, $A_m$ | interior and outline information, entity scatter |
| $D(\boldsymbol\delta)$, $w_\theta$ | encoder statistic under an injected perturbation, readout head |
| $\Sigma_\mu$, $\Sigma_\alpha$, $\varrho$ | variance components, repetition coefficient |
| $F_m$, $\zeta_m$, $\zeta_j(u)$, $\pi(\boldsymbol\phi_m)$ | entity fault state, entity verdict, pixel indicator, prior head |
| $V_m$, $\iota_m$ | variance of the entity gap, label information |
| $\varsigma$, $h$, $\delta^2$ | witness share, harm of a fault, noncentrality of its test |
